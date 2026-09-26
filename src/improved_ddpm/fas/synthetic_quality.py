"""Quality control for E6 synthetic spoof-amplitude samples.

The online gate uses thresholds calibrated once from source-train only. It
never looks at source-dev or target-test data, so rejecting a generated sample
cannot leak benchmark labels into training. Face/identity checks remain
offline diagnostics and are not hard gates for an amplitude generator.
"""

from __future__ import annotations

from collections import deque
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import torch
from improved_ddpm.fas.frequency_utils import FrequencyUtils


@dataclass(frozen=True)
class SyntheticQualityConfig:
    enabled: bool = True
    max_retries: int = 3
    normalized_amplitude_clip: float = 3.0
    max_clip_ratio: float = 0.01
    spectral_calibration_file: str | None = None
    rolling_window_batches: int = 100
    min_rolling_acceptance_rate: float = 0.80
    stop_on_nonfinite: bool = True

    @classmethod
    def from_mapping(cls, value: Mapping[str, object] | None) -> "SyntheticQualityConfig":
        section = dict(value or {})
        fields = cls.__dataclass_fields__
        unknown = sorted(set(section) - set(fields))
        if unknown:
            raise ValueError(f"Unknown stage6.quality_control keys: {unknown}")
        return cls(**section).validate()

    def validate(self) -> "SyntheticQualityConfig":
        if int(self.max_retries) < 0:
            raise ValueError("quality_control.max_retries must be >= 0")
        if float(self.normalized_amplitude_clip) <= 0:
            raise ValueError("quality_control.normalized_amplitude_clip must be > 0")
        if not 0.0 <= float(self.max_clip_ratio) <= 1.0:
            raise ValueError("quality_control.max_clip_ratio must be in [0,1]")
        if int(self.rolling_window_batches) <= 0:
            raise ValueError("quality_control.rolling_window_batches must be > 0")
        if not 0.0 <= float(self.min_rolling_acceptance_rate) <= 1.0:
            raise ValueError("quality_control.min_rolling_acceptance_rate must be in [0,1]")
        return self

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SpectralGateCalibration:
    """Source-train-only thresholds for the online synthetic spectral gate."""

    version: int
    policy: str
    radial_bins: int
    live_center: list[float]
    attack_center: list[float]
    attack_distance_scale: float
    min_attack_margin: float
    max_attack_distance_ratio: float
    min_high_frequency_ratio: float
    max_high_frequency_ratio: float
    live_count: int
    attack_count: int
    source_manifest: str
    train_manifest_fingerprint: str
    lower_quantile: float
    upper_quantile: float

    def validate(self) -> "SpectralGateCalibration":
        if self.policy != "source_train_only":
            raise ValueError(f"Unsafe spectral calibration policy: {self.policy!r}")
        if self.radial_bins <= 1:
            raise ValueError("spectral calibration radial_bins must be > 1")
        if len(self.live_center) != self.radial_bins or len(self.attack_center) != self.radial_bins:
            raise ValueError("spectral calibration center length does not match radial_bins")
        if self.attack_distance_scale <= 0 or self.max_attack_distance_ratio <= 0:
            raise ValueError("spectral calibration distances must be positive")
        if self.live_count < 2 or self.attack_count < 2:
            raise ValueError("spectral calibration requires at least two samples per class")
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "SpectralGateCalibration":
        return cls(**dict(payload)).validate()


def calibrate_spectral_gate(
    live_images: torch.Tensor,
    attack_images: torch.Tensor,
    *,
    radial_bins: int = 32,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
    source_manifest: str = "",
    train_manifest_fingerprint: str = "",
) -> SpectralGateCalibration:
    """Fit gate thresholds from source-train samples only.

    Thresholds are empirical quantiles of genuine source attack spectra.  No
    development or target sample participates in this calibration.
    """
    if live_images.ndim != 4 or attack_images.ndim != 4:
        raise ValueError("live_images and attack_images must be BCHW tensors")
    if live_images.shape[0] < 2 or attack_images.shape[0] < 2:
        raise ValueError("spectral gate calibration requires at least two images per class")
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("spectral quantiles must satisfy 0 <= lower < upper <= 1")
    live_psd = radial_power_spectrum(live_images, bins=radial_bins).float()
    attack_psd = radial_power_spectrum(attack_images, bins=radial_bins).float()
    live_center = live_psd.mean(dim=0)
    attack_center = attack_psd.mean(dim=0)
    distance_live = (attack_psd - live_center).abs().mean(dim=1)
    distance_attack = (attack_psd - attack_center).abs().mean(dim=1)
    scale = distance_attack.median().clamp_min(1e-8)
    margin = distance_live - distance_attack
    distance_ratio = distance_attack / scale
    hf = high_frequency_energy_ratio(attack_images).float()
    return SpectralGateCalibration(
        version=1,
        policy="source_train_only",
        radial_bins=int(radial_bins),
        live_center=live_center.cpu().tolist(),
        attack_center=attack_center.cpu().tolist(),
        attack_distance_scale=float(scale.item()),
        min_attack_margin=float(torch.quantile(margin, lower_quantile).item()),
        max_attack_distance_ratio=float(torch.quantile(distance_ratio, upper_quantile).item()),
        min_high_frequency_ratio=float(torch.quantile(hf, lower_quantile).item()),
        max_high_frequency_ratio=float(torch.quantile(hf, upper_quantile).item()),
        live_count=int(live_images.shape[0]),
        attack_count=int(attack_images.shape[0]),
        source_manifest=str(source_manifest),
        train_manifest_fingerprint=str(train_manifest_fingerprint),
        lower_quantile=float(lower_quantile),
        upper_quantile=float(upper_quantile),
    ).validate()


def calibrate_spectral_gate_from_records(
    records,
    *,
    data_root=None,
    dataset_roots=None,
    image_size: int = 224,
    radial_bins: int = 32,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
    source_manifest: str = "",
    batch_size: int = 32,
) -> SpectralGateCalibration:
    """Streaming source-train calibration that keeps only compact PSD vectors."""
    from improved_ddpm.data.manifest import (
        resolve_manifest_record_path,
        sort_records,
    )
    from improved_ddpm.fas.frequency_normalization import (
        image_to_model_tensor,
        manifest_fingerprint,
    )

    train_rows = [row for row in sort_records(records) if row.split == "train"]
    if batch_size <= 0:
        raise ValueError("spectral calibration batch_size must be positive")
    live_desc, attack_desc, attack_hf = [], [], []
    pending_images, pending_labels = [], []

    def flush_pending():
        if not pending_images:
            return
        images = torch.stack(pending_images)
        descriptors = radial_power_spectrum(images, bins=radial_bins).cpu()
        hf_values = high_frequency_energy_ratio(images).cpu()
        for descriptor, hf_value, label in zip(descriptors, hf_values, pending_labels):
            if label == 0:
                live_desc.append(descriptor)
            elif label == 1:
                attack_desc.append(descriptor)
                attack_hf.append(hf_value)
        pending_images.clear()
        pending_labels.clear()

    with torch.no_grad():
        for row in train_rows:
            path = resolve_manifest_record_path(row, data_root, dataset_roots)
            pending_images.append(image_to_model_tensor(path, image_size))
            pending_labels.append(int(row.label))
            if len(pending_images) >= batch_size:
                flush_pending()
        flush_pending()
    if len(live_desc) < 2 or len(attack_desc) < 2:
        raise RuntimeError(
            f"Need at least two source-train samples per class; live={len(live_desc)}, attack={len(attack_desc)}"
        )
    live_psd = torch.stack(live_desc).float()
    attack_psd = torch.stack(attack_desc).float()
    hf = torch.stack(attack_hf).float()
    live_center = live_psd.mean(dim=0)
    attack_center = attack_psd.mean(dim=0)
    distance_live = (attack_psd - live_center).abs().mean(dim=1)
    distance_attack = (attack_psd - attack_center).abs().mean(dim=1)
    scale = distance_attack.median().clamp_min(1e-8)
    return SpectralGateCalibration(
        version=1,
        policy="source_train_only",
        radial_bins=int(radial_bins),
        live_center=live_center.tolist(),
        attack_center=attack_center.tolist(),
        attack_distance_scale=float(scale.item()),
        min_attack_margin=float(torch.quantile(distance_live - distance_attack, lower_quantile).item()),
        max_attack_distance_ratio=float(torch.quantile(distance_attack / scale, upper_quantile).item()),
        min_high_frequency_ratio=float(torch.quantile(hf, lower_quantile).item()),
        max_high_frequency_ratio=float(torch.quantile(hf, upper_quantile).item()),
        live_count=len(live_desc),
        attack_count=len(attack_desc),
        source_manifest=str(source_manifest),
        train_manifest_fingerprint=manifest_fingerprint(train_rows),
        lower_quantile=float(lower_quantile),
        upper_quantile=float(upper_quantile),
    ).validate()


def save_spectral_gate_calibration(calibration: SpectralGateCalibration, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(calibration.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=str(path.parent), delete=False) as handle:
        handle.write(content)
        temporary = handle.name
    os.replace(temporary, path)
    return path


def load_spectral_gate_calibration(path) -> SpectralGateCalibration:
    with Path(path).open("r", encoding="utf-8") as handle:
        return SpectralGateCalibration.from_dict(json.load(handle))


def _per_sample_mean(value: torch.Tensor) -> torch.Tensor:
    return value.flatten(1).mean(dim=1)


def high_frequency_energy_ratio(images: torch.Tensor, radius_fraction: float = 0.35) -> torch.Tensor:
    """Return high-frequency energy / total FFT energy for every image."""
    amplitude, _ = FrequencyUtils.fft_decompose(images)
    height, width = amplitude.shape[-2:]
    yy = torch.linspace(-1.0, 1.0, height, device=images.device, dtype=images.dtype)
    xx = torch.linspace(-1.0, 1.0, width, device=images.device, dtype=images.dtype)
    radius = torch.sqrt(yy[:, None].square() + xx[None, :].square())
    mask = radius >= float(radius_fraction)
    energy = amplitude.square()
    high = energy[..., mask].sum(dim=-1)
    total = energy.flatten(2).sum(dim=-1).clamp_min(1e-12)
    return (high / total).mean(dim=1)


def radial_power_spectrum(images: torch.Tensor, bins: int = 32) -> torch.Tensor:
    """Compact, deterministic PSD descriptor used by the offline evaluator."""
    amplitude, _ = FrequencyUtils.fft_decompose(images)
    power = amplitude.square().mean(dim=1)
    height, width = power.shape[-2:]
    yy = torch.linspace(-1.0, 1.0, height, device=images.device, dtype=images.dtype)
    xx = torch.linspace(-1.0, 1.0, width, device=images.device, dtype=images.dtype)
    radius = torch.sqrt(yy[:, None].square() + xx[None, :].square())
    edges = torch.linspace(0.0, float(radius.max()), bins + 1, device=images.device)
    descriptors = []
    for index in range(bins):
        mask = (radius >= edges[index]) & (radius < edges[index + 1])
        if index == bins - 1:
            mask = (radius >= edges[index]) & (radius <= edges[index + 1])
        # Small test images can have more radial bins than distinct discrete
        # radii. In that case an annulus may contain no FFT coefficient and a
        # direct ``mean`` over the empty selection would produce NaN. The
        # weighted formulation below returns zero for an empty annulus while
        # remaining exactly equivalent to the original mean for non-empty
        # annuli. Production 224x224 inputs with 32 bins are unaffected.
        weights = mask.to(dtype=power.dtype)
        numerator = (power * weights).sum(dim=(-2, -1))
        denominator = weights.sum().clamp_min(1.0)
        descriptors.append(numerator / denominator)
    result = torch.stack(descriptors, dim=1)
    return torch.log1p(result)


def rbf_mmd_unbiased(x: torch.Tensor, y: torch.Tensor) -> float:
    """Unbiased RBF MMD with a pooled median-distance bandwidth."""
    x = x.float()
    y = y.float()
    if x.shape[0] < 2 or y.shape[0] < 2:
        return float("nan")
    pooled = torch.cat([x, y], dim=0)
    distances = torch.cdist(pooled, pooled).square()
    positive = distances[distances > 0]
    bandwidth = torch.median(positive) if positive.numel() else torch.tensor(1.0, device=x.device)
    bandwidth = bandwidth.clamp_min(1e-8)

    def kernel(a, b):
        return torch.exp(-torch.cdist(a, b).square() / (2.0 * bandwidth))

    kxx = kernel(x, x)
    kyy = kernel(y, y)
    kxy = kernel(x, y)
    nx, ny = x.shape[0], y.shape[0]
    xx = (kxx.sum() - kxx.diagonal().sum()) / (nx * (nx - 1))
    yy = (kyy.sum() - kyy.diagonal().sum()) / (ny * (ny - 1))
    return float((xx + yy - 2.0 * kxy.mean()).item())


class SyntheticQualityController:
    """Validate candidates, track acceptance, and fail closed to real attacks."""

    def __init__(
        self,
        config: SyntheticQualityConfig,
        calibration: SpectralGateCalibration | None = None,
    ):
        self.config = config.validate()
        if calibration is None and config.spectral_calibration_file:
            calibration = load_spectral_gate_calibration(config.spectral_calibration_file)
        self.calibration = calibration.validate() if calibration is not None else None
        if self.config.enabled and self.calibration is None:
            raise ValueError(
                "quality_control.spectral_calibration_file is required when the source-calibrated gate is enabled"
            )
        self._rolling = deque(maxlen=int(config.rolling_window_batches))
        self.disabled = False
        self.disable_reason = ""

    @property
    def rolling_acceptance_rate(self) -> float:
        if not self._rolling:
            return 1.0
        requested = sum(item[0] for item in self._rolling)
        accepted = sum(item[1] for item in self._rolling)
        return float(accepted / requested) if requested else 1.0

    def validate_candidates(
        self,
        synthetic: torch.Tensor,
        parent_live: torch.Tensor,
        source_live: torch.Tensor,
        source_attack: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        if synthetic.shape != parent_live.shape:
            raise RuntimeError(
                f"Synthetic shape {tuple(synthetic.shape)} does not match parent-live shape "
                f"{tuple(parent_live.shape)}"
            )
        if synthetic.ndim != 4 or synthetic.shape[1] != 3:
            raise RuntimeError(f"Expected synthetic BCHW RGB tensor, got {tuple(synthetic.shape)}")

        finite = torch.isfinite(synthetic).flatten(1).all(dim=1)
        if not bool(finite.all()) and self.config.stop_on_nonfinite:
            raise FloatingPointError("E6 generated NaN/Inf synthetic pixels; DDPM run stopped")
        safe = torch.nan_to_num(synthetic, nan=0.0, posinf=1.0, neginf=-1.0)
        clip_ratio = _per_sample_mean(((safe < -1.0) | (safe > 1.0)).float())
        assert self.calibration is not None or not self.config.enabled
        calibration = self.calibration
        if calibration is None:
            # Gate-disabled diagnostic mode: calculate descriptors but accept all
            # finite, range-safe samples.
            radial_bins = 32
            live_center = radial_power_spectrum(source_live, bins=radial_bins).mean(dim=0)
            attack_center = radial_power_spectrum(source_attack, bins=radial_bins).mean(dim=0)
            reference_attack_distance = torch.tensor(1.0, device=safe.device)
            min_attack_margin = -float("inf")
            max_attack_distance_ratio = float("inf")
            min_hf, max_hf = -float("inf"), float("inf")
        else:
            radial_bins = calibration.radial_bins
            live_center = torch.tensor(calibration.live_center, device=safe.device, dtype=safe.dtype)
            attack_center = torch.tensor(calibration.attack_center, device=safe.device, dtype=safe.dtype)
            reference_attack_distance = torch.tensor(
                calibration.attack_distance_scale, device=safe.device, dtype=safe.dtype
            )
            min_attack_margin = calibration.min_attack_margin
            max_attack_distance_ratio = calibration.max_attack_distance_ratio
            min_hf = calibration.min_high_frequency_ratio
            max_hf = calibration.max_high_frequency_ratio
        fake_psd = radial_power_spectrum(safe, bins=radial_bins)
        distance_live = (fake_psd - live_center).abs().mean(dim=1)
        distance_attack = (fake_psd - attack_center).abs().mean(dim=1)
        attack_margin = distance_live - distance_attack
        attack_distance_ratio = distance_attack / reference_attack_distance.clamp_min(1e-8)
        hf_ratio = high_frequency_energy_ratio(safe)

        checks = {
            "finite": finite,
            "clip": clip_ratio <= float(self.config.max_clip_ratio),
            "attack_margin": attack_margin >= float(min_attack_margin),
            "attack_distance": attack_distance_ratio <= float(max_attack_distance_ratio),
            "high_frequency": (hf_ratio >= float(min_hf)) & (hf_ratio <= float(max_hf)),
        }
        accepted = torch.ones_like(finite, dtype=torch.bool)
        for value in checks.values():
            accepted &= value

        report = {
            "generated": int(synthetic.shape[0]),
            "accepted": int(accepted.sum().item()),
            "rejected": int((~accepted).sum().item()),
            "clip_ratio_sum": float(clip_ratio.sum().item()),
            "attack_margin_sum": float(attack_margin.sum().item()),
            "attack_distance_ratio_sum": float(attack_distance_ratio.sum().item()),
            "high_frequency_ratio_sum": float(hf_ratio.sum().item()),
        }
        for name, value in checks.items():
            report[f"rejected_{name}"] = int((~value).sum().item())
        return accepted, report

    def record_batch(self, requested: int, accepted: int) -> dict:
        requested = int(requested)
        accepted = int(accepted)
        if requested > 0:
            self._rolling.append((requested, accepted))
        rate = self.rolling_acceptance_rate
        window_ready = len(self._rolling) >= int(self.config.rolling_window_batches)
        just_disabled = False
        if (
            self.config.enabled
            and not self.disabled
            and window_ready
            and rate < float(self.config.min_rolling_acceptance_rate)
        ):
            self.disabled = True
            self.disable_reason = (
                f"rolling acceptance {rate:.3f} < "
                f"{self.config.min_rolling_acceptance_rate:.3f} over "
                f"{self.config.rolling_window_batches} batches"
            )
            just_disabled = True
        return {
            "rolling_acceptance_rate": rate,
            "quality_disabled": int(self.disabled),
            "quality_just_disabled": int(just_disabled),
            "quality_disable_reason": self.disable_reason,
        }


def empty_quality_counters() -> dict:
    return {
        "quality_requested": 0,
        "quality_generated": 0,
        "quality_accepted": 0,
        "quality_rejected": 0,
        "quality_retries": 0,
        "quality_fallback_real": 0,
        "quality_clip_ratio_sum": 0.0,
        "quality_attack_margin_sum": 0.0,
        "quality_attack_distance_ratio_sum": 0.0,
        "quality_high_frequency_ratio_sum": 0.0,
        "quality_rejected_finite": 0,
        "quality_rejected_clip": 0,
        "quality_rejected_attack_margin": 0,
        "quality_rejected_attack_distance": 0,
        "quality_rejected_high_frequency": 0,
        "rolling_acceptance_rate": 1.0,
        "quality_disabled": 0,
        "quality_just_disabled": 0,
        "quality_disable_reason": "",
    }


def merge_quality_counters(target: dict, update: Mapping[str, object]) -> None:
    additive = {
        key for key in target
        if key.startswith("quality_") and key not in {
            "quality_disabled", "quality_just_disabled", "quality_disable_reason"
        }
    }
    for key in additive:
        if key in update:
            target[key] += update[key]
    for key in (
        "rolling_acceptance_rate",
        "quality_disabled",
        "quality_just_disabled",
        "quality_disable_reason",
    ):
        if key in update:
            target[key] = update[key]


def finalize_quality_counters(counters: Mapping[str, object]) -> dict:
    output = dict(counters)
    generated = max(int(output.get("quality_generated", 0)), 1)
    requested = int(output.get("quality_requested", 0))
    accepted = int(output.get("quality_accepted", 0))
    output["quality_acceptance_rate"] = float(accepted / requested) if requested else 1.0
    for source, destination in (
        ("quality_clip_ratio_sum", "quality_clip_ratio_mean"),
        ("quality_attack_margin_sum", "quality_attack_margin_mean"),
        ("quality_attack_distance_ratio_sum", "quality_attack_distance_ratio_mean"),
        ("quality_high_frequency_ratio_sum", "quality_high_frequency_ratio_mean"),
    ):
        output[destination] = float(output.get(source, 0.0)) / generated
        output.pop(source, None)
    return output
