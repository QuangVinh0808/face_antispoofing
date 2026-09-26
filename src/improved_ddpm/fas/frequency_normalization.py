"""Train-only FFT log-amplitude statistics and cache helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image

from improved_ddpm.data.manifest import ManifestRecord, resolve_manifest_record_path, resolve_record_path, sort_records
from improved_ddpm.fas.frequency_utils import FrequencyUtils


@dataclass(frozen=True)
class FFTNormalizationStats:
    version: int
    split: str
    mode: str
    mean: List[float]
    std: List[float]
    value_count: int
    image_count: int
    channels: int
    image_size: int
    rgb_range: str
    fft_norm: str
    manifest_fingerprint: str
    source_manifest: str = ""

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["fft"] = {
            "operation": "torch.fft.fft2 + torch.fft.fftshift",
            "amplitude": "abs",
            "phase": "angle",
            "log_amplitude": "log1p",
            "norm": self.fft_norm,
        }
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "FFTNormalizationStats":
        return cls(
            version=int(payload.get("version", 1)),
            split=str(payload["split"]),
            mode=str(payload["mode"]),
            mean=[float(value) for value in payload["mean"]],
            std=[float(value) for value in payload["std"]],
            value_count=int(payload["value_count"]),
            image_count=int(payload["image_count"]),
            channels=int(payload["channels"]),
            image_size=int(payload["image_size"]),
            rgb_range=str(payload["rgb_range"]),
            fft_norm=str(payload.get("fft_norm", payload.get("fft", {}).get("norm", "ortho"))),
            manifest_fingerprint=str(payload["manifest_fingerprint"]),
            source_manifest=str(payload.get("source_manifest", "")),
        )

    def tensors(self, reference: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = torch.tensor(self.mean, dtype=reference.dtype, device=reference.device)
        std = torch.tensor(self.std, dtype=reference.dtype, device=reference.device)
        if self.mode == "per_channel":
            mean = mean.view(1, -1, 1, 1)
            std = std.view(1, -1, 1, 1)
        else:
            mean = mean.reshape(1, 1, 1, 1)
            std = std.reshape(1, 1, 1, 1)
        return mean, std


class _RunningMoments:
    def __init__(self, dimensions: int) -> None:
        self.count = 0
        self.mean = np.zeros(dimensions, dtype=np.float64)
        self.m2 = np.zeros(dimensions, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        """Merge values shaped ``(dimensions, observations)``."""
        if values.ndim != 2 or values.shape[1] == 0:
            return
        chunk_count = values.shape[1]
        chunk_mean = values.mean(axis=1, dtype=np.float64)
        centered = values - chunk_mean[:, None]
        chunk_m2 = np.sum(centered * centered, axis=1, dtype=np.float64)
        if self.count == 0:
            self.count = chunk_count
            self.mean = chunk_mean
            self.m2 = chunk_m2
            return
        delta = chunk_mean - self.mean
        combined = self.count + chunk_count
        self.mean = self.mean + delta * (chunk_count / combined)
        self.m2 = self.m2 + chunk_m2 + delta * delta * self.count * chunk_count / combined
        self.count = combined

    def finalize(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.count < 2:
            raise RuntimeError("Need at least two FFT values to calculate standard deviation")
        variance = self.m2 / self.count
        return self.mean, np.sqrt(np.maximum(variance, 1e-12))


def image_to_model_tensor(path: Union[str, Path], image_size: int) -> torch.Tensor:
    """Match ManifestFASDataset preprocessing: RGB resize, tensor, [-1, 1]."""
    with Image.open(path) as opened:
        image = opened.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor * 2.0 - 1.0


def manifest_fingerprint(records: Sequence[ManifestRecord]) -> str:
    digest = hashlib.sha256()
    for row in sort_records(records):
        digest.update(
            f"{row.dataset}|{row.protocol}|{row.split}|{row.subject_id}|{row.video_id}|"
            f"{row.frame_id}|{row.path}|{row.label}\n".encode("utf-8")
        )
    return digest.hexdigest()


def compute_fft_normalization(
    records: Iterable[ManifestRecord],
    *,
    data_root: Optional[Union[str, Path]] = None,
    dataset_roots: Optional[Mapping[str, Union[str, Path]]] = None,
    image_size: int = 224,
    mode: str = "global",
    source_manifest: str = "",
    fft_norm: str = "ortho",
    labels: Optional[Sequence[int]] = None,
) -> FFTNormalizationStats:
    """Calculate mean/std strictly from selected ``split=train`` rows.

    ``labels=None`` preserves the classifier normalization over every source
    training image.  ``labels=(1,)`` creates the attack-only normalization used
    by the spoof-amplitude DDPM.  Development and target rows are never read.
    """

    mode = mode.strip().lower()
    if mode not in {"global", "per_channel"}:
        raise ValueError("mode must be global or per_channel")
    allowed_labels = None if labels is None else {int(label) for label in labels}
    train_rows = [
        row for row in sort_records(records)
        if row.split == "train"
        and (allowed_labels is None or int(row.label) in allowed_labels)
    ]
    if not train_rows:
        suffix = "" if allowed_labels is None else f" with labels={sorted(allowed_labels)}"
        raise RuntimeError(f"Cannot compute FFT normalization: no train rows{suffix}")

    dimensions = 1 if mode == "global" else 3
    moments = _RunningMoments(dimensions)
    for row in train_rows:
        path = resolve_manifest_record_path(row, data_root, dataset_roots)
        if not path.is_file():
            raise FileNotFoundError(f"Missing train image for normalization: {path}")
        image = image_to_model_tensor(path, image_size).unsqueeze(0)
        amplitude, _ = FrequencyUtils.fft_decompose(image, norm=fft_norm)
        log_amp = FrequencyUtils.log_amplitude(amplitude).squeeze(0).cpu().numpy().astype(np.float64)
        if mode == "global":
            moments.update(log_amp.reshape(1, -1))
        else:
            moments.update(log_amp.reshape(log_amp.shape[0], -1))

    mean, std = moments.finalize()
    value_count = moments.count * dimensions
    return FFTNormalizationStats(
        version=1,
        split="train",
        mode=mode,
        mean=mean.tolist(),
        std=std.tolist(),
        value_count=int(value_count),
        image_count=len(train_rows),
        channels=3,
        image_size=int(image_size),
        rgb_range="[-1,1]",
        fft_norm=fft_norm,
        manifest_fingerprint=manifest_fingerprint(train_rows),
        source_manifest=source_manifest,
    )


def save_fft_normalization(
    stats: FFTNormalizationStats,
    path: Union[str, Path],
    *,
    allow_changes: bool = False,
) -> Path:
    path = Path(path)
    content = (json.dumps(stats.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if path.exists() and path.read_bytes() == content:
        return path
    if path.exists() and not allow_changes:
        raise RuntimeError(f"Normalization file changed: {path}; use --force intentionally")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(path.parent), delete=False) as tmp:
        tmp.write(content)
        temp_name = tmp.name
    os.replace(temp_name, path)
    return path


def load_fft_normalization(path: Union[str, Path]) -> FFTNormalizationStats:
    with Path(path).open("r", encoding="utf-8") as handle:
        return FFTNormalizationStats.from_dict(json.load(handle))


def normalize_log_amplitude(
    log_amplitude: torch.Tensor,
    stats: Optional[FFTNormalizationStats],
) -> torch.Tensor:
    """Apply fixed train statistics. ``None`` preserves legacy raw log-amplitude."""
    if stats is None:
        return log_amplitude
    mean, std = stats.tensors(log_amplitude)
    return (log_amplitude - mean) / std.clamp_min(1e-6)


def denormalize_log_amplitude(
    normalized: torch.Tensor,
    stats: FFTNormalizationStats,
) -> torch.Tensor:
    mean, std = stats.tensors(normalized)
    return normalized * std + mean


def cache_log_amplitudes(
    records: Iterable[ManifestRecord],
    *,
    data_root: Optional[Union[str, Path]],
    output_dir: Union[str, Path],
    image_size: int,
    dataset_roots: Optional[Mapping[str, Union[str, Path]]] = None,
    allow_changes: bool = False,
) -> Path:
    """Cache deterministic raw log-amplitude arrays; random augmentation is excluded."""

    import csv
    import io
    import re

    output_dir = Path(output_dir)
    index_rows = []
    for row in sort_records(records):
        image_path = resolve_manifest_record_path(row, data_root, dataset_roots)
        image = image_to_model_tensor(image_path, image_size).unsqueeze(0)
        amplitude, _ = FrequencyUtils.fft_decompose(image)
        log_amp = FrequencyUtils.log_amplitude(amplitude).squeeze(0).cpu().numpy().astype(np.float32)
        safe_video = re.sub(r"[^A-Za-z0-9_.-]+", "_", row.video_id) or "unknown"
        safe_frame = re.sub(r"[^A-Za-z0-9_.-]+", "_", row.frame_id or image_path.stem) or "frame"
        cache_path = output_dir / row.split / safe_video / f"{safe_frame}.npy"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists() and not allow_changes:
            existing = np.load(cache_path)
            if not np.array_equal(existing, log_amp):
                raise RuntimeError(f"FFT cache changed: {cache_path}; use --force intentionally")
        else:
            np.save(cache_path, log_amp, allow_pickle=False)
        index_rows.append(
            {
                "split": row.split,
                "video_id": row.video_id,
                "frame_id": row.frame_id,
                "image_path": str(image_path),
                "log_amplitude_path": str(cache_path.resolve()),
            }
        )

    buffer = io.StringIO(newline="")
    columns = ("split", "video_id", "frame_id", "image_path", "log_amplitude_path")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(index_rows)
    index_path = output_dir / "cache_index.csv"
    content = buffer.getvalue().encode("utf-8")
    if index_path.exists() and index_path.read_bytes() != content and not allow_changes:
        raise RuntimeError(f"FFT cache index changed: {index_path}; use --force intentionally")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_bytes(content)
    return index_path
