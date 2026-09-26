"""
train.py
==========
Script huan luyen E0-E6 tren fixed manifest MSU-MFSD/SiW-Mv2.

Rap noi toan bo cac module da xay dung:
    1. data/manifest_dataset.py     -> fixed train/dev manifests
    2. fas/patch_utils.py            -> Random Crop scale [0.2,1.0] -> Patch RGB
    3. fas/frequency_utils.py        -> FFT decompose / IFFT reconstruct
    4. fas/amplitude_ddpm.py         -> Improved DDPM hoc sinh Amplitude gia mao
       (diffusion/schedule, variance, losses, diffusion.py o duoi)
    5. models/dual_branch.py         -> DualBranchFASNetwork (RGB + Amplitude)
    6. metrics/fas_metrics.py        -> APCER / BPCER / ACER / AUC

LUONG DU LIEU E6 MOI EPOCH (train classifier):
    Batch can bang 8 LIVE + 8 ATTACK that tu DataLoader
        |
        +--> nap DDPM da pretrain rieng tren source-train attack va FREEZE
        |
        +--> thay 4/8 ATTACK that bang 4 ATTACK synthetic dat quality gate
        |                 (khong noi them mau; batch classifier van la 16)
        |
        +--> Patch RGB = RandomCrop(scale=[0.2,1.0]) + resize  (Cap nhat 1)
        |
        v
    FFT -> Full_Amplitude, Patch_Amplitude (cho nhanh M_amp)
        |
        v
    DualBranchFASNetwork forward CA 2 luong (Full, Patch)
        |
        v
    L_cls = L_face + L_patch  +  Contrastive Loss + Center Loss   (Cap nhat 2)
        |
        v
    backward -> chi update Dual-Branch Classifier; DDPM khong thay doi

LUONG VALIDATION (cuoi moi epoch):
    CHI dung Full RGB (KHONG Patch, KHONG Diffusion) -> DualBranchFASNetwork.predict()
        |
        v
    evaluate_fas_metrics() -> APCER / BPCER / ACER / AUC               (Cap nhat 3)
        |
        v
    Luu checkpoint NEU ACER tot hon (model "best") + luon luu "last"

CHAY:
    python -m improved_ddpm.scripts.train --config configs/ms_e0_e6.yaml
"""

import argparse
import csv
import os
import random
import sys
import time

import numpy as np

# Tu dong them "src/" vao sys.path - KHONG can pip install -e . hay PYTHONPATH.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
_PROJECT_ROOT = os.path.abspath(os.path.join(_SRC_DIR, ".."))
_DEFAULT_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "configs", "ms_e0_e6.yaml")

import yaml
import torch
from torch.utils.data import DataLoader

from improved_ddpm.data.manifest_dataset import ManifestFASDataset
from improved_ddpm.data.manifest import read_manifest
from improved_ddpm.data.samplers import ClassBalancedBatchSampler, DomainClassBalancedBatchSampler
from improved_ddpm.fas.frequency_utils import FrequencyUtils
from improved_ddpm.fas.frequency_normalization import (
    FFTNormalizationStats,
    load_fft_normalization,
    normalize_log_amplitude,
    manifest_fingerprint,
)
from improved_ddpm.fas.amplitude_ddpm import ImprovedAmplitudeDDPM
from improved_ddpm.fas.synthetic_quality import (
    SyntheticQualityConfig,
    SyntheticQualityController,
    empty_quality_counters,
    finalize_quality_counters,
    merge_quality_counters,
)
from improved_ddpm.fas.patch_utils import random_crop_resize_batch
from improved_ddpm.models.factory import build_fas_model
from improved_ddpm.experiments.ablation import resolve_ablation_spec
from improved_ddpm.experiments.stage6 import (
    CONDITION_NAMES,
    canonical_attack_condition,
    classifier_mix_counts,
    resolve_stage6_spec,
    resolve_effective_stage6_spec,
)
from improved_ddpm.metrics.fas_metrics import (
    evaluate_fas_metrics,
    format_fas_metrics,
    select_threshold,
)
from improved_ddpm.evaluation.protocol import (
    attach_video_results,
    evaluate_rows,
    json_safe_metrics,
    tune_development_policy,
    write_json,
)


# ============================================================================
# Config
# ============================================================================
DEFAULT_CONFIG = {
    "experiment": {
        # Stage 5 canonical ablation: E0..E6. E6 preserves the previous full pipeline.
        "variant": "E6",
        "overrides": {},
    },
    "image_size": 224,
    "batch_size": 16,
    "amp_channels": 3,
    "data": {
        # Fixed manifests are mandatory to prevent subject leakage.
        "manifest_dir": None,
        "data_root": None,         # prefix neu cot path trong manifest la relative
        "dataset_roots": {},       # per-dataset roots (M/S)
        "num_workers": 4,
        "seed": 42,
        # Stage 6 can switch this to class_balanced so every batch contains
        # both live and attack samples.
        "sampler": "shuffle",
    },
    "training": {
        # Step-based budget is comparable when source datasets have very
        # different numbers of batches per epoch. Set max_steps to null to use
        # the legacy epoch loop instead.
        "max_steps": 3000,
        "eval_every_steps": 300,
        "min_steps": 1200,
        "early_stopping_patience": 4,
        "early_stopping_min_delta": 0.001,
        "epochs": 30,
        "log_every": 10,
        "checkpoint_dir": "outputs/checkpoints",
        "log_dir": "outputs/logs",
    },
    "frequency": {
        # Giai doan 3: mean/std chi duoc tinh tu split=train.
        "normalization_file": None,
        "require_train_stats": False,
    },
    "evaluation": {
        # Giai doan 4: threshold va aggregation CHI duoc chon tren development.
        "threshold_method": "eer",
        "aggregation_candidates": ["mean", "median", "trimmed_mean", "topk_mean"],
        "trim_ratio": 0.1,
        "topk_ratio": 0.25,
        "tpr_fprs": [0.01, 0.001],
    },
    "diffusion": {
        "base_channels": 64,
        "channel_mults": [1, 2, 4, 8],
        "attention_resolutions": [16, 8],
        "num_heads": 4,
        "T": 1000,
        "s": 0.008,
        "lam_vlb": 0.001,
        "lr": 2e-4,
        "sampling_k": 20,
        "sampling_k_candidates": [20, 50, 100],
        "grad_clip_norm": 1.0,
        # E6 uses a separately pretrained, frozen DDPM and attack-only stats.
        "normalization_file": None,
        "pretrained_checkpoint": None,
        "freeze_pretrained": True,
        "pretrain_steps": 3000,
    },
    "classifier": {
        "feat_dim": 256,
        "num_classes": 2,
        "alpha_center": 0.2,
        "lambda_contrastive": 1.0,
        "lr": 1e-4,
        "pretrained_backbone": True,
    },
    "patch": {
        "scale_min": 0.2,
        "scale_max": 1.0,
    },
    "stage6": {
        "enabled": False,
        "variant": "D2",
        "ddpm_mode": "spoof_amplitude",
        # Fraction of the fixed attack quota replaced by synthetic attacks.
        "augmentation_ratio": 0.5,
        "warmup_steps": 0,
        "class_balanced": True,
        # Canonical synthesis: A_syn=A_DDPM (no A_mix interpolation).
        "synthetic_strength": 1.0,
        # Source-only online safety gate. Rejected slots are retried, then
        # filled by real attacks so batch size and class prior never change.
        "quality_control": {
            "enabled": True,
            "max_retries": 3,
            "normalized_amplitude_clip": 3.0,
            "max_clip_ratio": 0.01,
            "spectral_calibration_file": None,
            "rolling_window_batches": 100,
            "min_rolling_acceptance_rate": 0.80,
            "stop_on_nonfinite": True,
        },
    },
}


def _deep_merge(default: dict, override: dict) -> dict:
    """Merge config tu YAML vao DEFAULT_CONFIG (chi can khai bao phan muon doi trong YAML)."""
    out = dict(default)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(config_path: str = None) -> dict:
    if config_path and os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        print(f"[config] Da doc config tu: {config_path}")
        return _deep_merge(DEFAULT_CONFIG, user_cfg)
    print("[config] Khong tim thay file config -> dung DEFAULT_CONFIG.")
    return DEFAULT_CONFIG


# ============================================================================
# FFT normalization (train-only, fixed for train/dev/test)
# ============================================================================
def load_frequency_stats(cfg: dict):
    frequency_cfg = cfg.get("frequency", {})
    path = frequency_cfg.get("normalization_file")
    required = bool(frequency_cfg.get("require_train_stats", False))
    if not path:
        if required:
            raise RuntimeError(
                "frequency.normalization_file is required. Run "
                "python -m improved_ddpm.scripts.prepare_cross_dataset_ms first."
            )
        print("[frequency] WARNING: no fixed train normalization; using legacy behavior.")
        return None
    if not os.path.isabs(path):
        path = os.path.join(_PROJECT_ROOT, path)
    stats = load_fft_normalization(path)
    if stats.split != "train":
        raise RuntimeError(f"Normalization must be train-only, got split={stats.split!r}")
    if stats.image_size != int(cfg["image_size"]):
        raise RuntimeError(
            f"Normalization image_size={stats.image_size} does not match config image_size={cfg['image_size']}"
        )
    print(
        f"[frequency] Loaded train-only {stats.mode} normalization from {path}: "
        f"mean={stats.mean}, std={stats.std}, images={stats.image_count}"
    )
    return stats


def load_ddpm_frequency_stats(cfg: dict):
    """Load attack-only source-train statistics used solely by E6 DDPM."""
    path = cfg.get("diffusion", {}).get("normalization_file")
    if not path:
        raise RuntimeError(
            "diffusion.normalization_file is required for E6 and must point to "
            "normalization_attack_<size>.json"
        )
    if not os.path.isabs(path):
        path = os.path.join(_PROJECT_ROOT, path)
    stats = load_fft_normalization(path)
    if stats.split != "train" or stats.image_size != int(cfg["image_size"]):
        raise RuntimeError("Invalid DDPM attack-only normalization metadata")
    print(
        f"[diffusion] Loaded attack-only normalization from {path}: "
        f"mean={stats.mean}, std={stats.std}, attacks={stats.image_count}"
    )
    return stats


# ============================================================================
# Khoi tao model
# ============================================================================
def set_global_seed(seed: int) -> None:
    """Make Stage-5 variants comparable by fixing all available RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms can be slower; benchmark reproducibility is preferred.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_models(cfg: dict, device: str, fft_stats=None, ddpm_fft_stats=None):
    """Build only components enabled by the selected Stage-5 variant."""
    diff_cfg = cfg["diffusion"]
    cls_cfg = cfg["classifier"]
    spec = resolve_ablation_spec(cfg)
    stage6_spec = resolve_effective_stage6_spec(cfg, spec.variant)
    if stage6_spec is not None and bool(spec.use_ddpm) != bool(stage6_spec.use_ddpm):
        raise RuntimeError(
            f"Stage-6 {stage6_spec.variant} expects experiment_variant="
            f"{stage6_spec.experiment_variant}, but resolved Stage-5 spec is {spec.variant}"
        )

    amp_ddpm = None
    ddpm_optimizer = None
    if spec.use_ddpm:
        amp_ddpm = ImprovedAmplitudeDDPM(
            in_channels=cfg["amp_channels"],
            image_size=cfg["image_size"],
            base_channels=diff_cfg["base_channels"],
            channel_mults=tuple(diff_cfg["channel_mults"]),
            attention_resolutions=tuple(diff_cfg["attention_resolutions"]),
            num_heads=diff_cfg["num_heads"],
            T=diff_cfg["T"],
            s=diff_cfg["s"],
            lam_vlb=diff_cfg["lam_vlb"],
            device=device,
            normalization_stats=ddpm_fft_stats,
            num_conditions=3 if (stage6_spec is not None and stage6_spec.conditional) else 0,
        )
        if not bool(diff_cfg.get("freeze_pretrained", True)):
            ddpm_optimizer = torch.optim.AdamW(
                amp_ddpm.unet.parameters(), lr=diff_cfg["lr"]
            )

    fas_net, resolved_spec = build_fas_model(cfg)
    fas_net = fas_net.to(device)
    fas_optimizer = torch.optim.AdamW(fas_net.parameters(), lr=cls_cfg["lr"])

    return amp_ddpm, ddpm_optimizer, fas_net, fas_optimizer, resolved_spec


def prepare_amplitude(rgb: torch.Tensor, fas_net, fft_stats=None):
    """Skip FFT entirely for RGB-only E0 to keep compute comparisons honest."""
    if not fas_net.use_amp:
        return None
    amplitude, _ = FrequencyUtils.fft_decompose(rgb)
    return normalize_log_amplitude(FrequencyUtils.log_amplitude(amplitude), fft_stats)


# ============================================================================
# Stage-6 augmentation helpers
# ============================================================================
# NOTE: the former live-amplitude `diffusion_augment_step` has been removed.
# Canonical E6 must go through `stage6_diffusion_augment_step`, whose default
# D2 mode trains the DDPM on source-domain spoof amplitudes. D1 live-amplitude
# behavior remains available only as an explicit legacy control in stage6.py.


def _metadata_list(metadata, key: str, batch_size: int):
    if metadata is None or key not in metadata:
        return [""] * batch_size
    value = metadata[key]
    if torch.is_tensor(value):
        return [item.item() for item in value]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value] * batch_size


def _sample_exact_indices(indices: torch.Tensor, count: int) -> torch.Tensor:
    """Sample an exact number of indices using the globally seeded Torch RNG."""
    count = int(count)
    if indices.numel() == 0 or count <= 0:
        return indices[:0]
    if count >= indices.numel():
        return indices
    order = torch.randperm(indices.numel(), device=indices.device)
    return indices[order[:count]]


def _compose_stage6_classifier_batch(rgb, labels, img_augmented):
    """Replace real attacks with synthetic attacks without growing the batch."""
    live_indices = torch.nonzero(labels == 0, as_tuple=False).flatten()
    attack_indices = torch.nonzero(labels == 1, as_tuple=False).flatten()
    synthetic_count = 0 if img_augmented is None else int(img_augmented.shape[0])
    if synthetic_count > attack_indices.numel():
        raise RuntimeError(
            f"Synthetic attacks ({synthetic_count}) exceed attack quota ({attack_indices.numel()})"
        )

    real_attack_count = int(attack_indices.numel()) - synthetic_count
    chosen_attacks = _sample_exact_indices(attack_indices, real_attack_count)
    parts = [rgb[live_indices], rgb[chosen_attacks]]
    if img_augmented is not None:
        parts.append(img_augmented)
    classifier_rgb = torch.cat(parts, dim=0)
    classifier_labels = torch.cat(
        [
            torch.zeros(live_indices.numel(), dtype=torch.long, device=labels.device),
            torch.ones(real_attack_count + synthetic_count, dtype=torch.long, device=labels.device),
        ],
        dim=0,
    )
    if classifier_rgb.shape[0] != rgb.shape[0]:
        raise RuntimeError(
            f"Stage-6 changed classifier batch size: {rgb.shape[0]} -> {classifier_rgb.shape[0]}"
        )
    return classifier_rgb, classifier_labels, {
        "classifier_live": int(live_indices.numel()),
        "classifier_real_attack": real_attack_count,
        "classifier_synthetic_attack": synthetic_count,
        "classifier_total": int(classifier_labels.numel()),
    }


def _empty_stage6_info(warmup: bool = False):
    info = {
        "ddpm_updated": 0,
        "synthetic_count": 0,
        "warmup": int(bool(warmup)),
    }
    info.update(empty_quality_counters())
    return info


def stage6_diffusion_augment_step(
    amp_ddpm,
    ddpm_optimizer,
    rgb,
    labels,
    metadata,
    image_size,
    amp_channels,
    sampling_k,
    stage6_spec,
    grad_clip_norm=1.0,
    generate_synthetic=True,
    quality_controller=None,
    update_ddpm=False,
):
    """Generate synthetic spoof samples from a pretrained DDPM.

    D1 learns the live-amplitude distribution (legacy control).
    D2 learns spoof amplitudes and transfers generated amplitude to live phase.
    D3 does the same as D2 but conditions the UNet on print/replay/other.

    Canonical E6 uses ``update_ddpm=False``: the generator is pretrained
    separately and frozen so its sampling distribution is stable throughout
    classifier training. ``update_ddpm=True`` remains only for explicit legacy
    controls. Synthetic samples are generated under no_grad and labelled attack.
    """
    live_indices = torch.nonzero(labels == 0, as_tuple=False).flatten()
    attack_indices = torch.nonzero(labels == 1, as_tuple=False).flatten()

    if stage6_spec.ddpm_mode == "live_amplitude":
        train_indices = live_indices
        train_condition = None
    elif stage6_spec.ddpm_mode in {"spoof_amplitude", "conditional_spoof"}:
        train_indices = attack_indices
        train_condition = None
    else:
        raise ValueError(f"Unsupported Stage-6 DDPM mode: {stage6_spec.ddpm_mode}")

    if train_indices.numel() == 0:
        # Class-balanced Stage-6 batches should prevent this. Keep a safe fallback
        # so custom datasets fail gracefully rather than producing NaN.
        return None, None, _empty_stage6_info(warmup=not generate_synthetic)

    train_rgb = rgb[train_indices]
    if stage6_spec.conditional:
        attack_types = _metadata_list(metadata, "attack_type", rgb.shape[0])
        condition_values = [
            canonical_attack_condition(attack_types[int(i)])
            for i in train_indices.detach().cpu().tolist()
        ]
        train_condition = torch.tensor(condition_values, device=rgb.device, dtype=torch.long)

    ddpm_loss_value = None
    ddpm_updated = 0
    if update_ddpm:
        if ddpm_optimizer is None:
            raise RuntimeError("DDPM optimizer is required when update_ddpm=True")
        ddpm_losses = amp_ddpm.training_step(train_rgb, condition=train_condition)
        if not torch.isfinite(ddpm_losses["loss"]):
            raise FloatingPointError(
                f"Non-finite DDPM loss in {stage6_spec.variant}/{stage6_spec.ddpm_mode}: "
                f"{ddpm_losses['loss'].item()}"
            )
        ddpm_optimizer.zero_grad(set_to_none=True)
        ddpm_losses["loss"].backward()
        if grad_clip_norm is not None and float(grad_clip_norm) > 0:
            torch.nn.utils.clip_grad_norm_(
                amp_ddpm.unet.parameters(), max_norm=float(grad_clip_norm)
            )
        ddpm_optimizer.step()
        ddpm_loss_value = float(ddpm_losses["loss"].detach().item())
        ddpm_updated = 1

    mix = classifier_mix_counts(
        int(live_indices.numel()),
        int(attack_indices.numel()),
        stage6_spec.augmentation_ratio,
        augmentation_active=bool(generate_synthetic),
    )
    chosen_live = _sample_exact_indices(live_indices, mix["synthetic_attack"])
    if chosen_live.numel() == 0:
        info = {
            "ddpm_updated": ddpm_updated,
            "synthetic_count": 0,
            "warmup": int(not generate_synthetic),
        }
        info.update(empty_quality_counters())
        return None, ddpm_loss_value, info

    quality_cfg = (
        quality_controller.config
        if quality_controller is not None
        else SyntheticQualityConfig(enabled=False)
    )
    quality_info = empty_quality_counters()
    requested_count = int(chosen_live.numel())
    quality_info["quality_requested"] = requested_count
    if quality_controller is not None and quality_controller.disabled:
        quality_info["quality_fallback_real"] = requested_count
        quality_info.update(quality_controller.record_batch(0, 0))
        return None, ddpm_loss_value, {
            "ddpm_updated": ddpm_updated,
            "synthetic_count": 0,
            "warmup": 0,
            **quality_info,
        }

    sample_conditions = None
    if stage6_spec.conditional:
        # Sample conditions from attack types actually present in this balanced
        # batch. This avoids inventing a condition absent from the source data.
        assert train_condition is not None
        repeat_idx = torch.arange(chosen_live.numel(), device=rgb.device) % train_condition.numel()
        sample_conditions = train_condition[repeat_idx]

    pending_live = chosen_live
    pending_conditions = sample_conditions
    accepted_images = []
    max_attempts = 1 + (int(quality_cfg.max_retries) if quality_cfg.enabled else 0)
    for attempt in range(max_attempts):
        if pending_live.numel() == 0:
            break
        parent_rgb = rgb[pending_live]
        _, parent_phase = FrequencyUtils.fft_decompose(parent_rgb)
        amplitude_fake = amp_ddpm.generate_fake_amplitude(
            shape=(pending_live.numel(), amp_channels, image_size, image_size),
            K=sampling_k,
            condition=pending_conditions,
            normalized_clip=quality_cfg.normalized_amplitude_clip,
        )
        # Canonical E6: A_syn=A_DDPM exactly. Only the phase comes from live.
        candidate = FrequencyUtils.ifft_reconstruct(amplitude_fake, parent_phase)
        quality_info["quality_generated"] += int(candidate.shape[0])
        if attempt > 0:
            quality_info["quality_retries"] += int(candidate.shape[0])

        if quality_controller is None or not quality_cfg.enabled:
            accepted_mask = torch.ones(candidate.shape[0], dtype=torch.bool, device=candidate.device)
            report = {"generated": int(candidate.shape[0]), "accepted": int(candidate.shape[0])}
        else:
            accepted_mask, report = quality_controller.validate_candidates(
                candidate,
                parent_rgb,
                rgb[live_indices],
                rgb[attack_indices],
            )
        quality_info["quality_accepted"] += int(report.get("accepted", 0))
        quality_info["quality_rejected"] += int(report.get("rejected", 0))
        for report_key, target_key in (
            ("clip_ratio_sum", "quality_clip_ratio_sum"),
            ("attack_margin_sum", "quality_attack_margin_sum"),
            ("attack_distance_ratio_sum", "quality_attack_distance_ratio_sum"),
            ("high_frequency_ratio_sum", "quality_high_frequency_ratio_sum"),
            ("rejected_finite", "quality_rejected_finite"),
            ("rejected_clip", "quality_rejected_clip"),
            ("rejected_attack_margin", "quality_rejected_attack_margin"),
            ("rejected_attack_distance", "quality_rejected_attack_distance"),
            ("rejected_high_frequency", "quality_rejected_high_frequency"),
        ):
            quality_info[target_key] += report.get(report_key, 0)

        if bool(accepted_mask.any()):
            # Explicit detach prevents classifier gradients from entering DDPM.
            accepted_images.append(candidate[accepted_mask].clamp(-1, 1).detach())
        rejected_mask = ~accepted_mask
        pending_live = pending_live[rejected_mask]
        if pending_conditions is not None:
            pending_conditions = pending_conditions[rejected_mask]

    accepted_count = sum(int(item.shape[0]) for item in accepted_images)
    quality_info["quality_fallback_real"] = requested_count - accepted_count
    if quality_controller is not None and quality_cfg.enabled:
        quality_info.update(quality_controller.record_batch(requested_count, accepted_count))
    if quality_info.get("quality_just_disabled"):
        print(
            "[stage6-quality] Synthetic augmentation disabled; real attacks will fill all "
            f"attack slots: {quality_info.get('quality_disable_reason', '')}"
        )

    img_augmented = torch.cat(accepted_images, dim=0) if accepted_images else None
    if update_ddpm:
        amp_ddpm.unet.train()
    else:
        amp_ddpm.unet.eval()
    return img_augmented, ddpm_loss_value, {
        "ddpm_updated": ddpm_updated,
        "synthetic_count": int(accepted_count),
        "warmup": 0,
        **quality_info,
    }


# ============================================================================
# Training interval helpers
# ============================================================================
def _iter_training_batches(loader, batch_count: int, sampler_epoch: int):
    """Yield an exact number of balanced batches, cycling the loader if needed."""
    batch_count = int(batch_count)
    if batch_count <= 0:
        raise ValueError("batch_count must be positive")
    produced = 0
    cycle = 0
    batch_sampler = getattr(loader, "batch_sampler", None)
    while produced < batch_count:
        if hasattr(batch_sampler, "set_epoch"):
            # Each validation interval receives a disjoint deterministic seed
            # range even when a small source loader must be cycled several times.
            batch_sampler.set_epoch(int(sampler_epoch) * 1000 + cycle)
        produced_this_cycle = 0
        for batch in loader:
            yield produced, batch
            produced += 1
            produced_this_cycle += 1
            if produced >= batch_count:
                break
        if produced_this_cycle == 0:
            raise RuntimeError("Training DataLoader yielded no batches")
        cycle += 1


def train_one_epoch(
    epoch,
    loader,
    amp_ddpm,
    ddpm_optimizer,
    fas_net,
    fas_optimizer,
    cfg,
    device,
    fft_stats=None,
    experiment_spec=None,
    global_step_start=0,
    max_batches=None,
    quality_controller=None,
):
    """Train one validation interval or one legacy epoch."""
    spec = experiment_spec or resolve_ablation_spec(cfg)
    stage6_spec = resolve_effective_stage6_spec(cfg, spec.variant)
    fas_net.train()
    if amp_ddpm is not None:
        if bool(cfg.get("diffusion", {}).get("freeze_pretrained", True)):
            amp_ddpm.unet.eval()
        else:
            amp_ddpm.unet.train()

    diff_cfg, cls_cfg, patch_cfg = cfg["diffusion"], cfg["classifier"], cfg["patch"]
    image_size, amp_channels = cfg["image_size"], cfg["amp_channels"]
    log_every = int(cfg["training"]["log_every"])

    alpha_center = float(cls_cfg["alpha_center"]) if spec.use_center else 0.0
    lambda_contrastive = (
        float(cls_cfg["lambda_contrastive"]) if spec.use_contrastive else 0.0
    )

    running = {
        "loss": 0.0,
        "l_face": 0.0,
        "l_patch": 0.0,
        "l_cls": 0.0,
        "l_contrastive": 0.0,
        "l_center": 0.0,
        "ddpm_loss": 0.0,
    }
    n_batches = 0
    n_ddpm_batches = 0
    synthetic_count = 0
    classifier_live_count = 0
    classifier_real_attack_count = 0
    classifier_synthetic_attack_count = 0
    warmup_batch_count = 0
    synthetic_active_batch_count = 0
    quality_counters = empty_quality_counters()
    target_batches = int(max_batches) if max_batches is not None else len(loader)

    for batch_idx, batch in _iter_training_batches(loader, target_batches, epoch):
        global_step = int(global_step_start) + batch_idx + 1
        augmentation_active = bool(
            stage6_spec is not None and global_step > int(stage6_spec.warmup_steps)
        )
        if stage6_spec is not None:
            if augmentation_active:
                synthetic_active_batch_count += 1
            else:
                warmup_batch_count += 1
        if len(batch) == 3:
            rgb, labels, metadata = batch
        else:
            rgb, labels = batch
            metadata = None
        rgb, labels = rgb.to(device), labels.to(device)

        # E6 only: train DDPM on spoof amplitudes, then replace attack slots
        # after warm-up without changing classifier batch size or class prior.
        img_augmented = None
        ddpm_loss_val = None
        ddpm_info = _empty_stage6_info(
            warmup=bool(stage6_spec is not None and not augmentation_active)
        )
        if spec.use_ddpm:
            freeze_ddpm = bool(diff_cfg.get("freeze_pretrained", True))
            if amp_ddpm is None or (not freeze_ddpm and ddpm_optimizer is None):
                raise RuntimeError("E6 requires a DDPM and an optimizer only when it is not frozen")
            if stage6_spec is None:
                raise RuntimeError(
                    "E6 requires an effective Stage-6 strategy; canonical E6 defaults to D2 spoof_amplitude"
                )
            img_augmented, ddpm_loss_val, ddpm_info = stage6_diffusion_augment_step(
                amp_ddpm,
                ddpm_optimizer,
                rgb,
                labels,
                metadata,
                image_size,
                amp_channels,
                diff_cfg["sampling_k"],
                stage6_spec,
                diff_cfg.get("grad_clip_norm", 1.0),
                generate_synthetic=augmentation_active,
                quality_controller=quality_controller,
                update_ddpm=not freeze_ddpm,
            )

        if spec.use_ddpm:
            rgb_full_batch, labels_full, classifier_mix = _compose_stage6_classifier_batch(
                rgb, labels, img_augmented
            )
        else:
            rgb_full_batch, labels_full = rgb, labels
            classifier_mix = {
                "classifier_live": int((labels == 0).sum().item()),
                "classifier_real_attack": int((labels == 1).sum().item()),
                "classifier_synthetic_attack": 0,
                "classifier_total": int(labels.numel()),
            }

        if ddpm_loss_val is not None:
            running["ddpm_loss"] += float(ddpm_loss_val)
            n_ddpm_batches += int(ddpm_info.get("ddpm_updated", 1))
        synthetic_count += int(ddpm_info.get("synthetic_count", 0))
        merge_quality_counters(quality_counters, ddpm_info)
        classifier_live_count += classifier_mix["classifier_live"]
        classifier_real_attack_count += classifier_mix["classifier_real_attack"]
        classifier_synthetic_attack_count += classifier_mix["classifier_synthetic_attack"]

        amp_full = prepare_amplitude(rgb_full_batch, fas_net, fft_stats)
        outputs_full = fas_net(rgb_full_batch if fas_net.use_rgb else None, amp_full)

        if spec.use_patch:
            rgb_patch_batch = random_crop_resize_batch(
                rgb_full_batch,
                scale_range=(patch_cfg["scale_min"], patch_cfg["scale_max"]),
                output_size=image_size,
            )
            amp_patch = prepare_amplitude(rgb_patch_batch, fas_net, fft_stats)
            outputs_patch = fas_net(
                rgb_patch_batch if fas_net.use_rgb else None,
                amp_patch,
            )
            fas_losses = fas_net.compute_multi_stream_loss(
                outputs_full,
                outputs_patch,
                labels_full,
                alpha_center=alpha_center,
                lambda_contrastive=lambda_contrastive,
            )
        else:
            fas_losses = fas_net.compute_loss(
                outputs_full,
                labels_full,
                alpha_center=alpha_center,
                lambda_contrastive=lambda_contrastive,
            )

        fas_optimizer.zero_grad(set_to_none=True)
        fas_losses["loss"].backward()
        fas_optimizer.step()

        for key in (
            "loss",
            "l_face",
            "l_patch",
            "l_cls",
            "l_contrastive",
            "l_center",
        ):
            running[key] += float(fas_losses[key].detach().item())
        n_batches += 1

        if log_every > 0 and (batch_idx + 1) % log_every == 0:
            ddpm_text = (
                f"{ddpm_loss_val:.4f}" if ddpm_loss_val is not None else "disabled"
            )
            if stage6_spec is not None and spec.use_ddpm:
                ddpm_text += (
                    f" mode={stage6_spec.ddpm_mode} "
                    f"warmup={int(ddpm_info.get('warmup', 0))} "
                    f"classifier={classifier_mix['classifier_live']}L+"
                    f"{classifier_mix['classifier_real_attack']}R+"
                    f"{classifier_mix['classifier_synthetic_attack']}S"
                )
            print(
                f"  [{spec.variant}][Step {global_step}] "
                f"Batch {batch_idx+1}/{target_batches} - "
                f"L_total={fas_losses['loss'].item():.4f}, "
                f"L_cls={fas_losses['l_cls'].item():.4f}, DDPM={ddpm_text}"
            )

    denominator = max(n_batches, 1)
    result = {
        key: value / denominator
        for key, value in running.items()
        if key != "ddpm_loss"
    }
    result["ddpm_loss"] = (
        running["ddpm_loss"] / n_ddpm_batches if n_ddpm_batches else 0.0
    )
    result["ddpm_enabled"] = bool(spec.use_ddpm)
    result["ddpm_updates"] = int(n_ddpm_batches)
    result["synthetic_count"] = int(synthetic_count)
    result["classifier_live_count"] = int(classifier_live_count)
    result["classifier_real_attack_count"] = int(classifier_real_attack_count)
    result["classifier_synthetic_attack_count"] = int(classifier_synthetic_attack_count)
    result["classifier_sample_count"] = int(
        classifier_live_count + classifier_real_attack_count + classifier_synthetic_attack_count
    )
    result["global_step"] = int(global_step_start) + int(n_batches)
    result["interval_steps"] = int(n_batches)
    result["warmup_batch_count"] = int(warmup_batch_count)
    result["synthetic_active_batch_count"] = int(synthetic_active_batch_count)
    result["warmup_active"] = bool(
        stage6_spec is not None and synthetic_active_batch_count == 0
    )
    result["warmup_steps"] = int(stage6_spec.warmup_steps) if stage6_spec is not None else 0
    result["stage6_variant"] = stage6_spec.variant if stage6_spec is not None else ""
    result["ddpm_mode"] = stage6_spec.ddpm_mode if stage6_spec is not None else "none"
    result["augmentation_ratio"] = (
        float(stage6_spec.augmentation_ratio) if stage6_spec is not None else 1.0
    )
    result.update(finalize_quality_counters(quality_counters))
    return result


# ============================================================================
# Validation (CHI dung luong Full - khong Patch, khong Diffusion)
# ============================================================================
@torch.no_grad()
def validate(loader, fas_net, device, fft_stats=None, evaluation_cfg=None):
    """Evaluate development scores without using argmax/0.5 as final policy.

    For manifest-backed video data, both the aggregation rule and threshold are
    selected on this development split. Legacy image-folder mode tunes only a
    frame-level threshold because it has no stable video metadata.
    """
    fas_net.eval()
    evaluation_cfg = evaluation_cfg or {}

    all_labels, all_scores = [], []
    frame_rows = []

    def batch_value(value, index):
        if torch.is_tensor(value):
            item = value[index]
            return item.item() if item.ndim == 0 else item.detach().cpu().tolist()
        if isinstance(value, (list, tuple)):
            return value[index]
        return value

    for batch in loader:
        if len(batch) == 3:
            rgb, labels, metadata = batch
        else:
            rgb, labels = batch
            metadata = None
        rgb, labels = rgb.to(device), labels.to(device)

        amplitude_full = prepare_amplitude(rgb, fas_net, fft_stats)
        outputs = fas_net(rgb if fas_net.use_rgb else None, amplitude_full)
        scores = torch.softmax(outputs["logits"], dim=-1)[:, 1]
        all_labels.append(labels.cpu())
        all_scores.append(scores.cpu())

        if metadata is not None:
            for index in range(labels.shape[0]):
                row = {key: batch_value(value, index) for key, value in metadata.items()}
                row["label"] = int(labels[index].item())
                row["frame_score"] = float(scores[index].item())
                frame_rows.append(row)

    all_labels = torch.cat(all_labels)
    all_scores = torch.cat(all_scores)
    tpr_fprs = evaluation_cfg.get("tpr_fprs", [0.01, 0.001])

    if frame_rows:
        policy = tune_development_policy(
            frame_rows,
            threshold_method=evaluation_cfg.get("threshold_method", "eer"),
            aggregation_candidates=evaluation_cfg.get(
                "aggregation_candidates", ["mean", "median", "trimmed_mean", "topk_mean"]
            ),
            trim_ratio=float(evaluation_cfg.get("trim_ratio", 0.1)),
            topk_ratio=float(evaluation_cfg.get("topk_ratio", 0.25)),
            tpr_fprs=tpr_fprs,
        )
        params = policy["aggregation_parameters"]
        attached_frames, video_rows = attach_video_results(
            frame_rows,
            aggregation=policy["aggregation"],
            threshold=policy["threshold"],
            trim_ratio=params["trim_ratio"],
            topk_ratio=params["topk_ratio"],
        )
        metrics = evaluate_rows(
            video_rows,
            score_field="video_score",
            threshold=policy["threshold"],
            tpr_fprs=tpr_fprs,
        )
        frame_metrics = evaluate_rows(
            attached_frames,
            score_field="frame_score",
            threshold=policy["threshold"],
            tpr_fprs=tpr_fprs,
        )
        metrics["frame_metrics"] = json_safe_metrics(frame_metrics)
        metrics["evaluation_policy"] = policy
        return metrics

    threshold_result = select_threshold(
        all_labels, all_scores, evaluation_cfg.get("threshold_method", "eer")
    )
    metrics = evaluate_fas_metrics(
        labels=all_labels,
        scores=all_scores,
        threshold=threshold_result["threshold"],
        tpr_fprs=tpr_fprs,
    )
    metrics["evaluation_policy"] = {
        "version": 1,
        "selection_split": "dev",
        "threshold_method": threshold_result["method"],
        "threshold": threshold_result["threshold"],
        "aggregation": "frame",
        "aggregation_parameters": {},
        "development_eer": metrics["EER"],
        "development_metrics": json_safe_metrics(metrics),
    }
    return metrics


# ============================================================================
# Checkpoint + CSV logging
# ============================================================================
def save_checkpoint(
    path,
    epoch,
    global_step,
    amp_ddpm,
    fas_net,
    ddpm_optimizer,
    fas_optimizer,
    metrics,
    fft_stats=None,
    experiment_spec=None,
    config=None,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "epoch": epoch,
        "global_step": int(global_step),
        "fas_net_state_dict": fas_net.state_dict(),
        "fas_optimizer_state_dict": fas_optimizer.state_dict(),
        "metrics": metrics,
        "evaluation_policy": metrics.get("evaluation_policy") if isinstance(metrics, dict) else None,
        "fft_normalization": fft_stats.to_dict() if fft_stats is not None else None,
        "ddpm_attack_normalization": (
            amp_ddpm.normalization_stats.to_dict()
            if amp_ddpm is not None and amp_ddpm.normalization_stats is not None
            else None
        ),
        "ddpm_frozen": bool(
            config is not None and config.get("diffusion", {}).get("freeze_pretrained", False)
        ),
        "ddpm_ema": False,
        "experiment": experiment_spec.to_dict() if experiment_spec is not None else None,
        "config": config,
        "stage6": (
            resolve_effective_stage6_spec(
                config,
                experiment_spec.variant if experiment_spec is not None else resolve_ablation_spec(config).variant,
            ).to_dict()
            if config is not None
            and resolve_effective_stage6_spec(
                config,
                experiment_spec.variant if experiment_spec is not None else resolve_ablation_spec(config).variant,
            ) is not None
            else None
        ),
    }
    if amp_ddpm is not None:
        payload["unet_state_dict"] = amp_ddpm.unet.state_dict()
    if ddpm_optimizer is not None:
        payload["ddpm_optimizer_state_dict"] = ddpm_optimizer.state_dict()
    torch.save(payload, path)
    print(f"  -> Da luu checkpoint: {path}")


def append_csv_log(log_path, row: dict):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    write_header = not os.path.isfile(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_checkpoint(path, amp_ddpm=None, fas_net=None, device="cpu"):
    """
    Tien ich nap lai checkpoint da luu boi save_checkpoint().

    LUU Y: tu PyTorch 2.6, torch.load() mac dinh weights_only=True, se BAO LOI
    voi checkpoint nay (vi 'metrics' chua numpy.ndarray - confusion_matrix).
    Phai dung weights_only=False (CHI nen lam vay voi checkpoint TU MINH luu ra,
    KHONG load checkpoint khong ro nguon goc voi weights_only=False vi co the
    chua ma doc hai khi unpickle).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if amp_ddpm is not None:
        if "unet_state_dict" not in ckpt:
            raise KeyError("Checkpoint does not contain DDPM weights (non-E6 experiment)")
        amp_ddpm.unet.load_state_dict(ckpt["unet_state_dict"])
    if fas_net is not None:
        fas_net.load_state_dict(ckpt["fas_net_state_dict"])
    print(f"[load_checkpoint] Da nap checkpoint epoch={ckpt['epoch']}, "
          f"metrics={ {k: v for k, v in ckpt['metrics'].items() if k != 'confusion_matrix'} }")
    return ckpt


def load_pretrained_ddpm(path, amp_ddpm, device="cpu"):
    """Load and freeze a source-only DDPM checkpoint for classifier training."""
    if not path:
        raise RuntimeError(
            "E6 requires diffusion.pretrained_checkpoint. Run "
            "python -m improved_ddpm.scripts.pretrain_amplitude_ddpm first."
        )
    resolved = path if os.path.isabs(path) else os.path.join(_PROJECT_ROOT, path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"Missing pretrained DDPM checkpoint: {resolved}")
    checkpoint = torch.load(resolved, map_location=device, weights_only=False)
    if checkpoint.get("policy") != "source_train_attack_only":
        raise RuntimeError("DDPM checkpoint is not marked source_train_attack_only")
    amp_ddpm.unet.load_state_dict(checkpoint["unet_state_dict"])
    amp_ddpm.unet.eval()
    for parameter in amp_ddpm.unet.parameters():
        parameter.requires_grad_(False)
    print(
        f"[diffusion] Loaded frozen pretrained DDPM: {resolved} "
        f"(step={checkpoint.get('global_step', -1)}, EMA=False)"
    )
    return checkpoint


def resolve_dataset_roots(data_cfg: dict) -> dict:
    roots = {}
    for key, value in (data_cfg.get("dataset_roots") or {}).items():
        if value is None:
            continue
        path = str(value)
        if not os.path.isabs(path):
            path = os.path.join(_PROJECT_ROOT, path)
        roots[str(key)] = path
    return roots


# ============================================================================
# Main
# ============================================================================
def main(cfg: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    experiment_spec = resolve_ablation_spec(cfg)
    stage6_spec = resolve_effective_stage6_spec(cfg, experiment_spec.variant)
    quality_controller = None
    quality_cfg = None
    if experiment_spec.use_ddpm:
        quality_cfg = SyntheticQualityConfig.from_mapping(
            cfg.get("stage6", {}).get("quality_control", {})
        )
        quality_controller = SyntheticQualityController(quality_cfg)
    seed = int(data_cfg.get("seed", 42))
    set_global_seed(seed)

    print(f"Device: {device}")
    print(
        f"[stage5] {experiment_spec.variant}: {experiment_spec.description} | "
        f"modalities={experiment_spec.active_modalities}, patch={experiment_spec.use_patch}, "
        f"contrastive={experiment_spec.use_contrastive}, center={experiment_spec.use_center}, "
        f"ddpm={experiment_spec.use_ddpm}, seed={seed}"
    )
    if stage6_spec is not None:
        per_class = int(cfg["batch_size"]) // 2
        planned_mix = classifier_mix_counts(
            per_class,
            per_class,
            stage6_spec.augmentation_ratio,
            augmentation_active=True,
        )
        print(
            f"[stage6] {stage6_spec.variant}: {stage6_spec.description} | "
            f"mode={stage6_spec.ddpm_mode}, synthetic_attack_fraction="
            f"{stage6_spec.augmentation_ratio:.2f}, warmup_steps={stage6_spec.warmup_steps}, "
            f"class_balanced={stage6_spec.class_balanced} | active_classifier_mix="
            f"{planned_mix['live']}L+{planned_mix['real_attack']}R+"
            f"{planned_mix['synthetic_attack']}S={planned_mix['total']}"
        )
        print(
            f"[stage6-quality] enabled={quality_cfg.enabled}, retries={quality_cfg.max_retries}, "
            f"amp_clip=±{quality_cfg.normalized_amplitude_clip:.1f}σ, "
            f"max_pixel_clip={100*quality_cfg.max_clip_ratio:.1f}%, "
            f"spectral_gate=source-calibrated, structure_gate=offline-only, "
            f"rolling_disable={100*quality_cfg.min_rolling_acceptance_rate:.0f}%/"
            f"{quality_cfg.rolling_window_batches} batches"
        )
        max_steps = train_cfg.get("max_steps")
        if (
            experiment_spec.use_ddpm
            and max_steps is not None
            and int(max_steps) <= int(stage6_spec.warmup_steps)
        ):
            print(
                "[stage6] WARNING: run nay ket thuc trong warm-up; classifier se khong dung "
                "synthetic attack. Chi dung cau hinh nay de smoke test."
            )

    # ---------------- Dataset & DataLoader ----------------
    # Khong tu chia lai du lieu neu da co manifest/protocol chinh thuc.
    manifest_dir = data_cfg.get("manifest_dir")
    if manifest_dir:
        manifest_path = manifest_dir
        if not os.path.isabs(manifest_path):
            manifest_path = os.path.join(_PROJECT_ROOT, manifest_path)

        data_root = data_cfg.get("data_root")
        if data_root and not os.path.isabs(data_root):
            data_root = os.path.join(_PROJECT_ROOT, data_root)
        dataset_roots = resolve_dataset_roots(data_cfg)

        print(f"[data] Dung fixed manifest: {manifest_path}")
        if dataset_roots:
            print(f"[data] Per-dataset roots: {sorted(dataset_roots)}")
        train_set = ManifestFASDataset(
            manifest=manifest_path,
            data_root=data_root,
            dataset_roots=dataset_roots,
            image_size=cfg["image_size"],
            split="train",
            return_metadata=bool(stage6_spec is not None and stage6_spec.conditional),
        )
        val_set = ManifestFASDataset(
            manifest=manifest_path,
            data_root=data_root,
            dataset_roots=dataset_roots,
            image_size=cfg["image_size"],
            split="dev",
            return_metadata=True,
        )
    else:
        raise RuntimeError(
            "data.manifest_dir is required. Automatic root_dir/val_ratio splitting "
            "was removed from this controlled package to prevent subject leakage."
        )

    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler_name = str(data_cfg.get("sampler", "shuffle")).lower()
    if sampler_name == "domain_class_balanced":
        if not hasattr(train_set, "records"):
            raise RuntimeError("Domain-balanced sampling requires manifest records")
        train_labels = [int(record.label) for record in train_set.records]
        train_domains = [str(record.dataset) for record in train_set.records]
        batch_sampler = DomainClassBalancedBatchSampler(
            train_labels, train_domains, batch_size=int(cfg["batch_size"]), seed=seed, drop_last=True
        )
        train_loader = DataLoader(
            train_set,
            batch_sampler=batch_sampler,
            num_workers=data_cfg["num_workers"],
            pin_memory=(device == "cuda"),
            worker_init_fn=seed_worker,
        )
        per = batch_sampler.per_domain_class
        print(
            f"[data] Domain+class balanced sampler: domains={batch_sampler.domain_names}, "
            f"per-domain={per} live + {per} attack, {len(batch_sampler)} batches/epoch"
        )
    else:
        use_class_balanced = bool(
            stage6_spec is not None and stage6_spec.class_balanced
        ) or sampler_name == "class_balanced"
        if use_class_balanced:
            if hasattr(train_set, "records"):
                train_labels = [int(record.label) for record in train_set.records]
            elif hasattr(train_set, "samples"):
                train_labels = [int(item[1]) for item in train_set.samples]
            else:
                raise RuntimeError("Class-balanced sampling requires dataset labels")
            batch_sampler = ClassBalancedBatchSampler(
                train_labels, batch_size=int(cfg["batch_size"]), seed=seed, drop_last=True
            )
            train_loader = DataLoader(
                train_set,
                batch_sampler=batch_sampler,
                num_workers=data_cfg["num_workers"],
                pin_memory=(device == "cuda"),
                worker_init_fn=seed_worker,
            )
            print(
                f"[data] Class-balanced sampler: {cfg['batch_size']//2} live + "
                f"{cfg['batch_size']//2} attack per batch, {len(batch_sampler)} batches/epoch"
            )
        else:
            train_loader = DataLoader(
                train_set, batch_size=cfg["batch_size"], shuffle=True,
                num_workers=data_cfg["num_workers"], pin_memory=(device == "cuda"), drop_last=True,
                worker_init_fn=seed_worker, generator=generator,
            )
    val_loader = DataLoader(
        val_set, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=data_cfg["num_workers"], pin_memory=(device == "cuda"),
        worker_init_fn=seed_worker,
    )

    # E0 does not touch FFT, so it should not depend on a normalization file.
    fft_stats = load_frequency_stats(cfg) if experiment_spec.use_fft else None
    if fft_stats is not None and manifest_dir:
        current_train = [row for row in read_manifest(manifest_path) if row.split == "train"]
        current_fp = manifest_fingerprint(current_train)
        if current_fp != fft_stats.manifest_fingerprint:
            raise RuntimeError(
                "FFT normalization fingerprint does not match the current source-train manifest. "
                "Re-run prepare_cross_dataset_ms; target data must never "
                "be used to recompute these statistics."
            )
        if quality_controller is not None and quality_controller.calibration is not None:
            gate_fp = quality_controller.calibration.train_manifest_fingerprint
            if gate_fp != manifest_fingerprint(current_train):
                raise RuntimeError(
                    "Spectral gate calibration does not match the current source-train manifest. "
                    "Re-run prepare_cross_dataset_ms."
                )
    ddpm_fft_stats = load_ddpm_frequency_stats(cfg) if experiment_spec.use_ddpm else None
    if ddpm_fft_stats is not None:
        current_attacks = [row for row in current_train if int(row.label) == 1]
        if manifest_fingerprint(current_attacks) != ddpm_fft_stats.manifest_fingerprint:
            raise RuntimeError(
                "DDPM normalization fingerprint does not match source-train attack rows. "
                "Re-run prepare_cross_dataset_ms."
            )
    amp_ddpm, ddpm_optimizer, fas_net, fas_optimizer, experiment_spec = build_models(
        cfg, device, fft_stats, ddpm_fft_stats
    )
    if experiment_spec.use_ddpm and bool(cfg["diffusion"].get("freeze_pretrained", True)):
        assert amp_ddpm is not None
        load_pretrained_ddpm(
            cfg["diffusion"].get("pretrained_checkpoint"), amp_ddpm, device
        )

    checkpoint_dir = os.path.join(_PROJECT_ROOT, train_cfg["checkpoint_dir"])
    log_path = os.path.join(_PROJECT_ROOT, train_cfg["log_dir"], "train_log.csv")
    quality_log_path = os.path.join(
        _PROJECT_ROOT, train_cfg["log_dir"], "synthetic_quality.csv"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(os.path.join(checkpoint_dir, "resolved_config.yaml"), "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, allow_unicode=True, sort_keys=False)
    write_json(os.path.join(checkpoint_dir, "experiment.json"), experiment_spec.to_dict())

    best_acer = float("inf")
    global_step = 0
    no_improvement_checks = 0
    stopped_early = False

    configured_max_steps = train_cfg.get("max_steps")
    step_mode = configured_max_steps is not None
    if step_mode:
        max_steps = int(configured_max_steps)
        eval_every_steps = int(train_cfg.get("eval_every_steps", 300))
        if max_steps <= 0 or eval_every_steps <= 0:
            raise ValueError("training.max_steps and eval_every_steps must be positive")
        validation_checks = (max_steps + eval_every_steps - 1) // eval_every_steps
        total_training_budget = max_steps
        print(
            f"[training] Step schedule: max_steps={max_steps}, "
            f"eval_every_steps={eval_every_steps}"
        )
    else:
        epochs = int(train_cfg.get("epochs", 30))
        if epochs <= 0:
            raise ValueError("training.epochs must be positive when max_steps is null")
        validation_checks = epochs
        total_training_budget = epochs * len(train_loader)
        print(f"[training] Legacy epoch schedule: epochs={epochs}")

    min_steps = int(train_cfg.get("min_steps", 0))
    early_stopping_patience = int(train_cfg.get("early_stopping_patience", 0))
    early_stopping_min_delta = float(train_cfg.get("early_stopping_min_delta", 0.0))
    if min_steps < 0 or early_stopping_patience < 0 or early_stopping_min_delta < 0:
        raise ValueError("Early-stopping parameters must be non-negative")

    for epoch in range(1, validation_checks + 1):
        interval_steps = (
            min(eval_every_steps, max_steps - global_step)
            if step_mode
            else None
        )
        t0 = time.time()
        if step_mode:
            print(
                f"\n========== Train steps {global_step + 1}-"
                f"{global_step + interval_steps}/{max_steps} =========="
            )
        else:
            print(f"\n========== Epoch {epoch}/{validation_checks} ==========")

        train_stats = train_one_epoch(
            epoch, train_loader, amp_ddpm, ddpm_optimizer, fas_net, fas_optimizer,
            cfg, device, fft_stats, experiment_spec,
            global_step_start=global_step,
            max_batches=interval_steps,
            quality_controller=quality_controller,
        )
        global_step = int(train_stats["global_step"])
        val_metrics = validate(val_loader, fas_net, device, fft_stats, cfg.get("evaluation", {}))

        elapsed = time.time() - t0
        print(f"[Step {global_step}] Train - L_total={train_stats['loss']:.4f}, "
              f"L_cls={train_stats['l_cls']:.4f}, DDPM_loss={train_stats['ddpm_loss']:.4f} "
              f"({elapsed:.1f}s)")
        if experiment_spec.use_ddpm:
            print(
                f"[Step {global_step}] E6 mix - warmup_batches="
                f"{train_stats['warmup_batch_count']}, synthetic_batches="
                f"{train_stats['synthetic_active_batch_count']} | "
                f"classifier totals: live={train_stats['classifier_live_count']}, "
                f"real_attack={train_stats['classifier_real_attack_count']}, "
                f"synthetic_attack={train_stats['classifier_synthetic_attack_count']}, "
                f"all={train_stats['classifier_sample_count']}"
            )
            print(
                f"[Step {global_step}] E6 quality - requested="
                f"{train_stats['quality_requested']}, generated={train_stats['quality_generated']}, "
                f"accepted={train_stats['quality_accepted']} "
                f"({100*train_stats['quality_acceptance_rate']:.1f}%), "
                f"fallback_real={train_stats['quality_fallback_real']}, "
                f"rolling={100*train_stats['rolling_acceptance_rate']:.1f}%, "
                f"disabled={bool(train_stats['quality_disabled'])}"
            )
        print(f"[Step {global_step}] Val   - {format_fas_metrics(val_metrics)}")

        # ---------------- Log CSV ----------------
        append_csv_log(log_path, {
            "variant": experiment_spec.variant,
            "stage6_variant": train_stats.get("stage6_variant", ""),
            "ddpm_mode": train_stats.get("ddpm_mode", ""),
            "augmentation_ratio": train_stats.get("augmentation_ratio", 0.0),
            "warmup_steps": train_stats.get("warmup_steps", 0),
            "warmup_active": train_stats.get("warmup_active", False),
            "seed": seed,
            "epoch": epoch,
            "global_step": global_step,
            "interval_steps": train_stats.get("interval_steps", 0),
            "use_rgb": experiment_spec.use_rgb,
            "use_fft": experiment_spec.use_fft,
            "use_patch": experiment_spec.use_patch,
            "use_contrastive": experiment_spec.use_contrastive,
            "use_center": experiment_spec.use_center,
            "use_ddpm": experiment_spec.use_ddpm,
            "train_loss": train_stats["loss"],
            "train_l_face": train_stats["l_face"],
            "train_l_patch": train_stats["l_patch"],
            "train_l_cls": train_stats["l_cls"],
            "train_l_contrastive": train_stats["l_contrastive"],
            "train_l_center": train_stats["l_center"],
            "train_ddpm_loss": train_stats["ddpm_loss"],
            "train_ddpm_updates": train_stats.get("ddpm_updates", 0),
            "train_synthetic_count": train_stats.get("synthetic_count", 0),
            "train_warmup_batch_count": train_stats.get("warmup_batch_count", 0),
            "train_synthetic_active_batch_count": train_stats.get("synthetic_active_batch_count", 0),
            "train_classifier_live_count": train_stats.get("classifier_live_count", 0),
            "train_classifier_real_attack_count": train_stats.get("classifier_real_attack_count", 0),
            "train_classifier_synthetic_attack_count": train_stats.get("classifier_synthetic_attack_count", 0),
            "train_classifier_sample_count": train_stats.get("classifier_sample_count", 0),
            "val_APCER": val_metrics["APCER"],
            "val_BPCER": val_metrics["BPCER"],
            "val_ACER": val_metrics["ACER"],
            "val_AUC": val_metrics["AUC"],
            "val_EER": val_metrics.get("EER", float("nan")),
            "val_threshold": val_metrics.get("threshold"),
            "val_aggregation": val_metrics.get("evaluation_policy", {}).get("aggregation", "frame"),
            "elapsed_sec": round(elapsed, 1),
        })
        if experiment_spec.use_ddpm:
            append_csv_log(quality_log_path, {
                "variant": experiment_spec.variant,
                "stage6_variant": train_stats.get("stage6_variant", ""),
                "seed": seed,
                "global_step": global_step,
                "interval_steps": train_stats.get("interval_steps", 0),
                "requested": train_stats.get("quality_requested", 0),
                "generated": train_stats.get("quality_generated", 0),
                "accepted": train_stats.get("quality_accepted", 0),
                "rejected": train_stats.get("quality_rejected", 0),
                "retries": train_stats.get("quality_retries", 0),
                "fallback_real": train_stats.get("quality_fallback_real", 0),
                "acceptance_rate": train_stats.get("quality_acceptance_rate", 1.0),
                "rolling_acceptance_rate": train_stats.get("rolling_acceptance_rate", 1.0),
                "clip_ratio_mean": train_stats.get("quality_clip_ratio_mean", 0.0),
                "attack_margin_mean": train_stats.get("quality_attack_margin_mean", 0.0),
                "attack_distance_ratio_mean": train_stats.get(
                    "quality_attack_distance_ratio_mean", 0.0
                ),
                "high_frequency_ratio_mean": train_stats.get(
                    "quality_high_frequency_ratio_mean", 0.0
                ),
                "rejected_nonfinite": train_stats.get("quality_rejected_finite", 0),
                "rejected_clip": train_stats.get("quality_rejected_clip", 0),
                "rejected_attack_margin": train_stats.get(
                    "quality_rejected_attack_margin", 0
                ),
                "rejected_attack_distance": train_stats.get(
                    "quality_rejected_attack_distance", 0
                ),
                "rejected_high_frequency": train_stats.get(
                    "quality_rejected_high_frequency", 0
                ),
                "quality_disabled": train_stats.get("quality_disabled", 0),
                "disable_reason": train_stats.get("quality_disable_reason", ""),
            })

        # ---------------- Checkpoint ----------------
        save_checkpoint(
            os.path.join(checkpoint_dir, "last.pt"),
            epoch, global_step, amp_ddpm, fas_net, ddpm_optimizer, fas_optimizer, val_metrics, fft_stats,
            experiment_spec=experiment_spec, config=cfg,
        )
        budget_has_post_warmup = bool(
            stage6_spec is not None
            and experiment_spec.use_ddpm
            and total_training_budget > int(stage6_spec.warmup_steps)
        )
        checkpoint_eligible = (
            not budget_has_post_warmup
            or global_step > int(stage6_spec.warmup_steps)
        )
        if not checkpoint_eligible:
            print(
                "  -> Warm-up: last.pt da luu; chua xet best.pt cho den khi "
                "synthetic attack duoc bat."
            )
        improved = bool(
            checkpoint_eligible
            and (
                best_acer == float("inf")
                or val_metrics["ACER"] < best_acer - early_stopping_min_delta
            )
        )
        if improved:
            best_acer = val_metrics["ACER"]
            no_improvement_checks = 0
            best_path = os.path.join(checkpoint_dir, "best.pt")
            save_checkpoint(
                best_path, epoch, global_step,
                amp_ddpm, fas_net, ddpm_optimizer, fas_optimizer, val_metrics, fft_stats,
                experiment_spec=experiment_spec, config=cfg,
            )
            policy = dict(val_metrics.get("evaluation_policy", {}))
            policy["checkpoint"] = best_path
            policy["checkpoint_epoch"] = epoch
            policy["checkpoint_global_step"] = global_step
            write_json(os.path.join(checkpoint_dir, "threshold.json"), policy)
            print(
                f"  *** ACER moi tot nhat: {best_acer*100:.2f}% tai step {global_step} "
                f"-> da luu best.pt "
                f"va threshold.json (dev-only) ***"
            )
        elif checkpoint_eligible and global_step >= min_steps:
            no_improvement_checks += 1

        if (
            checkpoint_eligible
            and global_step >= min_steps
            and early_stopping_patience > 0
            and no_improvement_checks >= early_stopping_patience
        ):
            stopped_early = True
            print(
                f"[early-stop] Source-dev khong cai thien >= "
                f"{early_stopping_min_delta:.6f} trong "
                f"{early_stopping_patience} lan validation; dung tai step {global_step}."
            )
            break

        if step_mode and global_step >= max_steps:
            break

    if best_acer == float("inf"):
        raise RuntimeError("No eligible best checkpoint was produced")
    stop_reason = "early stopping" if stopped_early else "training budget reached"
    print(
        f"\nHoan tat huan luyen tai step {global_step} ({stop_reason}). "
        f"ACER tot nhat tren source-dev: {best_acer*100:.2f}%"
    )
    print(f"Checkpoint: {checkpoint_dir}/best.pt | Log: {log_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Frequency-Aware DDPM FAS tren du lieu thuc")
    parser.add_argument("--config", type=str, default=_DEFAULT_CONFIG_PATH,
                         help=f"Duong dan file YAML config (mac dinh: {_DEFAULT_CONFIG_PATH})")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    main(cfg)
