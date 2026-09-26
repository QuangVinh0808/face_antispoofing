"""Offline, source-train-only quality report for an E6 DDPM checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import save_image

from improved_ddpm.data.manifest_dataset import ManifestFASDataset
from improved_ddpm.experiments.stage6 import resolve_effective_stage6_spec
from improved_ddpm.fas.frequency_utils import FrequencyUtils
from improved_ddpm.fas.synthetic_quality import (
    SyntheticQualityConfig,
    SyntheticQualityController,
    high_frequency_energy_ratio,
    radial_power_spectrum,
    rbf_mmd_unbiased,
)
from improved_ddpm.scripts.train import (
    build_models,
    load_config,
    load_ddpm_frequency_stats,
    load_frequency_stats,
    resolve_dataset_roots,
    set_global_seed,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (_PROJECT_ROOT / path).resolve()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate E6 synthetic quality using source-train references only"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--reference-count", type=int, default=64)
    parser.add_argument("--face-metrics", action="store_true")
    parser.add_argument(
        "--identity-metrics",
        action="store_true",
        help="Also compute FaceNet embedding cosine; may download VGGFace2 weights",
    )
    return parser.parse_args()


def _collect_source_examples(dataset, live_count: int, attack_count: int):
    live, attack = [], []
    for image, label in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0):
        destination = live if int(label.item()) == 0 else attack
        target = live_count if int(label.item()) == 0 else attack_count
        if len(destination) < target:
            destination.append(image[0])
        if len(live) >= live_count and len(attack) >= attack_count:
            break
    if len(live) < live_count or len(attack) < attack_count:
        raise RuntimeError(
            f"Source-train has insufficient examples: live={len(live)}, attack={len(attack)}"
        )
    return torch.stack(live), torch.stack(attack)


def _perceptual_hash(image: torch.Tensor) -> str:
    gray = image.mean(dim=0, keepdim=True).unsqueeze(0)
    small = torch.nn.functional.interpolate(gray, size=(16, 16), mode="area").flatten()
    bits = (small >= small.mean()).to(torch.uint8).tolist()
    return "".join(f"{sum(bits[i:i+4][j] << (3-j) for j in range(4)):x}" for i in range(0, 256, 4))


def _face_metrics(live: torch.Tensor, synthetic: torch.Tensor, identity: bool, device: str):
    try:
        from facenet_pytorch import InceptionResnetV1, MTCNN
    except ImportError as exc:
        raise ImportError(
            "Face metrics require facenet-pytorch. Install with: "
            "python -m pip install --no-deps facenet-pytorch==2.6.0"
        ) from exc

    mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, device=device)
    detected = 0
    aligned_live, aligned_synthetic = [], []
    for parent, generated in zip(live, synthetic):
        parent_pil = to_pil_image(((parent + 1.0) / 2.0).clamp(0, 1))
        generated_pil = to_pil_image(((generated + 1.0) / 2.0).clamp(0, 1))
        generated_face = mtcnn(generated_pil)
        if generated_face is not None:
            detected += 1
        if identity:
            parent_face = mtcnn(parent_pil)
            if parent_face is not None and generated_face is not None:
                aligned_live.append(parent_face)
                aligned_synthetic.append(generated_face)

    result = {"mtcnn_face_detection_rate": detected / max(len(synthetic), 1)}
    if identity:
        if not aligned_live:
            result["facenet_identity_cosine_mean"] = float("nan")
            result["facenet_identity_pair_count"] = 0
        else:
            model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
            first = torch.stack(aligned_live).to(device)
            second = torch.stack(aligned_synthetic).to(device)
            with torch.no_grad():
                first_embedding = model(first)
                second_embedding = model(second)
                cosine = torch.nn.functional.cosine_similarity(
                    first_embedding, second_embedding, dim=1
                )
            result["facenet_identity_cosine_mean"] = float(cosine.mean().item())
            result["facenet_identity_pair_count"] = int(cosine.numel())
    return result


def main():
    args = parse_args()
    if args.count <= 0 or args.batch_size <= 0 or args.reference_count < 2:
        raise ValueError("count/batch-size must be positive and reference-count must be >= 2")
    config_path = _resolve(args.config)
    checkpoint_path = _resolve(args.checkpoint)
    output_dir = _resolve(args.output_dir)
    assert config_path is not None and checkpoint_path is not None and output_dir is not None
    cfg = load_config(str(config_path))
    experiment_variant = str(cfg.get("experiment", {}).get("variant", "E6"))
    stage6_spec = resolve_effective_stage6_spec(cfg, experiment_variant)
    if stage6_spec is None or not stage6_spec.use_ddpm:
        raise RuntimeError("Synthetic-quality evaluation requires an E6/DDPM config")

    seed = int(cfg.get("data", {}).get("seed", 42))
    set_global_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fft_stats = load_frequency_stats(cfg)
    ddpm_fft_stats = load_ddpm_frequency_stats(cfg)
    amp_ddpm, _, _, _, _ = build_models(cfg, device, fft_stats, ddpm_fft_stats)
    assert amp_ddpm is not None
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    amp_ddpm.unet.load_state_dict(checkpoint["unet_state_dict"])
    amp_ddpm.unet.eval()

    data_cfg = cfg["data"]
    manifest = _resolve(data_cfg["manifest_dir"])
    data_root = _resolve(data_cfg.get("data_root"))
    dataset = ManifestFASDataset(
        manifest=manifest,
        data_root=data_root,
        dataset_roots=resolve_dataset_roots(data_cfg),
        image_size=int(cfg["image_size"]),
        split="train",
        return_metadata=False,
    )
    reference_count = max(int(args.reference_count), int(args.count))
    live, attack = _collect_source_examples(dataset, reference_count, args.reference_count)
    parent_live = live[: args.count]
    reference_live = live[: args.reference_count]
    reference_attack = attack[: args.reference_count]

    quality_cfg = SyntheticQualityConfig.from_mapping(
        cfg.get("stage6", {}).get("quality_control", {})
    )
    controller = SyntheticQualityController(quality_cfg)
    synthetic_parts, accepted_parts, rows = [], [], []
    generated_total = accepted_total = 0

    for start in range(0, args.count, args.batch_size):
        parent = parent_live[start : start + args.batch_size].to(device)
        _, parent_phase = FrequencyUtils.fft_decompose(parent)
        fake_amplitude = amp_ddpm.generate_fake_amplitude(
            shape=parent.shape,
            K=int(cfg["diffusion"].get("sampling_k", 20)),
            normalized_clip=quality_cfg.normalized_amplitude_clip,
        )
        raw_synthetic = FrequencyUtils.ifft_reconstruct(fake_amplitude, parent_phase)
        accepted, report = controller.validate_candidates(
            raw_synthetic,
            parent,
            reference_live.to(device),
            reference_attack.to(device),
        )
        clamped = raw_synthetic.clamp(-1, 1).detach().cpu()
        synthetic_parts.append(clamped)
        accepted_parts.append(accepted.detach().cpu())
        generated_total += int(report["generated"])
        accepted_total += int(report["accepted"])
        hf = high_frequency_energy_ratio(clamped).tolist()
        for offset in range(clamped.shape[0]):
            rows.append({
                "index": start + offset,
                "accepted_by_online_gate": int(accepted[offset].item()),
                "synthetic_strength": 1.0,
                "high_frequency_ratio": float(hf[offset]),
                "perceptual_hash": _perceptual_hash((clamped[offset] + 1.0) / 2.0),
            })

    synthetic = torch.cat(synthetic_parts, dim=0)
    accepted_mask = torch.cat(accepted_parts, dim=0)
    attack_compare = reference_attack[: min(reference_attack.shape[0], synthetic.shape[0])]
    synthetic_psd = radial_power_spectrum(synthetic)
    attack_psd = radial_power_spectrum(attack_compare)
    live_psd = radial_power_spectrum(parent_live[: synthetic.shape[0]])
    pairwise = torch.pdist(synthetic_psd)
    hashes = [row["perceptual_hash"] for row in rows]
    duplicate_rate = 1.0 - len(set(hashes)) / max(len(hashes), 1)
    nearest_attack = torch.cdist(synthetic_psd, attack_psd).min(dim=1).values

    summary = {
        "policy": "source_train_only_no_target",
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_global_step": int(checkpoint.get("global_step", -1)),
        "generated": generated_total,
        "accepted": accepted_total,
        "online_gate_acceptance_rate": accepted_total / max(generated_total, 1),
        "finite_rate": float(torch.isfinite(synthetic).flatten(1).all(dim=1).float().mean().item()),
        "radial_psd_mmd_attack": rbf_mmd_unbiased(synthetic_psd, attack_psd),
        "radial_psd_mmd_live": rbf_mmd_unbiased(synthetic_psd, live_psd),
        "radial_psd_pairwise_diversity_mean": float(pairwise.mean().item()) if pairwise.numel() else 0.0,
        "perceptual_hash_duplicate_rate": duplicate_rate,
        "nearest_source_attack_psd_distance_mean": float(nearest_attack.mean().item()),
        "high_frequency_ratio_synthetic_mean": float(high_frequency_energy_ratio(synthetic).mean().item()),
        "high_frequency_ratio_attack_mean": float(high_frequency_energy_ratio(attack_compare).mean().item()),
        "quality_control": quality_cfg.to_dict(),
        "interpretation": {
            "online_gate_acceptance_rate": "higher is better; rejected training slots fall back to real attacks",
            "radial_psd_mmd_attack": "lower means synthetic spectra are closer to source attacks",
            "radial_psd_pairwise_diversity_mean": "very low may indicate mode collapse",
            "perceptual_hash_duplicate_rate": "lower means fewer exact/near-exact low-resolution duplicates",
        },
    }
    if args.face_metrics or args.identity_metrics:
        summary.update(
            _face_metrics(
                parent_live[: synthetic.shape[0]],
                synthetic,
                identity=bool(args.identity_metrics),
                device=device,
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "synthetic_quality_samples.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "quality_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    grid_count = min(8, synthetic.shape[0])
    grid = []
    for index in range(grid_count):
        grid.extend([
            (parent_live[index] + 1.0) / 2.0,
            (synthetic[index] + 1.0) / 2.0,
        ])
    save_image(torch.stack(grid), output_dir / "live_vs_synthetic_grid.png", nrow=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Saved synthetic quality report: {output_dir}")


if __name__ == "__main__":
    main()
