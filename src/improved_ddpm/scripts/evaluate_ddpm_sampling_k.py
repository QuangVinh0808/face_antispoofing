"""Compare K=20/50/100 using source-train data only and select sampling K."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from improved_ddpm.data.manifest_dataset import ManifestFASDataset
from improved_ddpm.fas.frequency_utils import FrequencyUtils
from improved_ddpm.fas.synthetic_quality import (
    SyntheticQualityConfig,
    SyntheticQualityController,
    high_frequency_energy_ratio,
    radial_power_spectrum,
    rbf_mmd_unbiased,
)
from improved_ddpm.scripts.pretrain_amplitude_ddpm import _build_ddpm
from improved_ddpm.scripts.train import (
    load_config,
    load_ddpm_frequency_stats,
    resolve_dataset_roots,
    set_global_seed,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve(value) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (_PROJECT_ROOT / path).resolve()


def parse_args():
    parser = argparse.ArgumentParser(description="Source-only DDPM K sweep")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sampling-k", nargs="+", type=int, default=[20, 50, 100])
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def _examples(dataset, count):
    live, attack = [], []
    for image, label in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0):
        (live if int(label.item()) == 0 else attack).append(image[0])
        if len(live) >= count and len(attack) >= count:
            break
    if len(live) < count or len(attack) < count:
        raise RuntimeError(f"Insufficient source examples: live={len(live)}, attack={len(attack)}")
    return torch.stack(live[:count]), torch.stack(attack[:count])


def main():
    args = parse_args()
    if args.count < 2 or args.batch_size <= 0 or any(k <= 0 for k in args.sampling_k):
        raise ValueError("count>=2, batch-size>0 and sampling K>0 are required")
    cfg = load_config(str(_resolve(args.config)))
    output = _resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["data"].get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stats = load_ddpm_frequency_stats(cfg)
    model = _build_ddpm(cfg, device, stats)
    checkpoint = torch.load(_resolve(args.checkpoint), map_location=device, weights_only=False)
    if checkpoint.get("policy") != "source_train_attack_only":
        raise RuntimeError("K sweep requires a source_train_attack_only checkpoint")
    model.unet.load_state_dict(checkpoint["unet_state_dict"])
    model.unet.eval()

    data_cfg = cfg["data"]
    dataset = ManifestFASDataset(
        manifest=_resolve(data_cfg["manifest_dir"]),
        data_root=_resolve(data_cfg["data_root"]) if data_cfg.get("data_root") else None,
        dataset_roots=resolve_dataset_roots(data_cfg),
        image_size=int(cfg["image_size"]),
        split="train",
        return_metadata=False,
    )
    live, attack = _examples(dataset, args.count)
    quality_cfg = SyntheticQualityConfig.from_mapping(cfg["stage6"]["quality_control"])
    attack_psd = radial_power_spectrum(attack)
    attack_hf = float(high_frequency_energy_ratio(attack).mean().item())
    rows = []

    for sampling_k in sorted(set(args.sampling_k)):
        set_global_seed(seed)  # same parent images and initial RNG policy for every K
        controller = SyntheticQualityController(quality_cfg)
        synthetic_parts, accepted_parts = [], []
        for start in range(0, args.count, args.batch_size):
            parent = live[start:start + args.batch_size].to(device)
            _, phase = FrequencyUtils.fft_decompose(parent)
            amplitude = model.generate_fake_amplitude(
                parent.shape,
                K=sampling_k,
                normalized_clip=quality_cfg.normalized_amplitude_clip,
            )
            candidate = FrequencyUtils.ifft_reconstruct(amplitude, phase)
            accepted, _ = controller.validate_candidates(
                candidate, parent, parent, attack[:parent.shape[0]].to(device)
            )
            synthetic_parts.append(candidate.clamp(-1, 1).cpu())
            accepted_parts.append(accepted.cpu())
        synthetic = torch.cat(synthetic_parts)
        accepted = torch.cat(accepted_parts)
        synthetic_psd = radial_power_spectrum(synthetic)
        hf = float(high_frequency_energy_ratio(synthetic).mean().item())
        row = {
            "sampling_k": sampling_k,
            "generated": int(synthetic.shape[0]),
            "accepted": int(accepted.sum().item()),
            "acceptance_rate": float(accepted.float().mean().item()),
            "radial_psd_mmd_attack": rbf_mmd_unbiased(synthetic_psd, attack_psd),
            "high_frequency_ratio_synthetic": hf,
            "high_frequency_ratio_attack": attack_hf,
            "high_frequency_absolute_error": abs(hf - attack_hf),
            "finite_rate": float(torch.isfinite(synthetic).flatten(1).all(dim=1).float().mean().item()),
        }
        rows.append(row)
        grid = []
        for index in range(min(8, synthetic.shape[0])):
            grid.extend([(live[index] + 1) / 2, (synthetic[index] + 1) / 2])
        save_image(torch.stack(grid), output / f"live_vs_synthetic_k{sampling_k}.png", nrow=2)
        print(json.dumps(row, indent=2))

    recommended = min(
        rows,
        key=lambda row: (
            -row["acceptance_rate"],
            row["radial_psd_mmd_attack"],
            row["high_frequency_absolute_error"],
            row["sampling_k"],
        ),
    )
    with (output / "sampling_k_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    selection = {
        "policy": "source_train_only_no_target",
        "checkpoint": str(_resolve(args.checkpoint)),
        "checkpoint_global_step": int(checkpoint.get("global_step", -1)),
        "candidates": rows,
        "recommended_sampling_k": int(recommended["sampling_k"]),
        "selection_order": [
            "maximize_acceptance_rate",
            "minimize_radial_psd_mmd_attack",
            "minimize_high_frequency_absolute_error",
            "prefer_lower_k_on_tie",
        ],
        "ema": False,
    }
    (output / "sampling_k_selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Recommended K={selection['recommended_sampling_k']} (source-only)")


if __name__ == "__main__":
    main()
