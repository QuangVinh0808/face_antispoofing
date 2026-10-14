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
from improved_ddpm.fas.ddpm_representation import (
    validate_ddpm_checkpoint, spectral_comparison, source_reference_batches,
    representation_signature, radial_coordinate,
    save_spectral_diagnostics,
)
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
    parser.add_argument("--clip-denoised", choices=("auto", "on", "off"), default="auto",
                        help="R2 clean-sample stabilization; off enables a same-checkpoint diagnostic control")
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
    validate_ddpm_checkpoint(checkpoint, model)
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
    live, attack, attack_b = source_reference_batches(dataset, args.count, seed)
    quality_cfg = SyntheticQualityConfig.from_mapping(cfg["stage6"]["quality_control"])
    attack_psd = radial_power_spectrum(attack)
    attack_hf = float(high_frequency_energy_ratio(attack).mean().item())
    rows = []

    for sampling_k in sorted(set(args.sampling_k)):
        set_global_seed(seed)  # same parent images and initial RNG policy for every K
        controller = SyntheticQualityController(quality_cfg)
        synthetic_parts, accepted_parts, raw_clip_rates, direct_hf, normalized_clip_rates, amplitude_parts = [], [], [], [], [], []
        sampling_reports = []
        for start in range(0, args.count, args.batch_size):
            parent = live[start:start + args.batch_size].to(device)
            _, phase = FrequencyUtils.fft_decompose(parent)
            amplitude = model.generate_fake_amplitude(
                parent.shape,
                K=sampling_k,
                normalized_clip=quality_cfg.normalized_amplitude_clip,
                clip_denoised=None if args.clip_denoised == "auto" else args.clip_denoised == "on",
            )
            candidate = FrequencyUtils.ifft_reconstruct(amplitude, phase)
            amplitude_parts.append(amplitude.cpu())
            raw_clip_rates.append(((candidate < -1) | (candidate > 1)).float().flatten(1).mean(1).cpu())
            radius = radial_coordinate(*amplitude.shape[-2:], device=device) * (2 ** 0.5)
            energy = amplitude.square()
            direct_hf.append((energy[..., radius >= 0.35].sum(-1) / energy.flatten(2).sum(-1).clamp_min(1e-12)).mean(1).cpu())
            normalized_clip_rates.append(model.last_sampling_report["normalized_out_of_range_rate"])
            sampling_reports.append((dict(model.last_sampling_report), parent.shape[0]))
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
            "raw_synthetic_pixel_clip_rate": float(torch.cat(raw_clip_rates).mean()),
            "ddpm_amplitude_high_frequency_ratio": float(torch.cat(direct_hf).mean()),
            "normalized_out_of_range_rate": sum(normalized_clip_rates) / len(normalized_clip_rates),
            **spectral_comparison(synthetic, attack, attack_b),
        }
        for name in ("clean_prediction_out_of_range_rate", "clean_prediction_last_out_of_range_rate",
                     "normalized_boundary_rate"):
            row[name] = (sum(report[name] * count for report, count in sampling_reports) / args.count
                         if all(name in report for report, _ in sampling_reports) else None)
        for name in ("clean_prediction_abs_max", "clean_prediction_last_abs_max"):
            row[name] = max(report[name] for report, _ in sampling_reports) if all(name in report for report, _ in sampling_reports) else None
        row["clip_denoised"] = sampling_reports[0][0]["clip_denoised"]
        row["sampler_policy"] = sampling_reports[0][0]["sampler_policy"]
        rows.append(row)
        grid = []
        for index in range(min(8, synthetic.shape[0])):
            grid.extend([(live[index] + 1) / 2, (synthetic[index] + 1) / 2])
        save_image(torch.stack(grid), output / f"live_vs_synthetic_k{sampling_k}.png", nrow=2)
        save_spectral_diagnostics(output, live, attack, torch.cat(amplitude_parts), synthetic, suffix=f"_k{sampling_k}")
        print(json.dumps(row, indent=2))

    recommended = min(
        rows,
        key=lambda row: (
            -row["acceptance_rate"],
            row["radial_profile_l1_synthetic_real"],
            row["band_high_wasserstein_synthetic_real"],
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
            "minimize_radial_profile_l1_synthetic_real",
            "minimize_high_band_wasserstein",
            "minimize_high_frequency_absolute_error",
            "prefer_lower_k_on_tie",
        ],
        "ema": False,
        "ddpm_representation": representation_signature(stats, model.radius_conditioning),
        "quality_status": "all_candidates_rejected" if all(r["accepted"] == 0 for r in rows) else "inspect_source_diagnostics",
        "sampler_policy": rows[0]["sampler_policy"],
        "clip_denoised": rows[0]["clip_denoised"],
    }
    (output / "sampling_k_selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Recommended K={selection['recommended_sampling_k']} (source-only)")


if __name__ == "__main__":
    main()
