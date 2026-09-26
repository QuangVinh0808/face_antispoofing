"""Prepare leakage-safe M->S / S->M single-source cross-dataset manifests.

This uses official per-dataset train/dev/test manifests as input. If a source
manifest has no dev split, a SUBJECT-disjoint holdout is created from source
train. The target contributes TEST rows only. FFT normalization is estimated
from source TRAIN only and then frozen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from improved_ddpm.data.frame_sampling import FrameSamplingConfig
from improved_ddpm.data.manifest import read_manifest
from improved_ddpm.fas.frequency_normalization import compute_fft_normalization, save_fft_normalization
from improved_ddpm.fas.synthetic_quality import (
    calibrate_spectral_gate_from_records,
    save_spectral_gate_calibration,
)
from improved_ddpm.protocols.cross_dataset_ms import (
    CROSS_DATASET_MS_REGISTRY,
    get_cross_dataset_spec,
)
from improved_ddpm.protocols.protocol_a_builder import build_protocol_a_manifest

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (_PROJECT_ROOT / path).resolve()


def _frame_count(value: str):
    text = str(value).strip().lower()
    if text in {"all", "none", "full"}:
        return None
    count = int(text)
    if count <= 0:
        raise argparse.ArgumentTypeError("frame count must be positive or 'all'")
    return count


def parse_args():
    p = argparse.ArgumentParser(description="Prepare single-source MSU/SiW-Mv2 manifests")
    p.add_argument("--datasets-config", required=True)
    p.add_argument("--settings", nargs="+", default=["all"], help="all, M_to_S, S_to_M")
    p.add_argument("--output-root", default="data/benchmarks/cross_dataset_ms")
    p.add_argument("--source-dev-policy", choices=["auto", "existing", "holdout"], default="auto")
    p.add_argument("--source-dev-fraction", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--train-frames", type=_frame_count, default=16)
    p.add_argument("--dev-frames", type=_frame_count, default=32)
    p.add_argument("--test-frames", type=_frame_count, default=32)
    p.add_argument("--normalization-mode", choices=["global", "per_channel"], default="global")
    p.add_argument("--spectral-bins", type=int, default=32)
    p.add_argument("--spectral-lower-quantile", type=float, default=0.01)
    p.add_argument("--spectral-upper-quantile", type=float, default=0.99)
    p.add_argument("--skip-normalization", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(_resolve(args.datasets_config).read_text(encoding="utf-8")) or {}
    raw = cfg.get("datasets", {})
    datasets = {}
    for key in ("M", "S"):
        item = raw.get(key)
        if not item:
            raise KeyError(f"datasets-config is missing {key}")
        datasets[key] = {
            "manifest_dir": str(_resolve(item["manifest_dir"])),
            "data_root": str(_resolve(item["data_root"])),
        }

    if len(args.settings) == 1 and str(args.settings[0]).lower() == "all":
        specs = list(CROSS_DATASET_MS_REGISTRY.values())
    else:
        specs = [get_cross_dataset_spec(x) for x in args.settings]

    output_root = _resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    index = {}

    for spec in specs:
        out = output_root / spec.name
        print(f"\n=== Prepare {spec.name}: {''.join(spec.sources)} -> {spec.target} ===")
        metadata = build_protocol_a_manifest(
            spec,
            datasets,
            out,
            source_dev_policy=args.source_dev_policy,
            source_dev_fraction=args.source_dev_fraction,
            seed=args.seed,
            frame_sampling=FrameSamplingConfig(
                train=args.train_frames,
                dev=args.dev_frames,
                test=args.test_frames,
                protocol_policy="uniform",
            ),
            allow_changes=args.force,
        )
        metadata["benchmark"] = "Controlled single-source MSU-MFSD/SiW-Mv2"
        metadata["comparison_scope"] = "single_source_single_target"
        metadata["selection_safety"] = {
            "checkpoint_selected_on": "source_dev",
            "threshold_selected_on": "source_dev",
            "aggregation_selected_on": None,
            "aggregation_fixed_by_protocol": "mean",
            "target_used_for_training": False,
            "target_used_for_selection": False,
        }
        metadata["evaluation_rule"] = {
            "score_semantics": "attack_probability",
            "video_aggregation": "mean",
            "threshold_selection": "source_dev_eer",
            "target_threshold_recalibration": False,
        }

        if not args.skip_normalization:
            roots = json.loads((out / "dataset_roots.json").read_text(encoding="utf-8"))
            records = read_manifest(out)
            stats = compute_fft_normalization(
                records,
                dataset_roots=roots,
                image_size=args.image_size,
                mode=args.normalization_mode,
                source_manifest=str(out),
            )
            norm_path = out / f"normalization_{args.image_size}.json"
            save_fft_normalization(stats, norm_path, allow_changes=args.force)
            metadata["normalization_file"] = str(norm_path)
            print(f"FFT normalization: {norm_path} ({stats.image_count} source-train images)")

            attack_stats = compute_fft_normalization(
                records,
                dataset_roots=roots,
                image_size=args.image_size,
                mode=args.normalization_mode,
                source_manifest=str(out),
                labels=(1,),
            )
            attack_norm_path = out / f"normalization_attack_{args.image_size}.json"
            save_fft_normalization(attack_stats, attack_norm_path, allow_changes=args.force)
            metadata["attack_normalization_file"] = str(attack_norm_path)
            print(
                f"DDPM attack-only normalization: {attack_norm_path} "
                f"({attack_stats.image_count} source-train attack images)"
            )

            calibration = calibrate_spectral_gate_from_records(
                records,
                dataset_roots=roots,
                image_size=args.image_size,
                radial_bins=args.spectral_bins,
                lower_quantile=args.spectral_lower_quantile,
                upper_quantile=args.spectral_upper_quantile,
                source_manifest=str(out),
            )
            gate_path = out / f"spectral_gate_source_train_{args.image_size}.json"
            save_spectral_gate_calibration(calibration, gate_path)
            metadata["spectral_gate_calibration_file"] = str(gate_path)
            print(
                f"Source-calibrated spectral gate: {gate_path} "
                f"(live={calibration.live_count}, attack={calibration.attack_count})"
            )

        (out / "benchmark_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        index[spec.name] = metadata
        print(json.dumps(metadata["manifest_summary"], ensure_ascii=False, indent=2))

    (output_root / "cross_dataset_ms_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"\nPrepared: {output_root}")


if __name__ == "__main__":
    main()
