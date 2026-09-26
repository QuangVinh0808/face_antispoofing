"""Controlled E0-E6 single-source MSU-MFSD <-> SiW-Mv2 runner.

E0-E5 form the discriminative ablation; E6 adds canonical D2
spoof-amplitude DDPM to E5.

The runner assumes leakage-safe manifests prepared by prepare_cross_dataset_ms.
All variants share the same source train/dev split, source-only FFT stats when
the FFT branch is enabled,
class-balanced real-data sampling, seed and target-test protocol.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev

import yaml

from improved_ddpm.protocols.cross_dataset_ms import (
    CROSS_DATASET_MS_REGISTRY,
    get_cross_dataset_spec,
)
from improved_ddpm.experiments.ablation import ABLATION_REGISTRY, get_ablation_spec

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
METRIC_NAMES = (
    "APCER", "BPCER", "ACER", "HTER", "AUC", "EER",
    "macro_APCER", "worst_APCER",
)
ABLATION_COMPARISONS = (
    ("FFT_only_vs_RGB_only", "E1", "E0"),
    ("add_FFT_branch", "E2", "E0"),
    ("add_patch_stream", "E3", "E2"),
    ("add_contrastive_loss", "E4", "E3"),
    ("add_center_loss", "E5", "E4"),
    ("add_DDPM", "E6", "E5"),
    ("full_method_vs_RGB_only", "E6", "E0"),
)


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (_PROJECT_ROOT / path).resolve()


def _run(command: list[str], dry_run: bool):
    print("\n$ " + " ".join(command), flush=True)
    if dry_run:
        return
    env = os.environ.copy()
    src = str(_PROJECT_ROOT / "src")
    env["PYTHONPATH"] = src if not env.get("PYTHONPATH") else src + os.pathsep + env["PYTHONPATH"]
    subprocess.run(command, cwd=_PROJECT_ROOT, check=True, env=env)


def _load_base_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    required = {
        "image_size", "batch_size", "data", "frequency", "training",
        "diffusion", "classifier", "patch", "evaluation",
    }
    missing = sorted(required - set(cfg))
    if missing:
        raise KeyError(f"Base config is missing required sections: {missing}")
    return cfg


def _resolve_variants(values: list[str]) -> list[str]:
    if len(values) == 1 and str(values[0]).strip().lower() == "all":
        return list(ABLATION_REGISTRY)
    variants = [str(value).strip().upper() for value in values]
    unknown = [value for value in variants if value not in ABLATION_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown variants {unknown}; choose E0-E6 or all")
    if len(set(variants)) != len(variants):
        raise ValueError("--variants contains duplicates")
    return variants


def parse_args():
    p = argparse.ArgumentParser(description="Run fair E0-E6 MSU/SiW-Mv2 benchmark")
    p.add_argument("--base-config", required=True)
    p.add_argument("--benchmark-root", default="data/benchmarks/cross_dataset_ms")
    p.add_argument("--settings", nargs="+", default=["M_to_S", "S_to_M"])
    p.add_argument("--variants", nargs="+", default=["all"], help="all or any subset of E0 ... E6")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--epochs", type=int, default=None, help="Legacy epoch budget; disables max_steps")
    p.add_argument("--max-steps", type=int, default=None, help="Override training.max_steps")
    p.add_argument(
        "--augmentation-ratio", type=float, default=0.5,
        help="Fraction of E6 attack slots replaced by synthetic attacks (default: 0.5)",
    )
    p.add_argument("--ddpm-pretrain-steps", type=int, default=None)
    p.add_argument("--ddpm-save-every", type=int, default=1000)
    p.add_argument("--sampling-k-candidates", nargs="+", type=int, default=[20, 50, 100])
    p.add_argument("--sampling-quality-count", type=int, default=64)
    p.add_argument("--output-root", default="data/experiments/cross_dataset_ms")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    return p.parse_args()


def _extract_metrics(summary_path: Path) -> dict | None:
    if not summary_path.is_file():
        return None
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    target = payload.get("target", {})
    video = target.get("video", {})
    attacks = payload.get("target_attack_summary") or {}
    return {
        "APCER": float(video.get("APCER", float("nan"))),
        "BPCER": float(video.get("BPCER", float("nan"))),
        "ACER": float(video.get("ACER", float("nan"))),
        "HTER": float(video.get("HTER", float("nan"))),
        "AUC": float(video.get("AUC", float("nan"))),
        "EER": float(video.get("EER", float("nan"))),
        "macro_APCER": float(attacks.get("macro_APCER", float("nan"))),
        "worst_APCER": float(attacks.get("worst_APCER", float("nan"))),
    }


def _result_row(setting: str, variant: str, seed: int, metrics: dict) -> dict:
    spec = get_ablation_spec(variant)
    return {
        "setting": setting,
        "variant": variant,
        "seed": int(seed),
        "use_rgb": int(spec.use_rgb),
        "use_fft": int(spec.use_fft),
        "use_patch": int(spec.use_patch),
        "use_contrastive": int(spec.use_contrastive),
        "use_center": int(spec.use_center),
        "use_ddpm": int(spec.use_ddpm),
        **metrics,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_results(rows: list[dict]) -> list[dict]:
    output = []
    settings = sorted({str(row["setting"]) for row in rows})
    for setting in settings:
        for variant in ABLATION_REGISTRY:
            subset = [
                row for row in rows
                if row["setting"] == setting and row["variant"] == variant
            ]
            if not subset:
                continue
            item = {
                "setting": setting,
                "variant": variant,
                "seed_count": len(subset),
                "seeds": ",".join(str(row["seed"]) for row in sorted(subset, key=lambda x: x["seed"])),
            }
            for metric in METRIC_NAMES:
                values = [float(row[metric]) for row in subset]
                item[f"{metric}_mean"] = mean(values)
                item[f"{metric}_std"] = pstdev(values)
            output.append(item)
    return output


def _paired_ablation_deltas(rows: list[dict]) -> list[dict]:
    output = []
    settings = sorted({str(row["setting"]) for row in rows})
    for setting in settings:
        for comparison, after, before in ABLATION_COMPARISONS:
            before_by_seed = {
                int(row["seed"]): row for row in rows
                if row["setting"] == setting and row["variant"] == before
            }
            after_by_seed = {
                int(row["seed"]): row for row in rows
                if row["setting"] == setting and row["variant"] == after
            }
            seeds = sorted(set(before_by_seed) & set(after_by_seed))
            if not seeds:
                continue
            item = {
                "setting": setting,
                "comparison": comparison,
                "before": before,
                "after": after,
                "paired_seed_count": len(seeds),
                "seeds": ",".join(map(str, seeds)),
            }
            for metric in METRIC_NAMES:
                deltas = [
                    float(after_by_seed[seed][metric]) - float(before_by_seed[seed][metric])
                    for seed in seeds
                ]
                item[f"delta_{metric}_mean"] = mean(deltas)
                item[f"delta_{metric}_std"] = pstdev(deltas)
            output.append(item)
    return output


def _apply_ablation_controls(
    cfg: dict,
    variant: str,
    normalization: Path,
    attack_normalization: Path | None,
    spectral_gate: Path | None,
    ddpm_checkpoint: Path | None,
    sampling_k: int | None,
    augmentation_ratio: float,
):
    """Lock modality/loss/DDPM controls for one canonical E0-E6 variant."""

    ablation = get_ablation_spec(variant)
    stage6_defaults = copy.deepcopy(cfg.get("stage6", {}))
    cfg["experiment"] = {"variant": variant, "overrides": {}}
    cfg.setdefault("frequency", {})["normalization_file"] = (
        str(normalization) if ablation.use_fft else None
    )
    cfg["frequency"]["require_train_stats"] = bool(ablation.use_fft)
    if not ablation.use_ddpm:
        cfg["stage6"] = {
            "enabled": False,
            "variant": "D0",
            "experiment_variant": variant,
            "ddpm_mode": "none",
            "augmentation_ratio": 0.0,
            "warmup_steps": 0,
            "class_balanced": True,
            "description": f"{variant}: no DDPM; {ablation.description}",
        }
    else:
        cfg["stage6"] = {
            "enabled": True,
            "variant": "D2",
            "experiment_variant": "E6",
            "ddpm_mode": "spoof_amplitude",
            "augmentation_ratio": float(augmentation_ratio),
            "warmup_steps": 0,
            "class_balanced": True,
            "synthetic_strength": 1.0,
            "quality_control": copy.deepcopy(
                stage6_defaults.get("quality_control", {})
            ),
            "description": (
                "Canonical E6: frozen source-pretrained spoof-amplitude DDPM; A_syn=A_DDPM "
                "+ source bona-fide phase; replace a fixed fraction of real attack slots "
                "without changing classifier batch size"
            ),
        }
        if attack_normalization is None or spectral_gate is None or ddpm_checkpoint is None:
            raise ValueError("E6 requires attack normalization, spectral gate and pretrained DDPM")
        cfg.setdefault("diffusion", {})["normalization_file"] = str(attack_normalization)
        cfg["diffusion"]["pretrained_checkpoint"] = str(ddpm_checkpoint)
        cfg["diffusion"]["freeze_pretrained"] = True
        cfg["diffusion"]["sampling_k"] = int(sampling_k)
        cfg["stage6"]["quality_control"]["spectral_calibration_file"] = str(spectral_gate)
    return ablation


def main():
    args = parse_args()
    if not 0.0 <= args.augmentation_ratio <= 1.0:
        raise ValueError("--augmentation-ratio must be in [0,1]")
    if args.ddpm_pretrain_steps is not None and args.ddpm_pretrain_steps <= 0:
        raise ValueError("--ddpm-pretrain-steps must be positive")
    if args.ddpm_save_every <= 0 or args.sampling_quality_count < 2:
        raise ValueError("--ddpm-save-every must be positive and quality count >= 2")
    if any(value <= 0 for value in args.sampling_k_candidates):
        raise ValueError("--sampling-k-candidates must contain positive values")
    if args.epochs is not None and args.max_steps is not None:
        raise ValueError("Choose only one of --epochs or --max-steps")
    if args.epochs is not None and args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    base_cfg = _load_base_config(_resolve(args.base_config))
    if int(base_cfg["batch_size"]) != 16:
        raise ValueError(
            "The controlled E0-E6 benchmark locks batch_size=16 "
            "(8 live + 8 attack slots) for every variant"
        )
    variants = _resolve_variants(args.variants)
    benchmark_root = _resolve(args.benchmark_root)
    output_root = _resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if len(args.settings) == 1 and str(args.settings[0]).lower() == "all":
        specs = list(CROSS_DATASET_MS_REGISTRY.values())
    else:
        specs = [get_cross_dataset_spec(x) for x in args.settings]

    benchmark_plan = {
        "benchmark": "Controlled E0-E6 MSU-MFSD/SiW-Mv2",
        "settings": [spec.name for spec in specs],
        "variants": variants,
        "seeds": [int(seed) for seed in args.seeds],
        "training_schedule": (
            {"mode": "epochs", "epochs": int(args.epochs)}
            if args.epochs is not None
            else {
                "mode": "steps",
                "max_steps": int(
                    args.max_steps
                    if args.max_steps is not None
                    else base_cfg["training"]["max_steps"]
                ),
                "eval_every_steps": int(base_cfg["training"]["eval_every_steps"]),
                "min_steps": int(base_cfg["training"]["min_steps"]),
                "early_stopping_patience": int(
                    base_cfg["training"]["early_stopping_patience"]
                ),
                "early_stopping_min_delta": float(
                    base_cfg["training"]["early_stopping_min_delta"]
                ),
            }
        ),
        "run_count": len(specs) * len(variants) * len(args.seeds),
        "target_policy": "strict_source_dev_zero_target",
        "threshold": "source_dev_eer",
        "video_aggregation": "mean",
        "e6_classifier_batch_policy": "warmup=8L+8R; active=8L+4R+4S",
        "e6_synthetic_attack_fraction": float(args.augmentation_ratio),
        "e6_ddpm_training": "separate_source_attack_only_pretrain_then_freeze",
        "e6_ddpm_pretrain_steps": int(
            args.ddpm_pretrain_steps or base_cfg["diffusion"].get("pretrain_steps", 3000)
        ),
        "e6_sampling_k_candidates": list(args.sampling_k_candidates),
        "e6_quality_control": copy.deepcopy(
            base_cfg.get("stage6", {}).get("quality_control", {})
        ),
        "e6_synthetic_strength": 1.0,
        "e6_ema": False,
        "components": {
            variant: get_ablation_spec(variant).to_dict() for variant in variants
        },
    }
    (output_root / "benchmark_plan.json").write_text(
        json.dumps(benchmark_plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rows = []
    for spec in specs:
        benchmark_dir = benchmark_root / spec.name
        roots_path = benchmark_dir / "dataset_roots.json"
        metadata_path = benchmark_dir / "benchmark_metadata.json"
        if not args.dry_run and (not roots_path.is_file() or not metadata_path.is_file()):
            raise FileNotFoundError(
                f"Benchmark not prepared: {benchmark_dir}. Run prepare_cross_dataset_ms first."
            )
        roots = {} if args.dry_run and not roots_path.is_file() else json.loads(roots_path.read_text(encoding="utf-8"))
        normalization = benchmark_dir / f"normalization_{int(base_cfg['image_size'])}.json"
        attack_normalization = benchmark_dir / f"normalization_attack_{int(base_cfg['image_size'])}.json"
        spectral_gate = benchmark_dir / f"spectral_gate_source_train_{int(base_cfg['image_size'])}.json"

        for variant in variants:
            for seed in args.seeds:
                run_dir = output_root / spec.name / variant / f"seed_{seed}"
                ckpt_dir = run_dir / "checkpoints"
                log_dir = run_dir / "logs"
                eval_dir = run_dir / "target_evaluation"
                resolved = run_dir / "resolved_config.yaml"
                summary_path = eval_dir / "protocol_a_summary.json"
                if summary_path.is_file() and not args.force:
                    print(f"[skip] {spec.name}/{variant}/seed_{seed}")
                    m = _extract_metrics(summary_path)
                    if m:
                        rows.append(_result_row(spec.name, variant, seed, m))
                    continue

                run_dir.mkdir(parents=True, exist_ok=True)
                ddpm_checkpoint = None
                selected_sampling_k = None
                if variant == "E6":
                    pretrain_dir = output_root / "ddpm_pretrain" / spec.name / f"seed_{seed}"
                    pretrain_cfg_path = pretrain_dir / "pretrain_config.yaml"
                    ddpm_checkpoint = pretrain_dir / "ddpm_final.pt"
                    k_dir = pretrain_dir / "sampling_k_evaluation"
                    k_selection_path = k_dir / "sampling_k_selection.json"
                    pretrain_dir.mkdir(parents=True, exist_ok=True)

                    pretrain_cfg = copy.deepcopy(base_cfg)
                    pretrain_cfg["experiment"] = {"variant": "E6", "overrides": {}}
                    pretrain_cfg.setdefault("frequency", {})["normalization_file"] = str(normalization)
                    pretrain_cfg["frequency"]["require_train_stats"] = True
                    pretrain_cfg.setdefault("diffusion", {})["normalization_file"] = str(attack_normalization)
                    pretrain_cfg["diffusion"]["pretrain_steps"] = int(
                        args.ddpm_pretrain_steps
                        or base_cfg["diffusion"].get("pretrain_steps", 3000)
                    )
                    pretrain_cfg.setdefault("stage6", {})["enabled"] = True
                    pretrain_cfg["stage6"]["variant"] = "D2"
                    pretrain_cfg["stage6"]["experiment_variant"] = "E6"
                    pretrain_cfg["stage6"]["ddpm_mode"] = "spoof_amplitude"
                    pretrain_cfg["stage6"]["augmentation_ratio"] = float(args.augmentation_ratio)
                    pretrain_cfg["stage6"]["warmup_steps"] = 0
                    pretrain_cfg["stage6"]["synthetic_strength"] = 1.0
                    pretrain_cfg["stage6"].setdefault("quality_control", {})[
                        "spectral_calibration_file"
                    ] = str(spectral_gate)
                    pretrain_cfg.setdefault("data", {})["manifest_dir"] = str(benchmark_dir)
                    pretrain_cfg["data"]["data_root"] = None
                    pretrain_cfg["data"]["dataset_roots"] = roots
                    pretrain_cfg["data"]["seed"] = int(seed)
                    pretrain_cfg_path.write_text(
                        yaml.safe_dump(pretrain_cfg, allow_unicode=True, sort_keys=False),
                        encoding="utf-8",
                    )
                    if args.force or not ddpm_checkpoint.is_file():
                        _run([
                            sys.executable, "-m", "improved_ddpm.scripts.pretrain_amplitude_ddpm",
                            "--config", str(pretrain_cfg_path),
                            "--output-dir", str(pretrain_dir),
                            "--max-steps", str(pretrain_cfg["diffusion"]["pretrain_steps"]),
                            "--save-every", str(args.ddpm_save_every),
                        ], args.dry_run)

                    if args.force or not k_selection_path.is_file():
                        _run([
                            sys.executable, "-m", "improved_ddpm.scripts.evaluate_ddpm_sampling_k",
                            "--config", str(pretrain_cfg_path),
                            "--checkpoint", str(ddpm_checkpoint),
                            "--output-dir", str(k_dir),
                            "--sampling-k", *[str(value) for value in args.sampling_k_candidates],
                            "--count", str(args.sampling_quality_count),
                        ], args.dry_run)
                    selected_sampling_k = (
                        int(args.sampling_k_candidates[0])
                        if args.dry_run
                        else int(json.loads(k_selection_path.read_text(encoding="utf-8"))[
                            "recommended_sampling_k"
                        ])
                    )

                cfg = copy.deepcopy(base_cfg)
                _apply_ablation_controls(
                    cfg,
                    variant,
                    normalization,
                    attack_normalization if variant == "E6" else None,
                    spectral_gate if variant == "E6" else None,
                    ddpm_checkpoint,
                    selected_sampling_k,
                    args.augmentation_ratio,
                )
                cfg.setdefault("data", {})["manifest_dir"] = str(benchmark_dir)
                cfg["data"]["data_root"] = None
                cfg["data"]["dataset_roots"] = roots
                cfg["data"]["sampler"] = "class_balanced"
                cfg["data"]["seed"] = int(seed)
                cfg.setdefault("training", {})["checkpoint_dir"] = str(ckpt_dir)
                cfg["training"]["log_dir"] = str(log_dir)
                if args.epochs is not None:
                    cfg["training"]["epochs"] = int(args.epochs)
                    cfg["training"]["max_steps"] = None
                elif args.max_steps is not None:
                    cfg["training"]["max_steps"] = int(args.max_steps)
                cfg.setdefault("evaluation", {})["threshold_method"] = "eer"
                cfg["evaluation"]["aggregation_candidates"] = ["mean"]
                cfg.setdefault("benchmark", {})["single_source_cross_dataset"] = spec.to_dict()
                cfg["benchmark"]["selection_policy"] = "strict_source_dev_zero_target"

                resolved.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
                try:
                    _run([sys.executable, "-m", "improved_ddpm.scripts.train", "--config", str(resolved)], args.dry_run)
                    _run([
                        sys.executable, "-m", "improved_ddpm.scripts.evaluate_protocol_a",
                        "--config", str(resolved),
                        "--checkpoint", str(ckpt_dir / "best.pt"),
                        "--manifest-dir", str(benchmark_dir),
                        "--output-dir", str(eval_dir),
                    ], args.dry_run)
                except subprocess.CalledProcessError as exc:
                    if not args.continue_on_error:
                        raise
                    print(f"[error] {spec.name}/{variant}/seed_{seed}: {exc}")
                    continue

                if not args.dry_run:
                    m = _extract_metrics(summary_path)
                    if m:
                        rows.append(_result_row(spec.name, variant, seed, m))

    if rows:
        csv_path = output_root / "comparison.csv"
        _write_csv(csv_path, rows)
        summary_path = output_root / "ablation_summary.csv"
        delta_path = output_root / "ablation_paired_deltas.csv"
        _write_csv(summary_path, _aggregate_results(rows))
        _write_csv(delta_path, _paired_ablation_deltas(rows))

        print("\n=== SINGLE-SOURCE CROSS-DATASET RESULTS (video-level) ===")
        for setting in sorted({r["setting"] for r in rows}):
            print(f"\n{setting}")
            for variant in ABLATION_REGISTRY:
                subset = [r for r in rows if r["setting"] == setting and r["variant"] == variant]
                if not subset:
                    continue
                hter = [r["HTER"] for r in subset]
                auc = [r["AUC"] for r in subset]
                eer = [r["EER"] for r in subset]
                macro_apcer = [r["macro_APCER"] for r in subset]
                worst_apcer = [r["worst_APCER"] for r in subset]
                print(
                    f"  {variant}: HTER={100*mean(hter):.2f}% ± {100*pstdev(hter):.2f}, "
                    f"AUC={mean(auc):.4f} ± {pstdev(auc):.4f}, "
                    f"EER={100*mean(eer):.2f}% ± {100*pstdev(eer):.2f} "
                    f"macro-APCER={100*mean(macro_apcer):.2f}%, "
                    f"worst-APCER={100*mean(worst_apcer):.2f}% "
                    f"({len(subset)} seed(s))"
                )
            by_variant = {
                variant: [r for r in rows if r["setting"] == setting and r["variant"] == variant]
                for variant in ABLATION_REGISTRY
            }
            for after, before, label in (("E5", "E0", "classifier vs RGB"), ("E6", "E5", "DDPM"), ("E6", "E0", "full vs RGB")):
                if by_variant[after] and by_variant[before]:
                    delta = 100 * (
                        mean(r["HTER"] for r in by_variant[after])
                        - mean(r["HTER"] for r in by_variant[before])
                    )
                    print(f"  Delta HTER {after}-{before} ({label}): {delta:+.2f} percentage points")
        print(f"\nSaved: {csv_path}")
        print(f"Saved: {summary_path}")
        print(f"Saved: {delta_path}")


if __name__ == "__main__":
    main()
