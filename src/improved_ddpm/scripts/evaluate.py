"""Stage 4: protocol-safe development tuning and frozen-threshold test evaluation.

Typical full evaluation::

    python -m improved_ddpm.scripts.evaluate \
        --config configs/ms_e0_e6.yaml \
        --checkpoint outputs/checkpoints/best.pt \
        --output-dir experiments/run_001/evaluation

Test-only evaluation with a previously frozen policy::

    python -m improved_ddpm.scripts.evaluate \
        --config configs/ms_e0_e6.yaml \
        --checkpoint outputs/checkpoints/best.pt \
        --threshold-file experiments/run_001/evaluation/threshold.json \
        --test-only \
        --output-dir experiments/run_001/test_locked
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent.parent
_PROJECT_ROOT = _SRC_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from improved_ddpm.data.manifest_dataset import ManifestFASDataset
from improved_ddpm.evaluation.protocol import (
    FRAME_PREDICTION_COLUMNS,
    VIDEO_PREDICTION_COLUMNS,
    attach_video_results,
    evaluate_rows,
    json_safe_metrics,
    tune_development_policy,
    write_csv,
    write_json,
)
from improved_ddpm.fas.frequency_normalization import load_fft_normalization, normalize_log_amplitude
from improved_ddpm.fas.frequency_utils import FrequencyUtils
from improved_ddpm.models.factory import build_fas_model
from improved_ddpm.experiments.ablation import resolve_ablation_spec
from improved_ddpm.scripts.train import load_config


def _resolve(path: str | None) -> Path | None:
    if not path:
        return None
    value = Path(path).expanduser()
    return value if value.is_absolute() else (_PROJECT_ROOT / value).resolve()


def _batch_value(value, index: int):
    if torch.is_tensor(value):
        item = value[index]
        return item.item() if item.ndim == 0 else item.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        item = value[index]
        return item.item() if np.ndim(item) == 0 else item.tolist()
    if isinstance(value, (list, tuple)):
        return value[index]
    return value


def _metadata_rows(metadata: Mapping[str, object], batch_size: int) -> List[dict]:
    rows = []
    for index in range(batch_size):
        rows.append({key: _batch_value(value, index) for key, value in metadata.items()})
    return rows


@torch.no_grad()
def collect_frame_predictions(loader, model, device: str, fft_stats) -> List[dict]:
    """Collect continuous attack scores and immutable manifest metadata."""
    model.eval()
    rows: List[dict] = []
    for batch in loader:
        if len(batch) != 3:
            raise RuntimeError("Stage-4 evaluation requires return_metadata=True")
        rgb, labels, metadata = batch
        rgb = rgb.to(device)
        amplitude = None
        if model.use_amp:
            amplitude, _ = FrequencyUtils.fft_decompose(rgb)
            amplitude = normalize_log_amplitude(
                FrequencyUtils.log_amplitude(amplitude), fft_stats
            )
        outputs = model(rgb if model.use_rgb else None, amplitude)
        scores = torch.softmax(outputs["logits"], dim=-1)[:, 1].detach().cpu().numpy()
        label_values = labels.detach().cpu().numpy().astype(int)
        batch_metadata = _metadata_rows(metadata, len(label_values))
        for meta, label, score in zip(batch_metadata, label_values, scores):
            row = dict(meta)
            row["label"] = int(label)
            row["frame_score"] = float(score)
            rows.append(row)
    if not rows:
        raise RuntimeError("evaluation loader produced no samples")
    return rows


def load_classifier(cfg: dict, checkpoint_path: Path, device: str):
    model, requested_spec = build_fas_model(cfg, pretrained=False)
    model = model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    checkpoint_spec = checkpoint.get("experiment")
    if checkpoint_spec:
        saved_variant = str(checkpoint_spec.get("variant", "")).upper()
        if saved_variant and saved_variant != requested_spec.variant:
            raise RuntimeError(
                f"Config requests {requested_spec.variant}, but checkpoint was trained as "
                f"{saved_variant}. Use the resolved_config.yaml stored beside the checkpoint."
            )
        for key in ("use_rgb", "use_fft", "use_patch", "use_contrastive", "use_center", "use_ddpm"):
            requested_value = requested_spec.to_dict()[key]
            saved_value = bool(checkpoint_spec.get(key, requested_value))
            if saved_value != requested_value:
                raise RuntimeError(
                    f"Checkpoint/config mismatch for {key}: saved={saved_value}, "
                    f"requested={requested_value}"
                )

    model.load_state_dict(checkpoint["fas_net_state_dict"])
    model.eval()
    return model, checkpoint


def load_fft_stats(cfg: dict):
    if not resolve_ablation_spec(cfg).use_fft:
        return None
    path = _resolve(cfg.get("frequency", {}).get("normalization_file"))
    if path is None:
        if cfg.get("frequency", {}).get("require_train_stats", False):
            raise RuntimeError("frequency.normalization_file is required for Stage-4 evaluation")
        return None
    stats = load_fft_normalization(path)
    if stats.split != "train":
        raise RuntimeError(f"FFT normalization must be train-only, got {stats.split!r}")
    return stats


def _resolve_dataset_roots(cfg: dict) -> dict:
    roots = {}
    for key, value in (cfg.get("data", {}).get("dataset_roots") or {}).items():
        if value is None:
            continue
        roots[str(key)] = str(_resolve(str(value)))
    return roots


def make_loader(cfg: dict, split: str, manifest_dir: Path, data_root: Path | None, device: str):
    dataset = ManifestFASDataset(
        manifest=manifest_dir,
        image_size=int(cfg["image_size"]),
        split=split,
        data_root=data_root,
        dataset_roots=_resolve_dataset_roots(cfg),
        return_metadata=True,
    )
    return DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["data"].get("num_workers", 0)),
        pin_memory=(device == "cuda"),
    )


def _save_roc(rows: Sequence[Mapping[str, object]], score_field: str, output_prefix: Path, title: str):
    labels = np.asarray([int(row["label"]) for row in rows], dtype=int)
    scores = np.asarray([float(row[score_field]) for row in rows], dtype=float)
    if np.unique(labels).size < 2:
        return
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    csv_path = output_prefix.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["fpr", "tpr", "threshold"])
        writer.writerows(zip(fpr, tpr, thresholds))
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate (BPCER)")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_prefix.with_suffix(".png"), dpi=180)
    plt.close()


def _save_histogram(rows: Sequence[Mapping[str, object]], score_field: str, path: Path, title: str):
    live = [float(row[score_field]) for row in rows if int(row["label"]) == 0]
    attack = [float(row[score_field]) for row in rows if int(row["label"]) == 1]
    plt.figure(figsize=(7, 5))
    if live:
        plt.hist(live, bins=30, alpha=0.6, label="Live")
    if attack:
        plt.hist(attack, bins=30, alpha=0.6, label="Attack")
    plt.xlabel("Attack score")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _attack_type_metrics(video_rows: Sequence[Mapping[str, object]], threshold: float, tpr_fprs) -> dict:
    live_rows = [row for row in video_rows if int(row["label"]) == 0]
    attack_types = sorted({str(row.get("attack_type", "attack")) for row in video_rows if int(row["label"]) == 1})
    output = {}
    for attack_type in attack_types:
        subset = live_rows + [
            row for row in video_rows
            if int(row["label"]) == 1 and str(row.get("attack_type", "attack")) == attack_type
        ]
        output[attack_type] = json_safe_metrics(
            evaluate_rows(subset, score_field="video_score", threshold=threshold, tpr_fprs=tpr_fprs)
        )
    return output


def evaluate_split(
    frame_rows: Sequence[Mapping[str, object]],
    policy: Mapping[str, object],
    output_dir: Path,
    split_name: str,
    tpr_fprs: Sequence[float],
) -> dict:
    params = policy.get("aggregation_parameters", {})
    frames, videos = attach_video_results(
        frame_rows,
        aggregation=str(policy["aggregation"]),
        threshold=float(policy["threshold"]),
        trim_ratio=float(params.get("trim_ratio", 0.1)),
        topk_ratio=float(params.get("topk_ratio", 0.25)),
    )
    frame_metrics = evaluate_rows(
        frames,
        score_field="frame_score",
        threshold=float(policy["threshold"]),
        tpr_fprs=tpr_fprs,
    )
    video_metrics = evaluate_rows(
        videos,
        score_field="video_score",
        threshold=float(policy["threshold"]),
        tpr_fprs=tpr_fprs,
    )
    write_csv(output_dir / f"{split_name}_frame_predictions.csv", frames, FRAME_PREDICTION_COLUMNS)
    write_csv(output_dir / f"{split_name}_video_predictions.csv", videos, VIDEO_PREDICTION_COLUMNS)
    _save_roc(frames, "frame_score", output_dir / f"{split_name}_frame_roc", f"{split_name.title()} frame-level ROC")
    _save_roc(videos, "video_score", output_dir / f"{split_name}_video_roc", f"{split_name.title()} video-level ROC")
    _save_histogram(frames, "frame_score", output_dir / f"{split_name}_frame_histogram.png", f"{split_name.title()} frame scores")
    _save_histogram(videos, "video_score", output_dir / f"{split_name}_video_histogram.png", f"{split_name.title()} video scores")
    return {
        "frame": json_safe_metrics(frame_metrics),
        "video": json_safe_metrics(video_metrics),
        "by_attack_type": _attack_type_metrics(videos, float(policy["threshold"]), tpr_fprs),
        "frame_count": len(frames),
        "video_count": len(videos),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Stage-4 standard FAS evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest-dir", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold-method", choices=["eer", "min_acer"], default=None)
    parser.add_argument(
        "--aggregation-candidates",
        nargs="+",
        default=None,
        choices=["mean", "median", "trimmed_mean", "topk_mean"],
    )
    parser.add_argument("--trim-ratio", type=float, default=None)
    parser.add_argument("--topk-ratio", type=float, default=None)
    parser.add_argument("--tpr-fprs", nargs="+", type=float, default=None)
    parser.add_argument("--threshold-file", default=None)
    parser.add_argument("--test-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(str(_resolve(args.config)))
    eval_cfg = cfg.get("evaluation", {})
    manifest_dir = _resolve(args.manifest_dir or cfg["data"].get("manifest_dir"))
    if manifest_dir is None:
        raise RuntimeError("Stage-4 requires a fixed manifest_dir with dev/test splits")
    data_root = _resolve(args.data_root or cfg["data"].get("data_root"))
    checkpoint_path = _resolve(args.checkpoint)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, checkpoint = load_classifier(cfg, checkpoint_path, device)
    fft_stats = load_fft_stats(cfg)

    tpr_fprs = tuple(args.tpr_fprs or eval_cfg.get("tpr_fprs", [0.01, 0.001]))
    threshold_file = _resolve(args.threshold_file) if args.threshold_file else output_dir / "threshold.json"

    if args.test_only:
        if not threshold_file.is_file():
            raise FileNotFoundError(
                f"test-only requires an existing development threshold file: {threshold_file}"
            )
        policy = json.loads(threshold_file.read_text(encoding="utf-8"))
        if str(policy.get("selection_split")) != "dev":
            raise RuntimeError("threshold file was not selected on development")
    else:
        dev_loader = make_loader(cfg, "dev", manifest_dir, data_root, device)
        dev_rows = collect_frame_predictions(dev_loader, model, device, fft_stats)
        policy = tune_development_policy(
            dev_rows,
            threshold_method=args.threshold_method or eval_cfg.get("threshold_method", "eer"),
            aggregation_candidates=args.aggregation_candidates or eval_cfg.get(
                "aggregation_candidates", ["mean", "median", "trimmed_mean", "topk_mean"]
            ),
            trim_ratio=float(args.trim_ratio if args.trim_ratio is not None else eval_cfg.get("trim_ratio", 0.1)),
            topk_ratio=float(args.topk_ratio if args.topk_ratio is not None else eval_cfg.get("topk_ratio", 0.25)),
            tpr_fprs=tpr_fprs,
        )
        policy["checkpoint"] = str(checkpoint_path)
        policy["checkpoint_epoch"] = int(checkpoint.get("epoch", -1))
        policy["manifest_dir"] = str(manifest_dir)
        write_json(threshold_file, policy)
        development = evaluate_split(dev_rows, policy, output_dir, "development", tpr_fprs)
        write_json(output_dir / "development_metrics.json", development)

    # The test split is evaluated only with the frozen development policy above.
    test_loader = make_loader(cfg, "test", manifest_dir, data_root, device)
    test_rows = collect_frame_predictions(test_loader, model, device, fft_stats)
    test_result = evaluate_split(test_rows, policy, output_dir, "test", tpr_fprs)
    summary = {
        "protocol_safety": {
            "threshold_selected_on": "dev",
            "aggregation_selected_on": "dev",
            "test_used_for_selection": False,
        },
        "policy": policy,
        "test": test_result,
    }
    write_json(output_dir / "evaluation_summary.json", summary)
    write_json(output_dir / "test_metrics.json", test_result)

    print(f"Threshold ({policy['threshold_method']}): {policy['threshold']:.6f}")
    print(f"Aggregation: {policy['aggregation']}")
    print(
        "Test video: "
        f"HTER={test_result['video']['HTER']*100:.2f}% | "
        f"AUC={test_result['video']['AUC']:.4f} | "
        f"EER={test_result['video']['EER']*100:.2f}%"
    )
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
