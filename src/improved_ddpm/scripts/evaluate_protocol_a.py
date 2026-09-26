"""Target-only evaluation for Protocol A with a frozen source-dev policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from improved_ddpm.scripts.evaluate import (
    _resolve,
    collect_frame_predictions,
    evaluate_split,
    load_classifier,
    load_fft_stats,
    make_loader,
)
from improved_ddpm.scripts.train import load_config
from improved_ddpm.evaluation.protocol import write_json


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a frozen cross-dataset target without leakage")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest-dir", required=True)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    cfg_path = _resolve(args.config)
    cfg = load_config(str(cfg_path))
    checkpoint_path = _resolve(args.checkpoint)
    manifest_dir = _resolve(args.manifest_dir)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = manifest_dir / "benchmark_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing Protocol-A metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, checkpoint = load_classifier(cfg, checkpoint_path, device)
    policy = checkpoint.get("evaluation_policy")
    if not policy:
        raise RuntimeError("Checkpoint has no source-dev evaluation policy")
    if str(policy.get("selection_split")) != "dev":
        raise RuntimeError("Protocol A requires policy selected on source dev only")
    evaluation_rule = metadata.get("evaluation_rule", {})
    fixed_aggregation = evaluation_rule.get("video_aggregation")
    if fixed_aggregation and str(policy.get("aggregation")) != str(fixed_aggregation):
        raise RuntimeError(
            "Checkpoint aggregation does not match the locked benchmark rule: "
            f"checkpoint={policy.get('aggregation')}, required={fixed_aggregation}"
        )

    fft_stats = load_fft_stats(cfg)
    loader = make_loader(cfg, "test", manifest_dir, None, device)
    target_rows = collect_frame_predictions(loader, model, device, fft_stats)
    target_result = evaluate_split(
        target_rows,
        policy,
        output_dir,
        "target_test",
        tuple(cfg.get("evaluation", {}).get("tpr_fprs", [0.01, 0.001])),
    )
    setting = metadata["setting"]
    attack_metrics = target_result.get("by_attack_type", {})
    attack_apcers = {
        name: float(metrics["APCER"])
        for name, metrics in attack_metrics.items()
        if "APCER" in metrics
    }
    attack_summary = None
    if attack_apcers:
        attack_summary = {
            "scope": "Protocol-I target test stratified by attack type",
            "not_an_official_protocol_ii_score": True,
            "per_attack_APCER": attack_apcers,
            "macro_APCER": sum(attack_apcers.values()) / len(attack_apcers),
            "worst_APCER": max(attack_apcers.values()),
            "worst_attack_type": max(attack_apcers, key=attack_apcers.get),
        }
    summary = {
        "benchmark": metadata.get("benchmark", "Controlled cross-dataset FAS"),
        "setting": setting,
        "protocol_safety": {
            "checkpoint_selected_on": "source_dev",
            "threshold_selected_on": "source_dev",
            "aggregation_selected_on": None if fixed_aggregation else "source_dev",
            "aggregation_fixed_by_protocol": fixed_aggregation,
            "target_used_for_training": False,
            "target_used_for_selection": False,
            "target_adaptation": False,
        },
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "source_dev_policy": policy,
        "target": target_result,
        "target_attack_summary": attack_summary,
    }
    write_json(output_dir / "protocol_a_summary.json", summary)
    vm = target_result["video"]
    print(
        f"{setting['name']} | HTER={vm['HTER']*100:.2f}% | "
        f"AUC={vm['AUC']:.4f} | EER={vm['EER']*100:.2f}%"
    )
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
