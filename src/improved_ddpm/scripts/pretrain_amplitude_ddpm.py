"""Pretrain the E6 spoof-amplitude DDPM on source-train attacks only.

This stage never sees source-dev or target-test samples and never trains the
FAS classifier. The resulting final checkpoint is frozen by the E6 runner.
EMA is intentionally not implemented in this version.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from improved_ddpm.data.manifest import read_manifest
from improved_ddpm.data.manifest_dataset import ManifestFASDataset
from improved_ddpm.fas.amplitude_ddpm import ImprovedAmplitudeDDPM
from improved_ddpm.fas.frequency_normalization import manifest_fingerprint
from improved_ddpm.scripts.train import (
    load_config,
    load_ddpm_frequency_stats,
    resolve_dataset_roots,
    seed_worker,
    set_global_seed,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (_PROJECT_ROOT / path).resolve()


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain source attack-only amplitude DDPM")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def _build_ddpm(cfg: dict, device: str, stats):
    diff = cfg["diffusion"]
    return ImprovedAmplitudeDDPM(
        in_channels=int(cfg["amp_channels"]),
        image_size=int(cfg["image_size"]),
        base_channels=int(diff["base_channels"]),
        channel_mults=tuple(diff["channel_mults"]),
        attention_resolutions=tuple(diff["attention_resolutions"]),
        num_heads=int(diff["num_heads"]),
        T=int(diff["T"]),
        s=float(diff["s"]),
        lam_vlb=float(diff["lam_vlb"]),
        device=device,
        normalization_stats=stats,
        num_conditions=0,
    )


def _save(path: Path, model, optimizer, step: int, cfg: dict, stats, manifest: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "amplitude_ddpm_pretrain_v1",
            "policy": "source_train_attack_only",
            "global_step": int(step),
            "unet_state_dict": model.unet.state_dict(),
            "ddpm_optimizer_state_dict": optimizer.state_dict(),
            "attack_fft_normalization": stats.to_dict(),
            "source_manifest": str(manifest),
            "ema": False,
            "config": cfg,
        },
        path,
    )
    print(f"[pretrain-ddpm] Saved: {path}")


def main():
    args = parse_args()
    cfg = load_config(str(_resolve(args.config)))
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = int(args.max_steps or cfg["diffusion"].get("pretrain_steps", 3000))
    if steps <= 0 or args.save_every <= 0:
        raise ValueError("max-steps and save-every must be positive")

    seed = int(cfg["data"].get("seed", 42))
    set_global_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest = _resolve(cfg["data"]["manifest_dir"])
    data_root = cfg["data"].get("data_root")
    data_root = _resolve(data_root) if data_root else None
    roots = resolve_dataset_roots(cfg["data"])
    stats = load_ddpm_frequency_stats(cfg)

    all_train_rows = [row for row in read_manifest(manifest) if row.split == "train"]
    attack_rows = [row for row in all_train_rows if int(row.label) == 1]
    if manifest_fingerprint(attack_rows) != stats.manifest_fingerprint:
        raise RuntimeError("Attack-only normalization does not match source-train attack rows")

    dataset = ManifestFASDataset(
        manifest=manifest,
        data_root=data_root,
        dataset_roots=roots,
        image_size=int(cfg["image_size"]),
        split="train",
        return_metadata=False,
    )
    attack_indices = [i for i, row in enumerate(dataset.records) if int(row.label) == 1]
    if not attack_indices:
        raise RuntimeError("Source-train manifest has no attack samples")
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        Subset(dataset, attack_indices),
        batch_size=int(cfg["batch_size"] // 2),
        shuffle=True,
        drop_last=True,
        num_workers=int(cfg["data"].get("num_workers", 4)),
        pin_memory=(device == "cuda"),
        worker_init_fn=seed_worker,
        generator=generator,
    )

    model = _build_ddpm(cfg, device, stats)
    optimizer = torch.optim.AdamW(model.unet.parameters(), lr=float(cfg["diffusion"]["lr"]))
    start_step = 0
    if args.resume:
        resume = torch.load(_resolve(args.resume), map_location=device, weights_only=False)
        if resume.get("policy") != "source_train_attack_only":
            raise RuntimeError("Resume checkpoint has an incompatible policy")
        model.unet.load_state_dict(resume["unet_state_dict"])
        optimizer.load_state_dict(resume["ddpm_optimizer_state_dict"])
        start_step = int(resume["global_step"])
    if start_step >= steps:
        raise ValueError(f"Resume step {start_step} already reaches max_steps={steps}")

    log_path = output_dir / "pretrain_log.csv"
    write_header = not log_path.exists()
    iterator = iter(loader)
    model.unet.train()
    for step in range(start_step + 1, steps + 1):
        try:
            rgb, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            rgb, _ = next(iterator)
        rgb = rgb.to(device, non_blocking=True)
        losses = model.training_step(rgb)
        loss = losses["loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite DDPM loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip = float(cfg["diffusion"].get("grad_clip_norm", 1.0))
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.unet.parameters(), clip)
        optimizer.step()

        if step == 1 or step % int(cfg["training"].get("log_every", 10)) == 0:
            value = float(loss.detach().item())
            print(f"[pretrain-ddpm] step={step}/{steps} loss={value:.6f}")
            with log_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["step", "loss", "seed", "ema"])
                if write_header:
                    writer.writeheader()
                    write_header = False
                writer.writerow({"step": step, "loss": value, "seed": seed, "ema": 0})

        if step % args.save_every == 0 and step < steps:
            _save(output_dir / f"ddpm_step_{step}.pt", model, optimizer, step, cfg, stats, manifest)

    final_path = output_dir / "ddpm_final.pt"
    _save(final_path, model, optimizer, steps, cfg, stats, manifest)
    (output_dir / "pretrain_summary.json").write_text(
        json.dumps(
            {
                "policy": "source_train_attack_only",
                "steps": steps,
                "seed": seed,
                "checkpoint": str(final_path),
                "attack_images": len(attack_indices),
                "normalization_images": stats.image_count,
                "ema": False,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
