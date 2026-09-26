"""Export deterministic Stage-6 live/synthetic pairs for qualitative inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from improved_ddpm.data.manifest_dataset import ManifestFASDataset
from improved_ddpm.experiments.stage6 import CONDITION_NAMES, resolve_stage6_spec
from improved_ddpm.fas.frequency_utils import FrequencyUtils
from improved_ddpm.fas.synthetic_quality import SyntheticQualityConfig
from improved_ddpm.scripts.train import (
    build_models,
    load_config,
    load_ddpm_frequency_stats,
    load_frequency_stats,
    resolve_dataset_roots,
    set_global_seed,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve(path: str | None):
    if path is None:
        return None
    p = Path(path).expanduser()
    return p if p.is_absolute() else (_PROJECT_ROOT / p).resolve()


def parse_args():
    p = argparse.ArgumentParser(description="Export Stage-6 qualitative samples")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    p.add_argument("--count", type=int, default=8)
    p.add_argument("--condition", choices=["print", "replay", "other", "cycle"], default="cycle")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(str(_resolve(args.config)))
    spec = resolve_stage6_spec(cfg)
    if spec is None or not spec.use_ddpm:
        raise RuntimeError("Sample export requires a Stage-6 DDPM run (D1-D4)")
    set_global_seed(int(cfg.get("data", {}).get("seed", 42)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fft_stats = load_frequency_stats(cfg)
    ddpm_stats = load_ddpm_frequency_stats(cfg)
    amp_ddpm, _, _, _, _ = build_models(cfg, device, fft_stats, ddpm_stats)
    if amp_ddpm is None:
        raise RuntimeError("Config did not build DDPM")
    ckpt = torch.load(_resolve(args.checkpoint), map_location=device, weights_only=False)
    amp_ddpm.unet.load_state_dict(ckpt["unet_state_dict"])
    amp_ddpm.unet.eval()

    manifest = _resolve(cfg["data"]["manifest_dir"])
    data_root = _resolve(cfg["data"].get("data_root"))
    ds = ManifestFASDataset(
        manifest=manifest,
        data_root=data_root,
        dataset_roots=resolve_dataset_roots(cfg["data"]),
        image_size=int(cfg["image_size"]),
        split=args.split,
        return_metadata=True,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    lives = []
    meta_rows = []
    for rgb, labels, metadata in loader:
        if int(labels.item()) != 0:
            continue
        lives.append(rgb[0])
        meta_rows.append({key: value[0] if isinstance(value, list) else value for key, value in metadata.items()})
        if len(lives) >= args.count:
            break
    if not lives:
        raise RuntimeError(f"No live samples found in split={args.split}")

    live = torch.stack(lives).to(device)
    _, phase = FrequencyUtils.fft_decompose(live)
    condition = None
    condition_names = ["unconditional"] * live.shape[0]
    if spec.conditional:
        name_to_id = {v: k for k, v in CONDITION_NAMES.items()}
        if args.condition == "cycle":
            values = [i % 3 for i in range(live.shape[0])]
        else:
            values = [name_to_id[args.condition]] * live.shape[0]
        condition = torch.tensor(values, device=device, dtype=torch.long)
        condition_names = [CONDITION_NAMES[int(v)] for v in values]

    quality_cfg = SyntheticQualityConfig.from_mapping(
        cfg.get("stage6", {}).get("quality_control", {})
    )
    fake_amp = amp_ddpm.generate_fake_amplitude(
        shape=(live.shape[0], int(cfg["amp_channels"]), int(cfg["image_size"]), int(cfg["image_size"])),
        K=int(cfg["diffusion"].get("sampling_k", 20)),
        condition=condition,
        normalized_clip=quality_cfg.normalized_amplitude_clip,
    )
    synthetic = FrequencyUtils.ifft_reconstruct(fake_amp, phase).clamp(-1, 1)

    output = _resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paired = []
    report = []
    for i in range(live.shape[0]):
        orig = (live[i].detach().cpu() + 1.0) / 2.0
        fake = (synthetic[i].detach().cpu() + 1.0) / 2.0
        save_image(orig, output / f"{i:02d}_live.png")
        save_image(fake, output / f"{i:02d}_synthetic.png")
        paired.extend([orig, fake])
        report.append({
            "index": i,
            "condition": condition_names[i],
            "synthetic_strength": 1.0,
            "source": {k: str(v) for k, v in meta_rows[i].items()},
        })
    save_image(torch.stack(paired), output / "live_vs_synthetic_grid.png", nrow=2)
    (output / "samples.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved {len(report)} live/synthetic pairs to {output}")


if __name__ == "__main__":
    main()
