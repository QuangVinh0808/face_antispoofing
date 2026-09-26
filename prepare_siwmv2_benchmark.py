"""Prepare SiW-Mv2 Protocol I with deterministic MTCNN face crops."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def run_module(module: str, arguments: list[str]) -> None:
    command = [sys.executable, "-m", module, *arguments]
    print("\n$ " + " ".join(command), flush=True)
    environment = os.environ.copy()
    source_root = str(PROJECT_ROOT / "src")
    environment["PYTHONPATH"] = (
        source_root if not environment.get("PYTHONPATH")
        else source_root + os.pathsep + environment["PYTHONPATH"]
    )
    subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, help="Directory containing Live/ and Spoof/")
    parser.add_argument(
        "--protocol-dir",
        required=True,
        help="Official pro_3_text directory containing Protocol-I train/test lists",
    )
    parser.add_argument("--detector", choices=("mtcnn", "opencv_haar"), default="mtcnn")
    parser.add_argument("--mtcnn-device", default="auto")
    parser.add_argument("--crop-mode", choices=("face_tight", "face_context"), default="face_context")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_manifest = "data/manifests/SiW-Mv2/protocol_1_raw"
    frames_root = "data/interim/SiW-Mv2/frames_32_png"
    processed_root = "data/processed/SiW-Mv2/protocol_1"
    force = ["--force"] if args.force else []

    run_module("improved_ddpm.scripts.build_manifest", [
        "siw_mv2",
        "--root-dir", args.raw_root,
        "--protocol-dir", args.protocol_dir,
        "--frames-dir", frames_root,
        "--output-dir", raw_manifest,
        "--frames-per-video", "32",
        "--path-mode", "relative",
        *force,
    ])
    run_module("improved_ddpm.scripts.verify_protocol", [
        "--manifest-dir", raw_manifest,
        "--data-root", frames_root,
        "--allow-missing-dev",
    ])

    preprocess = [
        "--manifest-dir", raw_manifest,
        "--data-root", frames_root,
        "--output-dir", processed_root,
        "--train-frames", "all",
        "--dev-frames", "all",
        "--test-frames", "all",
        "--protocol-frame-policy", "uniform",
        "--crop-modes", args.crop_mode,
        "--image-size", "256",
        "--detector", args.detector,
        "--missing-face", "skip",
        "--output-format", "png",
        "--path-mode", "relative",
        "--fft-normalization", "none",
        *force,
    ]
    if args.detector == "mtcnn":
        preprocess.extend(["--mtcnn-device", args.mtcnn_device])
    run_module("improved_ddpm.scripts.preprocess_manifest", preprocess)

    processed_manifest = f"{processed_root}/{args.crop_mode}"
    run_module("improved_ddpm.scripts.verify_protocol", [
        "--manifest-dir", processed_manifest,
        "--data-root", processed_manifest,
        "--allow-missing-dev",
    ])
    print("\nSiW-Mv2 Protocol-I preprocessing completed.")
    print(f"Processed manifest: {processed_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
