"""CLI for dataset-specific manifest adapters."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from improved_ddpm.data.adapters import MSUMFSDAdapter, SiWMv2Adapter
from improved_ddpm.data.manifest import ManifestError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fixed FAS protocol manifests.")
    subparsers = parser.add_subparsers(dest="dataset", required=True)

    siw = subparsers.add_parser("siw_mv2", help="Build SiW-Mv2 Protocol-I train/test manifest.")
    siw.add_argument("--root-dir", required=True, type=Path)
    siw.add_argument("--protocol-dir", required=True, type=Path)
    siw.add_argument("--frames-dir", required=True, type=Path)
    siw.add_argument("--output-dir", required=True, type=Path)
    siw.add_argument("--frames-per-video", type=int, default=32)
    siw.add_argument("--path-mode", choices=("absolute", "relative"), default="relative")
    siw.add_argument("--allow-missing", action="store_true")
    siw.add_argument("--allow-decode-errors", action="store_true")
    siw.add_argument(
        "--force",
        action="store_true",
        help="Cho phep thay doi manifest da ton tai. Mac dinh se chan thay doi ngoai y muon.",
    )

    msu = subparsers.add_parser(
        "msu_mfsd",
        help="Build official MSU-MFSD train/test manifest from the 280 raw videos.",
    )
    msu.add_argument(
        "--root-dir",
        required=True,
        type=Path,
        help="MSU-MFSD-Publish or a parent directory containing it.",
    )
    msu.add_argument(
        "--frames-dir",
        required=True,
        type=Path,
        help="Output root for deterministic lossless PNG frames.",
    )
    msu.add_argument("--output-dir", required=True, type=Path)
    msu.add_argument("--frames-per-video", type=int, default=32)
    msu.add_argument("--path-mode", choices=("absolute", "relative"), default="relative")
    msu.add_argument(
        "--allow-nonofficial-counts",
        action="store_true",
        help="Disable the required 280/120/160/70/210 video-count checks.",
    )
    msu.add_argument(
        "--allow-decode-errors",
        action="store_true",
        help="Record unreadable videos/frames instead of stopping.",
    )
    msu.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.dataset == "siw_mv2":
            adapter = SiWMv2Adapter(
                root_dir=args.root_dir,
                protocol_dir=args.protocol_dir,
                frames_dir=args.frames_dir,
                output_dir=args.output_dir,
                frames_per_video=args.frames_per_video,
                path_mode=args.path_mode,
                strict_missing=not args.allow_missing,
                strict_decode=not args.allow_decode_errors,
            )
            result = adapter.build(force=args.force)
        elif args.dataset == "msu_mfsd":
            adapter = MSUMFSDAdapter(
                root_dir=args.root_dir,
                frames_dir=args.frames_dir,
                output_dir=args.output_dir,
                frames_per_video=args.frames_per_video,
                path_mode=args.path_mode,
                strict_counts=not args.allow_nonofficial_counts,
                strict_decode=not args.allow_decode_errors,
            )
            result = adapter.build(force=args.force)
        else:
            raise ManifestError(f"Adapter chua duoc ho tro: {args.dataset}")
    except (ManifestError, FileNotFoundError, OSError, json.JSONDecodeError) as exc:
        print(f"[build_manifest] ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Dataset: {result.dataset}")
    print(f"Protocol: {result.protocol}")
    print(f"Output: {result.output_dir}")
    print(json.dumps(result.summary, ensure_ascii=False, indent=2))
    if result.missing_entries:
        print(f"Warning: {len(result.missing_entries)} protocol entries bi thieu file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
