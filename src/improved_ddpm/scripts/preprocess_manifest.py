"""Stage 3 preprocessing: frame sampling, face crops, FFT stats, and cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from improved_ddpm.data.face_preprocessing import create_face_detector, preprocess_face_crops
from improved_ddpm.data.frame_sampling import FrameSamplingConfig, sample_manifest_records
from improved_ddpm.data.manifest import read_manifest, write_manifest_bundle
from improved_ddpm.fas.frequency_normalization import (
    cache_log_amplitudes,
    compute_fft_normalization,
    save_fft_normalization,
)


def _frame_count(value: str) -> Optional[int]:
    text = value.strip().lower()
    if text in {"all", "none", "full"}:
        return None
    count = int(text)
    if count <= 0:
        raise argparse.ArgumentTypeError("frame count must be positive or 'all'")
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Protocol-aware frame sampling + face crop + train-only FFT normalization"
    )
    parser.add_argument("--manifest-dir", required=True, help="Stage 2 manifest file/directory")
    parser.add_argument("--data-root", default=None, help="Prefix for relative source paths")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-frames", type=_frame_count, default=16)
    parser.add_argument("--dev-frames", type=_frame_count, default=32)
    parser.add_argument("--test-frames", type=_frame_count, default=32)
    parser.add_argument(
        "--protocol-frame-policy",
        choices=("official", "uniform"),
        default="official",
        help="official preserves known paper-specific frame rules",
    )
    parser.add_argument(
        "--crop-modes",
        nargs="+",
        choices=("face_tight", "face_context"),
        default=("face_tight", "face_context"),
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Materialized face-crop size. Models may consistently resize this to 224.",
    )
    parser.add_argument(
        "--detector",
        choices=("mtcnn", "opencv_haar", "full_frame"),
        default="mtcnn",
    )
    parser.add_argument("--mtcnn-device", default="auto")
    parser.add_argument("--mtcnn-min-probability", type=float, default=0.90)
    parser.add_argument("--mtcnn-min-face-size", type=int, default=40)
    parser.add_argument(
        "--missing-face",
        choices=("error", "skip", "full_frame"),
        default="skip",
    )
    parser.add_argument(
        "--output-format",
        choices=("png", "jpg"),
        default="png",
        help="Use PNG for FFT-sensitive benchmarks to avoid JPEG recompression artifacts.",
    )
    parser.add_argument("--path-mode", choices=("absolute", "relative"), default="relative")
    parser.add_argument(
        "--fft-normalization",
        choices=("global", "per_channel", "none"),
        default="global",
    )
    parser.add_argument("--cache-log-amplitude", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict:
    source_manifest = Path(args.manifest_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else None

    source_records = read_manifest(source_manifest)
    sampled_records, sampling_report = sample_manifest_records(
        source_records,
        FrameSamplingConfig(
            train=args.train_frames,
            dev=args.dev_frames,
            test=args.test_frames,
            protocol_policy=args.protocol_frame_policy,
        ),
    )
    sampled_dir = output_dir / "sampled"
    write_manifest_bundle(sampled_records, sampled_dir, allow_changes=args.force)
    sampling_report_path = sampled_dir / "sampling_report.json"
    sampling_payload = json.dumps(sampling_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if sampling_report_path.exists() and sampling_report_path.read_text(encoding="utf-8") != sampling_payload and not args.force:
        raise RuntimeError(f"Sampling report changed: {sampling_report_path}; use --force intentionally")
    sampling_report_path.write_text(sampling_payload, encoding="utf-8")

    detector_kwargs = {}
    if args.detector == "mtcnn":
        detector_kwargs = {
            "device": args.mtcnn_device,
            "min_probability": args.mtcnn_min_probability,
            "min_face_size": args.mtcnn_min_face_size,
        }
    detector = create_face_detector(args.detector, **detector_kwargs)
    records_by_mode, _, crop_report = preprocess_face_crops(
        sampled_records,
        data_root=data_root,
        output_root=output_dir,
        detector=detector,
        crop_modes=args.crop_modes,
        image_size=args.image_size,
        output_format=args.output_format,
        missing_face=args.missing_face,
        path_mode=args.path_mode,
        allow_changes=args.force,
    )

    result = {
        "source_manifest": str(source_manifest),
        "output_dir": str(output_dir),
        "sampling": sampling_report,
        "face_preprocessing": crop_report,
        "crop_modes": {},
    }
    for mode, mode_records in records_by_mode.items():
        mode_dir = output_dir / mode
        summary = write_manifest_bundle(mode_records, mode_dir, allow_changes=args.force)
        mode_data_root = mode_dir if args.path_mode == "relative" else None
        mode_result = {"manifest_summary": summary}

        if args.fft_normalization != "none":
            stats = compute_fft_normalization(
                mode_records,
                data_root=mode_data_root,
                image_size=args.image_size,
                mode=args.fft_normalization,
                source_manifest=str(mode_dir / "manifest.csv"),
            )
            normalization_path = save_fft_normalization(
                stats,
                mode_dir / "normalization.json",
                allow_changes=args.force,
            )
            mode_result["normalization"] = str(normalization_path)

        if args.cache_log_amplitude:
            cache_index = cache_log_amplitudes(
                mode_records,
                data_root=mode_data_root,
                output_dir=mode_dir / "fft_cache",
                image_size=args.image_size,
                allow_changes=args.force,
            )
            mode_result["fft_cache_index"] = str(cache_index)
        result["crop_modes"][mode] = mode_result

    pipeline_report = output_dir / "stage3_report.json"
    report_payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if pipeline_report.exists() and pipeline_report.read_text(encoding="utf-8") != report_payload and not args.force:
        raise RuntimeError(f"Stage 3 report changed: {pipeline_report}; use --force intentionally")
    pipeline_report.write_text(report_payload, encoding="utf-8")
    return result


def main() -> int:
    args = build_parser().parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
