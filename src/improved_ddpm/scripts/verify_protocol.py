"""Verify leakage and integrity of a fixed FAS protocol manifest.

Usage:
    python -m improved_ddpm.scripts.verify_protocol \
        --manifest-dir data/manifests/siw/protocol_1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from improved_ddpm.data.manifest import ManifestError, ManifestRecord, read_manifest, resolve_record_path


SPLITS = ("train", "dev", "test")
PAIR_NAMES = (("train", "dev"), ("train", "test"), ("dev", "test"))


def _sets_by_split(records: Sequence[ManifestRecord], field: str) -> Dict[str, Set[str]]:
    return {
        split: {str(getattr(row, field)) for row in records if row.split == split}
        for split in SPLITS
    }


def _duplicate_paths(records: Sequence[ManifestRecord]) -> List[str]:
    seen: Dict[str, str] = {}
    duplicates: List[str] = []
    for row in records:
        key = os.path.normcase(os.path.normpath(row.path))
        if key in seen:
            duplicates.append(row.path)
        else:
            seen[key] = row.path
    return sorted(set(duplicates))


def _verify_hashes(manifest_dir: Path) -> List[str]:
    hash_file = manifest_dir / "manifest.sha256.json"
    if not hash_file.is_file():
        return ["Khong co manifest.sha256.json de kiem tra tinh bat bien."]

    expected = json.loads(hash_file.read_text(encoding="utf-8"))
    errors: List[str] = []
    for name, digest in expected.items():
        path = manifest_dir / name
        if not path.is_file():
            errors.append(f"Thieu file da duoc bam: {path}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            errors.append(f"Checksum thay doi: {path}")
    return errors


def verify_protocol(
    manifest_dir: Path,
    *,
    data_root: Path = None,
    check_files: bool = True,
    require_splits: Sequence[str] = SPLITS,
    check_hashes: bool = True,
) -> Tuple[Dict[str, object], List[str]]:
    records = read_manifest(manifest_dir)
    if not records:
        raise ManifestError("Manifest rong.")

    datasets = sorted({row.dataset for row in records})
    protocols = sorted({row.protocol for row in records})
    errors: List[str] = []

    if len(datasets) != 1:
        errors.append(f"Manifest protocol phai chi chua 1 dataset, hien co: {datasets}")
    if len(protocols) != 1:
        errors.append(f"Manifest protocol phai chi chua 1 protocol, hien co: {protocols}")

    subjects = _sets_by_split(records, "subject_id")
    videos = _sets_by_split(records, "video_id")
    subject_overlap: Dict[str, int] = {}
    video_overlap: Dict[str, int] = {}

    for left, right in PAIR_NAMES:
        key = f"{left}-{right}"
        subject_items = subjects[left] & subjects[right]
        video_items = videos[left] & videos[right]
        subject_overlap[key] = len(subject_items)
        video_overlap[key] = len(video_items)
        if subject_items:
            preview = ", ".join(sorted(subject_items)[:10])
            errors.append(f"Subject overlap {key}: {len(subject_items)} ({preview})")
        if video_items:
            preview = ", ".join(sorted(video_items)[:10])
            errors.append(f"Video overlap {key}: {len(video_items)} ({preview})")

    duplicates = _duplicate_paths(records)
    if duplicates:
        errors.append(f"Co {len(duplicates)} duong dan trung trong manifest.")

    split_report: Dict[str, object] = {}
    for split in SPLITS:
        rows = [row for row in records if row.split == split]
        live = sum(row.label == 0 for row in rows)
        attack = sum(row.label == 1 for row in rows)
        split_report[split] = {
            "subjects": len(subjects[split]),
            "videos": len(videos[split]),
            "frames": len(rows),
            "live": live,
            "attack": attack,
        }
        if split in require_splits and not rows:
            errors.append(f"Thieu split bat buoc: {split}")
        if rows and (live == 0 or attack == 0):
            errors.append(f"Split {split} phai co ca live va attack (live={live}, attack={attack}).")

    missing_paths: List[str] = []
    if check_files:
        for row in records:
            path = resolve_record_path(row.path, data_root)
            if not path.is_file():
                missing_paths.append(str(path))
        if missing_paths:
            errors.append(f"Co {len(missing_paths)} file trong manifest khong ton tai.")

    if check_hashes:
        errors.extend(_verify_hashes(manifest_dir))

    report: Dict[str, object] = {
        "dataset": datasets[0] if len(datasets) == 1 else datasets,
        "protocol": protocols[0] if len(protocols) == 1 else protocols,
        "splits": split_report,
        "subject_overlap": subject_overlap,
        "video_overlap": video_overlap,
        "duplicate_paths": len(duplicates),
        "missing_paths": len(missing_paths),
        "status": "PASS" if not errors else "FAIL",
    }
    return report, errors


def format_report(report: Mapping[str, object], errors: Iterable[str]) -> str:
    lines = [
        f"Dataset: {report['dataset']}",
        f"Protocol: {report['protocol']}",
        "",
    ]
    split_title = {"train": "Train", "dev": "Development", "test": "Test"}
    splits = report["splits"]
    for split in SPLITS:
        values = splits[split]
        lines.extend(
            [
                f"{split_title[split]}:",
                f"  Subjects: {values['subjects']}",
                f"  Videos: {values['videos']}",
                f"  Frames: {values['frames']}",
                f"  Live: {values['live']}",
                f"  Attack: {values['attack']}",
                "",
            ]
        )

    lines.append("Subject overlap:")
    for key, value in report["subject_overlap"].items():
        lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append("Video overlap:")
    for key, value in report["video_overlap"].items():
        lines.append(f"  {key}: {value}")
    lines.extend(
        [
            "",
            f"Duplicate paths: {report['duplicate_paths']}",
            f"Missing paths: {report['missing_paths']}",
            f"Status: {report['status']}",
        ]
    )

    errors = list(errors)
    if errors:
        lines.append("")
        lines.append("Errors:")
        lines.extend(f"  - {error}" for error in errors)
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify FAS manifest protocol/leakage.")
    parser.add_argument("--manifest-dir", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--skip-file-check", action="store_true")
    parser.add_argument("--skip-hash-check", action="store_true")
    parser.add_argument(
        "--allow-missing-dev",
        action="store_true",
        help="Khong bat buoc split dev, nhung neu dev ton tai van kiem tra leakage/class.",
    )
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_dir = args.manifest_dir.expanduser().resolve()
    require_splits = ("train", "test") if args.allow_missing_dev else SPLITS
    try:
        report, errors = verify_protocol(
            manifest_dir,
            data_root=args.data_root,
            check_files=not args.skip_file_check,
            require_splits=require_splits,
            check_hashes=not args.skip_hash_check,
        )
    except (FileNotFoundError, ManifestError, json.JSONDecodeError) as exc:
        print(f"[verify_protocol] ERROR: {exc}", file=sys.stderr)
        return 2

    text = format_report(report, errors)
    print(text, end="")

    report_path = args.report or (manifest_dir / "verification_report.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")
    json_path = report_path.with_suffix(".json")
    json_path.write_text(
        json.dumps({"report": report, "errors": errors}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
