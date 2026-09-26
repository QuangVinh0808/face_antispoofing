"""Build leakage-safe controlled cross-dataset manifests."""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

from improved_ddpm.data.frame_sampling import FrameSamplingConfig, sample_manifest_records
from improved_ddpm.data.manifest import ManifestRecord, read_manifest, write_manifest_bundle
from improved_ddpm.protocols.protocol_a import ProtocolASpec


def _subject_holdout(
    train_rows: Sequence[ManifestRecord],
    *,
    fraction: float,
    seed: int,
) -> tuple[list[ManifestRecord], list[ManifestRecord]]:
    subjects = sorted({row.subject_id for row in train_rows})
    if len(subjects) < 2:
        raise RuntimeError("Need at least two source-train subjects to create source dev holdout")
    rng = random.Random(seed)
    labels_by_subject = {
        subject: frozenset(row.label for row in train_rows if row.subject_id == subject)
        for subject in subjects
    }
    strata: dict[frozenset[int], list[str]] = {}
    for subject, signature in labels_by_subject.items():
        strata.setdefault(signature, []).append(subject)
    dev_subjects: set[str] = set()
    for signature in sorted(strata, key=lambda value: tuple(sorted(value))):
        members = sorted(strata[signature])
        if len(members) < 2:
            continue
        rng.shuffle(members)
        n_dev = max(1, int(round(len(members) * float(fraction))))
        n_dev = min(n_dev, len(members) - 1)
        dev_subjects.update(members[:n_dev])
    if not dev_subjects:
        shuffled = list(subjects)
        rng.shuffle(shuffled)
        dev_subjects.add(shuffled[0])
    new_train = [row for row in train_rows if row.subject_id not in dev_subjects]
    new_dev = [row for row in train_rows if row.subject_id in dev_subjects]
    for label in (0, 1):
        if not any(row.label == label for row in new_train):
            raise RuntimeError(f"Source train holdout lost class {label}")
        if not any(row.label == label for row in new_dev):
            raise RuntimeError(
                f"Source dev holdout has no class {label}; choose a different holdout fraction/seed"
            )
    return new_train, new_dev


def _rewrite(
    row: ManifestRecord,
    *,
    dataset_key: str,
    benchmark_name: str,
    split: str,
) -> ManifestRecord:
    return replace(
        row,
        dataset=dataset_key,
        protocol=benchmark_name,
        split=split,
        subject_id=f"{dataset_key}::{row.subject_id}",
        video_id=f"{dataset_key}::{row.video_id}",
    )


def build_protocol_a_manifest(
    spec: ProtocolASpec,
    datasets: Mapping[str, Mapping[str, object]],
    output_dir: str | Path,
    *,
    source_dev_policy: str = "auto",
    source_dev_fraction: float = 0.20,
    seed: int = 42,
    frame_sampling: FrameSamplingConfig | None = None,
    allow_changes: bool = False,
) -> dict:
    """Create one merged benchmark manifest.

    Source-domain checkpoint/threshold selection is restricted to source dev.
    Target rows are written only as ``test`` and never enter train/dev.
    """
    policy = str(source_dev_policy).strip().lower()
    if policy not in {"auto", "existing", "holdout"}:
        raise ValueError("source_dev_policy must be auto/existing/holdout")
    if not 0.0 < float(source_dev_fraction) < 0.5:
        raise ValueError("source_dev_fraction must be in (0, 0.5)")

    required = set(spec.sources) | {spec.target}
    missing = required - set(datasets)
    if missing:
        raise KeyError(f"Dataset config missing: {sorted(missing)}")

    output_records: list[ManifestRecord] = []
    root_map: dict[str, str] = {}
    source_details = {}

    for dataset_key in required:
        cfg = datasets[dataset_key]
        manifest_dir = Path(str(cfg["manifest_dir"])).expanduser().resolve()
        data_root = Path(str(cfg["data_root"])).expanduser().resolve()
        if not manifest_dir.exists():
            raise FileNotFoundError(f"Missing manifest for {dataset_key}: {manifest_dir}")
        if not data_root.exists():
            raise FileNotFoundError(f"Missing data root for {dataset_key}: {data_root}")
        root_map[dataset_key] = str(data_root)
        records = read_manifest(manifest_dir)

        if dataset_key == spec.target:
            target_test = [row for row in records if row.split == "test"]
            if not target_test:
                raise RuntimeError(f"Target {dataset_key} has no test split")
            output_records.extend(
                _rewrite(row, dataset_key=dataset_key, benchmark_name=spec.name, split="test")
                for row in target_test
            )
            continue

        source_train = [row for row in records if row.split == "train"]
        source_dev = [row for row in records if row.split == "dev"]
        if not source_train:
            raise RuntimeError(f"Source {dataset_key} has no train split")

        use_holdout = policy == "holdout" or (policy == "auto" and not source_dev)
        if policy == "existing" and not source_dev:
            raise RuntimeError(f"Source {dataset_key} has no dev split but policy=existing")
        holdout_seed = int(seed) if len(spec.sources) == 1 else int(seed) + ord(dataset_key)
        if use_holdout:
            source_train, source_dev = _subject_holdout(
                source_train,
                fraction=source_dev_fraction,
                seed=holdout_seed,
            )
        if not source_dev:
            raise RuntimeError(f"Source {dataset_key} has no development rows")

        output_records.extend(
            _rewrite(row, dataset_key=dataset_key, benchmark_name=spec.name, split="train")
            for row in source_train
        )
        output_records.extend(
            _rewrite(row, dataset_key=dataset_key, benchmark_name=spec.name, split="dev")
            for row in source_dev
        )
        source_details[dataset_key] = {
            "train_rows": len(source_train),
            "dev_rows": len(source_dev),
            "dev_origin": "holdout_from_source_train" if use_holdout else "existing_source_dev",
            "holdout_seed": holdout_seed if use_holdout else None,
            "train_subjects": sorted({row.subject_id for row in source_train}),
            "dev_subjects": sorted({row.subject_id for row in source_dev}),
        }

    sampling_report = None
    if frame_sampling is not None:
        output_records, sampling_report = sample_manifest_records(
            output_records,
            frame_sampling,
        )
        frame_counts = Counter(
            (row.dataset, row.split, row.subject_id, row.video_id)
            for row in output_records
        )
        underfilled = []
        for key, count in frame_counts.items():
            target = frame_sampling.target_for_split(key[1])
            if target is not None and count < target:
                underfilled.append((key, count, target))
        if underfilled:
            preview = "; ".join(
                f"{key[0]}/{key[1]}/{key[3]}={count}<{target}"
                for key, count, target in underfilled[:10]
            )
            raise RuntimeError(
                f"{len(underfilled)} videos do not meet the locked frame budget: {preview}. "
                "Inspect MTCNN skipped frames; do not silently train with unequal frame counts."
            )

    output_dir = Path(output_dir)
    summary = write_manifest_bundle(output_records, output_dir, allow_changes=allow_changes)
    metadata = {
        "version": 1,
        "benchmark": "Controlled single-source cross-dataset FAS",
        "setting": spec.to_dict(),
        "selection_safety": {
            "checkpoint_selected_on": "source_dev",
            "threshold_selected_on": "source_dev",
            "aggregation_selected_on": None,
            "aggregation_fixed_by_protocol": "mean",
            "target_used_for_training": False,
            "target_used_for_selection": False,
        },
        "source_dev_policy": policy,
        "source_dev_fraction": float(source_dev_fraction),
        "seed": int(seed),
        "source_details": source_details,
        "frame_sampling": sampling_report,
        "dataset_roots": root_map,
        "manifest_summary": summary,
    }
    (output_dir / "dataset_roots.json").write_text(
        json.dumps(root_map, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "benchmark_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata
