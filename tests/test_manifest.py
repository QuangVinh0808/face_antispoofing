"""Tests for deterministic manifests and protocol leakage verification."""

from pathlib import Path

import pytest
import numpy as np
from PIL import Image

from improved_ddpm.data.manifest_dataset import ManifestFASDataset
from improved_ddpm.data.manifest import (
    ManifestError,
    ManifestRecord,
    read_manifest,
    write_manifest_bundle,
)
from improved_ddpm.scripts.verify_protocol import verify_protocol


def _record(tmp_path: Path, split: str, subject: str, video: str, label: int) -> ManifestRecord:
    image = tmp_path / f"{video}.jpg"
    image.write_bytes(b"fake-image")
    return ManifestRecord(
        dataset="UnitTest",
        protocol="1",
        split=split,
        subject_id=subject,
        video_id=video,
        frame_id="000001",
        path=str(image),
        label=label,
        attack_type="live" if label == 0 else "print",
    )


def test_manifest_is_deterministic_and_immutable(tmp_path):
    records = [
        _record(tmp_path, "train", "001", "001_1", 0),
        _record(tmp_path, "train", "002", "002_2", 1),
        _record(tmp_path, "dev", "101", "101_1", 0),
        _record(tmp_path, "dev", "102", "102_2", 1),
        _record(tmp_path, "test", "201", "201_1", 0),
        _record(tmp_path, "test", "202", "202_2", 1),
    ]
    output = tmp_path / "manifest"
    write_manifest_bundle(records, output)
    original = (output / "manifest.csv").read_bytes()

    # Rebuilding from the same protocol must be byte-identical and allowed.
    write_manifest_bundle(list(reversed(records)), output)
    assert (output / "manifest.csv").read_bytes() == original
    assert len(read_manifest(output)) == 6

    changed = list(records)
    changed[-1] = ManifestRecord(**{**changed[-1].to_dict(), "attack_type": "replay"})
    with pytest.raises(ManifestError):
        write_manifest_bundle(changed, output)


def test_verify_protocol_passes_disjoint_splits(tmp_path):
    records = [
        _record(tmp_path, "train", "001", "001_1", 0),
        _record(tmp_path, "train", "002", "002_2", 1),
        _record(tmp_path, "dev", "101", "101_1", 0),
        _record(tmp_path, "dev", "102", "102_2", 1),
        _record(tmp_path, "test", "201", "201_1", 0),
        _record(tmp_path, "test", "202", "202_2", 1),
    ]
    output = tmp_path / "manifest"
    write_manifest_bundle(records, output)
    report, errors = verify_protocol(output)
    assert errors == []
    assert report["status"] == "PASS"
    assert all(value == 0 for value in report["subject_overlap"].values())
    assert all(value == 0 for value in report["video_overlap"].values())


def test_verify_protocol_detects_subject_leakage(tmp_path):
    records = [
        _record(tmp_path, "train", "001", "001_train_live", 0),
        _record(tmp_path, "train", "002", "002_train_attack", 1),
        _record(tmp_path, "dev", "001", "001_dev_live", 0),
        _record(tmp_path, "dev", "102", "102_dev_attack", 1),
        _record(tmp_path, "test", "201", "201_test_live", 0),
        _record(tmp_path, "test", "202", "202_test_attack", 1),
    ]
    output = tmp_path / "manifest"
    write_manifest_bundle(records, output)
    report, errors = verify_protocol(output)
    assert report["status"] == "FAIL"
    assert report["subject_overlap"]["train-dev"] == 1
    assert any("Subject overlap train-dev" in error for error in errors)


def test_manifest_dataset_reads_relative_paths(tmp_path):
    data_root = tmp_path / "dataset"
    data_root.mkdir()
    rows = []
    for index, label in enumerate((0, 1), start=1):
        image_path = data_root / f"sample_{index}.jpg"
        Image.fromarray(np.zeros((12, 12, 3), dtype=np.uint8)).save(image_path)
        rows.append(
            ManifestRecord(
                dataset="UnitTest",
                protocol="1",
                split="train",
                subject_id=f"00{index}",
                video_id=f"video_{index}",
                frame_id="000001",
                path=image_path.name,
                label=label,
                attack_type="live" if label == 0 else "print",
            )
        )

    output = tmp_path / "manifest_relative"
    write_manifest_bundle(rows, output)
    dataset = ManifestFASDataset(
        output,
        split="train",
        image_size=16,
        data_root=data_root,
    )
    image, label = dataset[1]
    assert image.shape == (3, 16, 16)
    assert label in (0, 1)
