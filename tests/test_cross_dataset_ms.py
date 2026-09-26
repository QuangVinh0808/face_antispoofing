from pathlib import Path

from improved_ddpm.data.frame_sampling import FrameSamplingConfig
from improved_ddpm.data.manifest import ManifestRecord, read_manifest, write_manifest_bundle
from improved_ddpm.protocols.cross_dataset_ms import get_cross_dataset_spec
from improved_ddpm.protocols.protocol_a_builder import build_protocol_a_manifest


def _dataset(root: Path, name: str, subjects_by_label: dict[int, list[str]], split: str) -> Path:
    rows = []
    for label, subjects in subjects_by_label.items():
        for subject in subjects:
            for frame in range(32):
                rows.append(ManifestRecord(
                    dataset=name,
                    protocol="fixed",
                    split=split,
                    subject_id=subject,
                    video_id=f"{subject}_v",
                    frame_id=f"{frame:03d}",
                    path=f"{subject}_{frame:03d}.png",
                    label=label,
                    attack_type="live" if label == 0 else "attack",
                ))
    write_manifest_bundle(rows, root)
    return root


def test_s_to_m_holdout_is_class_safe_and_target_is_test_only(tmp_path):
    source = _dataset(
        tmp_path / "S",
        "SiW-Mv2",
        {0: [f"live{i}" for i in range(5)], 1: [f"spoof{i}" for i in range(5)]},
        "train",
    )
    target = _dataset(
        tmp_path / "M",
        "MSU-MFSD",
        {0: ["m_live"], 1: ["m_spoof"]},
        "test",
    )
    out = tmp_path / "benchmark"
    build_protocol_a_manifest(
        get_cross_dataset_spec("S_to_M"),
        {
            "S": {"manifest_dir": source, "data_root": source},
            "M": {"manifest_dir": target, "data_root": target},
        },
        out,
        source_dev_policy="holdout",
        source_dev_fraction=0.2,
        seed=42,
        frame_sampling=FrameSamplingConfig(train=16, dev=32, test=32, protocol_policy="uniform"),
    )
    rows = read_manifest(out)
    train = [row for row in rows if row.split == "train"]
    dev = [row for row in rows if row.split == "dev"]
    test = [row for row in rows if row.split == "test"]
    assert {row.label for row in train} == {0, 1}
    assert {row.label for row in dev} == {0, 1}
    assert {row.dataset for row in train + dev} == {"S"}
    assert {row.dataset for row in test} == {"M"}
    assert {row.subject_id for row in train}.isdisjoint({row.subject_id for row in dev})
    assert len(train) == 8 * 16
    assert len(dev) == 2 * 32
    assert len(test) == 2 * 32
