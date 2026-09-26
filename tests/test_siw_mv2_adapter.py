from pathlib import Path

import cv2
import numpy as np

from improved_ddpm.data.adapters.siw_mv2 import SiWMv2Adapter
from improved_ddpm.data.manifest import read_manifest


def _frames(directory: Path, count: int = 4) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        image = np.full((24, 32, 3), 20 + index, dtype=np.uint8)
        assert cv2.imwrite(str(directory / f"{index:03d}.png"), image)


def test_protocol_i_deduplicates_balancing_lines_and_extracts_frames(tmp_path):
    root = tmp_path / "SiW-Mv2"
    _frames(root / "Live" / "Live_1")
    _frames(root / "Live" / "Live_2")
    _frames(root / "Spoof" / "Makeup_Cosmetic" / "Makeup_Cosmetic_1")
    _frames(root / "Spoof" / "Replay" / "Replay_2")

    protocol = tmp_path / "pro_3_text"
    protocol.mkdir()
    (protocol / "trainlist_live.txt").write_text("Live_1\nLive_1\n", encoding="utf-8")
    (protocol / "testlist_live.txt").write_text("Live_2\n", encoding="utf-8")
    (protocol / "trainlist_all.txt").write_text("Co_1\nCo_1\n", encoding="utf-8")
    (protocol / "testlist_all.txt").write_text("Replay/Replay_2\n", encoding="utf-8")

    result = SiWMv2Adapter(
        root_dir=root,
        protocol_dir=protocol,
        frames_dir=tmp_path / "frames",
        output_dir=tmp_path / "manifest",
        frames_per_video=2,
    ).build()
    rows = read_manifest(result.output_dir)
    assert len(rows) == 8
    assert {(row.split, row.label) for row in rows} == {
        ("train", 0), ("train", 1), ("test", 0), ("test", 1)
    }
    assert {row.attack_type for row in rows if row.label == 1} == {"makeup_cosmetic", "replay"}
    assert all((tmp_path / "frames" / row.path).is_file() for row in rows)
    assert result.output_dir.joinpath("siwmv2_adapter_report.json").is_file()
