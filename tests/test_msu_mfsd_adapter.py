from pathlib import Path

import cv2
import numpy as np

from improved_ddpm.data.adapters.msu_mfsd import (
    MSUMFSDAdapter,
    infer_video_metadata,
    parse_subject_id,
)
from improved_ddpm.data.manifest import read_manifest


TRAIN = ("02", "03", "05", "06", "07", "08", "09", "11", "12", "21", "22", "34", "53", "54", "55")
TEST = ("01", "13", "14", "23", "24", "26", "28", "29", "30", "32", "33", "35", "36", "37", "39", "42", "48", "49", "50", "51")


def _write_video(path: Path, frames: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (32, 24),
    )
    assert writer.isOpened()
    for index in range(frames):
        image = np.zeros((24, 32, 3), dtype=np.uint8)
        image[:, :, index % 3] = 20 + index * 10
        writer.write(image)
    writer.release()


def test_name_parsing_and_metadata():
    subjects = set(TRAIN) | set(TEST)
    assert parse_subject_id("real_client002_android_SD_scene01.mp4", subjects) == "02"
    assert parse_subject_id("attack/client001_laptop_video.mov", subjects) == "01"
    assert infer_video_metadata(Path("real/real_client002_android.avi"))["label"] == 0
    attack = infer_video_metadata(Path("attack/attack_client001_ipad_video.avi"))
    assert attack["label"] == 1
    assert attack["attack_type"] == "replay_tablet"


def test_adapter_keeps_official_subject_split_and_extracts_png(tmp_path):
    release = tmp_path / "MSU-MFSD-Publish"
    release.mkdir()
    (release / "train_sub_list.txt").write_text("\n".join(TRAIN) + "\n", encoding="utf-8")
    (release / "test_sub_list.txt").write_text("\n".join(TEST) + "\n", encoding="utf-8")
    _write_video(release / "scene01" / "real" / "real_client002_android.avi")
    _write_video(release / "scene01" / "attack" / "attack_client001_ipad_video.avi")

    frames_root = tmp_path / "frames"
    manifest_root = tmp_path / "manifest"
    result = MSUMFSDAdapter(
        root_dir=release,
        frames_dir=frames_root,
        output_dir=manifest_root,
        frames_per_video=4,
        path_mode="relative",
        strict_counts=False,
        strict_decode=True,
    ).build()

    records = read_manifest(result.output_dir)
    assert len(records) == 8
    assert {row.subject_id for row in records if row.split == "train"} == {"02"}
    assert {row.subject_id for row in records if row.split == "test"} == {"01"}
    assert {row.label for row in records if row.split == "train"} == {0}
    assert {row.label for row in records if row.split == "test"} == {1}
    assert all((frames_root / row.path).is_file() for row in records)
    assert all(Path(row.path).suffix == ".png" for row in records)


def test_uniform_decode_recovers_from_overreported_container_count(tmp_path):
    video = tmp_path / "short.avi"
    _write_video(video, frames=10)

    decoded, effective_total, fallback_backend = MSUMFSDAdapter._decode_uniform_frames(
        video,
        reported_total=20,
        target=4,
    )

    assert fallback_backend == "ffmpeg_skip_corrupt"
    assert effective_total == 10
    assert len(decoded) == 4
    assert [index for index, _ in decoded] == [1, 3, 6, 8]
