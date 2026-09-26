"""Stage 3 tests: protocol-aware sampling, face crops, and train-only FFT stats."""

from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from improved_ddpm.data.face_preprocessing import FaceDetection, preprocess_face_crops
from improved_ddpm.data.frame_sampling import FrameSamplingConfig, sample_manifest_records
from improved_ddpm.data.manifest import ManifestRecord, read_manifest, write_manifest_bundle
from improved_ddpm.fas.frequency_normalization import (
    compute_fft_normalization,
    image_to_model_tensor,
    load_fft_normalization,
    normalize_log_amplitude,
)
from improved_ddpm.fas.frequency_utils import FrequencyUtils
from improved_ddpm.scripts.preprocess_manifest import run


class FixedFaceDetector:
    name = "fixed_test_detector"

    def detect(self, image_rgb):
        return FaceDetection(25, 25, 75, 75, backend=self.name)


def _row(split: str, video: str, frame: int, path: str, label: int = 0, dataset: str = "Unit"):
    return ManifestRecord(
        dataset=dataset,
        protocol="1",
        split=split,
        subject_id=video.split("_")[0],
        video_id=video,
        frame_id=f"{frame:06d}",
        path=path,
        label=label,
        attack_type="live" if label == 0 else "print",
    )


def test_uniform_sampling_is_deterministic():
    generic = [_row("train", "001_video", i, f"/tmp/generic_{i}.jpg") for i in range(100)]
    selected, report = sample_manifest_records(
        generic,
        FrameSamplingConfig(train=16, dev=32, test=32, protocol_policy="official"),
    )
    assert len(selected) == 16
    assert selected[0].frame_id != selected[1].frame_id
    assert report["splits"]["train"]["output_frames"] == 16



def test_face_tight_and_context_crops_are_materialized(tmp_path):
    source = tmp_path / "source.jpg"
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[25:75, 25:75] = 255
    Image.fromarray(image).save(source)
    records = [_row("train", "001_video", 1, str(source), 0)]

    records_by_mode, metadata, report = preprocess_face_crops(
        records,
        data_root=None,
        output_root=tmp_path / "processed",
        detector=FixedFaceDetector(),
        crop_modes=("face_tight", "face_context"),
        image_size=32,
        missing_face="error",
        path_mode="absolute",
    )

    for mode in ("face_tight", "face_context"):
        assert len(records_by_mode[mode]) == 1
        output = Path(records_by_mode[mode][0].path)
        assert output.is_file()
        assert Image.open(output).size == (32, 32)
    tight = metadata["face_tight"][0]
    context = metadata["face_context"][0]
    assert tight["bbox_x1"] == 25 and tight["bbox_x2"] == 75
    assert context["bbox_x1"] < tight["bbox_x1"]
    assert context["bbox_x2"] > tight["bbox_x2"]
    assert report["detected_frames"] == 1


def test_fft_statistics_use_train_only(tmp_path):
    train_path = tmp_path / "train.png"
    dev_path = tmp_path / "dev.png"
    train_array = np.zeros((16, 16, 3), dtype=np.uint8)
    train_array[:, 8:] = 180
    dev_array = np.full((16, 16, 3), 255, dtype=np.uint8)
    Image.fromarray(train_array).save(train_path)
    Image.fromarray(dev_array).save(dev_path)
    rows = [
        _row("train", "001_video", 1, str(train_path), 0),
        _row("dev", "101_video", 1, str(dev_path), 1),
    ]

    stats = compute_fft_normalization(rows, image_size=16, mode="global")
    train_tensor = image_to_model_tensor(train_path, 16).unsqueeze(0)
    amp, _ = FrequencyUtils.fft_decompose(train_tensor)
    expected = FrequencyUtils.log_amplitude(amp)
    assert abs(stats.mean[0] - expected.mean().item()) < 1e-6
    assert abs(stats.std[0] - expected.std(unbiased=False).item()) < 1e-6
    normalized = normalize_log_amplitude(expected, stats)
    assert abs(normalized.mean().item()) < 1e-5
    assert abs(normalized.std(unbiased=False).item() - 1.0) < 1e-5


def test_ddpm_statistics_can_be_restricted_to_source_train_attacks(tmp_path):
    live_path = tmp_path / "live.png"
    attack_path = tmp_path / "attack.png"
    Image.fromarray(np.full((16, 16, 3), 32, dtype=np.uint8)).save(live_path)
    attack_array = np.zeros((16, 16, 3), dtype=np.uint8)
    attack_array[:, ::2] = 255
    Image.fromarray(attack_array).save(attack_path)
    rows = [
        _row("train", "001_live", 1, str(live_path), 0),
        _row("train", "002_attack", 1, str(attack_path), 1),
    ]
    stats = compute_fft_normalization(rows, image_size=16, mode="global", labels=(1,))
    attack = image_to_model_tensor(attack_path, 16).unsqueeze(0)
    amplitude, _ = FrequencyUtils.fft_decompose(attack)
    expected = FrequencyUtils.log_amplitude(amplitude)
    assert stats.image_count == 1
    assert abs(stats.mean[0] - expected.mean().item()) < 1e-6


def test_stage3_pipeline_end_to_end_with_offline_full_frame_detector(tmp_path):
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    rows = []
    for split_index, split in enumerate(("train", "dev", "test")):
        for label in (0, 1):
            path = data_dir / f"{split}_{label}.png"
            array = np.zeros((24, 32, 3), dtype=np.uint8)
            array[:, :, label] = 80 + split_index * 30
            array[4:20, 8:24] += 80
            Image.fromarray(array).save(path)
            subject = f"{split_index}{label}1"
            rows.append(_row(split, f"{subject}_video", 1, str(path), label))
    manifest_dir = tmp_path / "manifest"
    write_manifest_bundle(rows, manifest_dir)

    output_dir = tmp_path / "processed"
    result = run(
        Namespace(
            manifest_dir=str(manifest_dir),
            data_root=None,
            output_dir=str(output_dir),
            train_frames=None,
            dev_frames=None,
            test_frames=None,
            protocol_frame_policy="uniform",
            crop_modes=("face_tight", "face_context"),
            image_size=16,
            detector="full_frame",
            mtcnn_device="auto",
            mtcnn_min_probability=0.90,
            mtcnn_min_face_size=40,
            missing_face="error",
            output_format="png",
            path_mode="absolute",
            fft_normalization="global",
            cache_log_amplitude=True,
            force=False,
        )
    )

    assert set(result["crop_modes"]) == {"face_tight", "face_context"}
    for mode in ("face_tight", "face_context"):
        mode_dir = output_dir / mode
        assert len(read_manifest(mode_dir)) == 6
        stats = load_fft_normalization(mode_dir / "normalization.json")
        assert stats.split == "train"
        assert stats.image_count == 2
        assert (mode_dir / "fft_cache" / "cache_index.csv").is_file()
        assert (mode_dir / "preprocess_metadata.csv").is_file()
