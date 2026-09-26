import json

import numpy as np
import pytest

from improved_ddpm.evaluation.protocol import (
    aggregate_frame_rows,
    attach_video_results,
    evaluate_rows,
    tune_development_policy,
)
from improved_ddpm.metrics.fas_metrics import (
    find_eer_threshold,
    find_min_acer_threshold,
    predictions_from_scores,
)


def _rows(split="dev"):
    rows = []
    videos = {
        "live_1": (0, [0.05, 0.10, 0.15], "live"),
        "live_2": (0, [0.15, 0.20, 0.25], "live"),
        "attack_1": (1, [0.55, 0.70, 0.80], "print"),
        "attack_2": (1, [0.60, 0.75, 0.95], "replay"),
    }
    for video_id, (label, scores, attack_type) in videos.items():
        for frame_index, score in enumerate(scores):
            rows.append(
                {
                    "dataset": "Synthetic",
                    "protocol": "1",
                    "split": split,
                    "subject_id": video_id,
                    "video_id": video_id,
                    "frame_id": f"{frame_index:06d}",
                    "path": f"/{video_id}/{frame_index}.png",
                    "label": label,
                    "frame_score": score,
                    "attack_type": attack_type,
                }
            )
    return rows


def test_threshold_selection_is_score_driven_not_fixed_half():
    labels = [0, 0, 1, 1]
    # Perfectly separable, but all useful scores are below 0.5.
    scores = [0.05, 0.10, 0.20, 0.25]
    eer = find_eer_threshold(labels, scores)
    min_acer = find_min_acer_threshold(labels, scores)
    assert eer["threshold"] == pytest.approx(0.20)
    assert min_acer["threshold"] == pytest.approx(0.20)
    assert eer["threshold"] != 0.5


def test_prediction_rule_uses_frozen_threshold():
    pred = predictions_from_scores([0.39, 0.40, 0.41], threshold=0.40)
    assert pred.tolist() == [0, 1, 1]


def test_video_aggregation_methods():
    rows = _rows()
    mean_rows = aggregate_frame_rows(rows, "mean")
    median_rows = aggregate_frame_rows(rows, "median")
    topk_rows = aggregate_frame_rows(rows, "topk_mean", topk_ratio=1 / 3)
    lookup_mean = {row["video_id"]: row["video_score"] for row in mean_rows}
    lookup_median = {row["video_id"]: row["video_score"] for row in median_rows}
    lookup_topk = {row["video_id"]: row["video_score"] for row in topk_rows}
    assert lookup_mean["attack_1"] == pytest.approx((0.55 + 0.70 + 0.80) / 3)
    assert lookup_median["attack_1"] == pytest.approx(0.70)
    assert lookup_topk["attack_1"] == pytest.approx(0.80)


def test_policy_tuning_accepts_development_only():
    with pytest.raises(ValueError, match="development rows only"):
        tune_development_policy(_rows(split="test"))


def test_policy_and_prediction_outputs_are_complete():
    rows = _rows()
    policy = tune_development_policy(
        rows,
        threshold_method="eer",
        aggregation_candidates=["mean", "median"],
    )
    assert policy["selection_split"] == "dev"
    assert policy["aggregation"] in {"mean", "median"}
    assert policy["development_metrics"]["ACER"] == pytest.approx(0.0)

    frames, videos = attach_video_results(
        rows,
        aggregation=policy["aggregation"],
        threshold=policy["threshold"],
    )
    assert all("frame_score" in row and "video_score" in row for row in frames)
    assert all("prediction" in row and "frame_prediction" in row for row in frames)
    metrics = evaluate_rows(
        videos,
        score_field="video_score",
        threshold=policy["threshold"],
    )
    assert metrics["HTER"] == pytest.approx(0.0)
    assert metrics["AUC"] == pytest.approx(1.0)
    assert metrics["EER"] == pytest.approx(0.0)


def test_video_label_must_be_consistent():
    rows = _rows()
    rows[0] = dict(rows[0], label=1)
    with pytest.raises(ValueError, match="inconsistent labels"):
        aggregate_frame_rows(rows, "mean")
