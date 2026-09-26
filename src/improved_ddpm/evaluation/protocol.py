"""Frame/video score aggregation and development-only policy selection."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np

from improved_ddpm.metrics.fas_metrics import (
    evaluate_fas_metrics,
    predictions_from_scores,
    select_threshold,
)


FRAME_PREDICTION_COLUMNS = (
    "dataset",
    "protocol",
    "split",
    "subject_id",
    "video_id",
    "frame_id",
    "path",
    "label",
    "frame_score",
    "video_score",
    "prediction",
    "frame_prediction",
    "attack_type",
    "device",
    "illumination",
    "environment",
    "session",
    "camera",
)

VIDEO_PREDICTION_COLUMNS = (
    "dataset",
    "protocol",
    "split",
    "subject_id",
    "video_id",
    "label",
    "video_score",
    "prediction",
    "attack_type",
    "frame_count",
    "aggregation",
)


def aggregate_values(
    values: Sequence[float],
    method: str,
    *,
    trim_ratio: float = 0.1,
    topk_ratio: float = 0.25,
) -> float:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError("cannot aggregate an empty score list")
    if not np.isfinite(array).all():
        raise ValueError("frame scores contain NaN or infinity")

    method = method.strip().lower()
    if method == "mean":
        return float(np.mean(array))
    if method == "median":
        return float(np.median(array))
    if method == "trimmed_mean":
        if not 0 <= trim_ratio < 0.5:
            raise ValueError("trim_ratio must be in [0, 0.5)")
        ordered = np.sort(array)
        trim_count = int(math.floor(ordered.size * trim_ratio))
        if trim_count > 0 and ordered.size - 2 * trim_count > 0:
            ordered = ordered[trim_count:-trim_count]
        return float(np.mean(ordered))
    if method == "topk_mean":
        if not 0 < topk_ratio <= 1:
            raise ValueError("topk_ratio must be in (0, 1]")
        count = max(1, int(math.ceil(array.size * topk_ratio)))
        return float(np.mean(np.sort(array)[-count:]))
    raise ValueError("aggregation must be mean, median, trimmed_mean, or topk_mean")


def _video_key(row: Mapping[str, object]) -> Tuple[str, str, str, str, str]:
    return (
        str(row.get("dataset", "")),
        str(row.get("protocol", "")),
        str(row.get("split", "")),
        str(row.get("subject_id", "")),
        str(row.get("video_id", "")),
    )


def aggregate_frame_rows(
    rows: Iterable[Mapping[str, object]],
    method: str,
    *,
    trim_ratio: float = 0.1,
    topk_ratio: float = 0.25,
) -> List[dict]:
    """Aggregate frame attack scores into one score per video."""
    groups: MutableMapping[Tuple[str, str, str, str, str], List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        if "frame_score" not in row:
            raise ValueError("each frame row must contain frame_score")
        groups[_video_key(row)].append(row)
    if not groups:
        raise ValueError("no frame rows to aggregate")

    videos: List[dict] = []
    for key in sorted(groups):
        group = groups[key]
        labels = {int(item["label"]) for item in group}
        if len(labels) != 1:
            raise ValueError(f"video has inconsistent labels: {key} -> {sorted(labels)}")
        attack_types = {str(item.get("attack_type", "")) for item in group if str(item.get("attack_type", ""))}
        if len(attack_types) > 1:
            raise ValueError(f"video has inconsistent attack_type values: {key}")
        score = aggregate_values(
            [float(item["frame_score"]) for item in group],
            method,
            trim_ratio=trim_ratio,
            topk_ratio=topk_ratio,
        )
        videos.append(
            {
                "dataset": key[0],
                "protocol": key[1],
                "split": key[2],
                "subject_id": key[3],
                "video_id": key[4],
                "label": labels.pop(),
                "video_score": score,
                "attack_type": next(iter(attack_types), ""),
                "frame_count": len(group),
                "aggregation": method,
            }
        )
    return videos


def evaluate_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    score_field: str,
    threshold: float,
    tpr_fprs: Sequence[float] = (0.01, 0.001),
) -> dict:
    labels = [int(row["label"]) for row in rows]
    scores = [float(row[score_field]) for row in rows]
    return evaluate_fas_metrics(
        labels=labels,
        scores=scores,
        threshold=float(threshold),
        tpr_fprs=tpr_fprs,
    )


def tune_development_policy(
    frame_rows: Sequence[Mapping[str, object]],
    *,
    threshold_method: str = "eer",
    aggregation_candidates: Sequence[str] = ("mean", "median", "trimmed_mean", "topk_mean"),
    trim_ratio: float = 0.1,
    topk_ratio: float = 0.25,
    tpr_fprs: Sequence[float] = (0.01, 0.001),
) -> dict:
    """Choose aggregation and threshold using development data only."""
    splits = {str(row.get("split", "")).lower() for row in frame_rows}
    if not splits or not splits.issubset({"dev", "development", "val", "validation"}):
        raise ValueError(
            "tune_development_policy accepts development rows only; test rows must never be used"
        )

    trials = []
    for aggregation in aggregation_candidates:
        video_rows = aggregate_frame_rows(
            frame_rows,
            aggregation,
            trim_ratio=trim_ratio,
            topk_ratio=topk_ratio,
        )
        labels = [int(row["label"]) for row in video_rows]
        scores = [float(row["video_score"]) for row in video_rows]
        threshold_result = select_threshold(labels, scores, threshold_method)
        metrics = evaluate_fas_metrics(
            labels=labels,
            scores=scores,
            threshold=threshold_result["threshold"],
            tpr_fprs=tpr_fprs,
        )
        trials.append(
            {
                "aggregation": aggregation,
                "threshold_method": threshold_result["method"],
                "threshold": threshold_result["threshold"],
                "development_eer": metrics["EER"],
                "development_metrics": _json_safe_metrics(metrics),
                "video_count": len(video_rows),
            }
        )

    method = threshold_method.strip().lower()
    if method == "eer":
        key = lambda item: (
            item["development_eer"],
            item["development_metrics"]["ACER"],
            -item["development_metrics"]["AUC"],
            item["aggregation"],
        )
    else:
        key = lambda item: (
            item["development_metrics"]["ACER"],
            item["development_eer"],
            -item["development_metrics"]["AUC"],
            item["aggregation"],
        )
    best = min(trials, key=key)
    return {
        "version": 1,
        "selection_split": "dev",
        "threshold_method": best["threshold_method"],
        "threshold": float(best["threshold"]),
        "aggregation": best["aggregation"],
        "aggregation_parameters": {
            "trim_ratio": float(trim_ratio),
            "topk_ratio": float(topk_ratio),
        },
        "development_eer": float(best["development_eer"]),
        "development_metrics": best["development_metrics"],
        "candidate_results": trials,
    }


def attach_video_results(
    frame_rows: Sequence[Mapping[str, object]],
    *,
    aggregation: str,
    threshold: float,
    trim_ratio: float = 0.1,
    topk_ratio: float = 0.25,
) -> Tuple[List[dict], List[dict]]:
    """Attach frozen video scores/predictions to frame rows and video rows."""
    video_rows = aggregate_frame_rows(
        frame_rows,
        aggregation,
        trim_ratio=trim_ratio,
        topk_ratio=topk_ratio,
    )
    lookup = {}
    for row in video_rows:
        row = dict(row)
        row["prediction"] = int(float(row["video_score"]) >= float(threshold))
        lookup[_video_key(row)] = row
    video_rows = [lookup[key] for key in sorted(lookup)]

    output_frames: List[dict] = []
    for original in frame_rows:
        row = dict(original)
        video = lookup[_video_key(row)]
        row["video_score"] = float(video["video_score"])
        row["prediction"] = int(video["prediction"])
        row["frame_prediction"] = int(float(row["frame_score"]) >= float(threshold))
        output_frames.append(row)
    return output_frames, video_rows


def _json_safe_metrics(metrics: Mapping[str, object]) -> dict:
    output = {}
    for key, value in metrics.items():
        if key == "confusion_matrix":
            output[key] = np.asarray(value).astype(int).tolist()
        elif isinstance(value, (np.floating, np.integer)):
            output[key] = value.item()
        else:
            output[key] = value
    return output


def json_safe_metrics(metrics: Mapping[str, object]) -> dict:
    return _json_safe_metrics(metrics)


def write_csv(path: Union[str, Path], rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    return path


def write_json(path: Union[str, Path], payload: Mapping[str, object]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=str(path.parent), delete=False) as temp:
        temp.write(content)
        temp_name = temp.name
    os.replace(temp_name, path)
    return path
