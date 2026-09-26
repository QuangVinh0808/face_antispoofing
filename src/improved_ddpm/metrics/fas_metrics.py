"""Standard metrics and threshold selection for face anti-spoofing.

Label/score convention used by the complete project:
    0 = bona fide/live
    1 = presentation attack/spoof
    higher score = more likely to be attack

A threshold is applied as ``prediction = int(score >= threshold)``.
Thresholds must be selected on a development split and then frozen for test.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


def _to_numpy(x) -> np.ndarray:
    """Convert a tensor/list/array to a one-dimensional NumPy array."""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().reshape(-1)
    return np.asarray(x).reshape(-1)


def _validate_labels_scores(labels, scores) -> tuple[np.ndarray, np.ndarray]:
    y_true = _to_numpy(labels).astype(int)
    y_score = _to_numpy(scores).astype(float)
    if y_true.size == 0:
        raise ValueError("labels/scores must not be empty")
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError(
            f"labels ({y_true.shape[0]}) and scores ({y_score.shape[0]}) must have equal length"
        )
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("labels must use 0=live and 1=attack")
    if not np.isfinite(y_score).all():
        raise ValueError("scores contain NaN or infinity")
    return y_true, y_score


def predictions_from_scores(scores, threshold: float) -> np.ndarray:
    """Convert continuous attack scores to labels using a frozen threshold."""
    score_array = _to_numpy(scores).astype(float)
    if not np.isfinite(score_array).all():
        raise ValueError("scores contain NaN or infinity")
    if not np.isfinite(float(threshold)):
        raise ValueError("threshold must be finite")
    return (score_array >= float(threshold)).astype(np.int64)


def _threshold_candidates(y_score: np.ndarray) -> np.ndarray:
    """Return every distinct operating point, including the all-live point."""
    unique = np.unique(y_score)
    above_max = np.nextafter(float(unique.max()), float("inf"))
    return np.concatenate(([above_max], unique[::-1])).astype(float)


def _rates_for_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute APCER, BPCER and ACER for threshold candidates efficiently."""
    order = np.argsort(-y_score, kind="mergesort")
    sorted_scores = y_score[order]
    sorted_labels = y_true[order]

    n_attack = int(np.sum(sorted_labels == 1))
    n_live = int(np.sum(sorted_labels == 0))
    if n_attack == 0 or n_live == 0:
        raise ValueError("threshold selection requires both live and attack samples")

    attack_cumsum = np.cumsum(sorted_labels == 1)
    live_cumsum = np.cumsum(sorted_labels == 0)

    # Number predicted attack for each threshold: scores >= threshold.
    counts = np.searchsorted(-sorted_scores, -thresholds, side="right")
    tp = np.where(counts > 0, attack_cumsum[np.maximum(counts - 1, 0)], 0)
    fp = np.where(counts > 0, live_cumsum[np.maximum(counts - 1, 0)], 0)
    fn = n_attack - tp

    apcer = fn.astype(float) / float(n_attack)
    bpcer = fp.astype(float) / float(n_live)
    acer = (apcer + bpcer) / 2.0
    return apcer, bpcer, acer


def find_eer_threshold(labels, scores) -> Dict[str, float]:
    """Select the development threshold where APCER and BPCER are closest.

    The returned EER is the mean of APCER and BPCER at the selected discrete
    operating point. Ties are resolved by lower ACER, then lower APCER/BPCER
    gap, then the larger threshold for deterministic output.
    """
    y_true, y_score = _validate_labels_scores(labels, scores)
    thresholds = _threshold_candidates(y_score)
    apcer, bpcer, acer = _rates_for_thresholds(y_true, y_score, thresholds)
    gap = np.abs(apcer - bpcer)
    order = np.lexsort((-thresholds, acer, gap))
    index = int(order[0])
    return {
        "method": "eer",
        "threshold": float(thresholds[index]),
        "eer": float(acer[index]),
        "APCER": float(apcer[index]),
        "BPCER": float(bpcer[index]),
        "ACER": float(acer[index]),
    }


def find_min_acer_threshold(labels, scores) -> Dict[str, float]:
    """Select the development threshold that minimizes ACER."""
    y_true, y_score = _validate_labels_scores(labels, scores)
    thresholds = _threshold_candidates(y_score)
    apcer, bpcer, acer = _rates_for_thresholds(y_true, y_score, thresholds)
    gap = np.abs(apcer - bpcer)
    order = np.lexsort((-thresholds, gap, acer))
    index = int(order[0])
    return {
        "method": "min_acer",
        "threshold": float(thresholds[index]),
        "eer": float((apcer[index] + bpcer[index]) / 2.0),
        "APCER": float(apcer[index]),
        "BPCER": float(bpcer[index]),
        "ACER": float(acer[index]),
    }


def select_threshold(labels, scores, method: str = "eer") -> Dict[str, float]:
    method = str(method).strip().lower()
    if method == "eer":
        return find_eer_threshold(labels, scores)
    if method in {"min_acer", "acer"}:
        return find_min_acer_threshold(labels, scores)
    raise ValueError("threshold method must be 'eer' or 'min_acer'")


def calculate_eer(labels, scores) -> float:
    """Calculate EER independently of the frozen classification threshold."""
    return float(find_eer_threshold(labels, scores)["eer"])


def tpr_at_fixed_fpr(labels, scores, targets: Sequence[float]) -> Dict[str, float]:
    """Return the highest observed TPR whose FPR does not exceed each target."""
    y_true, y_score = _validate_labels_scores(labels, scores)
    if np.unique(y_true).size < 2:
        return {f"TPR@FPR={float(target):g}": float("nan") for target in targets}
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    output: Dict[str, float] = {}
    for target in targets:
        target = float(target)
        if target < 0 or target > 1:
            raise ValueError("FPR targets must be in [0, 1]")
        allowed = tpr[fpr <= target + 1e-15]
        output[f"TPR@FPR={target:g}"] = float(allowed.max()) if allowed.size else 0.0
    return output


def evaluate_fas_metrics(
    predictions=None,
    labels=None,
    scores=None,
    *,
    threshold: Optional[float] = None,
    tpr_fprs: Optional[Sequence[float]] = None,
):
    """Compute APCER, BPCER, ACER/HTER, AUC and EER.

    Backward compatible usage is preserved::

        evaluate_fas_metrics(predictions, labels, scores)

    Preferred Stage-4 usage derives predictions from a development threshold::

        evaluate_fas_metrics(labels=labels, scores=scores, threshold=threshold)
    """
    if labels is None:
        raise ValueError("labels are required")
    y_true = _to_numpy(labels).astype(int)

    if threshold is not None:
        if scores is None:
            raise ValueError("scores are required when threshold is provided")
        y_pred = predictions_from_scores(scores, threshold)
    elif predictions is not None:
        y_pred = _to_numpy(predictions).astype(int)
    else:
        raise ValueError("provide predictions, or provide scores together with threshold")

    if y_pred.shape[0] != y_true.shape[0]:
        raise ValueError(
            f"predictions ({y_pred.shape[0]}) and labels ({y_true.shape[0]}) must have equal length"
        )
    if not np.isin(y_true, [0, 1]).all() or not np.isin(y_pred, [0, 1]).all():
        raise ValueError("labels and predictions must use 0=live and 1=attack")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    n_attack = fn + tp
    n_live = tn + fp
    apcer = fn / n_attack if n_attack > 0 else 0.0
    bpcer = fp / n_live if n_live > 0 else 0.0
    acer = (apcer + bpcer) / 2.0

    y_score = None
    if scores is not None:
        y_score = _to_numpy(scores).astype(float)
        if y_score.shape[0] != y_true.shape[0]:
            raise ValueError("scores and labels must have equal length")
        if not np.isfinite(y_score).all():
            if threshold is not None:
                raise ValueError("scores contain NaN or infinity")
            # Backward-compatible behavior for legacy callers that supplied
            # hard predictions together with unusable scores. Stage-4
            # thresholded evaluation remains strict and never uses this path.
            y_score = y_pred.astype(float)

    if y_score is None or np.unique(y_true).size < 2:
        auc = float("nan")
        eer = float("nan")
    else:
        auc = float(roc_auc_score(y_true, y_score))
        eer = calculate_eer(y_true, y_score)

    result = {
        "APCER": float(apcer),
        "BPCER": float(bpcer),
        "ACER": float(acer),
        "HTER": float(acer),
        "AUC": float(auc),
        "EER": float(eer),
        "threshold": float(threshold) if threshold is not None else None,
        "confusion_matrix": cm,
        "sample_count": int(y_true.size),
        "live_count": int(np.sum(y_true == 0)),
        "attack_count": int(np.sum(y_true == 1)),
    }
    if y_score is not None and tpr_fprs:
        result.update(tpr_at_fixed_fpr(y_true, y_score, tpr_fprs))
    return result


def format_fas_metrics(metrics: dict) -> str:
    """Format the main metrics for console logging."""
    text = (
        f"APCER={metrics['APCER']*100:.2f}% | "
        f"BPCER={metrics['BPCER']*100:.2f}% | "
        f"ACER={metrics['ACER']*100:.2f}% | "
        f"AUC={metrics['AUC']:.4f}"
    )
    if "EER" in metrics and np.isfinite(metrics["EER"]):
        text += f" | EER={metrics['EER']*100:.2f}%"
    if metrics.get("threshold") is not None:
        text += f" | threshold={metrics['threshold']:.6f}"
    return text
