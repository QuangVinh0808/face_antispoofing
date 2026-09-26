"""
tests/test_metrics.py
========================
Test cho evaluate_fas_metrics() - Cap nhat 3 (APCER/BPCER/ACER/AUC).
Chay: pytest tests/test_metrics.py -v
"""

import numpy as np
import torch
import pytest

from improved_ddpm.metrics.fas_metrics import evaluate_fas_metrics, format_fas_metrics


def test_perfect_classifier():
    """Du doan dung 100% -> APCER=BPCER=ACER=0, AUC=1.0."""
    labels = [0, 0, 0, 1, 1, 1]
    predictions = [0, 0, 0, 1, 1, 1]
    scores = [0.1, 0.05, 0.2, 0.9, 0.95, 0.8]  # diem Attack cao cho mau Attack, thap cho Live

    metrics = evaluate_fas_metrics(predictions, labels, scores)
    assert metrics["APCER"] == 0.0
    assert metrics["BPCER"] == 0.0
    assert metrics["ACER"] == 0.0
    assert metrics["AUC"] == 1.0


def test_worst_classifier():
    """Du doan SAI 100% (Live<->Attack lat nguoc) -> APCER=BPCER=ACER=1.0."""
    labels = [0, 0, 1, 1]
    predictions = [1, 1, 0, 0]  # nguoc hoan toan

    metrics = evaluate_fas_metrics(predictions, labels)
    assert metrics["APCER"] == 1.0
    assert metrics["BPCER"] == 1.0
    assert metrics["ACER"] == 1.0


def test_apcer_bpcer_known_confusion_matrix():
    """
    Kiem tra tay 1 truong hop cu the:
    Live: 4 mau (label=0), trong do 1 mau bi doan nham thanh Attack (FP=1) -> BPCER=1/4=0.25
    Attack: 4 mau (label=1), trong do 2 mau bi doan nham thanh Live (FN=2)  -> APCER=2/4=0.5
    """
    labels =      [0, 0, 0, 0, 1, 1, 1, 1]
    predictions = [0, 0, 0, 1, 1, 1, 0, 0]  # FP=1 (Live#4), FN=2 (Attack#3,#4)

    metrics = evaluate_fas_metrics(predictions, labels)
    assert metrics["BPCER"] == pytest.approx(0.25)
    assert metrics["APCER"] == pytest.approx(0.5)
    assert metrics["ACER"] == pytest.approx((0.25 + 0.5) / 2)


def test_accepts_torch_tensor_input():
    """Ham phai chay duoc voi input la torch.Tensor (giong outputs['pred']/['probs'] thuc te)."""
    labels = torch.tensor([0, 0, 1, 1])
    predictions = torch.tensor([0, 1, 1, 1])
    scores = torch.tensor([0.1, 0.6, 0.7, 0.9])

    metrics = evaluate_fas_metrics(predictions, labels, scores)
    assert isinstance(metrics["APCER"], float)
    assert isinstance(metrics["AUC"], float)
    assert np.isfinite(metrics["AUC"])


def test_single_class_labels_returns_nan_auc():
    """Neu labels chi co 1 class (vd toan bo la Live) -> AUC tra ve NaN, khong crash."""
    labels = [0, 0, 0, 0]
    predictions = [0, 0, 1, 0]
    metrics = evaluate_fas_metrics(predictions, labels)
    assert np.isnan(metrics["AUC"])
    # BPCER van tinh duoc binh thuong (1 Live bi doan nham thanh Attack / 4 Live)
    assert metrics["BPCER"] == pytest.approx(0.25)


def test_nan_scores_fallback_to_predictions_for_auc():
    """Neu scores chua NaN/inf thi ham phai khong crash va fallback ve predictions."""
    labels = [0, 0, 1, 1]
    predictions = [0, 1, 1, 1]
    scores = [0.1, np.nan, 0.7, 0.9]

    metrics = evaluate_fas_metrics(predictions, labels, scores)
    assert np.isfinite(metrics["AUC"])
    assert metrics["APCER"] == pytest.approx(0.0)
    assert metrics["BPCER"] == pytest.approx(0.5)


def test_format_fas_metrics_returns_string():
    metrics = evaluate_fas_metrics([0, 1], [0, 1], [0.1, 0.9])
    s = format_fas_metrics(metrics)
    assert "APCER" in s and "ACER" in s and "AUC" in s


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
