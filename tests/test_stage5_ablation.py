import pytest
import torch

from improved_ddpm.experiments.ablation import (
    ABLATION_REGISTRY,
    get_ablation_spec,
    resolve_ablation_spec,
)
from improved_ddpm.models.factory import build_fas_model


def _config(variant):
    return {
        "experiment": {"variant": variant, "overrides": {}},
        "amp_channels": 3,
        "classifier": {
            "feat_dim": 16,
            "num_classes": 2,
            "pretrained_backbone": False,
        },
    }


def test_canonical_stage5_registry_is_complete():
    assert list(ABLATION_REGISTRY) == ["E0", "E1", "E2", "E3", "E4", "E5", "E6"]
    assert get_ablation_spec("E0").use_rgb
    assert not get_ablation_spec("E0").use_fft
    assert get_ablation_spec("E1").use_fft
    assert not get_ablation_spec("E1").use_rgb
    assert not get_ablation_spec("E5").use_ddpm
    assert get_ablation_spec("E6").use_ddpm


def test_canonical_component_matrix_changes_one_intended_component():
    expected = {
        "E0": (True, False, False, False, False, False),
        "E1": (False, True, False, False, False, False),
        "E2": (True, True, False, False, False, False),
        "E3": (True, True, True, False, False, False),
        "E4": (True, True, True, True, False, False),
        "E5": (True, True, True, True, True, False),
        "E6": (True, True, True, True, True, True),
    }
    for variant, values in expected.items():
        spec = get_ablation_spec(variant)
        assert (
            spec.use_rgb,
            spec.use_fft,
            spec.use_patch,
            spec.use_contrastive,
            spec.use_center,
            spec.use_ddpm,
        ) == values


def test_invalid_override_is_rejected():
    cfg = _config("E0")
    cfg["experiment"]["overrides"] = {"use_rgb": False}
    with pytest.raises(ValueError, match="At least one input branch"):
        resolve_ablation_spec(cfg)


def test_e0_rgb_only_forward_and_loss():
    model, spec = build_fas_model(_config("E0"), pretrained=False)
    rgb = torch.randn(2, 3, 32, 32)
    labels = torch.tensor([0, 1])
    outputs = model(rgb, None)
    losses = model.compute_loss(outputs, labels, alpha_center=0.0, lambda_contrastive=0.0)
    assert spec.variant == "E0"
    assert outputs["f_amp"] is None
    assert outputs["logits"].shape == (2, 2)
    assert losses["l_contrastive"].item() == pytest.approx(0.0)
    assert losses["l_center"].item() == pytest.approx(0.0)


def test_e1_fft_only_forward():
    model, spec = build_fas_model(_config("E1"), pretrained=False)
    amplitude = torch.randn(2, 3, 32, 32)
    outputs = model(None, amplitude)
    assert spec.variant == "E1"
    assert outputs["f_rgb"] is None
    assert outputs["f_amp"].shape == (2, 16)
    assert outputs["logits"].shape == (2, 2)


def test_e5_multistream_losses_are_finite():
    cfg = _config("E5")
    model, spec = build_fas_model(cfg, pretrained=False)
    rgb = torch.randn(4, 3, 32, 32)
    amp = torch.randn(4, 3, 32, 32)
    labels = torch.tensor([0, 0, 1, 1])
    full = model(rgb, amp)
    patch = model(rgb.clone(), amp.clone())
    losses = model.compute_multi_stream_loss(
        full,
        patch,
        labels,
        alpha_center=0.2,
        lambda_contrastive=1.0,
    )
    assert spec.use_patch and spec.use_contrastive and spec.use_center
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
