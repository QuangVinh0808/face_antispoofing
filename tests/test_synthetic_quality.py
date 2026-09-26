import math

import torch

from improved_ddpm.fas.synthetic_quality import (
    SyntheticQualityConfig,
    SyntheticQualityController,
    SpectralGateCalibration,
    high_frequency_energy_ratio,
    radial_power_spectrum,
    rbf_mmd_unbiased,
)


def _source_batch(size=4, image_size=16):
    torch.manual_seed(42)
    live = torch.rand(size, 3, image_size, image_size) * 2.0 - 1.0
    attack = (live.roll(shifts=1, dims=-1) * 0.8).clamp(-1, 1)
    return live, attack


def _calibration(bins=32):
    return SpectralGateCalibration(
        version=1,
        policy="source_train_only",
        radial_bins=bins,
        live_center=[0.0] * bins,
        attack_center=[0.0] * bins,
        attack_distance_scale=1.0,
        min_attack_margin=-1e6,
        max_attack_distance_ratio=1e6,
        min_high_frequency_ratio=-1e6,
        max_high_frequency_ratio=1e6,
        live_count=4,
        attack_count=4,
        source_manifest="test",
        train_manifest_fingerprint="test",
        lower_quantile=0.01,
        upper_quantile=0.99,
    )


def test_lenient_gate_accepts_finite_shape_correct_candidates():
    live, attack = _source_batch()
    cfg = SyntheticQualityConfig(
        max_clip_ratio=1.0,
    )
    controller = SyntheticQualityController(cfg, _calibration())
    accepted, report = controller.validate_candidates(live.clone(), live, live, attack)
    assert accepted.tolist() == [True] * live.shape[0]
    assert report["generated"] == report["accepted"] == live.shape[0]


def test_clip_gate_rejects_saturated_candidates():
    live, attack = _source_batch()
    candidate = live.clone()
    candidate[:, :, :8, :] = 2.0
    cfg = SyntheticQualityConfig(
        max_clip_ratio=0.01,
    )
    accepted, report = SyntheticQualityController(cfg, _calibration()).validate_candidates(
        candidate, live, live, attack
    )
    assert not bool(accepted.any())
    assert report["rejected_clip"] == live.shape[0]


def test_nonfinite_candidate_stops_run_by_default():
    live, attack = _source_batch()
    candidate = live.clone()
    candidate[0, 0, 0, 0] = float("nan")
    controller = SyntheticQualityController(SyntheticQualityConfig(), _calibration())
    try:
        controller.validate_candidates(candidate, live, live, attack)
    except FloatingPointError as exc:
        assert "NaN/Inf" in str(exc)
    else:
        raise AssertionError("Non-finite synthetic sample did not stop the DDPM run")


def test_rolling_acceptance_disables_synthetic_but_not_classifier():
    cfg = SyntheticQualityConfig(
        rolling_window_batches=3,
        min_rolling_acceptance_rate=0.8,
    )
    controller = SyntheticQualityController(cfg, _calibration())
    controller.record_batch(4, 2)
    controller.record_batch(4, 2)
    state = controller.record_batch(4, 2)
    assert controller.disabled is True
    assert state["quality_just_disabled"] == 1
    assert math.isclose(state["rolling_acceptance_rate"], 0.5)


def test_frequency_quality_descriptors_are_finite():
    live, attack = _source_batch(size=5)
    hf = high_frequency_energy_ratio(live)
    # Deliberately use more radial bins than the 16x16 test grid has distinct
    # radii. Empty annuli must remain finite rather than producing NaN.
    psd_live = radial_power_spectrum(live, bins=32)
    psd_attack = radial_power_spectrum(attack, bins=32)
    mmd = rbf_mmd_unbiased(psd_live, psd_attack)
    assert hf.shape == (5,)
    assert psd_live.shape == (5, 32)
    assert torch.isfinite(hf).all()
    assert torch.isfinite(psd_live).all()
    assert math.isfinite(mmd)
