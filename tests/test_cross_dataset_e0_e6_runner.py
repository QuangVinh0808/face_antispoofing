from copy import deepcopy
from pathlib import Path

from improved_ddpm.experiments.ablation import ABLATION_REGISTRY
from improved_ddpm.scripts.run_cross_dataset_ms import (
    _aggregate_results,
    _apply_ablation_controls,
    _paired_ablation_deltas,
    _resolve_variants,
    _result_row,
)


def _base():
    return {"frequency": {}, "diffusion": {}, "stage6": {"quality_control": {}}}


def _metrics(value: float):
    return {
        "APCER": value,
        "BPCER": value,
        "ACER": value,
        "HTER": value,
        "AUC": 1.0 - value,
        "EER": value,
        "macro_APCER": value,
        "worst_APCER": value,
    }


def test_all_resolves_to_canonical_e0_e6_order():
    assert _resolve_variants(["all"]) == list(ABLATION_REGISTRY)


def test_each_variant_locks_fft_and_ddpm_correctly():
    normalization = Path("normalization_224.json")
    attack_normalization = Path("normalization_attack_224.json")
    spectral_gate = Path("spectral_gate_source_train_224.json")
    ddpm_checkpoint = Path("ddpm_final.pt")
    for variant in ABLATION_REGISTRY:
        cfg = deepcopy(_base())
        spec = _apply_ablation_controls(
            cfg,
            variant,
            normalization,
            attack_normalization if variant == "E6" else None,
            spectral_gate if variant == "E6" else None,
            ddpm_checkpoint if variant == "E6" else None,
            50 if variant == "E6" else None,
            0.5,
        )
        assert cfg["experiment"]["variant"] == variant
        assert cfg["frequency"]["require_train_stats"] is spec.use_fft
        assert (cfg["frequency"]["normalization_file"] is not None) is spec.use_fft
        assert cfg["stage6"]["enabled"] is spec.use_ddpm
        assert cfg["stage6"]["ddpm_mode"] == ("spoof_amplitude" if variant == "E6" else "none")
        assert cfg["stage6"]["augmentation_ratio"] == (0.5 if variant == "E6" else 0.0)
        assert cfg["stage6"]["warmup_steps"] == 0
        if variant == "E6":
            assert cfg["stage6"]["synthetic_strength"] == 1.0
            assert cfg["diffusion"]["freeze_pretrained"] is True
            assert cfg["diffusion"]["sampling_k"] == 50


def test_summary_and_paired_component_deltas_use_matching_seeds():
    rows = []
    for variant_index, variant in enumerate(ABLATION_REGISTRY):
        for seed in (42, 43, 44):
            rows.append(_result_row("M_to_S", variant, seed, _metrics(0.30 - variant_index * 0.01)))
    summary = _aggregate_results(rows)
    deltas = _paired_ablation_deltas(rows)
    assert len(summary) == 7
    assert all(row["seed_count"] == 3 for row in summary)
    ddpm = next(row for row in deltas if row["comparison"] == "add_DDPM")
    assert ddpm["before"] == "E5" and ddpm["after"] == "E6"
    assert ddpm["paired_seed_count"] == 3
    assert round(ddpm["delta_HTER_mean"], 8) == -0.01
