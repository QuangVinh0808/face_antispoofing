"""Experiment definitions used to reproduce controlled ablation studies."""

from .ablation import (
    ABLATION_REGISTRY,
    AblationSpec,
    get_ablation_spec,
    resolve_ablation_spec,
)
from .stage6 import (
    CONDITION_NAMES,
    STAGE6_REGISTRY,
    Stage6Spec,
    canonical_attack_condition,
    classifier_mix_counts,
    get_stage6_spec,
    resolve_stage6_spec,
    resolve_effective_stage6_spec,
)

__all__ = [
    "ABLATION_REGISTRY",
    "AblationSpec",
    "get_ablation_spec",
    "resolve_ablation_spec",
    "STAGE6_REGISTRY",
    "Stage6Spec",
    "get_stage6_spec",
    "resolve_stage6_spec",
    "resolve_effective_stage6_spec",
    "canonical_attack_condition",
    "classifier_mix_counts",
    "CONDITION_NAMES",
]
