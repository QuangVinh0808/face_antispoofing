"""Canonical Stage-5 baseline and ablation definitions.

The variants are intentionally cumulative so every experiment changes only one
major component compared with the preceding experiment:

E0: RGB baseline
E1: FFT-only baseline
E2: RGB + FFT
E3: E2 + patch stream
E4: E3 + frequency contrastive loss
E5: E4 + center loss
E6: E5 + spoof-artifact DDPM frequency augmentation
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Dict, Mapping


@dataclass(frozen=True)
class AblationSpec:
    variant: str
    description: str
    use_rgb: bool
    use_fft: bool
    use_patch: bool
    use_contrastive: bool
    use_center: bool
    use_ddpm: bool

    def validate(self) -> "AblationSpec":
        if not self.use_rgb and not self.use_fft:
            raise ValueError("At least one input branch (RGB or FFT) must be enabled")
        if self.use_contrastive and not self.use_fft:
            raise ValueError("Frequency contrastive loss requires the FFT branch")
        if self.use_ddpm and not self.use_fft:
            raise ValueError("DDPM amplitude augmentation requires the FFT branch")
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def active_modalities(self) -> str:
        if self.use_rgb and self.use_fft:
            return "rgb+fft"
        return "rgb" if self.use_rgb else "fft"


ABLATION_REGISTRY: Dict[str, AblationSpec] = {
    "E0": AblationSpec(
        variant="E0",
        description="RGB baseline: full image, cross-entropy only",
        use_rgb=True,
        use_fft=False,
        use_patch=False,
        use_contrastive=False,
        use_center=False,
        use_ddpm=False,
    ),
    "E1": AblationSpec(
        variant="E1",
        description="FFT log-amplitude baseline: full image, cross-entropy only",
        use_rgb=False,
        use_fft=True,
        use_patch=False,
        use_contrastive=False,
        use_center=False,
        use_ddpm=False,
    ),
    "E2": AblationSpec(
        variant="E2",
        description="Dual branch RGB + FFT, full image, cross-entropy only",
        use_rgb=True,
        use_fft=True,
        use_patch=False,
        use_contrastive=False,
        use_center=False,
        use_ddpm=False,
    ),
    "E3": AblationSpec(
        variant="E3",
        description="E2 + patch training stream",
        use_rgb=True,
        use_fft=True,
        use_patch=True,
        use_contrastive=False,
        use_center=False,
        use_ddpm=False,
    ),
    "E4": AblationSpec(
        variant="E4",
        description="E3 + frequency supervised contrastive loss",
        use_rgb=True,
        use_fft=True,
        use_patch=True,
        use_contrastive=True,
        use_center=False,
        use_ddpm=False,
    ),
    "E5": AblationSpec(
        variant="E5",
        description="E4 + center loss (complete classifier without DDPM)",
        use_rgb=True,
        use_fft=True,
        use_patch=True,
        use_contrastive=True,
        use_center=True,
        use_ddpm=False,
    ),
    "E6": AblationSpec(
        variant="E6",
        description="E5 + spoof-amplitude DDPM artifact augmentation (complete proposed model)",
        use_rgb=True,
        use_fft=True,
        use_patch=True,
        use_contrastive=True,
        use_center=True,
        use_ddpm=True,
    ),
}


def get_ablation_spec(variant: str) -> AblationSpec:
    key = str(variant).strip().upper()
    if key not in ABLATION_REGISTRY:
        available = ", ".join(ABLATION_REGISTRY)
        raise ValueError(f"Unknown Stage-5 variant {variant!r}; choose one of: {available}")
    return ABLATION_REGISTRY[key]


def resolve_ablation_spec(config: Mapping[str, object]) -> AblationSpec:
    """Resolve a canonical variant plus optional explicit research overrides.

    Overrides are useful for follow-up experiments, while the named E0-E6
    variants remain stable for the main ablation table. Example::

        experiment:
          variant: E5
          overrides:
            use_patch: false

    The resolved specification is stored in each checkpoint, so evaluation can
    verify that the matching architecture is reconstructed.
    """

    experiment = config.get("experiment", {}) if isinstance(config, Mapping) else {}
    if experiment is None:
        experiment = {}
    if not isinstance(experiment, Mapping):
        raise TypeError("config.experiment must be a mapping")

    spec = get_ablation_spec(str(experiment.get("variant", "E6")))
    overrides = experiment.get("overrides", {}) or {}
    if not isinstance(overrides, Mapping):
        raise TypeError("config.experiment.overrides must be a mapping")

    allowed = {
        "use_rgb",
        "use_fft",
        "use_patch",
        "use_contrastive",
        "use_center",
        "use_ddpm",
    }
    unknown = set(overrides) - allowed
    if unknown:
        raise ValueError(f"Unknown ablation override(s): {sorted(unknown)}")
    if overrides:
        spec = replace(spec, **{key: bool(value) for key, value in overrides.items()})
        spec = replace(spec, description=f"{spec.description} [custom overrides]")
    return spec.validate()
