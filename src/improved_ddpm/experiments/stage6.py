"""Stage-6 DDPM validation experiment definitions.

Stage 6 keeps the Stage-5 classifier fixed at E5/E6 and changes only how the
frequency DDPM is trained/used. This lets the report attribute any improvement
specifically to the generative augmentation strategy.

D0: E5 baseline, no DDPM.
D1: Train DDPM on live amplitudes; synthesize spoof-labelled samples (legacy).
D2: Train DDPM on spoof amplitudes; combine generated spoof amplitude with live phase.
    This is the canonical E6 mechanism used by default when experiment.variant=E6.
D3: Conditional spoof-amplitude DDPM (print/replay/other conditions).
D4: Ratio sweep using the best D1-D3 strategy; the runner materializes names such
    as D4_r10, D4_r25, D4_r50 and D4_r100.

For DDPM-enabled variants, ``augmentation_ratio`` is the fraction of the fixed
attack quota replaced by synthetic attacks. Canonical D2/D3 use a separately
pretrained frozen DDPM, so ``warmup_steps=0``. It never
increases classifier batch size or changes the 50/50 live/attack class prior.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Mapping


VALID_DDPM_MODES = {
    "none",
    "live_amplitude",
    "spoof_amplitude",
    "conditional_spoof",
}


@dataclass(frozen=True)
class Stage6Spec:
    variant: str
    description: str
    experiment_variant: str
    ddpm_mode: str
    augmentation_ratio: float
    warmup_steps: int = 0
    class_balanced: bool = True

    @property
    def use_ddpm(self) -> bool:
        return self.ddpm_mode != "none"

    @property
    def conditional(self) -> bool:
        return self.ddpm_mode == "conditional_spoof"

    def validate(self) -> "Stage6Spec":
        if self.experiment_variant not in {"E5", "E6"}:
            raise ValueError("Stage-6 experiment_variant must be E5 or E6")
        if self.ddpm_mode not in VALID_DDPM_MODES:
            raise ValueError(f"Unknown DDPM mode: {self.ddpm_mode}")
        if not 0.0 <= float(self.augmentation_ratio) <= 1.0:
            raise ValueError("augmentation_ratio must be in [0, 1]")
        if int(self.warmup_steps) < 0:
            raise ValueError("warmup_steps must be >= 0")
        if self.variant == "D0" and self.use_ddpm:
            raise ValueError("D0 must not enable DDPM")
        if self.variant != "D0" and not self.use_ddpm:
            raise ValueError("D1-D4 must enable a DDPM strategy")
        return self

    def to_dict(self) -> dict:
        return asdict(self)


STAGE6_REGISTRY: Dict[str, Stage6Spec] = {
    "D0": Stage6Spec(
        variant="D0",
        description="E5 baseline with class-balanced batches; no DDPM",
        experiment_variant="E5",
        ddpm_mode="none",
        augmentation_ratio=0.0,
        warmup_steps=0,
    ),
    "D1": Stage6Spec(
        variant="D1",
        description="Legacy DDPM trained on live amplitude; generated samples labelled spoof",
        experiment_variant="E6",
        ddpm_mode="live_amplitude",
        augmentation_ratio=0.5,
        warmup_steps=300,
    ),
    "D2": Stage6Spec(
        variant="D2",
        description="DDPM trained on spoof amplitude; generated spoof amplitude + live phase",
        experiment_variant="E6",
        ddpm_mode="spoof_amplitude",
        augmentation_ratio=0.5,
        warmup_steps=0,
    ),
    "D3": Stage6Spec(
        variant="D3",
        description="Conditional spoof-amplitude DDPM (print/replay/other) + live phase",
        experiment_variant="E6",
        ddpm_mode="conditional_spoof",
        augmentation_ratio=0.5,
        warmup_steps=0,
    ),
}


def get_stage6_spec(variant: str) -> Stage6Spec:
    key = str(variant).strip().upper()
    if key not in STAGE6_REGISTRY:
        raise ValueError(f"Unknown Stage-6 variant {variant!r}; choose one of {list(STAGE6_REGISTRY)}")
    return STAGE6_REGISTRY[key].validate()


def resolve_stage6_spec(config: Mapping[str, object]) -> Stage6Spec | None:
    section = config.get("stage6", {}) if isinstance(config, Mapping) else {}
    if not section or not isinstance(section, Mapping) or not bool(section.get("enabled", False)):
        return None

    variant = str(section.get("variant", "D1")).strip()
    upper = variant.upper()
    if upper in STAGE6_REGISTRY:
        base = get_stage6_spec(upper)
        ddpm_mode = str(section.get("ddpm_mode", base.ddpm_mode)).strip().lower()
        ratio = float(section.get("augmentation_ratio", base.augmentation_ratio))
        warmup_steps = int(section.get("warmup_steps", base.warmup_steps))
        class_balanced = bool(section.get("class_balanced", base.class_balanced))
        experiment_variant = str(section.get("experiment_variant", base.experiment_variant)).upper()
        description = str(section.get("description", base.description))
    elif upper.startswith("D4"):
        ddpm_mode = str(section.get("ddpm_mode", "spoof_amplitude")).strip().lower()
        ratio = float(section.get("augmentation_ratio", 0.25))
        warmup_steps = int(section.get("warmup_steps", 0))
        class_balanced = bool(section.get("class_balanced", True))
        experiment_variant = "E6"
        description = str(
            section.get(
                "description",
                f"D4 ratio sweep: {ddpm_mode}, synthetic attack-slot fraction={ratio:.2f}",
            )
        )
    else:
        raise ValueError("stage6.variant must be D0-D3 or a D4_* ratio-sweep label")

    return Stage6Spec(
        variant=variant,
        description=description,
        experiment_variant=experiment_variant,
        ddpm_mode=ddpm_mode,
        augmentation_ratio=ratio,
        warmup_steps=warmup_steps,
        class_balanced=class_balanced,
    ).validate()


def classifier_mix_counts(
    live_count: int,
    real_attack_count: int,
    synthetic_attack_fraction: float,
    augmentation_active: bool,
) -> dict[str, int]:
    """Resolve a fixed-size, class-balanced classifier batch for Stage 6.

    ``augmentation_ratio`` is interpreted as the fraction of the attack quota
    replaced by synthetic attacks, never as samples appended to the real batch.
    With the canonical 8-live/8-attack loader batch and ratio 0.5 this returns
    8 live, 4 real attacks and 4 synthetic attacks.
    """
    live_count = int(live_count)
    real_attack_count = int(real_attack_count)
    ratio = float(synthetic_attack_fraction)
    if live_count <= 0 or real_attack_count <= 0:
        raise ValueError("Stage-6 classifier batches require both live and attack samples")
    if live_count != real_attack_count:
        raise ValueError(
            "Stage-6 classifier input must be class-balanced before augmentation: "
            f"live={live_count}, attack={real_attack_count}"
        )
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("synthetic_attack_fraction must be in [0, 1]")

    synthetic = 0
    if augmentation_active and ratio > 0.0:
        synthetic = max(1, int(round(real_attack_count * ratio)))
        synthetic = min(synthetic, real_attack_count)
    return {
        "live": live_count,
        "real_attack": real_attack_count - synthetic,
        "synthetic_attack": synthetic,
        "total": live_count + real_attack_count,
    }



def resolve_effective_stage6_spec(
    config: Mapping[str, object],
    experiment_variant: str,
) -> Stage6Spec | None:
    """Resolve the DDPM strategy actually used by training.

    Canonical research definition:
      * E5 -> no DDPM.
      * E6 -> D2 spoof-amplitude DDPM unless an explicit Stage-6 control is set.

    Keeping D1 available is useful as a legacy/control experiment, but a plain
    ``experiment.variant: E6`` must never silently train the DDPM on bona-fide
    amplitudes.  This makes E6 match the stated hypothesis: learn the source
    spoof-artifact amplitude distribution, generate novel spoof amplitudes, and
    combine them with bona-fide phase before labelling the reconstruction as an
    attack.
    """
    explicit = resolve_stage6_spec(config)
    if explicit is not None:
        return explicit

    variant = str(experiment_variant).strip().upper()
    if variant == "E6":
        return get_stage6_spec("D2")
    return None

def canonical_attack_condition(attack_type: object) -> int:
    """Map arbitrary manifest attack_type text to stable D3 condition ids.

    0=print, 1=replay, 2=other. Live rows should not be passed here.
    """
    text = str(attack_type or "").strip().lower()
    if any(token in text for token in ("print", "paper", "photo")):
        return 0
    if any(token in text for token in ("replay", "video", "screen", "display", "ipad", "iphone", "mobile")):
        return 1
    return 2


CONDITION_NAMES = {0: "print", 1: "replay", 2: "other"}
