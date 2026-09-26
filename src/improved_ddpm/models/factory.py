"""Model factory shared by Stage-5 training and Stage-4/5 evaluation."""

from __future__ import annotations

from typing import Mapping, Optional

from improved_ddpm.experiments.ablation import AblationSpec, resolve_ablation_spec
from improved_ddpm.models.dual_branch import DualBranchFASNetwork


def build_fas_model(
    config: Mapping[str, object],
    *,
    pretrained: Optional[bool] = None,
) -> tuple[DualBranchFASNetwork, AblationSpec]:
    spec = resolve_ablation_spec(config)
    classifier = config["classifier"]
    if not isinstance(classifier, Mapping):
        raise TypeError("config.classifier must be a mapping")
    pretrained_value = (
        bool(classifier.get("pretrained_backbone", False))
        if pretrained is None
        else bool(pretrained)
    )
    model = DualBranchFASNetwork(
        feat_dim=int(classifier["feat_dim"]),
        amp_in_channels=int(config["amp_channels"]),
        num_classes=int(classifier["num_classes"]),
        pretrained=pretrained_value,
        use_rgb=spec.use_rgb,
        use_amp=spec.use_fft,
    )
    return model, spec
