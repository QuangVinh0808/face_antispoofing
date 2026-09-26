import torch

from improved_ddpm.data.samplers import ClassBalancedBatchSampler
from improved_ddpm.experiments.stage6 import (
    canonical_attack_condition,
    classifier_mix_counts,
    get_stage6_spec,
    resolve_stage6_spec,
    resolve_effective_stage6_spec,
)
from improved_ddpm.models.unet import UNet
from improved_ddpm.diffusion.diffusion import GaussianDiffusion
from improved_ddpm.diffusion.schedule import NoiseSchedule
from improved_ddpm.diffusion.variance import compute_learned_variance
from improved_ddpm.scripts.train import _compose_stage6_classifier_batch
from improved_ddpm.fas.frequency_utils import FrequencyUtils


def test_stage6_registry_and_dynamic_d4():
    assert get_stage6_spec("D0").use_ddpm is False
    assert get_stage6_spec("D1").ddpm_mode == "live_amplitude"
    assert get_stage6_spec("D2").ddpm_mode == "spoof_amplitude"
    assert get_stage6_spec("D2").augmentation_ratio == 0.5
    assert get_stage6_spec("D2").warmup_steps == 0
    assert get_stage6_spec("D3").conditional is True

    spec = resolve_stage6_spec(
        {
            "stage6": {
                "enabled": True,
                "variant": "D4_r25",
                "ddpm_mode": "spoof_amplitude",
                "augmentation_ratio": 0.25,
            }
        }
    )
    assert spec is not None
    assert spec.variant == "D4_r25"
    assert spec.augmentation_ratio == 0.25
    assert spec.experiment_variant == "E6"


def test_canonical_e6_batch_mix_is_fixed_during_and_after_warmup():
    warmup = classifier_mix_counts(8, 8, 0.5, augmentation_active=False)
    assert warmup == {
        "live": 8,
        "real_attack": 8,
        "synthetic_attack": 0,
        "total": 16,
    }
    active = classifier_mix_counts(8, 8, 0.5, augmentation_active=True)
    assert active == {
        "live": 8,
        "real_attack": 4,
        "synthetic_attack": 4,
        "total": 16,
    }


def test_tensor_batch_composition_replaces_attacks_instead_of_appending():
    torch.manual_seed(42)
    rgb = torch.randn(16, 3, 8, 8)
    labels = torch.tensor([0] * 8 + [1] * 8, dtype=torch.long)
    synthetic = torch.randn(4, 3, 8, 8)

    classifier_rgb, classifier_labels, info = _compose_stage6_classifier_batch(
        rgb, labels, synthetic
    )

    assert classifier_rgb.shape == rgb.shape
    assert classifier_labels.shape == labels.shape
    assert int((classifier_labels == 0).sum()) == 8
    assert int((classifier_labels == 1).sum()) == 8
    assert info == {
        "classifier_live": 8,
        "classifier_real_attack": 4,
        "classifier_synthetic_attack": 4,
        "classifier_total": 16,
    }


def test_attack_condition_mapping():
    assert canonical_attack_condition("printed_photo") == 0
    assert canonical_attack_condition("replay_ipad") == 1
    assert canonical_attack_condition("iphone_video") == 1
    assert canonical_attack_condition("silicone_mask") == 2


def test_class_balanced_sampler_always_has_both_classes():
    labels = [0] * 3 + [1] * 9
    sampler = ClassBalancedBatchSampler(labels, batch_size=4, seed=42, drop_last=True)
    sampler.set_epoch(1)
    batches = list(sampler)
    assert len(batches) == 3
    for batch in batches:
        ys = [labels[i] for i in batch]
        assert ys.count(0) == 2
        assert ys.count(1) == 2


def test_conditional_unet_forward_and_diffusion_loss():
    model = UNet(
        in_channels=1,
        base_channels=8,
        channel_mults=(1, 2),
        attention_resolutions=(),
        num_res_blocks=1,
        num_heads=1,
        image_size=16,
        time_emb_dim=32,
        num_conditions=3,
    )
    x = torch.randn(2, 1, 16, 16)
    t = torch.tensor([1, 2], dtype=torch.long)
    condition = torch.tensor([0, 1], dtype=torch.long)
    out = model(x, t, condition=condition)
    assert out.shape == (2, 2, 16, 16)

    diffusion = GaussianDiffusion(model, T=8, device="cpu")
    losses, _, _ = diffusion.training_loss(x, condition=condition)
    assert torch.isfinite(losses["loss"])


def test_learned_variance_is_bounded_and_finite_for_extreme_head_values():
    schedule = NoiseSchedule(T=16, device="cpu")
    v = torch.tensor([[[[-1000.0, 1000.0]]]])
    t = torch.tensor([8], dtype=torch.long)
    variance, log_variance = compute_learned_variance(v, t, schedule)
    assert torch.isfinite(variance).all()
    assert torch.isfinite(log_variance).all()
    log_beta = torch.log(schedule.betas[t]).item()
    log_post = schedule.posterior_log_variance_clipped[t].item()
    low, high = sorted((log_beta, log_post))
    assert float(log_variance.min()) >= low - 1e-6
    assert float(log_variance.max()) <= high + 1e-6


def test_plain_e6_defaults_to_spoof_amplitude():
    spec = resolve_effective_stage6_spec({"experiment": {"variant": "E6"}}, "E6")
    assert spec is not None
    assert spec.variant == "D2"
    assert spec.ddpm_mode == "spoof_amplitude"
    assert spec.class_balanced is True


def test_explicit_d1_remains_available_as_legacy_control():
    cfg = {
        "stage6": {
            "enabled": True,
            "variant": "D1",
            "experiment_variant": "E6",
            "ddpm_mode": "live_amplitude",
            "augmentation_ratio": 1.0,
        }
    }
    spec = resolve_effective_stage6_spec(cfg, "E6")
    assert spec is not None
    assert spec.ddpm_mode == "live_amplitude"


def test_generated_amplitude_projection_is_conjugate_symmetric():
    torch.manual_seed(7)
    shifted = torch.rand(2, 3, 16, 18)
    projected = FrequencyUtils.enforce_amplitude_conjugate_symmetry(shifted)
    unshifted = torch.fft.ifftshift(projected, dim=(-2, -1))
    partner = torch.roll(
        torch.flip(unshifted, dims=(-2, -1)), shifts=(1, 1), dims=(-2, -1)
    )
    assert torch.allclose(unshifted, partner, atol=1e-6)
