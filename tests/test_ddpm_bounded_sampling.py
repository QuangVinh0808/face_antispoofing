"""Numerical reverse-process tests; no real dataset or GPU is required."""
import pytest
import torch
from torch import nn

from improved_ddpm.diffusion.diffusion import GaussianDiffusion, RespacedSchedule, make_respaced_steps
from improved_ddpm.diffusion.schedule import NoiseSchedule
from improved_ddpm.fas.amplitude_ddpm import ImprovedAmplitudeDDPM
from improved_ddpm.fas.ddpm_representation import RadialAmplitudeStats


class Oracle(nn.Module):
    def __init__(self, schedule, clean):
        super().__init__()
        self.schedule, self.clean, self.inputs, self.timesteps = schedule, clean, [], []

    def forward(self, x, t, condition=None):
        self.inputs.append(x.detach().clone())
        self.timesteps.append(t.detach().clone())
        ac = self.schedule.alphas_cumprod[t].view(-1, 1, 1, 1).to(x)
        eps = (x - ac.sqrt() * self.clean) / (1 - ac).sqrt()
        return torch.cat((eps, torch.zeros_like(eps)), dim=1)


@pytest.mark.parametrize("k", [2, 20, 50, 100, 1000])
def test_respacing_preserves_original_forward_marginals(k):
    schedule = NoiseSchedule(T=1000)
    steps = make_respaced_steps(1000, k)
    respaced = RespacedSchedule(schedule, steps, preserve_marginals=True)
    torch.testing.assert_close(respaced.alphas.cumprod(0), schedule.alphas_cumprod[steps].double(),
                               atol=1e-16, rtol=1e-12)
    assert bool(((respaced.betas > 0) & (respaced.betas < 1)).all())


@pytest.mark.parametrize("k", [None, 4, 8])
def test_perfect_noise_prediction_recovers_clean_sample(k):
    clean = torch.full((2, 1, 4, 4), 0.35)
    diffusion = GaussianDiffusion(nn.Identity(), T=8)
    oracle = Oracle(diffusion.schedule, clean)
    diffusion.model = oracle
    torch.manual_seed(42)
    sample = diffusion.sample(clean.shape, K=k, denoised_bounds=(-1.0, 1.0), preserve_marginals=True)
    torch.testing.assert_close(sample, clean, atol=2e-5, rtol=2e-5)
    assert sample.dtype == torch.float32
    assert bool(torch.isfinite(sample).all())
    assert oracle.timesteps[-1].tolist() == [0, 0]


def test_bounds_are_on_clean_prediction_not_noisy_state_and_raw_metrics_survive():
    clean = torch.full((2, 1, 4, 4), 2.0)  # deliberately invalid clean prediction
    diffusion = GaussianDiffusion(nn.Identity(), T=8)
    oracle = Oracle(diffusion.schedule, clean)
    diffusion.model = oracle
    torch.manual_seed(42)
    sample = diffusion.sample(clean.shape, K=4, denoised_bounds=(-1.0, 1.0), preserve_marginals=True)
    torch.testing.assert_close(sample, torch.ones_like(clean), atol=0, rtol=0)
    assert oracle.inputs[0].abs().max() > 1  # noisy state was NOT clamped
    assert diffusion.last_sampling_stats["clean_prediction_last_out_of_range_rate"] == 1.0
    assert diffusion.last_sampling_stats["clean_prediction_last_abs_max"] > 1.9
    torch.manual_seed(42)
    control = diffusion.sample(clean.shape, K=4, preserve_marginals=True)
    torch.testing.assert_close(control, clean, atol=2e-4, rtol=2e-4)
    assert diffusion.last_sampling_stats["clean_prediction_last_out_of_range_rate"] == 1.0


def test_legacy_default_reverse_step_is_unchanged():
    class Constant(nn.Module):
        def forward(self, x, t):
            return torch.cat((torch.full_like(x, 0.2), torch.full_like(x, -0.7)), 1)
    d = GaussianDiffusion(Constant(), T=8)
    x, t = torch.randn(2, 1, 4, 4), torch.tensor([3, 6])
    beta = d.schedule.betas[t].view(-1, 1, 1, 1)
    alpha = d.schedule.alphas[t].view(-1, 1, 1, 1)
    ac = d.schedule.alphas_cumprod[t].view(-1, 1, 1, 1)
    mean = (x - beta / torch.sqrt(1 - ac) * 0.2) / alpha.sqrt()
    frac = torch.sigmoid(torch.full_like(x, -0.7))
    log_var = frac * beta.log() + (1-frac) * d.schedule.posterior_log_variance_clipped[t].view(-1, 1, 1, 1)
    torch.manual_seed(42)
    expected = mean + torch.exp(0.5 * log_var) * torch.randn_like(x)
    torch.manual_seed(42)
    actual = d.p_sample_step(x, t, t, d.schedule)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_r2_wrapper_defaults_to_bounds_and_keeps_diagnostic_control():
    stats = RadialAmplitudeStats(2, "train", "radial_q95_sqrt", 4,
        [[1.0]*4]*3, [[0.0]*4]*3, [[1.0]*4]*3, 1.0, 0.95, 0.995, 16, 42,
        1, 3, 16, "fixture", "fixture").validate()
    model = ImprovedAmplitudeDDPM(in_channels=3, image_size=16, base_channels=8,
        channel_mults=(1, 2), attention_resolutions=(), num_heads=1, T=8, normalization_stats=stats)
    model.diffusion.model = Oracle(model.diffusion.schedule, torch.full((2, 3, 16, 16), 2.0))
    sample = model.generate_fake_amplitude((2, 3, 16, 16), K=4)
    torch.testing.assert_close(sample, torch.ones_like(sample), atol=0, rtol=0)
    report = model.last_sampling_report
    assert report["clip_denoised"] is True
    assert report["normalized_out_of_range_rate"] == 0.0
    assert report["normalized_boundary_rate"] == 1.0
    assert report["clean_prediction_last_out_of_range_rate"] == 1.0
    model.generate_fake_amplitude((2, 3, 16, 16), K=4, clip_denoised=False)
    assert model.last_sampling_report["clip_denoised"] is False
    assert model.last_sampling_report["normalized_out_of_range_rate"] == 1.0
    model.normalization_stats = None
    with pytest.raises(ValueError, match="only valid"):
        model.generate_fake_amplitude((2, 3, 16, 16), K=4, clip_denoised=True)


def test_sampling_rejects_invalid_bounds():
    model = GaussianDiffusion(nn.Identity(), T=8)
    with pytest.raises(ValueError, match="increasing"):
        model.sample((2, 1, 4, 4), K=2, denoised_bounds=(1.0, -1.0))
    with pytest.raises(ValueError, match="positive"):
        model.sample((2, 1, 4, 4), K=0)
