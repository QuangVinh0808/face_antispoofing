"""
tests/test_diffusion.py
=========================
Test co ban (smoke test) cho ha tang Diffusion va pipeline FAS.
Chay: pytest tests/ -v
"""

import torch
import pytest

from improved_ddpm.diffusion.schedule import NoiseSchedule, cosine_beta_schedule
from improved_ddpm.models.unet import UNet
from improved_ddpm.diffusion.diffusion import GaussianDiffusion
from improved_ddpm.fas.frequency_utils import FrequencyUtils
from improved_ddpm.fas.patch_utils import random_crop_resize_batch
from improved_ddpm.models.dual_branch import DualBranchFASNetwork


def test_cosine_beta_schedule_clip():
    """beta_t khong vuot qua 0.999 va co do dai dung T."""
    T = 200
    betas = cosine_beta_schedule(T, s=0.008)
    assert betas.shape[0] == T
    assert torch.all(betas <= 0.999)
    assert torch.all(betas >= 0.0)


def test_noise_schedule_q_sample_shape():
    """q_sample() tra ve x_t cung shape voi x0."""
    T = 200
    sched = NoiseSchedule(T=T)
    x0 = torch.randn(4, 1, 16, 16)
    t = torch.randint(0, T, (4,))
    x_t, noise = sched.q_sample(x0, t)
    assert x_t.shape == x0.shape
    assert noise.shape == x0.shape


def test_unet_output_double_channels():
    """UNet phai output 2*C channel (eps_theta + v)."""
    model = UNet(
        in_channels=1, base_channels=16, channel_mults=(1, 2),
        attention_resolutions=(8,), num_res_blocks=1, num_heads=2, image_size=16,
        time_emb_dim=32,
    )
    x = torch.randn(2, 1, 16, 16)
    t = torch.randint(0, 100, (2,))
    out = model(x, t)
    assert out.shape == (2, 2, 16, 16)  # 2*in_channels


def test_training_loss_backward():
    """training_loss() phai tra ve loss hop le va backward khong loi, khong NaN."""
    model = UNet(
        in_channels=1, base_channels=16, channel_mults=(1, 2),
        attention_resolutions=(8,), num_res_blocks=1, num_heads=2, image_size=16,
        time_emb_dim=32,
    )
    diffusion = GaussianDiffusion(model, T=100, s=0.008, device="cpu")
    x0 = torch.randn(2, 1, 16, 16)

    losses, t, x_t = diffusion.training_loss(x0)
    assert torch.isfinite(losses["loss"])

    losses["loss"].backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0
    assert all(g == g for g in grad_norms)  # khong NaN


def test_fast_sampling_shape():
    """sample() voi K << T phai tra ve dung shape va khong NaN/Inf."""
    model = UNet(
        in_channels=1, base_channels=16, channel_mults=(1, 2),
        attention_resolutions=(8,), num_res_blocks=1, num_heads=2, image_size=16,
        time_emb_dim=32,
    )
    diffusion = GaussianDiffusion(model, T=100, s=0.008, device="cpu")
    samples = diffusion.sample((2, 1, 16, 16), K=10)
    assert samples.shape == (2, 1, 16, 16)
    assert torch.isfinite(samples).all()


def test_fft_ifft_roundtrip():
    """fft_decompose -> ifft_reconstruct phai khoi phuc lai anh ban dau (sai so nho)."""
    img = torch.randn(2, 3, 16, 16)
    amplitude, phase = FrequencyUtils.fft_decompose(img)
    recon = FrequencyUtils.ifft_reconstruct(amplitude, phase)
    assert torch.allclose(img, recon, atol=1e-4)


def test_log_amplitude_roundtrip_no_overflow():
    """log_amplitude -> inverse_log_amplitude phai khoi phuc dung va khong sinh Inf/NaN."""
    amp = torch.rand(2, 3, 16, 16) * 100  # amplitude duong, co the lon
    log_amp = FrequencyUtils.log_amplitude(amp)
    recon = FrequencyUtils.inverse_log_amplitude(log_amp)
    assert torch.allclose(amp, recon, atol=1e-2)
    assert torch.isfinite(recon).all()


def test_random_crop_resize_batch_shape_and_range():
    """Patch RGB phai dung kich thuoc output_size va scale nam trong [0.2, 1.0]."""
    images = torch.randn(4, 3, 64, 64)
    patches = random_crop_resize_batch(images, scale_range=(0.2, 1.0), output_size=32)
    assert patches.shape == (4, 3, 32, 32)
    assert torch.isfinite(patches).all()


def test_dual_branch_multi_stream_loss():
    """compute_multi_stream_loss phai tra ve loss huu han va backward khong loi."""
    net = DualBranchFASNetwork(feat_dim=16, amp_in_channels=3, num_classes=2)
    rgb_full = torch.randn(4, 3, 32, 32)
    amp_full = torch.randn(4, 3, 32, 32)
    rgb_patch = torch.randn(4, 3, 32, 32)
    amp_patch = torch.randn(4, 3, 32, 32)
    labels = torch.tensor([0, 0, 1, 1])

    out_full = net(rgb_full, amp_full)
    out_patch = net(rgb_patch, amp_patch)
    losses = net.compute_multi_stream_loss(out_full, out_patch, labels)

    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_dual_branch_predict_inference_full_only():
    """predict() chi can (rgb_full, amplitude_full), tra ve pred/probs hop le."""
    net = DualBranchFASNetwork(feat_dim=16, amp_in_channels=3, num_classes=2)
    rgb_full = torch.randn(3, 3, 32, 32)
    amp_full = torch.randn(3, 3, 32, 32)
    out = net.predict(rgb_full, amp_full)
    assert out["pred"].shape == (3,)
    assert torch.allclose(out["probs"].sum(dim=-1), torch.ones(3), atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
