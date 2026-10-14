"""
Module 2 - ImprovedAmplitudeDDPM
==================================
Mo hinh Diffusion hoc sinh ra ma tran Amplitude cua artifact gia mao
(thay vi sinh anh RGB nhu DDPM thong thuong).

Tai su dung lai cac module da xay dung truoc:
    - schedule.py   : cosine noise schedule (alpha_bar_t = cos(...)^2, clip beta<=0.999)
    - unet.py       : UNet + multi-head attention (4 heads) + scale-shift conditioning
    - variance.py   : noi suy phuong sai hoc duoc tu vector v
    - losses.py     : Hybrid loss (L_simple + 0.001*L_vlb, stop-gradient tren mu_theta)
    - diffusion.py  : GaussianDiffusion (training_loss, sample, fast sample)

Class nay chi la lop "wrapper" chuyen biet hoa cho input la AMPLITUDE
(thay vi anh RGB), giu nguyen toan bo co che toan hoc da cai dat.
"""

from typing import Optional

import torch
import torch.nn as nn

from improved_ddpm.models.unet import UNet
from improved_ddpm.diffusion.diffusion import GaussianDiffusion
from .frequency_utils import FrequencyUtils
from .frequency_normalization import FFTNormalizationStats
from .ddpm_representation import RadialAmplitudeStats, radial_coordinate


class RadiusConditionedUNet(UNet):
    """Clean radius is a conditioning channel, NOT a diffused/output channel."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        conv = self.input_conv
        self.input_conv = nn.Conv2d(conv.in_channels + 1, conv.out_channels, 3, padding=1)
        self.register_buffer("radius_map", radial_coordinate(kwargs["image_size"], kwargs["image_size"])[None, None], persistent=False)

    def forward(self, x, t, condition=None):
        radius = self.radius_map.to(x).expand(x.shape[0], -1, -1, -1)
        return super().forward(torch.cat((x, radius), dim=1), t, condition=condition)


class ImprovedAmplitudeDDPM(nn.Module):
    """
    Wrapper toan dien: UNet (backbone) + GaussianDiffusion (schedule/loss/sample),
    chuyen biet cho du lieu dau vao la log-amplitude da chuan hoa.

    Luu y quan trong: alpha_bar_t = cos( (1.008*t/T + 0.008) * pi/2 )^2 -- day la cong thuc
    cosine schedule chuan (Nichol & Dhariwal 2021) voi he so offset s=0.008,
    tuong duong: alpha_bar_t = f(t)/f(0) voi f(t)=cos( ((t/T+s)/(1+s)) * pi/2 )^2.
    Ham cosine_beta_schedule() trong schedule.py da cai dat dung cong thuc nay.
    """

    def __init__(
        self,
        in_channels: int = 3,       # 3 channel cho amplitude cua anh RGB (R,G,B rieng biet)
        image_size: int = 64,
        base_channels: int = 64,
        channel_mults=(1, 2, 4, 8),
        attention_resolutions=(16, 8),   # multi-head attention tai do phan giai thap
        num_heads: int = 4,
        T: int = 1000,
        s: float = 0.008,
        lam_vlb: float = 0.001,
        device: str = "cpu",
        normalization_stats: Optional[FFTNormalizationStats] = None,
        num_conditions: int = 0,
        radius_conditioning: bool = False,
    ):
        super().__init__()
        self.device = device
        self.lam_vlb = lam_vlb
        self.normalization_stats = normalization_stats
        self.radius_conditioning = bool(radius_conditioning)

        unet_type = RadiusConditionedUNet if self.radius_conditioning else UNet
        self.unet = unet_type(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
            attention_resolutions=attention_resolutions,
            num_res_blocks=2,
            num_heads=num_heads,
            image_size=image_size,
            time_emb_dim=256,
            num_conditions=num_conditions,
        ).to(device)

        # GaussianDiffusion da bao gom cosine schedule (clip beta<=0.999),
        # hybrid loss voi stop-gradient, va fast-sampling (Module 5 truoc).
        self.diffusion = GaussianDiffusion(self.unet, T=T, s=s, device=device)

    # ------------------------------------------------------------------
    # Buoc 1: chuan bi du lieu - tu anh RGB thuc -> log-amplitude chuan hoa
    # ------------------------------------------------------------------
    def preprocess_amplitude(self, rgb_image: torch.Tensor):
        """
        rgb_image: (B, 3, H, W) source images used to learn the amplitude distribution.
        For canonical E6 these are SPOOF/ATTACK images; D1 may use Live as a legacy control.
        Tra ve: (log_amp_norm, phase, mean, std)
            - log_amp_norm: input chuan cho Diffusion model (~N(0,1))
            - phase:        giu lai de tai tao anh sau nay (KHONG dua vao diffusion)
            - mean, std:    de denormalize khi sample xong
        """
        amplitude, phase = FrequencyUtils.fft_decompose(rgb_image)
        if isinstance(self.normalization_stats, RadialAmplitudeStats):
            encoded = self.normalization_stats.encode(amplitude)
            # Keep the four-value API for existing call sites; the transform is
            # nonlinear, so these legacy scalar slots are intentionally None.
            return encoded, phase, None, None
        log_amp = FrequencyUtils.log_amplitude(amplitude)
        if self.normalization_stats is None:
            # Legacy/demo fallback. Benchmark runs should always supply train-only stats.
            log_amp_norm, mean, std = FrequencyUtils.normalize_amplitude(log_amp)
        else:
            mean, std = self.normalization_stats.tensors(log_amp)
            log_amp_norm = (log_amp - mean) / std.clamp_min(1e-6)
        return log_amp_norm, phase, mean, std

    # ------------------------------------------------------------------
    # Buoc 2: huan luyen diffusion tren log-amplitude
    # ------------------------------------------------------------------
    def training_step(self, rgb_image: torch.Tensor, condition=None):
        """
        Mot buoc huan luyen: tu anh RGB -> trich amplitude -> tinh hybrid loss
        cho Diffusion model hoc phan bo amplitude. Canonical E6 truyen vao SPOOF images
        de DDPM hoc va sinh frequency artifacts cua attack.
        """
        log_amp_norm, _, _, _ = self.preprocess_amplitude(rgb_image)
        losses, t, x_t = self.diffusion.training_loss(
            log_amp_norm, lam=self.lam_vlb, condition=condition
        )
        return losses

    # ------------------------------------------------------------------
    # Buoc 3: sinh amplitude artifact moi (sau khi train xong)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate_fake_amplitude(
        self,
        shape,
        K: int = 50,
        mean=None,
        std=None,
        condition=None,
        normalized_clip: float = 3.0,
        clip_denoised: Optional[bool] = None,
    ):
        """
        Sinh ma tran amplitude gia mao MOI bang fast sampling (K buoc << T).
        shape: (B, C, H, W) - phai khop voi in_channels/image_size cua UNet.

        Tra ve: amplitude THO (da denormalize + inverse log), san sang ghep voi Phase thuc.
        """
        self.unet.eval()
        normalized_clip = float(normalized_clip)
        if normalized_clip <= 0:
            raise ValueError("normalized_clip must be > 0")
        is_r2 = isinstance(self.normalization_stats, RadialAmplitudeStats) and self.normalization_stats.mode == "radial_q95_sqrt"
        clip_denoised = is_r2 if clip_denoised is None else bool(clip_denoised)
        if clip_denoised and not is_r2:
            raise ValueError("[-1,1] clean-sample bounds are only valid for radial_q95_sqrt (R2)")
        log_amp_norm_fake = self.diffusion.sample(
            shape, K=K, condition=condition,
            denoised_bounds=(-1.0, 1.0) if clip_denoised else None,
            preserve_marginals=is_r2,
        )  # (B,C,H,W), khong gian chuan hoa
        if not bool(torch.isfinite(log_amp_norm_fake).all()):
            raise FloatingPointError("Nonfinite DDPM sample before representation decoding")
        self.last_sampling_report = {
            "normalized_out_of_range_rate": float((log_amp_norm_fake.abs() > 1).float().mean()),
            "normalized_boundary_rate": float((log_amp_norm_fake.abs() >= 0.9999).float().mean()),
            "clip_denoised": clip_denoised,
            "sampler_policy": ("r2_posterior_x0_bounded_v1" if clip_denoised else
                               "r2_eps_consistent_respacing_v1" if is_r2 else "legacy_eps_v1"),
            **self.diffusion.last_sampling_stats,
        }
        if isinstance(self.normalization_stats, RadialAmplitudeStats):
            if self.normalization_stats.mode == "radial_q95_sqrt":
                amplitude_fake = self.normalization_stats.decode(log_amp_norm_fake)
            else:
                amplitude_fake = self.normalization_stats.decode(log_amp_norm_fake.clamp(-normalized_clip, normalized_clip))
            return FrequencyUtils.enforce_amplitude_conjugate_symmetry(amplitude_fake)
        # The synthetic-quality policy limits the generated value to the central
        # source-train normalization range before inverse-log expansion.  A
        # configurable 3-sigma default is far safer than the old [-10, 10]
        # range, which could turn a moderate DDPM error into enormous FFT energy.
        log_amp_norm_fake = torch.clamp(
            log_amp_norm_fake,
            min=-normalized_clip,
            max=normalized_clip,
        )
        if mean is None or std is None:
            if self.normalization_stats is not None:
                mean, std = self.normalization_stats.tensors(log_amp_norm_fake)
            else:
                mean, std = 0.0, 1.0
        log_amp_fake = FrequencyUtils.denormalize_amplitude(log_amp_norm_fake, mean, std)
        amplitude_fake = FrequencyUtils.inverse_log_amplitude(log_amp_fake)
        # A real-valued image has a conjugate-symmetric Fourier spectrum.  The
        # DDPM does not enforce that constraint by architecture, so project its
        # amplitude before combining it with a real bona-fide phase.
        return FrequencyUtils.enforce_amplitude_conjugate_symmetry(amplitude_fake)
