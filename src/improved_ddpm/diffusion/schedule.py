"""
Module 1 - Lich trinh nhieu (Cosine Noise Schedule)
====================================================
Cai dat cosine schedule theo Improved DDPM (Nichol & Dhariwal, 2021),
thay cho linear schedule cua DDPM goc (Ho et al., 2020).

Cong thuc:
    f(t) = cos( ((t/T + s) / (1+s)) * pi/2 )^2
    alpha_bar_t = f(t) / f(0)
    beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}
    beta_t = clip(beta_t, max=0.999)   # tranh singularity khi t -> T
"""

import torch
import numpy as np


def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """
    Tra ve tensor beta co kich thuoc (T,), beta[0] tuong ung voi buoc t=1.

    Args:
        T: tong so buoc diffusion (vd 4000)
        s: he so offset nho, tranh beta_t qua nho gan t=0 (mac dinh 0.008)
    """
    steps = T + 1
    t = torch.linspace(0, T, steps, dtype=torch.float64)  # t = 0, 1, ..., T

    # f(t) = cos( ((t/T + s)/(1+s)) * pi/2 )^2
    f_t = torch.cos(((t / T + s) / (1 + s)) * (np.pi / 2)) ** 2

    # alpha_bar_t = f(t)/f(0)
    alphas_cumprod = f_t / f_t[0]

    # beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}, voi t = 1..T
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

    # Clip de tranh diem di thuong (singularities) cuoi qua trinh diffusion
    betas = torch.clamp(betas, max=0.999)

    return betas.float()  # shape (T,)


class NoiseSchedule:
    """
    Goi cac dai luong dan xuat tu beta_t de dung xuyen suot
    qua trinh forward (them nhieu) va loss/sampling.

    Tat ca tensor co shape (T,), index 0 <-> buoc t=1 (theo quy uoc pytorch,
    ta se truy cap bang gather voi t trong [0, T-1]).
    """

    def __init__(self, T: int = 4000, s: float = 0.008, device="cpu"):
        self.T = T
        betas = cosine_beta_schedule(T, s).to(device)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=device), alphas_cumprod[:-1]]
        )

        self.betas = betas                              # beta_t
        self.alphas = alphas                             # alpha_t = 1 - beta_t
        self.alphas_cumprod = alphas_cumprod              # alpha_bar_t
        self.alphas_cumprod_prev = alphas_cumprod_prev    # alpha_bar_{t-1}

        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        # Posterior variance: beta_tilde_t = beta_t * (1-alpha_bar_{t-1})/(1-alpha_bar_t)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        # t=0 (buoc dau) co the cho ra 0 -> clip nho de log() khong loi
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )

        # He so cho posterior mean: mu_tilde(x_t, x_0)
        self.posterior_mean_coef1 = (
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

    def to(self, device):
        for k, v in self.__dict__.items():
            if torch.is_tensor(v):
                setattr(self, k, v.to(device))
        return self

    @staticmethod
    def _extract(a: torch.Tensor, t: torch.Tensor, x_shape):
        """Lay gia tri a[t] cho moi sample trong batch, reshape de broadcast voi x."""
        b = t.shape[0]
        out = a.gather(0, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """
        Forward diffusion: x_t = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*eps
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_omac = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_ac * x0 + sqrt_omac * noise, noise

    def q_posterior_mean_variance(self, x0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        """
        Tra ve mu_tilde(x_t, x_0) va beta_tilde_t (posterior thuc, dung lam target cho VLB).
        """
        coef1 = self._extract(self.posterior_mean_coef1, t, x_t.shape)
        coef2 = self._extract(self.posterior_mean_coef2, t, x_t.shape)
        posterior_mean = coef1 * x0 + coef2 * x_t

        posterior_var = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_var = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_var, posterior_log_var
