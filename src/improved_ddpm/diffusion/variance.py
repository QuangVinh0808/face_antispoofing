"""
Module 3 - Tinh toan phuong sai (Learning Variance)
=====================================================
Vector v (mot nua so channel output cua UNet) khong qua ham kich hoat rang buoc
(tanh/sigmoid...) - giu nguyen gia tri thuc (raw).

Phuong sai Sigma_theta duoc noi suy trong khong gian LOGARIT giua:
    - beta_t        (gioi han tren cua phuong sai - "noisiest" choice)
    - beta_tilde_t   (posterior variance thuc - "tightest" choice)

Cong thuc:
    Sigma_theta(x_t, t) = exp( v * log(beta_t) + (1-v) * log(beta_tilde_t) )
"""

import torch
from .schedule import NoiseSchedule


def model_output_to_eps_and_v(model_out: torch.Tensor):
    """
    Tach output 2*C channel cua UNet thanh (eps_theta, v).
    model_out: (B, 2*C, H, W)
    """
    eps_theta, v = torch.chunk(model_out, 2, dim=1)
    return eps_theta, v


def compute_learned_variance(v: torch.Tensor, t: torch.Tensor, schedule: NoiseSchedule):
    """
    Tinh Sigma_theta(x_t, t) = exp( v*log(beta_t) + (1-v)*log(beta_tilde_t) )

    v: (B, C, H, W) - raw variance head output. Stage 6 maps this value
       through sigmoid to obtain an interpolation fraction in [0,1]. This keeps
       the learned log-variance inside the valid [posterior, beta] range while
       retaining smooth gradients.
    t:  (B,) - timestep

    Tra ve: log_variance (B, C, H, W) va variance (B, C, H, W)
    """
    log_beta_t = schedule._extract(torch.log(schedule.betas), t, v.shape)
    log_beta_tilde_t = schedule._extract(schedule.posterior_log_variance_clipped, t, v.shape)

    # Stage 6 stability correction: constrain the learned interpolation weight
    # smoothly to [0,1]. The previous raw extrapolation could drive log-variance
    # far outside [log(beta_tilde), log(beta_t)] and produce Inf/NaN, especially
    # after adding conditional DDPM embeddings. Sigmoid preserves gradients while
    # enforcing a valid learned-range interpolation.
    v_frac = torch.sigmoid(v)

    log_variance = v_frac * log_beta_t + (1.0 - v_frac) * log_beta_tilde_t
    variance = torch.exp(log_variance)
    return variance, log_variance
