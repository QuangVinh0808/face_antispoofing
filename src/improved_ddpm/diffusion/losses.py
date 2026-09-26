"""
Module 4 - Ham mat mat lai (Hybrid Objective Loss)
=====================================================
L_hybrid = L_simple + lambda * L_vlb,  lambda = 0.001

- L_simple: MSE giua nhieu thuc (eps) va nhieu du doan (eps_theta).
  Day la thanh phan chinh, dieu khien viec hoc mu_theta (qua eps_theta).

- L_vlb: uoc luong Variational Lower Bound (KL divergence giua posterior
  thuc q(x_{t-1}|x_t,x_0) va model p_theta(x_{t-1}|x_t)), nhung QUAN TRONG:
  ta ap dung stop-gradient (detach) cho phan trung binh mu_theta truoc khi
  dua vao L_vlb. Nho vay, L_vlb CHI tao gradient cho nhanh du doan v
  (-> Sigma_theta), khong lam nhieu qua trinh hoc mu_theta cua L_simple.

Tham khao: Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic
Models", 2021 (muc 3.2 - 3.3).
"""

import torch
import torch.nn.functional as F

from .schedule import NoiseSchedule
from .variance import model_output_to_eps_and_v, compute_learned_variance


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL( N(mean1, exp(logvar1)) || N(mean2, exp(logvar2)) ), tinh theo tung phan tu.
    Cong thuc dong dang chuan cho 2 phan phoi Gauss (Cong thuc duoc dung
    rong rai trong cac repo Improved-DDPM goc cua OpenAI).
    """
    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def predicted_mean_from_eps(x_t, t, eps_theta, schedule: NoiseSchedule):
    """
    mu_theta(x_t, t) = 1/sqrt(alpha_t) * ( x_t - beta_t/sqrt(1-alpha_bar_t) * eps_theta )
    (Cong thuc mu chuan trong DDPM, dao nguoc tu eps_theta.)
    """
    alpha_t = schedule._extract(schedule.alphas, t, x_t.shape)
    beta_t = schedule._extract(schedule.betas, t, x_t.shape)
    sqrt_one_minus_ac = schedule._extract(schedule.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

    mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - (beta_t / sqrt_one_minus_ac) * eps_theta)
    return mean


def hybrid_loss(
    model_out: torch.Tensor,
    x0: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
    eps_true: torch.Tensor,
    schedule: NoiseSchedule,
    lam: float = 0.001,
):
    """
    Tinh L_hybrid = L_simple + lambda * L_vlb

    Args:
        model_out: output tho cua UNet, shape (B, 2*C, H, W)
        x0:       anh goc khong nhieu (B, C, H, W)
        x_t:      anh da bi lam nhieu o buoc t (B, C, H, W)
        t:        timestep (B,)
        eps_true: nhieu Gauss thuc te da dung de tao x_t (B, C, H, W)
        schedule: doi tuong NoiseSchedule (cosine schedule)
        lam:      he so lambda cho L_vlb (mac dinh 0.001)

    Tra ve: dict {"loss", "l_simple", "l_vlb"}
    """
    eps_theta, v = model_output_to_eps_and_v(model_out)

    # ---------- L_simple ----------
    l_simple = F.mse_loss(eps_theta, eps_true)

    # ---------- L_vlb ----------
    # 1) Posterior THUC (target): mu_tilde, beta_tilde_t (tu x0, x_t thuc)
    true_mean, _, true_log_var = schedule.q_posterior_mean_variance(x0, x_t, t)

    # 2) Mean cua model: stop-gradient! Chi dung eps_theta.detach() de tinh mu_theta,
    #    nhu vay nhanh L_vlb KHONG anh huong gradient cua eps_theta (-> mu_theta)
    eps_theta_detached = eps_theta.detach()
    model_mean_detached = predicted_mean_from_eps(x_t, t, eps_theta_detached, schedule)

    # 3) Phuong sai hoc duoc Sigma_theta (day la phan DUY NHAT con gradient,
    #    vi v khong bi detach -> L_vlb chi huan luyen nhanh v)
    _, model_log_var = compute_learned_variance(v, t, schedule)

    # 4) KL divergence giua N(true_mean, true_var) va N(model_mean_detached, Sigma_theta)
    kl = normal_kl(true_mean, true_log_var, model_mean_detached, model_log_var)
    # Trung binh theo so chieu du lieu (C,H,W), giu batch -> rat giong "bits-per-dim"
    kl = kl.mean(dim=list(range(1, kl.dim())))  # (B,)

    # Stage 6 stability correction: the original Improved-DDPM objective uses a
    # decoder NLL at t=0 instead of this KL term. The compact implementation in
    # earlier stages applied KL at t=0 as well, which can create enormous finite
    # values (and eventually Inf/NaN) because the posterior variance is nearly
    # zero. Keep L_simple for every timestep, but exclude t=0 from the simplified
    # VLB approximation. This is safer than pretending the t=0 KL is valid.
    valid_vlb = t > 0
    if torch.any(valid_vlb):
        l_vlb = kl[valid_vlb].mean()
    else:
        l_vlb = torch.zeros((), device=kl.device, dtype=kl.dtype)

    loss = l_simple + lam * l_vlb

    return {"loss": loss, "l_simple": l_simple, "l_vlb": l_vlb}
