"""
Module 5 - Toc do lay mau (Fast Sampling / Respacing)
=======================================================
Cho phep lay mau voi K buoc (K << T), bang cach chon K chi so buoc t
phan bo deu trong [1, T] (lam tron ve so nguyen gan nhat, loai trung).

Voi moi buoc da chon S_t, ta tinh LAI:
    - alpha_bar_{S_t}            (lay truc tiep tu schedule goc tai vi tri S_t)
    - beta_{S_t}  = 1 - alpha_bar_{S_t} / alpha_bar_{S_{t-1}}
    - beta_tilde_{S_t} = beta_{S_t} * (1-alpha_bar_{S_{t-1}}) / (1-alpha_bar_{S_t})

=> Day chinh la "respacing": mang no-ron KHONG can huan luyen lai, vi
gia tri t truyen vao mang van la t GOC (S_t) trong khong gian [0, T-1],
chi co cac he so beta/alpha dung de SAMPLING la duoc tinh lai cho phu hop
voi khoang cach moi giua cac buoc.

Day cung la class GaussianDiffusion tong hop: ket noi NoiseSchedule + UNet +
losses + sampling thanh 1 API don gian (training_loss(), sample()).
"""

from typing import List
import torch

from .schedule import NoiseSchedule
from .losses import hybrid_loss
from .variance import model_output_to_eps_and_v, compute_learned_variance


def make_respaced_steps(T: int, K: int) -> List[int]:
    """
    Chon K chi so buoc t (gia tri nguyen trong [0, T-1]) phan bo deu.
    Vi du T=1000, K=50 -> lay [0, 20, 40, ..., 980] (xap xi).
    """
    if K >= T:
        return list(range(T))

    # Chia deu khoang cach tu 0 den T-1, lam tron so nguyen gan nhat
    indices = torch.linspace(0, T - 1, K)
    steps = sorted(set(int(round(i.item())) for i in indices))
    return steps


class RespacedSchedule:
    """
    Bao boc mot NoiseSchedule goc (T buoc) thanh phien ban rut gon K buoc,
    tinh lai beta_S va beta_tilde_S theo cong thuc o tren.
    """

    def __init__(self, base_schedule: NoiseSchedule, respaced_steps: List[int]):
        self.base = base_schedule
        self.steps = respaced_steps  # vi du [0, 20, 40, ..., 980], do dai K
        device = base_schedule.alphas_cumprod.device

        # alpha_bar tai cac buoc da chon
        ac = base_schedule.alphas_cumprod[respaced_steps]                  # (K,)
        ac_prev = torch.cat([torch.tensor([1.0], device=device), ac[:-1]])  # (K,)

        betas_new = 1.0 - (ac / ac_prev)
        betas_new = torch.clamp(betas_new, max=0.999)

        self.alphas_cumprod = ac
        self.alphas_cumprod_prev = ac_prev
        self.betas = betas_new
        self.alphas = 1.0 - betas_new

        self.posterior_variance = betas_new * (1.0 - ac_prev) / (1.0 - ac)
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - ac)

        # map tu vi tri "moi" (0..K-1) -> timestep "goc" (0..T-1), de truyen vao UNet
        self.original_t = torch.tensor(respaced_steps, device=device, dtype=torch.long)

    def __len__(self):
        return len(self.steps)


class GaussianDiffusion:
    """
    API tong hop: training_loss() de huan luyen, sample()/fast_sample() de sinh anh.
    """

    def __init__(self, model: torch.nn.Module, T: int = 4000, s: float = 0.008, device="cpu"):
        self.model = model
        self.T = T
        self.device = device
        self.schedule = NoiseSchedule(T=T, s=s, device=device)

    # ----------------------- Training -----------------------
    def training_loss(self, x0: torch.Tensor, lam: float = 0.001, condition=None):
        """
        Lay mau t ngau nhien cho moi sample trong batch, forward-diffuse x0 -> x_t,
        cho UNet du doan, roi tinh hybrid loss.
        """
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=self.device, dtype=torch.long)

        x_t, eps_true = self.schedule.q_sample(x0, t)
        model_out = (
            self.model(x_t, t) if condition is None else self.model(x_t, t, condition=condition)
        )  # (B, 2C, H, W)

        losses = hybrid_loss(model_out, x0, x_t, t, eps_true, self.schedule, lam=lam)
        return losses, t, x_t

    # ----------------------- Sampling (full T buoc) -----------------------
    @torch.no_grad()
    def p_sample_step(self, x_t, t_tensor, t_index, schedule, condition=None):
        """
        Mot buoc sampling DDPM (ho tro ca full-step va respaced):
            x_{t-1} = mu_theta(x_t,t) + sigma_theta(x_t,t) * z ,  z ~ N(0,I) (z=0 khi t_index==0)
        """
        model_out = (
            self.model(x_t, t_tensor)
            if condition is None
            else self.model(x_t, t_tensor, condition=condition)
        )
        eps_theta, v = model_output_to_eps_and_v(model_out)

        beta_t = schedule.betas[t_index].view(-1, 1, 1, 1)
        alpha_t = schedule.alphas[t_index].view(-1, 1, 1, 1)
        ac_t = schedule.alphas_cumprod[t_index].view(-1, 1, 1, 1)
        sqrt_one_minus_ac = torch.sqrt(1.0 - ac_t)

        # mu_theta tu eps_theta (cong thuc DDPM chuan)
        mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - (beta_t / sqrt_one_minus_ac) * eps_theta)

        # Sigma_theta = exp(v*log(beta_t) + (1-v)*log(beta_tilde_t))  (Module 3)
        log_beta_t = torch.log(beta_t)
        log_beta_tilde_t = schedule.posterior_log_variance_clipped[t_index].view(-1, 1, 1, 1)
        # Match training-time learned-range stabilization.
        v_frac = torch.sigmoid(v)
        log_var = v_frac * log_beta_t + (1.0 - v_frac) * log_beta_tilde_t
        sigma = torch.exp(0.5 * log_var)

        noise = torch.randn_like(x_t)
        # Khong them nhieu o buoc cuoi cung (t_index == 0)
        mask = (t_index > 0).float().view(-1, 1, 1, 1)
        x_prev = mean + mask * sigma * noise
        return x_prev

    @torch.no_grad()
    def sample(self, shape, K: int = None, condition=None):
        """
        Sinh anh moi.
        Args:
            shape: (B, C, H, W)
            K: so buoc rut gon. Neu None -> dung full T buoc.
               Neu duoc cung cap (vd 50, 100) -> dung respaced schedule (fast sampling).
        """
        if K is None:
            return self._sample_full(shape, condition=condition)
        return self._sample_fast(shape, K, condition=condition)

    @torch.no_grad()
    def _sample_full(self, shape, condition=None):
        x = torch.randn(shape, device=self.device)
        B = shape[0]
        for i in reversed(range(self.T)):
            t_index = torch.full((B,), i, device=self.device, dtype=torch.long)
            x = self.p_sample_step(x, t_index, t_index, self.schedule, condition=condition)
        return x

    @torch.no_grad()
    def _sample_fast(self, shape, K: int, condition=None):
        """
        Fast sampling voi K buoc << T (vd K=50, T=4000).
        1) Chon K chi so buoc phan bo deu trong [0, T-1].
        2) Tinh lai beta_S, beta_tilde_S tren cac buoc da chon (RespacedSchedule).
        3) Lap nguoc tu K-1 -> 0, truyen vao UNet timestep GOC (original_t)
           de mang tu dieu chinh ty le (khong can huan luyen lai).
        """
        respaced_steps = make_respaced_steps(self.T, K)
        rs = RespacedSchedule(self.schedule, respaced_steps)

        x = torch.randn(shape, device=self.device)
        B = shape[0]
        K_actual = len(rs)

        for i in reversed(range(K_actual)):
            idx_tensor = torch.full((B,), i, device=self.device, dtype=torch.long)
            # timestep GOC (trong khong gian 0..T-1) de dua vao UNet
            t_original = rs.original_t[i].expand(B)
            x = self.p_sample_step(x, t_original, idx_tensor, rs, condition=condition)
        return x
