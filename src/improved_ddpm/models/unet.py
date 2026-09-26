"""
Module 2 - Kien truc mang (UNet Architecture)
==============================================
- Timestep embedding kieu sinusoidal (giong Transformer positional encoding).
- ResBlock dung "scale-shift norm" (Adaptive GroupNorm):
      h = GroupNorm(h)
      h = h * (w + 1) + b
  voi (w, b) duoc tinh tu timestep embedding (thay vi cong truc tiep nhu DDPM goc).
- Attention block: Multi-head self-attention (4 heads), chi ap dung
  o cac do phan giai 16x16 va 8x8 (truyen vao qua attention_resolutions).
- Output: nhan doi so channel (2*C). C dau la epsilon_theta (du doan nhieu),
  C sau la vector v (dung de noi suy phuong sai).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Timestep embedding (sinusoidal), giong positional encoding cua Transformer
# ----------------------------------------------------------------------------
def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:  # dim le -> pad them 1 cot 0
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb  # (B, dim)


# ----------------------------------------------------------------------------
# ResBlock voi Adaptive GroupNorm: GroupNorm(h) * (w+1) + b
# ----------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, num_groups=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        # Linear chieu timestep embedding -> (w, b), moi cai co out_ch chieu
        self.time_proj = nn.Linear(time_emb_dim, out_ch * 2)

        self.norm2 = nn.GroupNorm(num_groups, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # --- Scale-shift conditioning: GroupNorm(h) * (w + 1) + b ---
        w, b = self.time_proj(t_emb).chunk(2, dim=-1)   # (B, out_ch) moi cai
        w = w[:, :, None, None]
        b = b[:, :, None, None]

        h = self.norm2(h)
        h = h * (w + 1) + b      # <-- cong thuc theo yeu cau
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip(x)


# ----------------------------------------------------------------------------
# Multi-head Self-Attention block (ap dung tai 16x16 va 8x8)
# ----------------------------------------------------------------------------
class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4, num_groups=8):
        super().__init__()
        assert channels % num_heads == 0, "channels phai chia het cho num_heads"
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(num_groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]   # (B, heads, C/heads, HW)

        scale = 1.0 / math.sqrt(C // self.num_heads)
        attn = torch.einsum("bhcn,bhcm->bhnm", q, k) * scale
        attn = attn.softmax(dim=-1)

        out = torch.einsum("bhnm,bhcm->bhcn", attn, v)
        out = out.reshape(B, C, H, W)
        out = self.proj_out(out)
        return x + out


# ----------------------------------------------------------------------------
# Cac block tien ich: Downsample / Upsample
# ----------------------------------------------------------------------------
class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.op(x)


# ----------------------------------------------------------------------------
# UNet day du
# ----------------------------------------------------------------------------
class UNet(nn.Module):
    """
    UNet thu nho, du dung cho FAS amplitude-conditioned diffusion.

    Args:
        in_channels: so channel input (vd 1 cho log-amplitude grayscale, 3 cho RGB)
        base_channels: so channel o lop dau
        channel_mults: he so nhan channel qua moi lan downsample
        attention_resolutions: tap cac do phan giai (tinh theo H/W cua feature map)
            se ap dung attention block. Theo yeu cau: {16, 8}.
        num_res_blocks: so ResBlock moi resolution
        image_size: kich thuoc anh dau vao (H=W=image_size), dung de xac dinh
            resolution hien tai khi di qua cac tang downsample/upsample.
    """

    def __init__(
        self,
        in_channels=1,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        attention_resolutions=(16, 8),
        num_res_blocks=2,
        num_heads=4,
        image_size=64,
        time_emb_dim=256,
        dropout=0.1,
        num_conditions: int = 0,
    ):
        super().__init__()
        self.image_size = image_size
        self.attention_resolutions = set(attention_resolutions)
        self.num_conditions = int(num_conditions)

        # ---- Timestep embedding MLP ----
        self.time_emb_dim = time_emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )
        # Stage 6 D3: optional attack-type conditioning.  The condition
        # embedding is added to the timestep embedding, so all ResBlocks receive
        # the same print/replay/other context without changing their interface.
        self.condition_emb = (
            nn.Embedding(self.num_conditions, time_emb_dim)
            if self.num_conditions > 0
            else None
        )
        if self.condition_emb is not None:
            # Small initialization prevents the new Stage-6 conditioning path
            # from overwhelming the already-scaled timestep embedding early in training.
            nn.init.normal_(self.condition_emb.weight, mean=0.0, std=0.02)

        # ---- Input conv ----
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # ---- Encoder (Downsampling path) ----
        self.down_blocks = nn.ModuleList()
        ch = base_channels
        cur_res = image_size
        channels_track = [ch]  # luu lai channel tai moi diem de noi skip-connection

        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                block = nn.ModuleList([ResBlock(ch, out_ch, time_emb_dim, dropout=dropout)])
                if cur_res in self.attention_resolutions:
                    block.append(AttentionBlock(out_ch, num_heads=num_heads))
                self.down_blocks.append(block)
                ch = out_ch
                channels_track.append(ch)

            if level != len(channel_mults) - 1:
                self.down_blocks.append(nn.ModuleList([Downsample(ch)]))
                channels_track.append(ch)
                cur_res //= 2

        # ---- Bottleneck ----
        self.mid_block1 = ResBlock(ch, ch, time_emb_dim, dropout=dropout)
        self.mid_attn = AttentionBlock(ch, num_heads=num_heads)
        self.mid_block2 = ResBlock(ch, ch, time_emb_dim, dropout=dropout)

        # ---- Decoder (Upsampling path) ----
        self.up_blocks = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks + 1):
                skip_ch = channels_track.pop()
                block = nn.ModuleList(
                    [ResBlock(ch + skip_ch, out_ch, time_emb_dim, dropout=dropout)]
                )
                if cur_res in self.attention_resolutions:
                    block.append(AttentionBlock(out_ch, num_heads=num_heads))
                self.up_blocks.append(block)
                ch = out_ch

            if level != 0:
                self.up_blocks.append(nn.ModuleList([Upsample(ch)]))
                cur_res *= 2

        # ---- Output: nhan doi so channel (eps + v) ----
        self.out_norm = nn.GroupNorm(8, ch)
        self.out_conv = nn.Conv2d(ch, in_channels * 2, 3, padding=1)

    def forward(self, x, t, condition=None):
        """
        x: (B, C, H, W) - anh nhieu x_t (hoac log-amplitude nhieu)
        t: (B,) - timestep (long tensor)

        Tra ve: tensor (B, 2*C, H, W)
            - kenh dau (C dau) = epsilon_theta (du doan nhieu)
            - kenh sau (C sau) = v (dung noi suy phuong sai)
        """
        t_emb = timestep_embedding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)
        if self.condition_emb is not None:
            if condition is None:
                raise ValueError("Conditional UNet requires condition labels")
            condition = condition.to(device=x.device, dtype=torch.long)
            if condition.ndim != 1 or condition.shape[0] != x.shape[0]:
                raise ValueError("condition must have shape (B,)")
            t_emb = t_emb + 0.1 * self.condition_emb(condition)

        h = self.input_conv(x)
        skips = [h]

        for block in self.down_blocks:
            if isinstance(block[0], Downsample):
                h = block[0](h)
            else:
                h = block[0](h, t_emb)
                if len(block) > 1:
                    h = block[1](h)
            skips.append(h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        for block in self.up_blocks:
            if isinstance(block[0], Upsample):
                h = block[0](h)
            else:
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = block[0](h, t_emb)
                if len(block) > 1:
                    h = block[1](h)

        h = self.out_norm(h)
        h = F.silu(h)
        out = self.out_conv(h)
        return out  # (B, 2*C, H, W)
