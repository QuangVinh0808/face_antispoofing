"""
patch_utils.py
================
Cap nhat 1 - Co che Patch-based trong Module Tien xu ly du lieu.

Truoc khi dua qua fft_decompose(), anh RGB goc (Full RGB) duoc cat ngau nhien
mot vung (Random Cropping) voi ty le scale dao dong trong [0.2, 1.0] (tinh theo
DIEN TICH so voi anh goc), roi resize ve kich thuoc chuan (mac dinh 224x224)
de tao ra Patch RGB.

    Full RGB  (anh goc, da resize chuan 224x224)
        |
        |  (KHONG bien doi gi them)
        v
    Full RGB  ──────────────────────────────────► dung truc tiep cho luong "Full"

    Full RGB
        |
        v  RandomCropResize(scale=[0.2,1.0]) -> resize 224x224
    Patch RGB ─────────────────────────────────► dung cho luong "Patch"

Ca Full RGB va Patch RGB sau do deu duoc dua qua FrequencyUtils.fft_decompose()
o ben ngoai (trong train_fas_pipeline.py) de tao ra 2 cap (Amplitude, Phase)
doc lap: (Full_Amplitude, Full_Phase) va (Patch_Amplitude, Patch_Phase).
"""

import random
import torch
import torch.nn.functional as F


def random_crop_resize_batch(
    images: torch.Tensor,
    scale_range=(0.2, 1.0),
    output_size: int = 224,
) -> torch.Tensor:
    """
    Ap dung Random Cropping (RC) + Resize cho TUNG anh trong batch (moi anh
    duoc cat voi 1 scale va 1 vi tri ngau nhien KHAC NHAU, giong augmentation
    chuan trong huan luyen CNN/ViT).

    Args:
        images: (B, C, H, W) - Full RGB (da o kich thuoc chuan, vd 224x224)
        scale_range: (min_scale, max_scale) - ty le DIEN TICH vung crop so voi
            anh goc, lay ngau nhien Uniform(min_scale, max_scale) cho moi anh.
            Vi du scale=0.5 -> dien tich vung crop = 50% dien tich anh goc.
        output_size: kich thuoc canh vuong sau khi resize (mac dinh 224).

    Tra ve:
        patches: (B, C, output_size, output_size) - Patch RGB, cung dtype/device
                 voi `images`.
    """
    B, C, H, W = images.shape
    patches = []

    for i in range(B):
        # 1) Lay scale ngau nhien trong [scale_range[0], scale_range[1]]
        scale = random.uniform(scale_range[0], scale_range[1])

        # 2) Dien tich vung crop = scale * (H*W); gia su crop hinh vuong
        #    -> canh crop = sqrt(scale * H * W), gioi han khong vuot kich thuoc anh.
        crop_area = scale * H * W
        crop_side = int(round(crop_area ** 0.5))
        crop_h = min(max(crop_side, 1), H)
        crop_w = min(max(crop_side, 1), W)

        # 3) Chon vi tri crop ngau nhien (top-left corner)
        top = random.randint(0, H - crop_h)
        left = random.randint(0, W - crop_w)

        # 4) Cat vung anh (1, C, crop_h, crop_w)
        cropped = images[i : i + 1, :, top : top + crop_h, left : left + crop_w]

        # 5) Resize ve kich thuoc chuan (output_size x output_size)
        resized = F.interpolate(
            cropped, size=(output_size, output_size), mode="bilinear", align_corners=False
        )
        patches.append(resized)

    return torch.cat(patches, dim=0)  # (B, C, output_size, output_size)


def resize_to_standard(images: torch.Tensor, output_size: int = 224) -> torch.Tensor:
    """
    Tien ich: dam bao Full RGB luon o kich thuoc chuan truoc khi dua vao pipeline
    (vd anh goc tu dataset co kich thuoc khac nhau).
    """
    if images.shape[-1] == output_size and images.shape[-2] == output_size:
        return images
    return F.interpolate(images, size=(output_size, output_size), mode="bilinear", align_corners=False)
