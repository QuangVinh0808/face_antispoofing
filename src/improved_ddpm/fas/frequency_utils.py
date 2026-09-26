"""
Module 1 - FrequencyUtils
==========================
Trich xuat va tai tao anh trong mien tan so (FFT), dung lam nen tang cho
ca Amplitude Diffusion Model va Dual-Branch Classifier.

- Amplitude: chua thong tin ket cau, nhieu Moire, vet in giay (artifact gia mao)
- Phase:     chua cau truc hinh hoc, hinh dang khuon mat (can giu nguyen tu anh thuc)
"""

import torch


class FrequencyUtils:
    """
    Tat ca ham la staticmethod -> co the goi truc tiep FrequencyUtils.fft_decompose(...)
    ma khong can khoi tao object. Ho tro batch tensor (B, C, H, W).
    """

    @staticmethod
    def fft_decompose(image: torch.Tensor, norm: str = "ortho"):
        """
        Chuyen anh RGB tu mien khong gian -> mien tan so bang FFT 2D day du
        (dung fft2 tren 2 chieu cuoi H, W - khong dung rfft2 vi ta can amplitude/phase
        doi xung de de dang ghep lai chinh xac).

        Args:
            image: tensor (B, C, H, W), gia tri thuc (real), khong can chuan hoa [0,1]/[-1,1] deu duoc.
            norm:  chuan hoa FFT ("ortho" giu nang luong on dinh, khuyen dung cho deep learning)

        Tra ve:
            amplitude: (B, C, H, W) - |F(image)|, da duoc fftshift de tan so thap o giua
            phase:     (B, C, H, W) - angle(F(image)), gia tri trong [-pi, pi]
        """
        # FFT 2D tren 2 chieu cuoi (H, W), giu nguyen batch & channel
        freq = torch.fft.fft2(image, dim=(-2, -1), norm=norm)
        freq = torch.fft.fftshift(freq, dim=(-2, -1))  # dua tan so thap (DC) ve giua anh

        amplitude = torch.abs(freq)
        phase = torch.angle(freq)
        return amplitude, phase

    @staticmethod
    def ifft_reconstruct(amplitude: torch.Tensor, phase: torch.Tensor, norm: str = "ortho"):
        """
        Tai tao anh tu Amplitude + Phase (nguoc lai voi fft_decompose).

        Args:
            amplitude: (B, C, H, W)
            phase:     (B, C, H, W)

        Tra ve:
            image: (B, C, H, W) - phan thuc (real) cua IFFT, day chinh la anh RGB tai tao.
        """
        real = amplitude * torch.cos(phase)
        imag = amplitude * torch.sin(phase)
        freq = torch.complex(real, imag)

        freq = torch.fft.ifftshift(freq, dim=(-2, -1))  # dao nguoc fftshift truoc khi ifft
        image = torch.fft.ifft2(freq, dim=(-2, -1), norm=norm)
        return image.real  # phan ao ly thuyet xap xi 0, bo qua

    @staticmethod
    def enforce_amplitude_conjugate_symmetry(amplitude: torch.Tensor) -> torch.Tensor:
        """Project a shifted amplitude map onto the real-image FFT subspace.

        For a real spatial image, the unshifted spectrum obeys
        ``F[k,l] = conj(F[-k,-l])``; consequently its amplitude must be
        centrosymmetric. A DDPM predicts every frequency bin independently and
        does not guarantee this property. Averaging each bin with its conjugate
        partner makes the generated amplitude compatible with the phase of a
        real bona-fide parent before IFFT.
        """
        if amplitude.ndim < 2:
            raise ValueError("amplitude must have at least two spatial dimensions")
        unshifted = torch.fft.ifftshift(amplitude, dim=(-2, -1))
        partner = torch.roll(
            torch.flip(unshifted, dims=(-2, -1)),
            shifts=(1, 1),
            dims=(-2, -1),
        )
        symmetric = 0.5 * (unshifted + partner)
        return torch.fft.fftshift(symmetric, dim=(-2, -1))

    @staticmethod
    def log_amplitude(amplitude: torch.Tensor) -> torch.Tensor:
        """Nen log de on dinh huan luyen Diffusion model (amplitude raw co dynamic range rat lon)."""
        return torch.log1p(amplitude)

    @staticmethod
    def inverse_log_amplitude(log_amp: torch.Tensor) -> torch.Tensor:
        # Clamp truoc khi expm1: model (dac biet luc chua train hoac sample voi K rat nho)
        # co the sinh ra gia tri rat lon -> expm1 overflow thanh inf/nan.
        # 20 la nguong an toan (e^20 ~ 4.8e8, du lon cho amplitude thuc te, tranh overflow fp32).
        log_amp = torch.clamp(log_amp, min=-20.0, max=20.0)
        return torch.expm1(log_amp).clamp(min=0)  # amplitude phai >= 0

    @staticmethod
    def normalize_amplitude(log_amp: torch.Tensor, mean: float = None, std: float = None):
        """
        Chuan hoa log-amplitude ve phan phoi ~N(0,1) de dua vao Diffusion model
        (giong chuan hoa anh RGB ve [-1,1] trong DDPM thong thuong).
        Neu khong truyen mean/std, tu tinh tren batch hien tai (chi dung cho demo;
        thuc te nen tinh mean/std tren toan bo training set va luu lai co dinh).
        """
        if mean is None:
            mean = log_amp.mean()
        if std is None:
            std = log_amp.std() + 1e-6
        return (log_amp - mean) / std, mean, std

    @staticmethod
    def denormalize_amplitude(norm_log_amp: torch.Tensor, mean, std):
        return norm_log_amp * std + mean
