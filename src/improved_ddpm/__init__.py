"""
improved_ddpm
==============
Package cai dat "Frequency-Aware DDPM Augmentation & Dual-Branch Contrastive
Learning" cho Face Anti-Spoofing.

Submodules:
    - diffusion : ha tang Diffusion dung chung (schedule, variance, losses, sampling)
    - models    : kien truc mang (UNet, Dual-Branch classifier)
    - fas       : cac thanh phan chuyen biet cho FAS (FFT utils, Amplitude DDPM)
    - scripts   : entrypoint chay train/demo
"""

__version__ = "0.3.0"
