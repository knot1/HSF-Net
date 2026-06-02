from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .frequency_modules import FrequencyModule

try:
    from pytorch_wavelets import DWTForward, DWTInverse
except ImportError:  # pragma: no cover (optional dependency at runtime)
    DWTForward = None
    DWTInverse = None


class FDRModule(nn.Module):
    def __init__(
        self,
        dim: int,
        mode: str = "fft",
        wave: str = "haar",
        levels: int = 1,
        gamma_init: float = 0.1,
    ):
        super().__init__()
        self.mode = mode.lower()
        valid_modes = {"fft", "wavelet", "both"}
        if self.mode not in valid_modes:
            raise ValueError(f"Unsupported FDR mode: {mode}. Options: fft, wavelet, both.")
        self.use_fft = self.mode in ("fft", "both")
        self.use_wavelet = self.mode in ("wavelet", "both")

        if self.use_fft:
            self.fft_module = FrequencyModule(dim)
            self.fft_fuse = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        if self.use_wavelet:
            if DWTForward is None or DWTInverse is None:
                raise ImportError("pytorch_wavelets is required for wavelet mode.")
            self.dwt = DWTForward(J=levels, wave=wave, mode="zero")
            self.idwt = DWTInverse(wave=wave, mode="zero")
            self.wavelet_fuse = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        self.gamma = nn.Parameter(torch.tensor(gamma_init, dtype=torch.float32))

    def _fft_freq(self, x: torch.Tensor) -> torch.Tensor:
        high, low = self.fft_module(x)
        freq = high + low
        return self.fft_fuse(freq)

    def _wavelet_freq(self, x: torch.Tensor) -> torch.Tensor:
        yl, yh = self.dwt(x)
        zero_low = torch.zeros_like(yl)
        high_only = self.idwt((zero_low, yh))
        if high_only.shape[-2:] != x.shape[-2:]:
            high_only = F.interpolate(
                high_only,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return self.wavelet_fuse(high_only)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        freq_feats = []
        if self.use_fft:
            freq_feats.append(self._fft_freq(feat))
        if self.use_wavelet:
            freq_feats.append(self._wavelet_freq(feat))
        if not freq_feats:
            return feat

        freq_sum = torch.stack(freq_feats, dim=0).sum(dim=0)
        return feat + self.gamma * freq_sum
