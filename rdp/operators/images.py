# Portions of this file are adapted from InverseBench:
# https://github.com/devzhk/InverseBench
# Original source: inverse_problems/image_restore.py
#
# Copyright (c) 2025 Hongkai Zheng, Wenda Chu, Bingliang Zhang, Zihui Wu,
# Austin Wang, Berthy T. Feng, Caifeng Zou, Yu Sun, Nikola Kovachki,
# Zachary E. Ross, Katherine L. Bouman, and Yisong Yue.
# SPDX-License-Identifier: MIT
# See LICENSE for the full license text.
# Modified for the RDP implementation.

"""The three FFHQ measurement models used in the paper."""

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch.nn import functional as F

from .base import Operator


class GaussianBlur(Operator):
    def __init__(self, resolution=256, kernel_size=61, intensity=3.0, channels=3, **kwargs):
        super().__init__(**kwargs)
        self.shape = (channels, resolution, resolution)
        impulse = np.zeros((kernel_size, kernel_size))
        impulse[kernel_size // 2, kernel_size // 2] = 1
        kernel = gaussian_filter(impulse, sigma=intensity)
        self.kernel = torch.tensor(kernel, dtype=torch.float32, device=self.device)[None, None].repeat(channels, 1, 1, 1)
        self.padding = kernel_size // 2

    def linear_forward(self, x):
        return F.conv2d(F.pad(x, [self.padding] * 4, mode="reflect"),
                        self.kernel.to(x.dtype), groups=x.shape[1])


class Inpainting(Operator):
    def __init__(self, resolution=256, mask_len_range=(105, 180), channels=3, **kwargs):
        super().__init__(**kwargs)
        self.shape = (channels, resolution, resolution)
        lo, hi = mask_len_range
        if not 0 < lo <= hi < resolution:
            raise ValueError("Mask sizes must be positive and smaller than the image.")
        height, width = [np.random.randint(lo, hi) if lo < hi else lo for _ in range(2)]
        margin = min(resolution // 8, (resolution - max(height, width) - 1) // 2)
        top = np.random.randint(margin, resolution - margin - height)
        left = np.random.randint(margin, resolution - margin - width)
        self.mask = torch.ones(1, 1, resolution, resolution, device=self.device)
        self.mask[..., top:top + height, left:left + width] = 0

    def linear_forward(self, x):
        return x * self.mask

    def adjoint(self, y):
        return y * self.mask

    def noisy_linear_vec(self, residual, damping, weights, **kwargs):
        return self.mask * residual / (self.mask + damping / weights.clamp_min(1e-6))


class PhaseRetrieval(Operator):
    linear = False

    def __init__(self, resolution=256, oversample=2.0, channels=3, **kwargs):
        super().__init__(**kwargs)
        self.shape = (channels, resolution, resolution)
        self.padding = int(oversample / 8 * resolution)

    def linear_forward(self, x):
        x = F.pad(x, [self.padding] * 4).to(torch.complex64)
        return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)),
                                               norm="ortho"), dim=(-2, -1)).abs()
