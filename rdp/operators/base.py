# Portions of this file are adapted from InverseBench:
# https://github.com/devzhk/InverseBench
# Original source: inverse_problems/base.py
#
# Copyright (c) 2025 Hongkai Zheng, Wenda Chu, Bingliang Zhang, Zihui Wu,
# Austin Wang, Berthy T. Feng, Caifeng Zou, Yu Sun, Nikola Kovachki,
# Zachary E. Ross, Katherine L. Bouman, and Yisong Yue.
# SPDX-License-Identifier: MIT
# See LICENSE for the full license text.
# Modified for the RDP implementation.

"""Forward operators share guidance and a matrix-free PiGDM update."""

import torch

from ..guidance import Guidance
from ..noise import corrupt


class Operator:
    linear = True
    mask = None

    def __init__(self, noise=None, guidance=None, device="cpu", shift=0.0, scale=1.0):
        self.device = torch.device(device)
        self.noise = noise or {}
        self.sigma_noise = self.noise.get("sigma_noise", 0.0)
        self.guidance = Guidance(**(guidance or {}))
        self.shift, self.scale = shift, scale

    def unnormalize(self, x):
        return (x + self.shift) * self.scale

    def forward(self, x):
        return self.linear_forward(self.unnormalize(x))

    def observe(self, x):
        observation = corrupt(self.forward(x), **self.noise)
        return observation if self.mask is None else observation * self.mask

    def gradient(self, prediction, observation, return_loss=False):
        return self.guidance.gradient(self, prediction, observation, return_loss)

    def robust_weights(self, prediction, observation):
        return self.guidance.weights(prediction, observation, self.mask)

    def adjoint(self, y):
        with torch.enable_grad():
            probe = torch.zeros(y.shape[0], *self.shape, device=y.device, dtype=torch.float32, requires_grad=True)
            product = (self.linear_forward(probe).conj() * y).real.sum()
            return torch.autograd.grad(product, probe)[0]

    def noisy_linear_vec(self, residual, damping, weights, iterations=50, tolerance=1e-6):
        """H^T (H H^T + damping W^-1)^-1 r, without a dense matrix/SVD."""
        if not self.linear:
            raise ValueError("PiGDM requires a linear operator in this release.")
        regularizer = damping / weights.clamp_min(1e-6)
        dims = tuple(range(1, residual.ndim))

        def inner(a, b):
            return (a.conj() * b).real.sum(dim=dims, keepdim=True)

        def matvec(value):
            return self.linear_forward(self.adjoint(value)) + regularizer * value

        x = torch.zeros_like(residual)
        r = residual.clone()
        p = r.clone()
        squared = inner(r, r)
        initial = squared.sqrt()
        tiny = torch.finfo(residual.real.dtype).tiny
        for _ in range(iterations):
            if torch.all(squared.sqrt() <= tolerance * initial.clamp_min(tiny)):
                break
            mapped = matvec(p)
            alpha = squared / inner(p, mapped).clamp_min(tiny)
            x = x + alpha * p
            r = r - alpha * mapped
            next_squared = inner(r, r)
            p = r + next_squared / squared.clamp_min(tiny) * p
            squared = next_squared
        return self.adjoint(x)
