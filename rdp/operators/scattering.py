# Portions of this file are adapted from InverseBench:
# https://github.com/devzhk/InverseBench
# Original source: inverse_problems/inverse_scatter.py
#
# Copyright (c) 2025 Hongkai Zheng, Wenda Chu, Bingliang Zhang, Zihui Wu,
# Austin Wang, Berthy T. Feng, Caifeng Zou, Yu Sun, Nikola Kovachki,
# Zachary E. Ross, Katherine L. Bouman, and Yisong Yue.
# SPDX-License-Identifier: MIT
# See LICENSE for the full license text and NOTICE.md for further attribution.
# Modified for the RDP implementation.

"""Linear Born scattering model, adapted from InverseBench."""

import numpy as np
import torch
from scipy.special import hankel1

from .base import Operator


class InverseScattering(Operator):
    def __init__(self, Lx=0.18, Ly=0.18, Nx=128, Ny=128, wave=6,
                 numRec=360, numTrans=20, sensorRadius=1.6, **kwargs):
        super().__init__(shift=1.0, scale=0.5, **kwargs)
        self.shape = (1, Ny, Nx)
        self.dx, self.dy = Lx / Nx, Ly / Ny
        x, y = np.meshgrid(np.arange(-Nx / 2, Nx / 2) * self.dx,
                           np.arange(-Ny / 2, Ny / 2) * self.dy)
        wavenumber = 2 * np.pi / (self.dx * wave)

        def fields(count):
            angles = np.linspace(0, 359, count) * np.pi / 180
            distance = np.hypot(x[..., None] - sensorRadius * np.cos(angles),
                                y[..., None] - sensorRadius * np.sin(angles))
            return torch.as_tensor(0.25j * hankel1(0, wavenumber * distance), device=self.device)

        self.incident = fields(numTrans).reshape(-1, numTrans)
        self.sensors = (wavenumber ** 2 * fields(numRec)).reshape(-1, numRec)

    def forward(self, x):
        return self.linear_forward(self.unnormalize(x.to(torch.float64)))

    def linear_forward(self, x):
        source = x.to(torch.float64).flatten(1).to(self.incident.dtype)
        return self.dx * self.dy * torch.einsum("bp,pt,pr->btr", source, self.incident.conj(), self.sensors)

    def adjoint(self, y):
        field = torch.einsum("btr,pt,pr->bp", y.to(self.sensors.dtype),
                             self.incident, self.sensors.conj())
        return (self.dx * self.dy * field.real).reshape(-1, *self.shape)
