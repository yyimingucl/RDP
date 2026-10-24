# Portions of this file are adapted from InverseBench:
# https://github.com/devzhk/InverseBench
# Original source: algo/pigdm.py
#
# Copyright (c) 2025 Hongkai Zheng, Wenda Chu, Bingliang Zhang, Zihui Wu,
# Austin Wang, Berthy T. Feng, Caifeng Zou, Yu Sun, Nikola Kovachki,
# Zachary E. Ross, Katherine L. Bouman, and Yisong Yue.
# SPDX-License-Identifier: MIT
# See LICENSE for the full license text and NOTICE.md for further attribution.
# Modified for the RDP implementation.

"""Noisy PiGDM with matrix-free robust linear guidance."""

import numpy as np
import torch
from tqdm.auto import trange

from ..schedule import Schedule


class PiGDM:
    def __init__(self, net, operator, num_steps=100, eta=0.2, sigma_max=100.0,
                 sigma_min=0.01, cg_iterations=50, cg_tolerance=1e-6):
        if not operator.linear:
            raise ValueError("PiGDM supports scattering, deblurring, and inpainting; use DPS/LGD for phase retrieval.")
        if not 0 <= eta <= 1:
            raise ValueError("eta must lie in [0, 1].")
        self.net, self.operator, self.eta = net, operator, eta
        self.schedule = Schedule(num_steps, "ve", sigma_max, sigma_min)
        self.cg_iterations, self.cg_tolerance = cg_iterations, cg_tolerance

    def inference(self, observation, num_samples=1, save_every=0, progress=True):
        if num_samples < 1:
            raise ValueError("num_samples must be positive.")
        observation = observation.expand(num_samples, *observation.shape[1:])
        x = torch.randn(num_samples, self.net.img_channels, self.net.img_resolution,
                        self.net.img_resolution, device=self.operator.device) * self.schedule.sigma_max
        sigma_y = max(self.operator.sigma_noise, 1e-4)
        self.trajectory = []
        for step in trange(self.schedule.num_steps, desc="PiGDM", disable=not progress):
            x = x.detach().requires_grad_(True)
            sigma, next_sigma = self.schedule.sigma_steps[step:step + 2]
            denoised = self.net(x, torch.as_tensor(sigma, device=x.device))
            physical = self.operator.unnormalize(denoised)
            with torch.no_grad():
                prediction = self.operator.linear_forward(physical)
                weights = self.operator.robust_weights(prediction, observation)
                vector = self.operator.noisy_linear_vec(
                    observation - prediction, sigma_y ** 2 * (sigma ** 2 + 1) / sigma ** 2,
                    weights, iterations=self.cg_iterations, tolerance=self.cg_tolerance,
                )
            gradient = torch.autograd.grad(physical, x, vector.to(physical.dtype))[0]
            with torch.no_grad():
                x = denoised + np.sqrt(1 - self.eta ** 2) * next_sigma / sigma * (x - denoised)
                x += self.eta * next_sigma * torch.randn_like(x) + gradient
            if save_every and (step % save_every == 0 or step == self.schedule.num_steps - 1):
                self.trajectory.append(x.cpu())
        return x.detach()
