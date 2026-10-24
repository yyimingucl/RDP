# Portions of this file are adapted from InverseBench:
# https://github.com/devzhk/InverseBench
# Original source: algo/dps.py
#
# Copyright (c) 2025 Hongkai Zheng, Wenda Chu, Bingliang Zhang, Zihui Wu,
# Austin Wang, Berthy T. Feng, Caifeng Zou, Yu Sun, Nikola Kovachki,
# Zachary E. Ross, Katherine L. Bouman, and Yisong Yue.
# SPDX-License-Identifier: MIT
# See LICENSE for the full license text.
# Modified for the RDP implementation.

"""DPS in the InverseBench/EDM parameterization."""

import numpy as np
import torch
from tqdm.auto import trange

from ..schedule import Schedule


class DPS:
    def __init__(self, net, operator, num_steps=1000, guidance_scale=10.0,
                 sde=True, schedule="vp", sigma_max=100.0, sigma_min=0.01):
        self.net, self.operator = net, operator
        self.guidance_scale, self.sde = guidance_scale, sde
        self.schedule = Schedule(num_steps, schedule, sigma_max, sigma_min)

    def _likelihood(self, denoised, observation, sigma):
        return self.operator.gradient(denoised, observation, return_loss=True)

    def inference(self, observation, num_samples=1, save_every=0, progress=True):
        if num_samples < 1:
            raise ValueError("num_samples must be positive.")
        observation = observation.expand(num_samples, *observation.shape[1:])
        x = torch.randn(num_samples, self.net.img_channels, self.net.img_resolution,
                        self.net.img_resolution, device=self.operator.device) * self.schedule.sigma_max
        self.trajectory = []
        for step in trange(self.schedule.num_steps, desc=type(self).__name__, disable=not progress):
            x = x.detach().requires_grad_(True)
            sigma = self.schedule.sigma_steps[step]
            scaling = self.schedule.scaling_steps[step]
            factor = self.schedule.factor_steps[step]
            denoised = self.net(x / scaling, torch.as_tensor(sigma, device=x.device))
            gradient, loss = self._likelihood(denoised, observation, sigma)
            likelihood = torch.autograd.grad(denoised, x, gradient)[0]
            likelihood = likelihood * 0.5 / loss.clamp_min(1e-24).sqrt()
            score = (denoised - x / scaling) / sigma ** 2 / scaling
            with torch.no_grad():
                x = x * self.schedule.scaling_factor[step] + factor * score * (1 if self.sde else 0.5)
                if self.sde:
                    x += np.sqrt(factor) * torch.randn_like(x)
                x -= self.guidance_scale * likelihood
            if save_every and (step % save_every == 0 or step == self.schedule.num_steps - 1):
                self.trajectory.append(x.cpu())
        return x.detach()
