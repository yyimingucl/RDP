# Portions of this file are adapted from InverseBench:
# https://github.com/devzhk/InverseBench
# Original source: algo/lgd.py
#
# Copyright (c) 2025 Hongkai Zheng, Wenda Chu, Bingliang Zhang, Zihui Wu,
# Austin Wang, Berthy T. Feng, Caifeng Zou, Yu Sun, Nikola Kovachki,
# Zachary E. Ross, Katherine L. Bouman, and Yisong Yue.
# SPDX-License-Identifier: MIT
# See LICENSE for the full license text and NOTICE.md for further attribution.
# Modified for the RDP implementation.

"""LGD shares the DPS reverse process and averages Monte Carlo guidance."""

import numpy as np
import torch

from .dps import DPS


class LGD(DPS):
    def __init__(self, net, operator, mc_samples=3, **kwargs):
        super().__init__(net, operator, **kwargs)
        if mc_samples < 1:
            raise ValueError("mc_samples must be positive.")
        self.mc_samples = mc_samples

    def _likelihood(self, denoised, observation, sigma):
        batch = denoised.shape[0]
        samples = denoised[:, None] + torch.randn(
            batch, self.mc_samples, *denoised.shape[1:], device=denoised.device
        ) * (sigma / np.sqrt(1 + sigma ** 2))
        gradient, loss = self.operator.gradient(
            samples.flatten(0, 1), observation.repeat_interleave(self.mc_samples, dim=0),
            return_loss=True,
        )
        return gradient.reshape_as(samples).mean(dim=1), loss / self.mc_samples
