# Portions of this file are adapted from InverseBench:
# https://github.com/devzhk/InverseBench
# Original source: inverse_problems/base.py (operator-gradient interface)
#
# Copyright (c) 2025 Hongkai Zheng, Wenda Chu, Bingliang Zhang, Zihui Wu,
# Austin Wang, Berthy T. Feng, Caifeng Zou, Yu Sun, Nikola Kovachki,
# Zachary E. Ross, Katherine L. Bouman, and Yisong Yue.
# SPDX-License-Identifier: MIT
# See LICENSE for the full license text.
# Modified for the RDP implementation.
# Robust weighting and guidance extensions are contributed by the RDP authors.

"""Shared robust weights and data-consistency gradients."""

from dataclasses import dataclass

import torch


@dataclass
class Guidance:
    kind: str = "none"
    quantile: float = 0.9
    power: int = 1
    per_sample: bool = False

    def __post_init__(self):
        self.kind = self.kind.lower()
        if self.kind not in {"none", "imq", "huber"}:
            raise ValueError("Guidance must be none, imq, or huber.")
        if not 0 < self.quantile <= 1:
            raise ValueError("quantile must be in (0, 1].")
        if self.power not in {1, 2}:
            raise ValueError("Weight power must be 1 or 2.")

    def weights(self, prediction, observation, mask=None):
        residual = (prediction - observation).abs()
        squared = residual.square()
        valid = torch.ones_like(squared, dtype=torch.bool)
        if mask is not None:
            valid = mask.to(device=squared.device, dtype=torch.bool).expand_as(squared)
        result = torch.zeros_like(squared, dtype=torch.float64)
        if self.kind == "none":
            return valid.to(torch.float64)

        groups = range(len(squared)) if self.per_sample else [slice(None)]
        for group in groups:
            selected = squared[group][valid[group]]
            if selected.numel() == 0:
                continue
            if self.kind == "huber":
                magnitude = residual[group][valid[group]]
                threshold = torch.quantile(magnitude.detach(), self.quantile).clamp_min(1e-6)
                weights = (threshold / magnitude.clamp_min(1e-12)).clamp_max(1)
            else:
                threshold = torch.quantile(selected.detach().sqrt(), self.quantile).clamp_min(1e-12)
                weights = threshold / (selected + threshold.square()).sqrt()
            weights = weights.to(torch.float64)
            weights = weights * weights.numel() / weights.sum().clamp_min(1e-12)
            result[group][valid[group]] = weights
        return result

    def loss(self, prediction, observation, mask=None):
        weights = self.weights(prediction, observation, mask)
        return (weights.pow(self.power) * (prediction - observation).abs().square().to(torch.float64)).sum()

    def gradient(self, operator, prediction, observation, return_loss=False):
        with torch.enable_grad():
            probe = prediction.detach().requires_grad_(True)
            loss = self.loss(operator.forward(probe), observation, getattr(operator, "mask", None))
            gradient = torch.autograd.grad(loss, probe)[0]
        return (gradient, loss.detach()) if return_loss else gradient
