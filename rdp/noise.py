# Portions of this file are adapted from InverseBench:
# https://github.com/devzhk/InverseBench
# Original source: inverse_problems/base.py (measurement-noise utilities)
#
# Copyright (c) 2025 Hongkai Zheng, Wenda Chu, Bingliang Zhang, Zihui Wu,
# Austin Wang, Berthy T. Feng, Caifeng Zou, Yu Sun, Nikola Kovachki,
# Zachary E. Ross, Katherine L. Bouman, and Yisong Yue.
# SPDX-License-Identifier: MIT
# See LICENSE for the full license text and NOTICE.md for further attribution.
# Modified for the RDP implementation.

"""Measurement corruption used by the released experiments."""

import torch


def corrupt(signal, name="gaussian", sigma_noise=0.0, df=3.0,
            outlier_fraction=0.01, outlier_magnitude_factor=10.0):
    if name not in {"gaussian", "student_t", "outlier"}:
        raise ValueError("Noise must be gaussian, student_t, or outlier.")
    if sigma_noise < 0 or not 0 <= outlier_fraction <= 1 or outlier_magnitude_factor < 0:
        raise ValueError("Invalid noise scale or outlier fraction.")
    if name == "student_t" and df <= 2:
        raise ValueError("Student-t degrees of freedom must exceed 2.")

    def draw(component):
        if name == "student_t":
            distribution = torch.distributions.StudentT(df, loc=0.0, scale=sigma_noise)
            return distribution.sample(component.shape).to(component)
        return sigma_noise * torch.randn_like(component)

    if signal.is_complex():
        noisy = signal + torch.complex(draw(signal.real), draw(signal.imag))
    else:
        noisy = signal + draw(signal)
    if name != "outlier":
        return noisy
    if signal.is_complex():
        std = ((noisy.real.std(unbiased=False).square() + noisy.imag.std(unbiased=False).square()) / 2).sqrt()
        count = max(int(noisy.numel() * outlier_fraction), 2) // 2
        indices = torch.randperm(noisy.numel(), device=noisy.device)[:count]
        real = outlier_magnitude_factor * std * torch.randn(count, device=noisy.device, dtype=noisy.real.dtype)
        imag = outlier_magnitude_factor * std * torch.randn(count, device=noisy.device, dtype=noisy.real.dtype)
        noisy.real.view(-1)[indices] += real
        noisy.imag.view(-1)[indices] += imag
    else:
        count = max(int(noisy.numel() * outlier_fraction), 1)
        indices = torch.randperm(noisy.numel(), device=noisy.device)[:count]
        values = outlier_magnitude_factor * noisy.std(unbiased=False) * torch.randn(count, device=noisy.device, dtype=noisy.dtype)
        noisy.view(-1)[indices] += values
    return noisy
