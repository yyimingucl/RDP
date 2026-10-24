# Portions of this file are adapted from InverseBench:
# https://github.com/devzhk/InverseBench
# Original source: utils/scheduler.py
#
# Copyright (c) 2025 Hongkai Zheng, Wenda Chu, Bingliang Zhang, Zihui Wu,
# Austin Wang, Berthy T. Feng, Caifeng Zou, Yu Sun, Nikola Kovachki,
# Zachary E. Ross, Katherine L. Bouman, and Yisong Yue.
# SPDX-License-Identifier: MIT
# See LICENSE for the full license text.
# Modified for the RDP implementation.

"""VP and VE discretizations used by the three released samplers."""

import numpy as np


class Schedule:
    def __init__(self, num_steps=1000, schedule="vp", sigma_max=100.0, sigma_min=0.01):
        if num_steps < 2 or not 0 < sigma_min < sigma_max:
            raise ValueError("Use at least two steps and 0 < sigma_min < sigma_max.")
        self.num_steps = num_steps
        if schedule == "vp":
            def sigma(t):
                return np.sqrt(np.exp(19.9 * t ** 2 / 2 + 0.1 * t) - 1)

            def derivative(t):
                return (19.9 * t + 0.1) * np.exp(19.9 * t ** 2 / 2 + 0.1 * t) / (2 * sigma(t))

            def scaling(t):
                return np.exp(-(19.9 * t ** 2 / 2 + 0.1 * t) / 2)

            end = np.sqrt(0.1 ** 2 + 2 * 19.9 * np.log1p(sigma_min ** 2)) / 19.9 - 0.1 / 19.9
            times = np.append(np.linspace(1, 1e-3, num_steps), end)
            scales = scaling(times)
            self.scaling_factor = 1 + (19.9 * times[:-1] + 0.1) / 2 * np.diff(-times)
            self.sigma_max = float(sigma(1) * scaling(1))
        elif schedule == "ve":
            sigma = np.sqrt
            derivative = lambda t: 0.5 / np.sqrt(t)
            times = np.append(np.geomspace(sigma_max ** 2, sigma_min ** 2, num_steps), sigma_min ** 2)
            scales = np.ones_like(times)
            self.scaling_factor = np.ones(num_steps)
            self.sigma_max = sigma_max
        else:
            raise ValueError("Schedule must be vp or ve.")
        self.sigma_steps = sigma(times)
        self.scaling_steps = scales
        self.factor_steps = np.maximum(0, 2 * scales[:-1] ** 2 * sigma(times[:-1]) *
                                       derivative(times[:-1]) * np.diff(-times))
