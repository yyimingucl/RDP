# Portions of this file are adapted from InverseBench:
# https://github.com/devzhk/InverseBench
# Original source: models/fp16_util.py
#
# Copyright (c) 2025 Hongkai Zheng, Wenda Chu, Bingliang Zhang, Zihui Wu,
# Austin Wang, Berthy T. Feng, Caifeng Zou, Yu Sun, Nikola Kovachki,
# Zachary E. Ross, Katherine L. Bouman, and Yisong Yue.
# SPDX-License-Identifier: MIT
# See LICENSE for the full license text.
# Modified for the RDP implementation.

import torch as th
import torch.nn as nn

def convert_module_to_f16(l):
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.half()
        if l.bias is not None:
            l.bias.data = l.bias.data.half()

def convert_module_to_f32(l):
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.float()
        if l.bias is not None:
            l.bias.data = l.bias.data.float()
