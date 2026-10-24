# Portions of this file are adapted from InverseBench:
# https://github.com/devzhk/InverseBench
# Original source: main.py; configs/pretrain/inv-scatter.yaml; configs/pretrain/ffhq256.yaml
#
# Copyright (c) 2025 Hongkai Zheng, Wenda Chu, Bingliang Zhang, Zihui Wu,
# Austin Wang, Berthy T. Feng, Caifeng Zou, Yu Sun, Nikola Kovachki,
# Zachary E. Ross, Katherine L. Bouman, and Yisong Yue.
# SPDX-License-Identifier: MIT
# See LICENSE for the full license text and NOTICE.md for further attribution.
# Modified for the RDP implementation.

"""Load the two released InverseBench state-dict checkpoints."""

from pathlib import Path

import torch

from .precond import EDMPrecond, VPPrecond


def load_model(path, kind, device="cpu"):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}. See README.md for download links.")
    if kind == "scattering":
        model = EDMPrecond(img_resolution=128, img_channels=1, model_type="DhariwalUNet",
                           model_channels=128, channel_mult=[1, 1, 1, 2, 2],
                           attn_resolutions=[16], num_blocks=1, dropout=0.0)
    elif kind == "ffhq":
        model = VPPrecond(img_resolution=256, img_channels=3, model_type="ADM",
                          model_channels=128, num_res_blocks=1, channel_mult=[1, 1, 2, 2, 4, 4],
                          attention_resolutions=[16], dropout=0.0, num_heads=4,
                          num_head_channels=64, use_scale_shift_norm=True, resblock_updown=True)
    else:
        raise ValueError("Model kind must be scattering or ffhq.")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("ema", checkpoint.get("net", checkpoint))
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False)
