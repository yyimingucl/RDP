# Portions of this file are adapted from InverseBench:
# https://github.com/devzhk/InverseBench
# Original source: training/dataset.py
#
# Copyright (c) 2025 Hongkai Zheng, Wenda Chu, Bingliang Zhang, Zihui Wu,
# Austin Wang, Berthy T. Feng, Caifeng Zou, Yu Sun, Nikola Kovachki,
# Zachary E. Ross, Katherine L. Bouman, and Yisong Yue.
# SPDX-License-Identifier: MIT
# See LICENSE for the full license text.
# Modified for the RDP implementation.

"""Only image-folder and inverse-scattering LMDB readers are retained."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


def parse_ids(value):
    if value is None or isinstance(value, list):
        return value
    result = []
    for part in str(value).split(","):
        bounds = [int(x) for x in part.split("-")]
        result.extend(range(bounds[0], bounds[-1] + 1))
    if len(set(result)) != len(result) or any(i < 0 for i in result):
        raise ValueError("Sample IDs must be distinct and nonnegative.")
    return result


class TestData(Dataset):
    def __init__(self, root, kind, resolution, ids="0", mean=0.5, std=0.25):
        self.root, self.kind, self.resolution = Path(root), kind, resolution
        self.mean, self.std = mean, std
        if kind == "lmdb":
            import lmdb
            self.env = lmdb.open(str(self.root), readonly=True, lock=False, create=False)
            self.txn = self.env.begin(write=False)
            self.ids = parse_ids(ids)
            if self.ids is None:
                self.ids = list(range(self.txn.stat()["entries"]))
        elif kind == "images":
            if not self.root.is_dir():
                raise FileNotFoundError(f"Image directory does not exist: {self.root}")
            self.paths = sorted(p for p in self.root.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
            self.ids = parse_ids(ids)
            if self.ids is None:
                self.ids = list(range(len(self.paths)))
        else:
            raise ValueError("Data kind must be lmdb or images.")
        if not self.ids:
            raise ValueError(f"No test samples found in {self.root}.")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        sample_id = self.ids[index]
        if self.kind == "lmdb":
            value = self.txn.get(str(sample_id).encode())
            if value is None:
                raise KeyError(f"LMDB has no sample {sample_id}.")
            image = np.frombuffer(value, dtype=np.float32).copy().reshape(1, self.resolution, self.resolution)
            return torch.from_numpy((image - self.mean) / (2 * self.std))
        # Match InverseBench's explicit-ID convention; also support arbitrary image names.
        candidates = [self.root / f"{sample_id:05d}{suffix}" for suffix in (".png", ".jpg", ".jpeg")]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            path = self.paths[sample_id]
        with Image.open(path) as image:
            image = torch.from_numpy(np.array(image.convert("RGB")).copy()).permute(2, 0, 1).float() / 127.5 - 1
        return TF.resize(image, [self.resolution, self.resolution], antialias=True)

    def close(self):
        if self.kind == "lmdb":
            self.txn.abort()
            self.env.close()
