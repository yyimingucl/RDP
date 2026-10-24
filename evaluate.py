# Portions of this file are adapted from InverseBench:
# https://github.com/devzhk/InverseBench
# Original source: eval.py
#
# Copyright (c) 2025 Hongkai Zheng, Wenda Chu, Bingliang Zhang, Zihui Wu,
# Austin Wang, Berthy T. Feng, Caifeng Zou, Yu Sun, Nikola Kovachki,
# Zachary E. Ross, Katherine L. Bouman, and Yisong Yue.
# SPDX-License-Identifier: MIT
# See LICENSE for the full license text and NOTICE.md for further attribution.
# Modified for the RDP implementation.

"""Evaluate saved image-space result_*.pt tensors."""

import argparse
import csv
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch


@lru_cache(maxsize=4)
def perceptual_metrics(device):
    from piq import LPIPS, VIFLoss
    return (LPIPS(replace_pooling=True, reduction="none").to(device).eval(),
            VIFLoss(data_range=1, reduction="none").to(device).eval())


@torch.no_grad()
def image_metrics(prediction, target, kind="scattering"):
    from piq import psnr, ssim
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("Evaluation inputs must be finite.")
    if prediction.min() < -1e-4 or prediction.max() > 1 + 1e-4 or target.min() < -1e-4 or target.max() > 1 + 1e-4:
        raise ValueError("Evaluation expects saved images in [0, 1].")
    prediction, target = prediction.clamp(0, 1), target.clamp(0, 1).expand_as(prediction)
    values = {"psnr": psnr(prediction, target, data_range=1, reduction="none").mean().item(),
              "ssim": ssim(prediction, target, data_range=1, reduction="none").mean().item()}
    if kind == "scattering":
        values["nmae"] = ((prediction - target).abs().mean() / target.abs().mean().clamp_min(1e-6)).item()
    elif kind == "ffhq":
        lpips, vif = perceptual_metrics(str(prediction.device))
        # piq.LPIPS expects [0, 1] and applies ImageNet normalization internally.
        values["lpips"] = lpips(prediction, target).mean().item()
        values["vifl"] = vif(prediction, target).mean().item()
    else:
        raise ValueError("Metric kind must be scattering or ffhq.")
    return values


def write_metrics(rows, path):
    if not rows:
        raise ValueError("No results to evaluate.")
    fields = list(rows[0])
    summaries = [{"sample_id": "mean"}, {"sample_id": "std"}]
    for field in fields:
        if field != "sample_id":
            values = [row[field] for row in rows]
            summaries[0][field] = np.mean(values)
            summaries[1][field] = np.std(values, ddof=1) if len(values) > 1 else ""
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows + summaries)


@torch.no_grad()
def source_frechet(paths, device="cpu", batch_size=16, ridge=500.0):
    """Preserve source FID.py's Inception logits, preprocessing, and ridge.

    This is a source-compatible Frechet score, not standard pool3 FID.
    """
    from torchvision.models import inception_v3, Inception_V3_Weights
    from torchvision.transforms import Compose, Resize, CenterCrop, Normalize

    if len(paths) < 2:
        raise ValueError("Frechet evaluation requires at least two images.")
    model = inception_v3(weights=Inception_V3_Weights.DEFAULT).to(device).eval()
    transform = Compose([Resize(256), CenterCrop(224),
                         Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    features = {"recon": [], "target": []}
    for start in range(0, len(paths), batch_size):
        results = [torch.load(path, map_location="cpu", weights_only=True) for path in paths[start:start + batch_size]]
        for key in features:
            batch = torch.stack([transform(result[key][0].clamp(0, 1)) for result in results]).to(device)
            features[key].append(model(batch).cpu().numpy().astype(np.float64))
    moments = []
    for key in features:
        data = np.concatenate(features[key])
        covariance = np.cov(data, rowvar=False)
        moments.append((data.mean(0), (covariance + covariance.T) / 2 + ridge * np.eye(data.shape[1])))
    (mean_a, cov_a), (mean_b, cov_b) = moments
    eigenvalues, vectors = np.linalg.eigh(cov_a)
    root = (vectors * np.sqrt(eigenvalues.clip(0))) @ vectors.T
    product = root @ cov_b @ root
    trace_root = np.sqrt(np.linalg.eigvalsh((product + product.T) / 2).clip(0)).sum()
    return max(float(np.square(mean_a - mean_b).sum() + np.trace(cov_a + cov_b) - 2 * trace_root), 0.0)


def main():
    import json
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--source-frechet", action="store_true", help="Also compute the original FID.py convention (see README)")
    args = parser.parse_args()
    config = yaml.safe_load((args.root / "config.yaml").read_text())
    paths = sorted(args.root.glob("result_*.pt"))
    rows = []
    for path in paths:
        result = torch.load(path, map_location=args.device, weights_only=True)
        rows.append(dict(sample_id=path.stem.removeprefix("result_"),
                         **image_metrics(result["recon"], result["target"], config["model"])))
    write_metrics(rows, args.root / "metrics.csv")
    if args.source_frechet:
        value = source_frechet(paths, args.device)
        (args.root / "source_frechet.json").write_text(json.dumps({"source_frechet": value, "ridge": 500.0}, indent=2))
        print("source_frechet:", value)
    print(args.root / "metrics.csv")


if __name__ == "__main__":
    main()
