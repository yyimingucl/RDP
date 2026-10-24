# Portions of this file are adapted from InverseBench:
# https://github.com/devzhk/InverseBench
# Original source: main.py
#
# Copyright (c) 2025 Hongkai Zheng, Wenda Chu, Bingliang Zhang, Zihui Wu,
# Austin Wang, Berthy T. Feng, Caifeng Zou, Yu Sun, Nikola Kovachki,
# Zachary E. Ross, Katherine L. Bouman, and Yisong Yue.
# SPDX-License-Identifier: MIT
# See LICENSE for the full license text and NOTICE.md for further attribution.
# Modified for the RDP implementation.

"""Run one task with DPS, LGD, or PiGDM; see demo.ipynb for a comparison."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from rdp.data import TestData
from rdp.models import load_model
from rdp.operators.images import GaussianBlur, Inpainting, PhaseRetrieval
from rdp.operators.scattering import InverseScattering
from rdp.samplers.dps import DPS
from rdp.samplers.lgd import LGD
from rdp.samplers.pigdm import PiGDM


ROOT = Path(__file__).resolve().parent
SAMPLERS = {"dps": DPS, "lgd": LGD, "pigdm": PiGDM}
OPERATORS = {"scattering": InverseScattering, "deblur": GaussianBlur,
             "inpaint": Inpainting, "phase": PhaseRetrieval}


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(task):
    with (ROOT / "configs" / f"{task}.yaml").open() as handle:
        config = yaml.safe_load(handle)
    with (ROOT / "configs" / "samplers.yaml").open() as handle:
        config["samplers"] = yaml.safe_load(handle)
    return config


def make_operator(config, device):
    kwargs = copy.deepcopy(config["operator"])
    kind = kwargs.pop("kind")
    return OPERATORS[kind](**kwargs, noise=config["noise"], guidance=config["guidance"], device=device)


def image_space(tensor, config):
    return ((tensor + 1) / 2).clamp(0, 1)


def run(config, sampler="dps", seed=0, num_samples=1, output=None, device=None, save_every=0):
    from evaluate import image_metrics, write_metrics

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(seed)
    operator = make_operator(config, device)
    net = load_model(config["checkpoint"], config["model"], device)
    algorithm = SAMPLERS[sampler](net, operator, **config["samplers"][sampler])
    dataset = TestData(**config["data"])
    output = Path(output or ROOT / "results" / config["operator"]["kind"] / sampler /
                  config["noise"]["name"] / config["guidance"]["kind"])
    output.mkdir(parents=True, exist_ok=True)
    saved_config = dict(config, sampler=sampler, seed=seed, num_samples=num_samples)
    (output / "config.yaml").write_text(yaml.safe_dump(saved_config, sort_keys=False))
    rows = []
    try:
        for index, sample_id in enumerate(dataset.ids):
            target = dataset[index][None].to(device)
            seed_all(seed + sample_id)
            observation = operator.observe(target)
            seed_all(seed + sample_id)
            reconstruction = algorithm.inference(observation, num_samples, save_every=save_every)
            if not torch.isfinite(reconstruction).all():
                raise RuntimeError(f"Nonfinite reconstruction for sample {sample_id}.")
            result = {"target": image_space(target, config).cpu(),
                      "recon": image_space(reconstruction, config).cpu(),
                      "observation": observation.detach().cpu()}
            if save_every:
                result["trajectory"] = torch.stack([image_space(x, config) for x in algorithm.trajectory])
            torch.save(result, output / f"result_{sample_id}.pt")
            row = dict(sample_id=sample_id, **image_metrics(result["recon"].to(device),
                       result["target"].to(device), config["model"]))
            rows.append(row)
            print(json.dumps(row))
    finally:
        dataset.close()
    write_metrics(rows, output / "metrics.csv")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=OPERATORS, default="scattering")
    parser.add_argument("--sampler", choices=SAMPLERS, default="dps")
    parser.add_argument("--robust", choices=["none", "imq", "huber"], default="imq")
    parser.add_argument("--noise", choices=["gaussian", "student_t", "outlier"], default="outlier")
    parser.add_argument("--data", help="Test LMDB or FFHQ image directory")
    parser.add_argument("--checkpoint")
    parser.add_argument("--ids", default="0", help="e.g. 0-9,12")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--quantile", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--output")
    args = parser.parse_args()
    config = load_config(args.task)
    config["noise"]["name"] = args.noise
    config["guidance"]["kind"] = args.robust
    config["data"]["ids"] = args.ids
    config["data"]["root"] = args.data or str(ROOT / config["data"]["root"])
    config["checkpoint"] = args.checkpoint or str(ROOT / config["checkpoint"])
    if args.steps is not None:
        config["samplers"][args.sampler]["num_steps"] = args.steps
    if args.guidance_scale is not None:
        if args.sampler == "pigdm":
            parser.error("PiGDM uses eta; guidance-scale applies to DPS and LGD.")
        config["samplers"][args.sampler]["guidance_scale"] = args.guidance_scale
    if args.quantile is not None:
        config["guidance"]["quantile"] = args.quantile
    print(run(config, args.sampler, args.seed, args.num_samples, args.output, args.device, args.save_every))


if __name__ == "__main__":
    main()
