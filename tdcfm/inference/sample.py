"""Conditional sampling: checkpoint -> generated time-domain interval velocities.

    python -m tdcfm.inference.sample --ckpt runs/train/otcfm/ckpt_step_79999.pt \\
        --data-dir data/stage3/val --n 8 --best-of 1 --out runs/samples

The conditioning setup is restored from the checkpoint, never from the command
line: evaluating a model under settings other than the ones it was trained with
produces plausible-looking, wrong numbers without raising an error.

Results are written as ``.npy`` files in physical units (m/s).
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..models.unet import build_unet, euler_sample
from ..training.config import cfg_from_ckpt
from ..training.dataset import ShardConditionDataset
from ..training.normalize import unit_to_vel


def load_net(ckpt_path: str, weights: str = "net", device=None):
    """Load a checkpoint and rebuild the network with its own configuration.

    ``weights='net'`` uses the raw training weights, ``'ema'`` the exponential
    moving average. The EMA lags badly before it has warmed up, so early
    checkpoints evaluated with ``ema`` understate the model.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = cfg_from_ckpt(ckpt["cfg"])
    net = build_unet(cfg).to(device)
    key = "ema_model" if weights == "ema" else "net_model"
    if key not in ckpt:
        raise KeyError(f"{ckpt_path} has no {key}; available keys: {list(ckpt.keys())}")
    net.load_state_dict(ckpt[key], strict=True)
    net.eval()
    return net, cfg, ckpt


def main():
    ap = argparse.ArgumentParser(description="Sample velocity models from a checkpoint")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True, help="stage 3 directory supplying conditions")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--best-of", type=int, default=1,
                    help="draws per condition, for assessing sample diversity")
    ap.add_argument("--ode-steps", type=int, default=100)
    ap.add_argument("--out", default="runs/samples")
    ap.add_argument("--weights", choices=["net", "ema"], default="net")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, cfg, ckpt = load_net(args.ckpt, args.weights, device)
    print(f"[sample] {args.ckpt} (step {ckpt.get('step')}), {args.weights} weights, "
          f"device {device}")

    ds = ShardConditionDataset.from_cfg(cfg, args.data_dir, random_flip=False, overfit=0)
    loader = DataLoader(ds, batch_size=args.n, shuffle=False)
    x1, cond, y = next(iter(loader))
    cond, y = cond.to(device), y.to(device)

    for b in range(args.best_of):
        gen = euler_sample(net, cond, y, args.ode_steps, cfg.image_size).cpu().numpy()
        vel = unit_to_vel(gen[:, 0], cfg.v_min, cfg.v_max)          # (n, H, W) m/s
        np.save(os.path.join(args.out, f"generated_velocity_{b}.npy"), vel)
        print(f"  draw {b}: {vel.shape}  [{vel.min():.0f}, {vel.max():.0f}] m/s")

    np.save(os.path.join(args.out, "true_velocity.npy"),
            unit_to_vel(x1.numpy()[:, 0], cfg.v_min, cfg.v_max))
    np.save(os.path.join(args.out, "class_id.npy"), y.cpu().numpy())
    print(f"[sample] written to {args.out}")


if __name__ == "__main__":
    main()
