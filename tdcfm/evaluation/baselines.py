"""Baselines the generative model has to beat.

    python -m tdcfm.evaluation.baselines --data-dir data/stage3/val \\
        --time-axis data/stage2/train/time_axis.json \\
        --ckpt runs/train/otcfm/ckpt_step_79999.pt --n 64 --model-mae 120

Two groups, deliberately separated:

**A. The conditions used unchanged.** The RMS field and the well section exactly
as the network receives them. Under weak conditioning these are poor, and they
exist only as a lower reference; beating them proves nothing.

**B. Conditions after the conventional processing a practitioner would apply.**
Dix inversion of the picked RMS field, lateral interpolation between wells,
their average, and the same with the un-degraded baseline RMS smoothing. These
are the *fair strong baselines*: this is how the problem is solved today, and
only an improvement over the best of them is a real contribution.

The conditioning setup is read from the checkpoint, so the baselines are
computed on exactly the inputs the model was trained with.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from ..conditions.dix import dix_inverse
from ..conditions.wells import interp_well_horizontal
from ..training.config import TrainCfg, apply_overrides, cfg_from_ckpt
from ..training.dataset import ShardConditionDataset
from ..training.normalize import unit_to_vel
from .metrics import corr, dix_rms_consistency, mae, rmse, ssim, well_consistency


def main():
    ap = argparse.ArgumentParser(description="Stage 5: trivial and conventional baselines")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--time-axis", required=True)
    ap.add_argument("--ckpt", default="",
                    help="read the conditioning setup from this checkpoint (recommended)")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model-mae", type=float, default=None,
                    help="the model's MAE, to print the ratio against the best baseline")
    ap.add_argument("overrides", nargs="*",
                    help="key=value settings, used only when --ckpt is not given")
    args = ap.parse_args()

    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        cfg = cfg_from_ckpt(ck.get("cfg", {}))
        print(f"conditioning taken from {args.ckpt} (step {ck.get('step')})")
    else:
        cfg = apply_overrides(TrainCfg(), args.overrides)
    print(f"  rms_sigma=({cfg.rms_smooth_tau}, {cfg.rms_smooth_x})  "
          f"n_wells={cfg.n_wells}  well_width={cfg.well_width}  "
          f"well_interp={cfg.well_interp}")

    with open(args.time_axis) as f:
        tau_axis = np.asarray(json.load(f)["tau_axis"], np.float64)
    lo, hi = cfg.v_min, cfg.v_max
    data_range = hi - lo

    # Two views of the same samples: the training conditioning, and the
    # stage-3 baseline smoothing (the RMS field that was never degraded).
    ds = ShardConditionDataset.from_cfg(cfg, args.data_dir, random_flip=False, overfit=0)
    cfg_std = TrainCfg(**{**cfg.to_dict(),
                          "rms_smooth_tau": ds.base_tau,
                          "rms_smooth_x": ds.base_x,
                          "well_interp": False})
    ds_std = ShardConditionDataset.from_cfg(cfg_std, args.data_dir,
                                            random_flip=False, overfit=0)

    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(len(ds), size=min(args.n, len(ds)), replace=False)
    print(f"  {len(ds)} samples available, evaluating {len(idxs)}\n")

    acc: dict[str, list] = {}
    for gi in idxs:
        x1, c, _ = ds[int(gi)]
        _, c_std, _ = ds_std[int(gi)]
        c, c_std = c.numpy(), c_std.numpy()
        tv = unit_to_vel(x1.numpy()[0], lo, hi)
        mask = c[2] > 0.5

        rms_fed = unit_to_vel(c[0], lo, hi)
        rms_std = unit_to_vel(c_std[0], lo, hi)
        well_fed = unit_to_vel(c[1], lo, hi)
        # Interpolation is applied to the raw wells, on physical velocities.
        well_raw = unit_to_vel(c_std[1], lo, hi)
        well_itp = interp_well_horizontal(well_raw,
                                          (c_std[2] > 0.5).astype(np.float32))

        dix_fed = np.clip(dix_inverse(rms_fed, tau_axis, lo, hi), lo, hi)
        dix_std = np.clip(dix_inverse(rms_std, tau_axis, lo, hi), lo, hi)

        candidates = {
            "A  RMS field as fed": rms_fed,
            "A  well section as fed": well_fed,
            "B  Dix inversion (RMS as fed)": dix_fed,
            "B  lateral well interpolation": well_itp,
            "B  mean of Dix and wells": 0.5 * (dix_fed + well_itp),
            "B  Dix inversion (baseline RMS)": dix_std,
            "B  mean of baseline Dix and wells": 0.5 * (dix_std + well_itp),
        }
        for k, v in candidates.items():
            acc.setdefault(k, []).append(
                (mae(v, tv), rmse(v, tv), ssim(v, tv, data_range=data_range),
                 corr(v, tv), well_consistency(v, tv, mask),
                 dix_rms_consistency(v, tv)))

    header = (f"{'baseline':<36}{'MAE':>10}{'RMSE':>10}{'SSIM':>9}"
              f"{'corr':>9}{'well':>10}{'Dix':>10}")
    print(header)
    print("-" * len(header))
    best_name, best_mae = None, np.inf
    for k, v in acc.items():
        m = np.nanmean(np.asarray(v), axis=0)
        print(f"{k:<36}{m[0]:>10.1f}{m[1]:>10.1f}{m[2]:>9.4f}{m[3]:>9.4f}"
              f"{m[4]:>10.1f}{m[5]:>10.1f}")
        if m[0] < best_mae:
            best_name, best_mae = k, m[0]

    print(f"\nstrongest baseline: {best_name}   MAE = {best_mae:.1f} m/s")
    if args.model_mae is not None:
        r = best_mae / max(args.model_mae, 1e-9)
        print(f"model MAE = {args.model_mae:.1f} m/s  ->  {r:.2f}x better than the "
              "strongest baseline")
    else:
        print("pass --model-mae <value> to print the ratio directly.")


if __name__ == "__main__":
    main()
