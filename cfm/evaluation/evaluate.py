"""Stage 5 entry point: quantitative evaluation of a trained checkpoint.

    python -m cfm.evaluation.evaluate \\
        --ckpt runs/train/otcfm/ckpt_step_79999.pt \\
        --data-dir data/stage3/val --time-axis data/stage2/train/time_axis.json \\
        --n 500 --best-of 4 --ode-steps 100 --batch-size 8 --out runs/eval

Reports, per geological class and overall:

* accuracy      MAE, RMSE, correlation against the true interval velocity
* consistency   Dix/RMS residual and the misfit at the well pixels
* diversity     best-of-N RMSE, i.e. how much the spread of draws buys

and writes ``report.json``, ``samples/*.png``, ``closed_loop.png`` and
``diversity.png``. The conditioning setup is restored from the checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..inference.sample import load_net
from ..models.unet import euler_sample
from ..training.dataset import ShardConditionDataset
from ..training.normalize import unit_to_vel
from .figures import save_closed_loop_figure, save_diversity_figure, save_sample_figure
from .metrics import (
    aggregate_by_class,
    corr,
    dix_rms_consistency,
    mae,
    rmse,
    ssim,
    well_consistency,
)

CLASS_NAMES = {0: "near_horizontal", 1: "dip_wedge", 2: "fold", 3: "fault", 4: "salt"}


def main():
    ap = argparse.ArgumentParser(description="Stage 5: evaluate a checkpoint")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True, help="stage 3 validation directory")
    ap.add_argument("--time-axis", required=True,
                    help="training-split time_axis.json (never the validation one)")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--best-of", type=int, default=4)
    ap.add_argument("--ode-steps", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="runs/eval")
    ap.add_argument("--weights", choices=["net", "ema"], default="net")
    ap.add_argument("--max-figures", type=int, default=20,
                    help="write at most this many per-sample figures")
    args = ap.parse_args()

    samples_dir = os.path.join(args.out, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, cfg, ckpt = load_net(args.ckpt, args.weights, device)
    print(f"[eval] {args.ckpt} (step {ckpt.get('step')}), {args.weights} weights, "
          f"device {device}")

    with open(args.time_axis) as f:
        ta = json.load(f)
    tau_axis = np.asarray(ta["tau_axis"], np.float64)
    dz = float(ta["dz"])
    data_range = cfg.v_max - cfg.v_min

    ds = ShardConditionDataset.from_cfg(cfg, args.data_dir, random_flip=False, overfit=0)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    records: list[dict] = []
    example_true, example_gen = [], []
    done = 0
    for x1, cond, y in loader:
        if done >= args.n:
            break
        cond_np = cond.numpy()
        cond, y = cond.to(device), y.to(device)
        true_v = unit_to_vel(x1.numpy()[:, 0], cfg.v_min, cfg.v_max)     # (B, H, W)
        mask = cond_np[:, 2] > 0.5

        gens = np.stack([
            unit_to_vel(
                euler_sample(net, cond, y, args.ode_steps,
                             cfg.image_size).cpu().numpy()[:, 0],
                cfg.v_min, cfg.v_max)
            for _ in range(args.best_of)
        ])                                                                # (bo, B, H, W)

        for i in range(true_v.shape[0]):
            if done >= args.n:
                break
            tv = true_v[i]
            gv = gens[0, i]                     # the first draw: the honest single sample
            bo_rmse = [rmse(gens[b, i], tv) for b in range(args.best_of)]
            class_id = int(y[i].item())
            record = {
                "class_id": class_id,
                "mae": mae(gv, tv), "rmse": rmse(gv, tv), "corr": corr(gv, tv),
                "ssim": ssim(gv, tv, data_range=data_range),
                "rmse_best_of_n": float(min(bo_rmse)),
                "dix": dix_rms_consistency(gv, tv),
                "well": well_consistency(gv, tv, mask[i]),
            }
            records.append(record)
            if done < args.max_figures:
                save_sample_figure(
                    true_v=tv, generated_v=gv, cond=cond_np[i], class_id=class_id,
                    class_name=CLASS_NAMES[class_id], sample_index=done,
                    out_path=os.path.join(samples_dir,
                                          f"sample_{done:05d}_cls{class_id}.png"),
                    metrics=record, v_min=cfg.v_min, v_max=cfg.v_max)
            if len(example_true) < 4:
                example_true.append(tv)
                example_gen.append(gv)
            done += 1

    if not records:
        raise SystemExit("no samples evaluated; check --data-dir and --n")
    print(f"[eval] evaluated {len(records)} samples")

    report = aggregate_by_class(records, CLASS_NAMES)
    report["n_eval"] = len(records)
    report["best_of_n"] = args.best_of
    report["ode_steps"] = args.ode_steps
    report["checkpoint"] = args.ckpt
    report["weights"] = args.weights
    report["config"] = cfg.to_dict()
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    o = report["overall"]
    print(f"  overall: MAE {o['mae']:.1f}  RMSE {o['rmse']:.1f}  corr {o['corr']:.4f}  "
          f"SSIM {o['ssim']:.4f}  Dix {o['dix']:.1f}  well {o['well']:.1f}  "
          f"RMSE(best of {args.best_of}) {o['rmse_best_of_n']:.1f}")
    for c, d in report["per_class"].items():
        print(f"  class {c} {d['name']:16s} n={d['n']:4d}  MAE {d['mae']:.1f}  "
              f"RMSE {d['rmse']:.1f}  corr {d['corr']:.4f}  SSIM {d['ssim']:.4f}  "
              f"Dix {d['dix']:.1f}  well {d['well']:.1f}")

    save_closed_loop_figure(example_true, example_gen, tau_axis, dz, cfg.image_size,
                            os.path.join(args.out, "closed_loop.png"),
                            cfg.v_min, cfg.v_max)
    _diversity_figure(net, cfg, ds, args.out, args.ode_steps, device)
    print(f"[eval] per-sample figures -> {samples_dir}")
    print(f"[eval] done -> {args.out}")


@torch.no_grad()
def _diversity_figure(net, cfg, ds, out, ode_steps, device, n_draws=6):
    _, cond, y = ds[0]
    cond = cond.unsqueeze(0).to(device)
    y = y.view(1).to(device)
    draws = [unit_to_vel(
        euler_sample(net, cond, y, ode_steps, cfg.image_size).cpu().numpy()[0, 0],
        cfg.v_min, cfg.v_max) for _ in range(n_draws)]
    save_diversity_figure(np.stack(draws), os.path.join(out, "diversity.png"),
                          class_id=int(y.item()), v_min=cfg.v_min, v_max=cfg.v_max)


if __name__ == "__main__":
    main()
