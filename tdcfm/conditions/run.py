"""Stage 3 entry point: derive the training conditions for every sample.

Reads the stage-1 depth-domain shards and the stage-2 ``time_axis.json``, and
writes ``target`` plus the four condition channels.

    python -m tdcfm.conditions.run --source-dir data/stage1/train \\
        --time-axis data/stage2/train/time_axis.json \\
        --out-dir data/stage3/train --shard-size 500

Every split must be given the *training* time axis; a split that recomputed its
own axis would sit on a different time scale.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter

import numpy as np

from ..shards import (
    ConditionShardWriter,
    load_manifest,
    read_velocity_shard,
    shard_paths,
    write_manifest,
)
from .config import COND_NAMES, CondCfg, ImagingCfg, RmsCfg
from .derive import derive_conditions, stack_cond


def main():
    d = CondCfg()
    ap = argparse.ArgumentParser(description="Stage 3: derive network conditions")
    ap.add_argument("--source-dir", required=True, help="stage 1 output directory")
    ap.add_argument("--time-axis", required=True,
                    help="time_axis.json produced by stage 2 for the training split")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shard-size", type=int, default=256)
    ap.add_argument("--fmt", type=str, default="hdf5", choices=["hdf5", "npz"])
    ap.add_argument("--no-compress", action="store_true")
    ap.add_argument("--rms-smooth-tau", type=float, default=d.rms.smooth_tau,
                    help="baseline RMS smoothing written to disk; the dataloader "
                         "can only add to it, never undo it")
    ap.add_argument("--rms-smooth-x", type=float, default=d.rms.smooth_x)
    args = ap.parse_args()

    cfg = CondCfg(
        rms=RmsCfg(smooth_tau=args.rms_smooth_tau, smooth_x=args.rms_smooth_x),
        imaging=ImagingCfg(),
    )

    src_man = load_manifest(args.source_dir)
    dz = float(src_man["grid"]["dz"])
    with open(args.time_axis) as f:
        ta = json.load(f)
    tau_axis = np.asarray(ta["tau_axis"], dtype=np.float64)
    n_t = len(tau_axis)
    paths = shard_paths(args.source_dir, src_man)
    total = src_man["n_samples"]
    print(f"[stage3] source {args.source_dir} | {len(paths)} shards | {total} samples | "
          f"dz={dz} | n_t={n_t} T_max={ta['T_max_s'] * 1000:.0f} ms")
    print(f"         baseline RMS smoothing sigma=({cfg.rms.smooth_tau}, "
          f"{cfg.rms.smooth_x}) | wells: {cfg.well.n_wells[0]}-{cfg.well.n_wells[1] - 1} "
          "single columns (rebuilt by the dataloader at training time)")

    writer = ConditionShardWriter(args.out_dir, args.shard_size, args.fmt,
                                  compress=not args.no_compress,
                                  cond_names=COND_NAMES)
    assign, gidx = [], 0
    t0 = time.time()
    for p in paths:
        v, cid, sd = read_velocity_shard(p)
        for i in range(v.shape[0]):
            rng = np.random.default_rng(int(sd[i]))
            out = derive_conditions(v[i], dz, tau_axis, rng, cfg)
            writer.add(out["target"], stack_cond(out), int(cid[i]), int(sd[i]),
                       gidx=gidx)
            assign.append(int(cid[i]))
            gidx += 1
        print(f"  derived {gidx}/{total}  "
              f"({gidx / max(1e-9, time.time() - t0):.1f} samples/s)")
    shards = writer.close()

    write_manifest(args.out_dir, {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stage": "stage3_conditions",
        "source_dir": args.source_dir,
        "time_axis": args.time_axis,
        "n_samples": gidx,
        "class_distribution": {int(k): int(n) for k, n in sorted(Counter(assign).items())},
        "cond_names": list(COND_NAMES),
        "T_max_s": ta["T_max_s"], "n_t": n_t, "nx": int(src_man["grid"]["nx"]),
        "fmt": args.fmt, "shard_size": args.shard_size,
        "compressed": not args.no_compress,
        "shards": shards,
        "config": cfg.to_dict(),
    })
    print(f"[stage3] done: {len(shards)} shards -> {args.out_dir} "
          f"({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
