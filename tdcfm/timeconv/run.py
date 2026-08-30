"""Stage 2 entry point: batch depth-to-time conversion.

Two passes: (1) scan the whole set for the global ``T_max``, (2) convert every
sample onto the resulting common time axis.

    # training split -- derives the axis and writes time_axis.json
    python -m tdcfm.timeconv.run --source-dir data/stage1/train \\
        --out-dir data/stage2/train --n-t 256 --shard-size 500

    # validation split -- MUST reuse the training axis, never recompute it
    python -m tdcfm.timeconv.run --source-dir data/stage1/val \\
        --out-dir data/stage2/val --time-axis data/stage2/train/time_axis.json

Outputs
-------
``shard_*.{h5,npz}``  time-domain ``V_int (N, n_t, nx)`` mirroring stage 1
``time_axis.json``    ``T_max`` / ``n_t`` / ``dz`` / ``tau_axis[n_t]`` -- the
                      reference file every later stage and evaluation script
                      must be pointed at
``manifest.json``     shard list, class distribution, provenance
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter

import numpy as np

from ..shards import (
    VelocityShardWriter,
    load_manifest,
    read_velocity_shard,
    shard_paths,
    write_manifest,
)
from .convert import depth_to_time, make_tau_axis

DEFAULT_N_T = 256
DEFAULT_TMAX_PERCENTILE = 90.0


def _batched_tau_max(v: np.ndarray, dz: float) -> np.ndarray:
    """Total two-way time per trace for a batch ``(N, nz, nx)`` -> ``(N, nx)``."""
    return 2.0 * np.sum(dz / v, axis=1)


def main():
    ap = argparse.ArgumentParser(description="Stage 2: depth-to-time conversion")
    ap.add_argument("--source-dir", required=True, help="stage 1 output directory")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-t", type=int, default=DEFAULT_N_T,
                    help="samples on the common time axis (ignored with --time-axis)")
    ap.add_argument("--tmax-percentile", type=float, default=DEFAULT_TMAX_PERCENTILE,
                    help="percentile of the pooled per-trace tau_max that sets T_max")
    ap.add_argument("--shard-size", type=int, default=256)
    ap.add_argument("--fmt", type=str, default="hdf5", choices=["hdf5", "npz"])
    ap.add_argument("--no-compress", action="store_true")
    ap.add_argument("--time-axis", type=str, default=None,
                    help="reuse an existing time_axis.json instead of scanning; "
                         "required for every split other than the one that defined it")
    args = ap.parse_args()

    src_man = load_manifest(args.source_dir)
    dz = float(src_man["grid"]["dz"])
    paths = shard_paths(args.source_dir, src_man)
    print(f"[stage2] source {args.source_dir} | {len(paths)} shards | "
          f"{src_man['n_samples']} samples | dz={dz}")

    t0 = time.time()
    if args.time_axis:
        with open(args.time_axis) as f:
            ref = json.load(f)
        t_max = float(ref["T_max_s"])
        tau_axis = np.asarray(ref["tau_axis"], dtype=np.float64)
        if args.n_t != DEFAULT_N_T and args.n_t != len(tau_axis):
            print(f"  note: --n-t {args.n_t} ignored; the reused axis has "
                  f"n_t={len(tau_axis)}")
        n_t = len(tau_axis)
        percentile = float(ref.get("tmax_percentile", args.tmax_percentile))
        tau_max_all = np.concatenate(
            [_batched_tau_max(read_velocity_shard(p)[0], dz).ravel() for p in paths])
        frac_trunc = float((tau_max_all > t_max).mean())
        print(f"[stage2] reusing {args.time_axis}: T_max={t_max * 1000:.1f} ms  "
              f"n_t={n_t}  traces truncated at depth: {frac_trunc * 100:.1f}%")
    else:
        # Pass 1: pool tau_max over every trace of every model.
        tau_max_all = np.concatenate(
            [_batched_tau_max(read_velocity_shard(p)[0], dz).ravel() for p in paths])
        t_max = float(np.percentile(tau_max_all, args.tmax_percentile))
        n_t = args.n_t
        percentile = args.tmax_percentile
        tau_axis = make_tau_axis(t_max, n_t)
        frac_trunc = float((tau_max_all > t_max).mean())
        print(f"[stage2] T_max({args.tmax_percentile:.0f}th pct)={t_max * 1000:.1f} ms  "
              f"dtau={t_max / (n_t - 1) * 1000:.2f} ms  "
              f"traces truncated at depth: {frac_trunc * 100:.1f}%  "
              f"({time.time() - t0:.1f}s)")
    d_tau = t_max / (n_t - 1)

    # Pass 2: convert and write.
    writer = VelocityShardWriter(args.out_dir, args.shard_size, args.fmt,
                                 compress=not args.no_compress)
    assign, gidx = [], 0
    for p in paths:
        v, c, s = read_velocity_shard(p)
        for i in range(v.shape[0]):
            writer.add(depth_to_time(v[i], dz, tau_axis), int(c[i]), int(s[i]),
                       gidx=gidx)
            assign.append(int(c[i]))
            gidx += 1
        print(f"  converted {gidx}/{src_man['n_samples']}")
    shards = writer.close()

    with open(os.path.join(args.out_dir, "time_axis.json"), "w") as f:
        json.dump({
            "T_max_s": t_max, "n_t": n_t, "dz": dz,
            "tmax_percentile": percentile,
            "d_tau_s": d_tau, "unit": "s",
            "tau_axis": tau_axis.tolist(),
        }, f, indent=2)

    write_manifest(args.out_dir, {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stage": "stage2_timeconv",
        "source_dir": args.source_dir,
        "n_samples": gidx,
        "class_distribution": {int(k): int(n) for k, n in sorted(Counter(assign).items())},
        "T_max_s": t_max, "n_t": n_t, "d_tau_s": d_tau,
        "tmax_percentile": percentile,
        "frac_truncated": frac_trunc,
        "reused_time_axis": args.time_axis or "",
        "fmt": args.fmt, "shard_size": args.shard_size,
        "compressed": not args.no_compress,
        "grid": {"n_t": n_t, "nx": int(src_man["grid"]["nx"]), "dz": dz},
        "shards": shards,
    })

    print(f"[stage2] done: {len(shards)} shards -> {args.out_dir} "
          f"({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
