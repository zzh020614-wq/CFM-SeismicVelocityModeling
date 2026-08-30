"""Stage 1 entry point: batch synthesis of depth-domain velocity models.

    python -m cfm.synth.generate --n-samples 20000 --seed-base 20260630 \\
        --out-dir data/stage1/train --shard-size 500 --fmt hdf5

Guarantees
----------
* **Class balance** -- classes are assigned round-robin over the global index,
  so counts differ by at most one and every shard is class-mixed.
* **Determinism** -- ``sample_seed = seed_base + global_index`` fixes every
  random parameter of a sample. Use disjoint ``--seed-base`` ranges for the
  training and validation splits; that is what makes the split leak-free.
"""
from __future__ import annotations

import argparse
import time
from collections import Counter

from ..shards import VelocityShardWriter, write_manifest
from .config import CLASS_TABLE, N_CLASSES, SynthCfg
from .generator import ModelGenerator


def build_assignment(n_samples: int, classes: list[int]) -> list[int]:
    """Round-robin class assignment: sample i gets ``classes[i % len(classes)]``."""
    m = len(classes)
    return [classes[i % m] for i in range(n_samples)]


def main():
    d = SynthCfg()
    ap = argparse.ArgumentParser(
        description="Stage 1: synthesise depth-domain velocity models")
    ap.add_argument("--n-samples", type=int, default=d.n_samples)
    ap.add_argument("--out-dir", type=str, default=d.out_dir)
    ap.add_argument("--seed-base", type=int, default=d.seed_base)
    ap.add_argument("--shard-size", type=int, default=d.shard_size)
    ap.add_argument("--fmt", type=str, default=d.fmt, choices=["hdf5", "npz"])
    ap.add_argument("--classes", type=str, default=None,
                    help="comma-separated class subset, e.g. '0,3,4' (default: all)")
    ap.add_argument("--no-compress", action="store_true",
                    help="disable shard compression (larger files, faster writes)")
    args = ap.parse_args()

    classes = ([int(x) for x in args.classes.split(",")]
               if args.classes else list(range(N_CLASSES)))
    for c in classes:
        if c not in CLASS_TABLE:
            raise SystemExit(f"unknown class id: {c} (valid: {sorted(CLASS_TABLE)})")

    cfg = SynthCfg(n_samples=args.n_samples, out_dir=args.out_dir,
                   seed_base=args.seed_base, shard_size=args.shard_size,
                   fmt=args.fmt)
    gen = ModelGenerator(cfg)
    writer = VelocityShardWriter(args.out_dir, args.shard_size, args.fmt,
                                 compress=not args.no_compress)
    assign = build_assignment(args.n_samples, classes)

    print(f"[stage1] {args.n_samples} samples | classes {classes} | "
          f"fmt={args.fmt} shard={args.shard_size} -> {args.out_dir}")
    t0 = time.time()
    for i, cid in enumerate(assign):
        out = gen.generate(cid, args.seed_base + i)
        writer.add(out["v"], out["class_id"], out["seed"], gidx=i)
        if (i + 1) % max(1, args.n_samples // 20) == 0 or i + 1 == args.n_samples:
            dt = time.time() - t0
            print(f"  {i + 1}/{args.n_samples}  ({(i + 1) / dt if dt else 0:.1f} samples/s)")

    shards = writer.close()
    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stage": "stage1_synth",
        "n_samples": args.n_samples,
        "n_classes": N_CLASSES,
        "classes_used": classes,
        "class_names": {int(k): v.name for k, v in CLASS_TABLE.items()},
        "class_distribution": {int(k): int(n) for k, n in sorted(Counter(assign).items())},
        "seed_base": args.seed_base,
        "seed_formula": "sample_seed = seed_base + global_index",
        "fmt": args.fmt,
        "shard_size": args.shard_size,
        "compressed": not args.no_compress,
        "grid": cfg.grid.__dict__,
        "shards": shards,
        "config": cfg.to_dict(),
    }
    path = write_manifest(args.out_dir, manifest)

    print(f"[stage1] done: {len(shards)} shards in {time.time() - t0:.1f}s, "
          f"class distribution {manifest['class_distribution']}")
    print(f"         manifest: {path}")


if __name__ == "__main__":
    main()
