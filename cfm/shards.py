"""Sharded dataset I/O shared by all pipeline stages.

Every stage writes one directory containing

    shard_00000.h5 ... shard_000NN.h5    (or .npz)
    manifest.json

The manifest records the shard list, the class distribution, the seeding rule
and the full stage configuration, so a stage output is self-describing and the
next stage never has to be told anything that is not in the directory itself.

Two schemas exist:

``VelocityShardWriter``   ``v / class_id / seed``
    Depth-domain models (stage 1) and time-domain models (stage 2) use the
    same schema, so stage 2 mirrors stage 1 shard for shard.

``ConditionShardWriter``  ``target / cond / class_id / seed``
    Stage 3: the time-domain ground truth plus the stacked condition channels.
"""
from __future__ import annotations

import json
import os
from collections import Counter

import numpy as np


def _extension(fmt: str) -> str:
    if fmt not in ("hdf5", "npz"):
        raise ValueError(f"unknown shard format: {fmt!r} (expected 'hdf5' or 'npz')")
    return "h5" if fmt == "hdf5" else "npz"


class _BaseShardWriter:
    """Buffers samples and flushes a shard once ``shard_size`` is reached."""

    def __init__(self, out_dir: str, shard_size: int, fmt: str = "hdf5",
                 compress: bool = True):
        self.ext = _extension(fmt)
        self.out_dir = out_dir
        self.shard_size = int(shard_size)
        self.fmt = fmt
        self.compress = compress
        os.makedirs(out_dir, exist_ok=True)

        self._class_id: list[int] = []
        self._seed: list[int] = []
        self._gidx: list[int] = []          # global sample index, for provenance
        self._shard_id = 0
        self.shards: list[dict] = []

    # -- subclass hooks ------------------------------------------------
    def _arrays(self) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def _clear_arrays(self) -> None:
        raise NotImplementedError

    def _n_buffered(self) -> int:
        raise NotImplementedError

    # -- public --------------------------------------------------------
    def close(self) -> list[dict]:
        if self._n_buffered():
            self._flush()
        return self.shards

    def _tick(self, class_id: int, seed: int, gidx: int) -> None:
        self._class_id.append(int(class_id))
        self._seed.append(int(seed))
        self._gidx.append(int(gidx))
        if self._n_buffered() >= self.shard_size:
            self._flush()

    def _flush(self) -> None:
        fname = f"shard_{self._shard_id:05d}.{self.ext}"
        fpath = os.path.join(self.out_dir, fname)

        arrays = dict(self._arrays())
        arrays["class_id"] = np.asarray(self._class_id, dtype=np.int8)
        arrays["seed"] = np.asarray(self._seed, dtype=np.int64)
        n = len(self._class_id)

        if self.fmt == "hdf5":
            import h5py

            kw = dict(compression="gzip", compression_opts=4) if self.compress else {}
            with h5py.File(fpath, "w") as f:
                for key, value in arrays.items():
                    # Small 1-D label arrays are not worth compressing.
                    f.create_dataset(key, data=value, **(kw if value.ndim > 1 else {}))
        else:
            saver = np.savez_compressed if self.compress else np.savez
            saver(fpath, **arrays)

        self.shards.append({
            "file": fname,
            "n": n,
            "global_index_range": [self._gidx[0], self._gidx[-1]],
            "class_dist": {int(k): int(v)
                           for k, v in sorted(Counter(self._class_id).items())},
        })

        self._shard_id += 1
        self._clear_arrays()
        self._class_id.clear()
        self._seed.clear()
        self._gidx.clear()


class VelocityShardWriter(_BaseShardWriter):
    """Shards of velocity models: ``v (N, H, W) float32``."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._v: list[np.ndarray] = []

    def add(self, v: np.ndarray, class_id: int, seed: int, gidx: int) -> None:
        self._v.append(np.asarray(v, dtype=np.float32))
        self._tick(class_id, seed, gidx)

    def _arrays(self):
        return {"v": np.stack(self._v)}

    def _clear_arrays(self):
        self._v.clear()

    def _n_buffered(self):
        return len(self._v)


class ConditionShardWriter(_BaseShardWriter):
    """Shards of training samples: ``target (N, H, W)`` and ``cond (N, C, H, W)``."""

    def __init__(self, *args, cond_names: tuple[str, ...] = (), **kwargs):
        super().__init__(*args, **kwargs)
        self.cond_names = tuple(cond_names)
        self._target: list[np.ndarray] = []
        self._cond: list[np.ndarray] = []

    def add(self, target: np.ndarray, cond: np.ndarray, class_id: int,
            seed: int, gidx: int) -> None:
        self._target.append(np.asarray(target, dtype=np.float32))
        self._cond.append(np.asarray(cond, dtype=np.float32))
        self._tick(class_id, seed, gidx)

    def _arrays(self):
        return {"target": np.stack(self._target), "cond": np.stack(self._cond)}

    def _clear_arrays(self):
        self._target.clear()
        self._cond.clear()

    def _n_buffered(self):
        return len(self._target)


# ----------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------
def load_manifest(directory: str) -> dict:
    path = os.path.join(directory, "manifest.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found -- {directory} is not a cfm stage output directory")
    with open(path) as f:
        return json.load(f)


def shard_paths(directory: str, manifest: dict | None = None) -> list[str]:
    man = manifest if manifest is not None else load_manifest(directory)
    return [os.path.join(directory, s["file"]) for s in man["shards"]]


def read_velocity_shard(path: str):
    """Return ``(v, class_id, seed)`` from one stage-1/stage-2 shard."""
    if path.endswith(".h5"):
        import h5py

        with h5py.File(path, "r") as f:
            return f["v"][:], f["class_id"][:], f["seed"][:]
    d = np.load(path)
    return d["v"], d["class_id"], d["seed"]


def write_manifest(out_dir: str, manifest: dict) -> str:
    path = os.path.join(out_dir, "manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return path
