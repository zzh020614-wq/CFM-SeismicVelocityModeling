"""Stage-3 shards -> normalised ``(x1, cond, y)`` training tensors.

* A flat ``(shard_path, local_index)`` index spans all shards; HDF5 handles are
  opened lazily per worker, which keeps the dataset fork-safe.
* The RMS condition is smoothed up to the target sigma here rather than in
  stage 3, so its strength can be varied without regenerating the dataset.
* The well condition is rebuilt from the label here, so the number and width of
  the wells can be varied without regenerating the dataset either.
* Horizontal flip augmentation flips ``x1`` and every condition channel
  together; the class label is flip-invariant and is left alone.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset

from ..conditions.config import RmsCfg
from ..conditions.wells import build_wells
from .normalize import normalize_sample


class ShardConditionDataset(Dataset):
    def __init__(self, data_dir: str, v_min: float, v_max: float,
                 well_interp: bool = True, random_flip: bool = True,
                 overfit: int = 0,
                 rms_smooth_tau: float = 18.0, rms_smooth_x: float = 8.0,
                 wells_from_target: bool = True, n_wells: int = 10,
                 well_width: int = 3, use_imaging: bool = True):
        if wells_from_target and (n_wells < 1 or well_width < 1):
            raise ValueError(
                f"n_wells and well_width must be >= 1, got {n_wells}/{well_width}")
        self.data_dir = data_dir
        self.v_min, self.v_max = v_min, v_max
        self.well_interp = well_interp
        self.random_flip = random_flip
        self.wells_from_target = wells_from_target
        self.n_wells = n_wells
        self.well_width = well_width
        self.use_imaging = use_imaging

        with open(os.path.join(data_dir, "manifest.json")) as f:
            man = json.load(f)
        self.index: list[tuple[str, int]] = []
        for s in man["shards"]:
            p = os.path.join(data_dir, s["file"])
            self.index.extend((p, i) for i in range(s["n"]))
        if overfit > 0:
            self.index = self.index[:overfit]
        self._handles: dict[str, object] = {}      # per-worker HDF5 handle cache

        # Gaussians compose: sigma_total^2 = sigma_base^2 + sigma_extra^2.
        base = (man.get("config") or {}).get("rms") or {}
        defaults = RmsCfg()
        self.base_tau = float(base.get("smooth_tau", defaults.smooth_tau))
        self.base_x = float(base.get("smooth_x", defaults.smooth_x))
        self._extra_tau = self._extra_sigma(rms_smooth_tau, self.base_tau, "tau")
        self._extra_x = self._extra_sigma(rms_smooth_x, self.base_x, "x")

    @staticmethod
    def _extra_sigma(target: float, base: float, axis: str) -> float:
        """Sigma still to be applied. Smoothing cannot be undone."""
        if target < base:
            print(f"[dataset] warning: rms_smooth_{axis}={target} is below the "
                  f"stage-3 baseline {base}; smoothing cannot be reversed, so the "
                  f"effective sigma stays at the baseline")
            return 0.0
        return float(np.sqrt(max(0.0, target ** 2 - base ** 2)))

    @classmethod
    def from_cfg(cls, cfg, data_dir: str, random_flip=None, overfit=None):
        """Build from a :class:`~cfm.training.config.TrainCfg`.

        Training, sampling and evaluation all construct the dataset this way, so
        they cannot drift apart in how the conditions are prepared.
        """
        return cls(
            data_dir, cfg.v_min, cfg.v_max,
            well_interp=cfg.well_interp,
            random_flip=cfg.random_flip if random_flip is None else random_flip,
            overfit=cfg.overfit if overfit is None else overfit,
            rms_smooth_tau=cfg.rms_smooth_tau,
            rms_smooth_x=cfg.rms_smooth_x,
            wells_from_target=cfg.wells_from_target,
            n_wells=cfg.n_wells,
            well_width=cfg.well_width,
            use_imaging=cfg.use_imaging,
        )

    def __len__(self):
        return len(self.index)

    def _handle(self, path):
        h = self._handles.get(path)
        if h is None:
            import h5py

            h = h5py.File(path, "r")
            self._handles[path] = h
        return h

    def __getitem__(self, idx):
        path, li = self.index[idx]
        f = self._handle(path)
        target = np.asarray(f["target"][li], dtype=np.float32)     # (H, W)
        cond = np.asarray(f["cond"][li], dtype=np.float32)         # (4, H, W)
        y = int(f["class_id"][li])
        seed = int(f["seed"][li])

        # Extra RMS smoothing, applied on physical velocities. The Gaussian is
        # linear and therefore commutes with the affine normalisation that follows.
        if self._extra_tau > 0.0 or self._extra_x > 0.0:
            cond[0] = gaussian_filter(cond[0], (self._extra_tau, self._extra_x),
                                      mode="nearest")

        if self.wells_from_target:
            cond[1], cond[2] = build_wells(target, self.n_wells,
                                           self.well_width, seed)

        t, c = normalize_sample(target, cond, self.v_min, self.v_max, self.well_interp)

        if not self.use_imaging:
            c = c[:3]

        if self.random_flip and torch.rand(()) < 0.5:
            # torch's RNG is automatically re-seeded per worker.
            t = t[:, :, ::-1].copy()
            c = c[:, :, ::-1].copy()

        return (torch.from_numpy(t), torch.from_numpy(c),
                torch.tensor(y, dtype=torch.long))
