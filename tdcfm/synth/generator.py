"""Assemble one depth-domain velocity model according to its class recipe.

Order of operations::

    layered background -> dip -> fold -> Gaussian blur -> fault -> salt -> clip

The blur sits deliberately *before* faulting and salt emplacement: it should
soften only the continuous sedimentary background, leaving the fault planes and
the salt boundary sharp.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from . import operators as ops
from .config import CLASS_TABLE, SynthCfg


class ModelGenerator:
    """Deterministic generator: ``generate(class_id, seed)`` is a pure function."""

    def __init__(self, cfg: SynthCfg):
        self.cfg = cfg

    def generate(self, class_id: int, sample_seed: int) -> dict:
        cfg = self.cfg
        g = cfg.grid
        rng = np.random.default_rng(sample_seed)
        recipe = CLASS_TABLE[class_id]

        v = ops.layered_background(g, cfg.background, rng)
        if recipe.use_dip:
            v = ops.apply_dip(v, g, cfg.dip, rng)
        if recipe.use_fold:
            v = ops.apply_fold(v, g, cfg.fold, rng)

        s = cfg.smooth
        if s.sigma_z > 0 or s.sigma_x > 0:
            v = gaussian_filter(v, sigma=(s.sigma_z, s.sigma_x), mode="nearest")

        if recipe.use_fault:                      # after the blur: keep sharp planes
            v = ops.apply_fault(v, g, cfg.fault, rng)

        if recipe.use_salt:                       # class 4 only: hard-edged salt
            info = ops.generate_salt(g.nz, g.nx, rng, cfg.salt)
            v = ops.apply_salt_body(v, info["mask"], info["salt_v"],
                                    cfg.salt.feather, cfg.salt.feather_sigma)

        v = np.clip(v, g.v_min, g.v_max).astype(np.float32)
        return {"v": v, "class_id": int(class_id), "seed": int(sample_seed)}
