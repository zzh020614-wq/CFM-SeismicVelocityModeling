"""Derive all conditions of one sample from the depth-domain ground truth.

Data flow::

    v_depth (nz, nx)
     |
     +-- depth_to_time --> V_int (n_t, nx)            <- label x1
          |
          +-- dix_forward + Gaussian smoothing -----> rms         <- condition 0
          +-- columns of V_int at the well positions -> well_val  <- condition 1
          |                                            well_mask  <- condition 2
          +-- reflectivity * Ricker ----------------> imaging     <- condition 3

Because every condition comes from the same single truth, conditions and label
are self-consistent by construction: there is no forward-modelling mismatch to
account for.

One ``default_rng(seed)`` drives the whole derivation, and the draw order below
is fixed: well count, well columns, Ricker frequency, then the imaging noise.
Changing that order changes the dataset for a given seed.

The well condition written here is a placeholder. Depth-to-time conversion is
purely vertical and per trace, so the time-domain log at trace ``c`` is exactly
column ``c`` of the label; the training dataloader therefore rebuilds the wells
from the label with the number and width set in ``TrainCfg`` (see
``tdcfm.conditions.wells.build_wells``), and what stage 3 stored is overwritten.
Storing it anyway keeps a shard self-contained and readable on its own.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from ..timeconv.convert import depth_to_time
from .config import CondCfg
from .dix import dix_forward
from .seismic import imaging_profile


def derive_conditions(v_depth: np.ndarray, dz: float, tau_axis: np.ndarray,
                      rng: np.random.Generator, cfg: CondCfg) -> dict:
    """Return the label and every condition of one sample, in physical units."""
    d_tau = float(tau_axis[1] - tau_axis[0])
    nz, nx = v_depth.shape
    n_t = len(tau_axis)

    # Label: time-domain interval velocity.
    v_int = depth_to_time(v_depth, dz, tau_axis)                    # (n_t, nx)

    # Condition 0: RMS velocity field, Dix forward model plus baseline smoothing.
    rms = gaussian_filter(dix_forward(v_int),
                          (cfg.rms.smooth_tau, cfg.rms.smooth_x),
                          mode="nearest").astype(np.float32)

    # Conditions 1 and 2: wells at random single columns (see the module docstring
    # -- the dataloader replaces these).
    n_w = int(rng.integers(cfg.well.n_wells[0], cfg.well.n_wells[1]))
    cols = np.sort(rng.choice(nx, size=n_w, replace=False))
    well_val = np.zeros((n_t, nx), np.float32)
    well_val[:, cols] = v_int[:, cols]
    well_mask = np.zeros((n_t, nx), np.float32)
    well_mask[:, cols] = 1.0

    # Condition 3: migrated image.
    f = float(rng.uniform(*cfg.imaging.f_ricker))
    imaging = imaging_profile(v_int, d_tau, f, rng,
                              cfg.imaging.noise_std, cfg.imaging.lateral_smooth)

    return {
        "target": v_int,          # (n_t, nx)  label x1
        "rms": rms,               # (n_t, nx)  condition channel 0
        "well_val": well_val,     # (n_t, nx)  condition channel 1
        "well_mask": well_mask,   # (n_t, nx)  condition channel 2
        "imaging": imaging,       # (n_t, nx)  condition channel 3
        "well_cols": cols,
        "ricker_f": f,
    }


def stack_cond(d: dict) -> np.ndarray:
    """Stack the conditions into ``(C, n_t, nx)`` following ``COND_NAMES``."""
    return np.stack([d["rms"], d["well_val"], d["well_mask"], d["imaging"]])
