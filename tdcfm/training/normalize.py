"""Per-channel conversion between physical units and ``[-1, 1]`` (NumPy only).

Channel conventions (see ``tdcfm.conditions.config.COND_NAMES``)::

    target      interval velocity   [v_min, v_max] -> [-1, 1]
    cond[0] rms RMS velocity        [v_min, v_max] -> [-1, 1]
    cond[1] well velocity           [v_min, v_max] -> [-1, 1]
    cond[2] well mask               kept as {0, 1}
    cond[3] imaging                 per-sample max|amplitude| -> [-1, 1]

The velocity range is a *fixed global constant*, not a per-sample statistic:
a per-sample scaling would make the network's output scale depend on
information it does not receive at inference time.
"""
from __future__ import annotations

import numpy as np

from ..conditions.wells import interp_well_horizontal


def vel_to_unit(v, v_min, v_max):
    x = (v - v_min) / (v_max - v_min)
    return np.clip(x * 2.0 - 1.0, -1.0, 1.0)


def unit_to_vel(x, v_min, v_max):
    return (np.clip(x, -1.0, 1.0) + 1.0) / 2.0 * (v_max - v_min) + v_min


def imaging_to_unit(img, eps=1e-6):
    """Scale the migrated image by its own peak amplitude.

    Unlike velocity, the image has no absolute physical scale worth preserving:
    only the relative reflectivity pattern carries information.
    """
    a = np.abs(img).max()
    return (img / (a + eps)).astype(np.float32)


def normalize_sample(target, cond, v_min, v_max, well_interp=True):
    """``target (H, W)`` and ``cond (4, H, W)`` in physical units ->
    ``(1, H, W)`` and ``(4, H, W)`` normalised."""
    t = vel_to_unit(target, v_min, v_max)[None].astype(np.float32)

    rms = vel_to_unit(cond[0], v_min, v_max)
    well_val, well_mask = cond[1], cond[2]
    if well_interp:
        # Interpolation must happen on physical velocities, before normalisation.
        well_val = interp_well_horizontal(well_val, well_mask)
    well_val = vel_to_unit(well_val, v_min, v_max)
    imaging = imaging_to_unit(cond[3])

    c = np.stack([rms, well_val,
                  (well_mask > 0.5).astype(np.float32), imaging]).astype(np.float32)
    return t, c
