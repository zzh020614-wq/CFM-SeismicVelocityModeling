"""Depth-to-time conversion and its inverse (pure functions, no I/O).

Conventions
-----------
* Depth-domain ``v`` is ``(nz, nx)`` in m/s, rows increasing downwards.
* Two-way vertical traveltime ``tau[:, x] = 2 * cumsum(dz / v[:, x])``, strictly
  monotonic per trace because ``v > 0``. Layer ordering therefore cannot be
  scrambled by the conversion; what changes is relative layer thickness
  (``dtau = 2*dz/v``: shallow layers stretch, deep layers compress), which is
  the correct time-domain geometry.
* Resampling onto the common ``tau_axis`` is a per-trace ``np.interp``, whose
  default end-point behaviour holds the first/last value. No validity mask is
  produced: the held tail is a constant-velocity extension, which is a
  well-defined model rather than missing data.
* ``tau`` is in seconds; the values stay physical velocities in m/s.

The conversion ignores ray bending, so ``V_int(tau, x)`` is an interval velocity
in the *vertical two-way time* sense.
"""
from __future__ import annotations

import numpy as np


def column_tau(v_col: np.ndarray, dz: float) -> np.ndarray:
    """Native (non-uniform) time axis of a single trace, shape ``(nz,)``."""
    return 2.0 * np.cumsum(dz / v_col)


def column_tau_max(v_depth: np.ndarray, dz: float) -> np.ndarray:
    """Total two-way time of every trace, ``2 * sum(dz/v)``, shape ``(nx,)``."""
    return 2.0 * np.sum(dz / v_depth, axis=0)


def depth_to_time(v_depth: np.ndarray, dz: float,
                  tau_axis: np.ndarray) -> np.ndarray:
    """``(nz, nx)`` depth-domain model -> ``(n_t, nx)`` time-domain ``V_int``."""
    nz, nx = v_depth.shape
    tau = 2.0 * np.cumsum(dz / v_depth, axis=0)        # (nz, nx), monotonic per trace
    out = np.empty((len(tau_axis), nx), dtype=np.float32)
    for x in range(nx):
        # below tau[0] -> v[0]; beyond tau[-1] -> v[-1] (held)
        out[:, x] = np.interp(tau_axis, tau[:, x], v_depth[:, x])
    return out


def scan_tmax(v_iter, dz: float, percentile: float = 90.0) -> tuple[float, np.ndarray]:
    """Pool ``tau_max`` over every trace of every model and take a percentile.

    Returns ``(T_max, all_tau_max)``. A *single global* ``T_max`` shared by the
    whole dataset is essential: deriving it per sample would give every sample a
    different time scale, and the network would learn a systematically drifting
    depth position.
    """
    all_tau_max = np.concatenate([column_tau_max(v, dz) for v in v_iter])
    return float(np.percentile(all_tau_max, percentile)), all_tau_max


def make_tau_axis(t_max: float, n_t: int) -> np.ndarray:
    """Uniformly sampled common time axis shared by every sample and trace."""
    return np.linspace(0.0, t_max, n_t).astype(np.float64)


def time_to_depth(v_int_time: np.ndarray, tau_axis: np.ndarray,
                  dz: float, nz: int) -> np.ndarray:
    """Inverse mapping ``V_int(tau, x) -> V(z, x)``, closing the loop.

    Per trace ``z(tau) = cumsum(v * dtau / 2)`` inverts ``tau = 2z/v``; the
    velocities are then resampled onto a uniform depth axis. Where the time
    model was in its held tail the result is a constant-velocity interval, which
    is expected. Used to show that a generated time-domain model can be turned
    back into a depth model suitable for depth imaging.

    Both this function and :func:`depth_to_time` accumulate at the right-hand
    endpoint of each interval, so the round trip carries a fixed sub-sample
    offset (well under one depth sample) that does not shrink as ``n_t`` grows.
    It is immaterial for the closed-loop demonstration this function serves; a
    depth model intended for imaging should be built on a purpose-made axis.
    """
    n_t, nx = v_int_time.shape
    d_tau = float(tau_axis[1] - tau_axis[0])
    z_axis = np.arange(nz) * dz
    out = np.empty((nz, nx), np.float32)
    for x in range(nx):
        z_of_tau = np.cumsum(v_int_time[:, x] * d_tau / 2.0)   # monotonically increasing
        out[:, x] = np.interp(z_axis, z_of_tau, v_int_time[:, x])
    return out
