"""Dix forward and inverse transforms in the time domain, trace by trace.

The common time axis is uniformly sampled, so ``dtau`` cancels and the Dix
relation degenerates to a running root-mean-square along ``tau``.
"""
from __future__ import annotations

import numpy as np


def dix_forward(v_int: np.ndarray) -> np.ndarray:
    """Interval velocity -> RMS velocity.

    ``V_rms^2(tau_n) = sum_{k<=n} V_int,k^2 * dtau / sum dtau``; with a uniform
    ``dtau`` this is a cumulative mean of the squared interval velocity.
    Input and output are ``(n_t, nx)``.
    """
    n_t = v_int.shape[0]
    n = np.arange(1, n_t + 1, dtype=np.float64)[:, None]
    rms2 = np.cumsum(v_int.astype(np.float64) ** 2, axis=0) / n
    return np.sqrt(rms2).astype(np.float32)


def dix_inverse(v_rms: np.ndarray, tau_axis: np.ndarray,
                v_min: float = 1500.0, v_max: float = 6000.0) -> np.ndarray:
    """RMS velocity -> interval velocity, paired with :func:`dix_forward`.

    ``dix_forward`` has accumulated ``n+1`` equal-width intervals at sample
    ``n``, so the inversion differentiates ``V_rms^2 * (n+1)``. Multiplying by
    ``tau_axis`` directly would be wrong here, because ``tau_axis`` starts at 0
    and would count the first, already-accumulated sample as having zero
    duration.

    ``tau_axis`` is only used to validate the time dimension; the differencing
    is sensitive to noise, so the input should be a smoothed RMS field.
    """
    v_rms = v_rms.astype(np.float64)
    if len(tau_axis) != v_rms.shape[0]:
        raise ValueError(
            f"tau_axis length {len(tau_axis)} != v_rms n_t {v_rms.shape[0]}")

    n_t = v_rms.shape[0]
    n = np.arange(1, n_t + 1, dtype=np.float64)[:, None]
    num = v_rms ** 2 * n                                  # accumulated V_int^2
    vint2 = np.empty_like(v_rms)
    vint2[0] = num[0]
    vint2[1:] = np.diff(num, axis=0)
    vint2 = np.clip(vint2, v_min ** 2, v_max ** 2)        # guard against negatives
    return np.sqrt(vint2).astype(np.float32)
