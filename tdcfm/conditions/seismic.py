"""Migrated-image condition: reflectivity convolved with a Ricker wavelet."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve1d, gaussian_filter1d


def ricker(f: float, dt: float) -> np.ndarray:
    """Ricker wavelet with dominant frequency ``f`` (Hz) sampled at ``dt`` (s)."""
    half = max(3, int(round(1.5 / (f * dt))))             # main lobe plus side lobes
    t = np.arange(-half, half + 1) * dt
    a = (np.pi * f * t) ** 2
    return ((1.0 - 2.0 * a) * np.exp(-a)).astype(np.float64)


def reflectivity(v_int: np.ndarray) -> np.ndarray:
    """``r_i = (v_{i+1} - v_i) / (v_{i+1} + v_i)`` along tau, placed above the interface.

    Density is assumed constant, so the reflection coefficient reduces to the
    velocity contrast.
    """
    r = np.zeros_like(v_int, dtype=np.float64)
    up, dn = v_int[:-1], v_int[1:]
    r[:-1] = (dn - up) / (dn + up)
    return r


def imaging_profile(v_int: np.ndarray, d_tau: float, f: float,
                    rng: np.random.Generator, noise_std: float = 0.0,
                    lateral_smooth: float = 0.0) -> np.ndarray:
    """Zero-offset migrated image ``(n_t, nx)``.

    The additive noise is passed through the same wavelet, so it is band-limited
    like the signal instead of being white.
    """
    r = reflectivity(v_int)
    w = ricker(f, d_tau)
    img = convolve1d(r, w, axis=0, mode="constant", cval=0.0)

    if noise_std > 0:
        n = convolve1d(rng.standard_normal(img.shape), w, axis=0,
                       mode="constant", cval=0.0)
        s = img.std()
        if n.std() > 0:
            n = n / n.std() * (noise_std * (s if s > 0 else 1.0))
        img = img + n

    if lateral_smooth > 0:
        img = gaussian_filter1d(img, lateral_smooth, axis=1, mode="nearest")

    return img.astype(np.float32)
