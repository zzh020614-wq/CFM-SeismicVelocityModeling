"""Well condition: placement of the well columns and extraction of the log values.

A well log is measured in depth, but depth-to-time conversion is a purely
vertical, per-trace mapping, so the time-domain log at trace ``c`` is exactly
column ``c`` of the time-domain ground truth. The well condition can therefore
be rebuilt from the label at any time, which is what lets the number and width
of the wells be changed without regenerating the dataset.

Wells are widened to several columns on purpose. The UNet downsamples six
times (stride 32 at the coarsest level), so a single-pixel well disappears from
the deep feature maps entirely and its constraint never reaches them. Inside a
widened band each column carries *its own* true value -- equivalent to a few
closely spaced wells -- rather than the centre column replicated sideways,
which would contradict the label.
"""
from __future__ import annotations

import numpy as np


def well_rng(seed: int) -> np.random.Generator:
    """Deterministic RNG for the training-time well placement.

    Seeded from the sample's own seed, so the well positions are stable across
    epochs and reproducible from the dataset alone, while still differing from
    sample to sample.
    """
    return np.random.default_rng(int(seed))


def well_columns(nx: int, n_wells: int, width: int,
                 rng: np.random.Generator) -> np.ndarray:
    """Stratified well placement: split the section into ``n_wells`` bands and
    draw one column at random inside each.

    This spreads the wells over the whole section while still varying their
    positions from sample to sample, so the network cannot memorise fixed
    locations.

    A minimum gap of ``width + 1`` between successive wells is enforced.
    Without it, two neighbouring bands can each pick a column just either side
    of their shared boundary; the two ``width``-wide bands then merge into a
    single band of twice the width, silently giving one well fewer than asked.
    """
    # A band spans [c - left, c + right] for a total of `width` columns. For an
    # even width the split must be asymmetric, otherwise c-half : c+half+1
    # yields 2*half+1 columns (3 columns for width=2).
    left = (width - 1) // 2
    right = width - 1 - left
    lo_b, hi_b = left, nx - 1 - right
    min_gap = width + 1
    if n_wells > 1 and hi_b - lo_b < (n_wells - 1) * min_gap:
        raise ValueError(
            f"nx={nx} cannot hold {n_wells} non-touching wells of width {width}: "
            f"at least {(n_wells - 1) * min_gap + width} columns are needed")

    edges = np.linspace(lo_b, hi_b + 1, n_wells + 1)
    cols, prev = [], None
    for k in range(n_wells):
        a, b = int(np.floor(edges[k])), int(np.ceil(edges[k + 1])) - 1
        lo = max(a, lo_b if prev is None else prev + min_gap)
        # Leave enough room on the right for the wells that still have to fit.
        hi = min(b, hi_b - (n_wells - 1 - k) * min_gap)
        c = int(rng.integers(lo, hi + 1)) if hi >= lo else lo
        cols.append(int(np.clip(c, lo_b, hi_b)))
        prev = cols[-1]
    return np.asarray(cols, dtype=int)


def build_wells(target: np.ndarray, n_wells: int, width: int,
                seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Build ``(well_val, well_mask)`` from the time-domain ground truth.

    This is what the training dataloader feeds the network. It overrides
    whatever stage 3 wrote, which is why the number and width of the wells are
    free parameters at training time.
    """
    n_t, nx = target.shape
    cols = well_columns(nx, n_wells, width, well_rng(seed))
    left = (width - 1) // 2                # must match well_columns
    well_val = np.zeros((n_t, nx), np.float32)
    well_mask = np.zeros((n_t, nx), np.float32)
    for c in cols:
        a = max(0, c - left)
        b = min(nx, a + width)
        well_val[:, a:b] = target[:, a:b]
        well_mask[:, a:b] = 1.0
    return well_val, well_mask


def interp_well_horizontal(well_val: np.ndarray, well_mask: np.ndarray) -> np.ndarray:
    """Fill between the wells by linear interpolation along each time slice.

    Applied to physical velocities (not normalised values). Outside the outer
    wells ``np.interp`` holds the nearest value. Interpolating removes the
    vertical-stripe shortcut in which the network only has to reproduce sharp
    edges at the well positions; it is also the standard manual workflow, and
    hence the fair strong baseline this method has to beat.
    """
    cols = np.where((well_mask > 0.5).any(axis=0))[0]
    if len(cols) < 2:
        return well_val
    n_t, nx = well_val.shape
    out = well_val.copy()
    all_x = np.arange(nx, dtype=np.float32)
    cx = cols.astype(np.float32)
    for r in range(n_t):
        out[r, :] = np.interp(all_x, cx, well_val[r, cols])
    return out.astype(np.float32)
