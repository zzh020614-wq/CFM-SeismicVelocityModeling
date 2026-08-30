"""Depth-domain structural operators (pure functions, no I/O).

Conventions
-----------
* A velocity model ``v`` is ``float32`` of shape ``(nz, nx)``: rows are depth
  (increasing downwards), columns are traces.
* Every operator takes the current model, the per-sample ``rng``, its own
  configuration section and the grid, and returns a new model.
* Dip and fold are smooth warps evaluated with ``scipy.ndimage.map_coordinates``.
  Faulting instead shifts one side by an integer number of samples and splices
  along the fault plane (nearest-neighbour indexing, no interpolation), which is
  what keeps the fault a sharp discontinuity.
"""
from __future__ import annotations

import numpy as np
from matplotlib.path import Path
from scipy.ndimage import (
    binary_erosion,
    binary_fill_holes,
    gaussian_filter,
    label,
    map_coordinates,
)

from .config import BackgroundCfg, DipCfg, FaultCfg, FoldCfg, GridCfg, SaltCfg


# ----------------------------------------------------------------------
# (1) Layered background with a compaction trend
# ----------------------------------------------------------------------
def layered_background(grid: GridCfg, cfg: BackgroundCfg,
                       rng: np.random.Generator) -> np.ndarray:
    """v(z) = v0 + k*z with random layer interfaces superposed."""
    nz, nx = grid.nz, grid.nx
    z = np.arange(nz) * grid.dz                       # depth (m)

    v0 = rng.uniform(*cfg.v0_range)
    k = rng.uniform(*cfg.k_range)
    prof = v0 + k * z                                 # compaction trend, shape (nz,)

    # Random interfaces; each jump is cumulative for everything below it.
    n_layers = int(rng.integers(cfg.n_layers[0], cfg.n_layers[1] + 1))
    if n_layers > 1:
        bnds = np.sort(rng.choice(np.arange(int(0.05 * nz), int(0.95 * nz)),
                                  size=n_layers - 1, replace=False))
        for b in bnds:
            jump = rng.uniform(*cfg.layer_jump)
            if rng.random() > cfg.jump_sign_pos:
                jump = -jump
            prof[b:] += jump

    return np.repeat(prof[:, None], nx, axis=1).astype(np.float32)


def _vertical_warp(v: np.ndarray, shift_samples: np.ndarray) -> np.ndarray:
    """Warp by a vertical displacement field (in samples): zsrc = z - shift."""
    nz, nx = v.shape
    zz, xx = np.mgrid[0:nz, 0:nx]
    zsrc = zz - shift_samples
    out = map_coordinates(v, [zsrc, xx], order=1, mode="nearest")
    return out.astype(np.float32)


# ----------------------------------------------------------------------
# (2) Dip / wedge
# ----------------------------------------------------------------------
def apply_dip(v: np.ndarray, grid: GridCfg, cfg: DipCfg,
              rng: np.random.Generator) -> np.ndarray:
    nz, nx = v.shape
    angle = np.deg2rad(rng.uniform(*cfg.angle_deg))
    # Vertical shift per trace in metres, converted to samples.
    x_phys = np.arange(nx) * grid.dx
    shift_m = np.tan(angle) * x_phys                  # (nx,)
    shift = (shift_m / grid.dz)[None, :] * np.ones((nz, 1))

    # Wedge: the shift grows with depth, so layer thickness varies laterally.
    if rng.random() < cfg.wedge_prob:
        gain = rng.uniform(*cfg.wedge_gain)
        depth_frac = (np.arange(nz) / nz)[:, None]
        shift = shift * (1.0 + gain * depth_frac)

    return _vertical_warp(v, shift)


# ----------------------------------------------------------------------
# (3) Folding
# ----------------------------------------------------------------------
def apply_fold(v: np.ndarray, grid: GridCfg, cfg: FoldCfg,
               rng: np.random.Generator) -> np.ndarray:
    nz, nx = v.shape
    x = np.arange(nx)
    disp_m = np.zeros(nx)
    n_harm = int(rng.integers(cfg.n_harm[0], cfg.n_harm[1] + 1))
    for _ in range(n_harm):
        amp = rng.uniform(*cfg.amp_m)
        lam = rng.uniform(*cfg.wavelen_frac) * nx     # wavelength in samples
        phi = rng.uniform(0, 2 * np.pi)
        disp_m += amp * np.sin(2 * np.pi * x / lam + phi)

    shift = (disp_m / grid.dz)[None, :] * np.ones((nz, 1))
    # Folds are strongest near the surface and flatten with depth.
    if cfg.depth_decay > 0:
        depth_frac = (np.arange(nz) / nz)[:, None]
        shift = shift * (1.0 - cfg.depth_decay * depth_frac)

    return _vertical_warp(v, shift)


# ----------------------------------------------------------------------
# (4) Faulting
# ----------------------------------------------------------------------
def _single_fault(v: np.ndarray, grid: GridCfg, cfg: FaultCfg,
                  rng: np.random.Generator) -> np.ndarray:
    nz, nx = v.shape

    dip = np.deg2rad(rng.uniform(*cfg.dip_deg))       # from horizontal
    # Fault-plane slope in pixel space: physically dx/dz = 1/tan(dip),
    # converted to samples by the dz/dx aspect ratio.
    slope = (1.0 / np.tan(dip)) * (grid.dz / grid.dx)  # x samples per z sample
    x0 = rng.uniform(*cfg.x0_frac) * nx

    throw_samples = int(round(rng.uniform(*cfg.throw_m) / grid.dz))
    if throw_samples == 0:
        return v

    z_idx = np.arange(nz)[:, None]
    x_idx = np.arange(nx)[None, :]
    x_fault = x0 + slope * z_idx                       # fault position per depth

    # Pick which side is the hanging wall.
    hanging = x_idx > x_fault if rng.random() < 0.5 else x_idx < x_fault
    hanging = np.broadcast_to(hanging, (nz, nx))

    # Move the hanging wall down by `throw_samples`: source row = z - throw.
    zsrc = z_idx - throw_samples * hanging
    zsrc = np.clip(zsrc, 0, nz - 1)                    # above the top: hold
    xsrc = np.broadcast_to(x_idx, (nz, nx))

    return v[zsrc, xsrc].astype(np.float32)            # integer indexing, no blending


def apply_fault(v: np.ndarray, grid: GridCfg, cfg: FaultCfg,
                rng: np.random.Generator) -> np.ndarray:
    n = int(rng.integers(cfg.n_faults[0], cfg.n_faults[1] + 1))
    for _ in range(n):
        v = _single_fault(v, grid, cfg, rng)
    return v


# ----------------------------------------------------------------------
# (5) Salt body (class 4 only)
# ----------------------------------------------------------------------
def _clean(mask: np.ndarray) -> np.ndarray:
    """Keep the largest connected component and fill holes, so the body is solid."""
    lab, n = label(mask)
    if n == 0:
        return mask
    sizes = [(lab == i).sum() for i in range(1, n + 1)]
    return binary_fill_holes(lab == (np.argmax(sizes) + 1))


def _random_polygon(bh: int, bw: int, cfg: SaltCfg,
                    rng: np.random.Generator) -> np.ndarray:
    """`mid_flat` shape: a random polygon whose radii are heavily smoothed.

    Smoothing the radius sequence with a wide circular kernel rounds off the
    corners, giving the lobate outline of a salt sheet rather than a star.
    """
    cz, cx = bh / 2, bw / 2
    n_v = rng.integers(cfg.poly_n_vert[0], cfg.poly_n_vert[1])
    ang = np.sort(rng.uniform(0, 2 * np.pi, size=n_v))
    rr = rng.uniform(cfg.poly_radius[0], cfg.poly_radius[1], size=n_v)
    kern = cfg.poly_smooth_kernel
    for _ in range(rng.integers(cfg.poly_smooth_passes[0], cfg.poly_smooth_passes[1])):
        pad = np.concatenate([rr[-2:], rr, rr[:2]])    # circular padding
        rr = np.convolve(pad, np.ones(kern) / kern, mode="same")[2:-2]
    vx = cx + rr * np.cos(ang) * (bw / 2)
    vz = cz + rr * np.sin(ang) * (bh / 2)
    verts = np.column_stack([vx, vz])
    zz, xx = np.mgrid[0:bh, 0:bw]
    pts = np.column_stack([xx.ravel(), zz.ravel()])
    return _clean(Path(verts).contains_points(pts).reshape(bh, bw))


def _dome_perturbed(bh: int, bw: int, cfg: SaltCfg,
                    rng: np.random.Generator) -> np.ndarray:
    """`bottom` shape: a semi-elliptical dome with a low-frequency perturbed crest."""
    xs = np.arange(bw)
    center = rng.uniform(*cfg.dome_center_frac) * bw
    width = rng.uniform(*cfg.dome_width_frac) * bw
    base = np.sqrt(np.clip(1 - ((xs - center) / width) ** 2, 0, None))
    pert = np.zeros(bw)
    for k in range(1, rng.integers(cfg.dome_n_harm[0], cfg.dome_n_harm[1])):
        pert += (rng.uniform(*cfg.dome_pert_amp) / k) * \
            np.sin(2 * np.pi * k * xs / bw + rng.uniform(0, 2 * np.pi))
    height = np.clip(base + pert, 0, cfg.dome_height_cap)
    top = (bh - 1) - height * (bh - 1) * rng.uniform(*cfg.dome_top_scale)
    zz = np.arange(bh)[:, None]
    mask = zz >= top[None, :]
    mask = gaussian_filter(mask.astype(float), sigma=cfg.dome_gauss_sigma) > cfg.dome_thresh
    return _clean(mask)


_SHAPE = {"mid_flat": _random_polygon, "bottom": _dome_perturbed}


def _off_top_sides(m: np.ndarray, cfg: SaltCfg, protect_bottom: bool) -> np.ndarray:
    """Erode until the body no longer touches the top or side boundaries."""
    for _ in range(cfg.off_edge_max_iters):
        touch = m[0, :].any() or m[:, 0].any() or m[:, -1].any()
        if not protect_bottom:
            touch = touch or m[-1, :].any()
        if not touch:
            break
        m = _clean(binary_erosion(m, iterations=cfg.off_edge_erosion_iters))
    return m


def generate_salt(nz: int, nx: int, rng: np.random.Generator,
                  cfg: SaltCfg) -> dict:
    """Build one salt-body mask plus metadata; the velocity field is untouched."""
    # With uniform layout probabilities we deliberately call `rng.choice`
    # without `p`. Passing `p` draws from the RNG stream differently, and the
    # released dataset was generated on the no-`p` branch; keeping the branch
    # preserves seed-level reproducibility. `p` is only used when the user
    # configures genuinely non-uniform probabilities.
    probs = np.asarray(cfg.layout_probs, dtype=float)
    if np.allclose(probs, probs[0]):
        layout = rng.choice(cfg.layout_names)
    else:
        layout = rng.choice(cfg.layout_names, p=probs / probs.sum())

    z0f, z1f, x0f, x1f = cfg.layouts[layout]
    z0, z1 = int(z0f * nz), int(z1f * nz)
    x0, x1 = int(x0f * nx), int(x1f * nx)
    region_h, region_w = z1 - z0, x1 - x0
    fw, fh = cfg.box_frac[layout]
    bh, bw = max(8, int(region_h * fh)), max(8, int(region_w * fw))

    m = _clean(_SHAPE[layout](bh, bw, cfg, rng))
    m = _off_top_sides(m, cfg, protect_bottom=(layout == "bottom"))

    ys, xs = np.where(m)
    mask = np.zeros((nz, nx), bool)
    if len(ys) == 0:                                   # eroded away; no salt body
        return {"mask": mask, "layout": str(layout),
                "box": (z0, z1, x0, x1), "salt_v": float(cfg.salt_v)}

    crop = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    hh, ww = crop.shape
    px = x0 + rng.integers(0, max(1, region_w - ww))   # random lateral placement
    # A bottom-rooted dome is pinned to the base of its region; a salt sheet is
    # vertically centred in its region.
    pz = (z1 - hh) if layout == "bottom" else (z0 + (region_h - hh) // 2)
    mask[pz:pz + hh, px:px + ww] = crop

    return {"mask": mask, "layout": str(layout),
            "box": (z0, z1, x0, x1), "salt_v": float(cfg.salt_v)}


def apply_salt_body(v: np.ndarray, mask: np.ndarray, salt_v: float,
                    feather: bool = False, feather_sigma: float = 1.5) -> np.ndarray:
    """Stamp the salt body into the velocity field."""
    if not mask.any():
        return v
    if not feather:
        return np.where(mask, np.float32(salt_v), v).astype(np.float32)
    w = np.clip(gaussian_filter(mask.astype(np.float32), feather_sigma), 0.0, 1.0)
    return ((1.0 - w) * v + w * salt_v).astype(np.float32)
