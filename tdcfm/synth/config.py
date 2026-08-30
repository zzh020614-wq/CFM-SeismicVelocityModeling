"""Stage 1 configuration: grid, per-operator parameter ranges, class recipes.

Everything is a dataclass and no path is hard-coded, so a full-scale run only
needs ``n_samples`` / ``out_dir`` / ``seed_base`` to change. All tuple-valued
ranges are ``[lo, hi]`` intervals sampled uniformly.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Tuple


# ----------------------------------------------------------------------
# Global grid and physical bounds
# ----------------------------------------------------------------------
@dataclass
class GridCfg:
    """Model grid. 256 x 256 is used throughout the whole pipeline."""

    nz: int = 256           # depth samples
    nx: int = 256           # traces
    dz: float = 16.0        # depth sampling interval (m); 256 * 16 ~ 4096 m
    dx: float = 25.0        # trace spacing (m); 256 * 25 ~ 6400 m
    v_min: float = 1500.0   # lower velocity bound (m/s)
    v_max: float = 6000.0   # upper velocity bound (m/s)


# ----------------------------------------------------------------------
# Operator parameters
# ----------------------------------------------------------------------
@dataclass
class BackgroundCfg:
    """Layered background with a compaction trend v = v0 + k*z plus interface jumps."""

    n_layers: Tuple[int, int] = (4, 9)                  # number of layers
    v0_range: Tuple[float, float] = (1600.0, 2200.0)    # near-surface velocity (m/s)
    k_range: Tuple[float, float] = (0.3, 0.9)           # compaction gradient ((m/s)/m)
    layer_jump: Tuple[float, float] = (150.0, 600.0)    # interface velocity contrast (m/s)
    jump_sign_pos: float = 0.85                         # probability the jump is positive


@dataclass
class DipCfg:
    """Dipping / wedge geometry: vertical shift varying linearly with x."""

    angle_deg: Tuple[float, float] = (-12.0, 12.0)      # apparent dip
    wedge_prob: float = 0.4                             # probability of a wedge
    wedge_gain: Tuple[float, float] = (0.3, 1.0)        # depth gain of the wedge


@dataclass
class FoldCfg:
    """Folding: a vertical warp built from superposed sine harmonics."""

    n_harm: Tuple[int, int] = (1, 3)                    # number of harmonics
    amp_m: Tuple[float, float] = (40.0, 160.0)          # amplitude per harmonic (m)
    wavelen_frac: Tuple[float, float] = (0.3, 1.0)      # wavelength / model width
    depth_decay: float = 0.4                            # decay of the shift with depth


@dataclass
class FaultCfg:
    """Faulting: rigid throw of the hanging wall, nearest-neighbour indexing.

    Integer index shifts (no interpolation) keep the fault plane a sharp step.
    """

    n_faults: Tuple[int, int] = (1, 3)                  # number of faults
    throw_m: Tuple[float, float] = (40.0, 200.0)        # vertical throw (m)
    dip_deg: Tuple[float, float] = (50.0, 85.0)         # fault dip from horizontal
    x0_frac: Tuple[float, float] = (0.2, 0.8)           # surface intercept / model width


@dataclass
class SmoothCfg:
    """Gaussian blur emulating gradational velocity interfaces.

    Applied to every class, but only to the continuous background: the
    generator runs it *before* faulting and salt emplacement so that the fault
    planes and the salt boundary stay sharp. ``sigma`` is in samples; 0 disables.
    """

    sigma_z: float = 2.5
    sigma_x: float = 2.5


@dataclass
class SaltCfg:
    """Strongly heterogeneous salt body, used by class 4 only.

    Two layouts of equal probability: a rounded random polygon in the middle of
    the section (``mid_flat``) and a bottom-rooted perturbed dome (``bottom``).
    The body is filled with a constant high velocity and a hard edge.
    See ``docs/salt_body.md`` for the full specification.
    """

    salt_v: float = 5000.0                              # constant salt velocity (m/s)

    # layout name -> normalised bounding region (z0, z1, x0, x1) of the model
    layout_names: Tuple[str, ...] = ("mid_flat", "bottom")
    layout_probs: Tuple[float, ...] = (0.5, 0.5)
    layouts: Dict[str, Tuple[float, float, float, float]] = field(default_factory=lambda: {
        "mid_flat": (0.40, 0.60, 0.10, 0.90),
        "bottom": (0.60, 1.00, 0.15, 0.85),
    })
    # fraction of the region occupied by the shape construction box (width, height)
    box_frac: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "mid_flat": (0.78, 0.78),
        "bottom": (0.90, 0.80),
    })

    # rounded random polygon (mid_flat)
    poly_n_vert: Tuple[int, int] = (14, 23)             # exclusive upper bound -> 14..22
    poly_radius: Tuple[float, float] = (0.80, 0.98)
    poly_smooth_passes: Tuple[int, int] = (2, 4)        # exclusive upper bound -> 2..3
    poly_smooth_kernel: int = 5

    # perturbed dome (bottom)
    dome_center_frac: Tuple[float, float] = (0.40, 0.60)
    dome_width_frac: Tuple[float, float] = (0.42, 0.55)
    dome_n_harm: Tuple[int, int] = (3, 6)               # harmonics 1..(K-1)
    dome_pert_amp: Tuple[float, float] = (0.05, 0.16)
    dome_height_cap: float = 1.0
    dome_top_scale: Tuple[float, float] = (0.78, 0.92)
    dome_gauss_sigma: float = 0.8
    dome_thresh: float = 0.5

    # keep the body off the top and side boundaries (the bottom edge is allowed
    # for the `bottom` layout, where the salt is rooted at the base of the model)
    off_edge_max_iters: int = 6
    off_edge_erosion_iters: int = 1

    # optional soft blending of the salt boundary; off reproduces the hard edge
    feather: bool = False
    feather_sigma: float = 1.5


# ----------------------------------------------------------------------
# Class recipes: background + blur is common to all classes, the class only
# decides which structural operators are superposed on top.
# ----------------------------------------------------------------------
@dataclass
class ClassRecipe:
    name: str
    use_dip: bool = False
    use_fold: bool = False
    use_fault: bool = False
    use_salt: bool = False


CLASS_TABLE: Dict[int, ClassRecipe] = {
    0: ClassRecipe("near_horizontal"),                          # background + blur
    1: ClassRecipe("dip_wedge", use_dip=True),                  # dipping / wedge
    2: ClassRecipe("fold", use_fold=True),                      # folding
    3: ClassRecipe("fault", use_fault=True, use_dip=True),      # faulting on a gentle dip
    4: ClassRecipe("salt", use_salt=True),                      # salt body (hard class)
}
N_CLASSES = len(CLASS_TABLE)


# ----------------------------------------------------------------------
# Run configuration
# ----------------------------------------------------------------------
@dataclass
class SynthCfg:
    grid: GridCfg = field(default_factory=GridCfg)
    background: BackgroundCfg = field(default_factory=BackgroundCfg)
    dip: DipCfg = field(default_factory=DipCfg)
    fold: FoldCfg = field(default_factory=FoldCfg)
    fault: FaultCfg = field(default_factory=FaultCfg)
    smooth: SmoothCfg = field(default_factory=SmoothCfg)
    salt: SaltCfg = field(default_factory=SaltCfg)

    n_samples: int = 32
    class_balanced: bool = True                 # round-robin class assignment
    seed_base: int = 20260630                   # sample_seed = seed_base + global_index
    out_dir: str = "data/stage1"
    shard_size: int = 256
    fmt: str = "hdf5"                           # "hdf5" | "npz"

    def to_dict(self) -> dict:
        return asdict(self)
