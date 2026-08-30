"""Stage 5: metrics, trivial and conventional baselines, result figures.

``metrics`` is NumPy-only and can be imported without PyTorch; the command-line
entry points (``evaluate``, ``baselines``) need the training stack.
"""

from .metrics import (
    aggregate_by_class,
    corr,
    dix_rms_consistency,
    mae,
    rmse,
    ssim,
    well_consistency,
)

__all__ = [
    "aggregate_by_class",
    "corr",
    "dix_rms_consistency",
    "mae",
    "rmse",
    "ssim",
    "well_consistency",
]
