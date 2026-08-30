"""Stage 2: vertical depth-to-time conversion onto a global common time axis."""

from .convert import (
    column_tau,
    column_tau_max,
    depth_to_time,
    make_tau_axis,
    scan_tmax,
    time_to_depth,
)

__all__ = [
    "column_tau",
    "column_tau_max",
    "depth_to_time",
    "make_tau_axis",
    "scan_tmax",
    "time_to_depth",
]
