"""Stage 3: forward modelling of the network conditions from the ground truth.

Every condition is derived from the *same* depth-domain truth, which is what
makes the conditions and the label mutually consistent by construction.
"""

from .config import COND_NAMES, CondCfg
from .derive import derive_conditions, stack_cond
from .dix import dix_forward, dix_inverse
from .seismic import imaging_profile, reflectivity, ricker
from .wells import build_wells, well_columns

__all__ = [
    "COND_NAMES",
    "CondCfg",
    "derive_conditions",
    "stack_cond",
    "dix_forward",
    "dix_inverse",
    "imaging_profile",
    "reflectivity",
    "ricker",
    "build_wells",
    "well_columns",
]
