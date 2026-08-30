"""Stage 3 configuration. Everything here is in physical units; normalisation
to ``[-1, 1]`` happens in the training dataloader.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Tuple


@dataclass
class WellCfg:
    """Well count drawn per sample at stage 3, as ``[lo, hi)`` for ``rng.integers``.

    These stage-3 wells are single columns and are a placeholder: the training
    dataloader rebuilds the well condition from the label with the count and
    width set in ``TrainCfg``. See ``derive.py`` and ``wells.py``.
    """

    n_wells: Tuple[int, int] = (3, 9)      # -> 3..8 columns


@dataclass
class RmsCfg:
    """Baseline Gaussian smoothing of the RMS field, in samples.

    Picked RMS velocities are smooth by construction, so an unsmoothed Dix
    forward model would be an unrealistically informative condition. This is the
    *baseline* smoothing written to disk; the dataloader can add more on top
    without regenerating the dataset, because Gaussians compose as
    ``sigma_total^2 = sigma_base^2 + sigma_extra^2``.
    """

    smooth_tau: float = 9.0
    smooth_x: float = 4.0


@dataclass
class ImagingCfg:
    """Migrated-image condition."""

    f_ricker: Tuple[float, float] = (25.0, 30.0)   # dominant frequency range (Hz)
    noise_std: float = 0.05                        # band-limited noise, relative to image std
    lateral_smooth: float = 0.0                    # lateral Gaussian sigma (samples), 0 = off


@dataclass
class CondCfg:
    well: WellCfg = field(default_factory=WellCfg)
    rms: RmsCfg = field(default_factory=RmsCfg)
    imaging: ImagingCfg = field(default_factory=ImagingCfg)

    v_min: float = 1500.0
    v_max: float = 6000.0

    def to_dict(self) -> dict:
        return asdict(self)


#: Channel order of the stacked condition tensor written to disk and fed to the network.
COND_NAMES = ("rms", "well_val", "well_mask", "imaging")
