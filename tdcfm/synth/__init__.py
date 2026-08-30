"""Stage 1: parametric synthesis of depth-domain velocity models."""

from .config import CLASS_TABLE, N_CLASSES, GridCfg, SynthCfg
from .generator import ModelGenerator

__all__ = ["CLASS_TABLE", "N_CLASSES", "GridCfg", "SynthCfg", "ModelGenerator"]
