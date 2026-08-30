"""Stage 4: dataloader, OT-CFM objective, well consistency loss, training loop.

Only the torch-free configuration is re-exported here, so that modules which do
not need PyTorch (and the unit tests that cover them) can be imported without it.
"""

from .config import TrainCfg, apply_overrides, cfg_from_ckpt

__all__ = ["TrainCfg", "apply_overrides", "cfg_from_ckpt"]
