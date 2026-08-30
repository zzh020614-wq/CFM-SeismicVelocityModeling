"""Conditional UNet velocity field and the Euler ODE sampler.

How the conditions enter the network
------------------------------------
The rule is simply whether a condition has a per-pixel spatial location.

``c`` -- **channel concatenation.** The RMS velocity field, the migrated image,
    the well values and the well mask all live on the same ``(n_t, nx)`` grid as
    ``x_t``, pixel for pixel. They are concatenated straight onto the network
    input. They are deliberately *not* given their own downsampling encoder: an
    encoder would destroy the pixel-wise alignment that makes them useful.

``y`` -- **label embedding.** The geological pattern is a single global
    attribute with no spatial extent. It goes through ``label_emb(y)`` and is
    added to the timestep embedding. Broadcasting a one-hot class code into
    spatial channels would waste capacity and hurt.

The backbone is the ``UNetModel`` from ``torchcfm`` (which itself derives from
the guided-diffusion architecture); only the input/output channel counts and
the conditioning wiring are ours.
"""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    from torchcfm.models.unet.unet import UNetModel
except ImportError as exc:  # pragma: no cover - import-time guidance only
    raise ImportError(
        "torchcfm is required for the model definition: pip install torchcfm==1.0.7"
    ) from exc

from ..training.config import TrainCfg, active_cond_channels, channel_mult_for


def build_unet(cfg: TrainCfg) -> nn.Module:
    """Build the velocity-field network for a given training configuration.

    ``in_channels = 1 (x_t) + number of active condition channels``;
    ``out_channels = 1``, the predicted velocity field of the flow.
    """
    size = cfg.image_size
    return UNetModel(
        image_size=size,
        in_channels=1 + active_cond_channels(cfg),
        model_channels=cfg.num_channel,
        out_channels=1,
        num_res_blocks=cfg.num_res_blocks,
        attention_resolutions=(size // cfg.attention_res,),
        dropout=cfg.dropout,
        channel_mult=channel_mult_for(size),
        num_classes=cfg.num_classes,           # enables the label embedding
        num_heads=cfg.num_heads,
        num_head_channels=cfg.num_head_channels,
        use_scale_shift_norm=False,
        resblock_updown=False,
        dims=2,
    )


@torch.no_grad()
def euler_sample(model: nn.Module, cond: torch.Tensor, y: torch.Tensor,
                 ode_steps: int = 100, image_size: int = 256,
                 x0: torch.Tensor | None = None) -> torch.Tensor:
    """Integrate the probability-flow ODE from t=0 to t=1 with explicit Euler.

    Returns ``(B, 1, H, W)`` in ``[-1, 1]``.

    Pass ``x0`` to pin the starting noise. Every controlled comparison in this
    work fixes both the sample set and ``x0``, so that the only thing differing
    between two runs is the variable actually under study.
    """
    model.eval()
    device = next(model.parameters()).device
    B = cond.shape[0]
    x = (torch.randn(B, 1, image_size, image_size, device=device)
         if x0 is None else x0.to(device))
    dt = 1.0 / ode_steps
    for i in range(ode_steps):
        t = torch.full((B,), i * dt, device=device)
        v = model(t, torch.cat([x, cond], dim=1), y=y)
        x = x + v * dt
    return x.clamp(-1, 1)
