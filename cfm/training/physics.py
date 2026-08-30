"""Well consistency loss.

Total objective::

    L = lambda_FM * L_FM + lambda_well * L_well

The constraint applies to the *final* velocity model ``x1``, but the network
outputs the flow velocity field ``v_theta`` at an intermediate time. The flow
matching identities connect the two::

    x_t = (1-t) * x0 + t * x1,   u_t = x1 - x0
      =>  x1_hat = x_t + (1-t) * v_theta(x_t, t, c)

so one forward pass -- the one the flow-matching loss already performs -- yields
an estimate of ``x1`` at essentially no extra cost; the ODE solve is not needed
during training. Note that ``(1-t)`` multiplies rather than divides the error:
an error ``eps`` in ``v_theta`` reaches ``x1_hat`` as ``(1-t)*eps``, which is
bounded over the whole interval and smallest as ``t -> 1``. No time-dependent
weighting or truncation is required.

``L_well`` requires ``x1_hat`` to match the logged velocity at the masked well
pixels. The values and the mask are channels the network itself received, so
this term says "obey the conditions you were given".
"""
from __future__ import annotations

import torch

from .config import TrainCfg


def to_velocity(x: torch.Tensor, v_min: float, v_max: float) -> torch.Tensor:
    """``[-1, 1]`` -> physical velocity (m/s). Not clamped, to keep gradients."""
    return (x + 1.0) * 0.5 * (v_max - v_min) + v_min


def well_loss(x1_hat: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    """Mean squared error at the well pixels, in the normalised ``[-1, 1]`` scale.

    ``x1_hat``  ``(B, 1, H, W)``
    ``cond``    ``(B, C, H, W)``, the OT-aligned conditions, matching ``x1_hat``
        row for row. Channels are ``[0] rms, [1] well value, [2] well mask,
        [3] imaging (optional)``.

    Normalising by the number of masked pixels rather than by the image size
    keeps the term's scale independent of how many wells are present.
    """
    well_val = cond[:, 1:2]
    mask = (cond[:, 2:3] > 0.5).to(x1_hat.dtype)
    return (((x1_hat - well_val) * mask) ** 2).sum() / mask.sum().clamp(min=1.0)


def physics_loss(x1_hat: torch.Tensor, cond: torch.Tensor,
                 cfg: TrainCfg) -> dict[str, torch.Tensor]:
    """Return the unweighted constraint terms; a zero-weight term is not computed."""
    out: dict[str, torch.Tensor] = {}
    if cfg.lambda_well > 0:
        out["well"] = well_loss(x1_hat, cond)
    return out


@torch.no_grad()
def physics_metrics(x1_hat: torch.Tensor, cond: torch.Tensor) -> dict[str, float]:
    """The same quantity, monitored only and never backpropagated.

    Runs are compared on how well the constraint is satisfied, so a
    ``lambda_well=0`` control run has to log the same number as the constrained
    run; otherwise the comparison is only possible after a full evaluation pass.
    """
    return {"well": float(well_loss(x1_hat, cond))}


def combine_losses(l_fm: torch.Tensor, parts: dict[str, torch.Tensor],
                   cfg: TrainCfg) -> torch.Tensor:
    """Weighted sum.

    Returns a tensor and deliberately does not take any scalar: ``float()``
    forces a CUDA synchronisation, and this runs before ``backward()``, where a
    sync would serialise the forward and backward passes. The logging scalars
    are read by the caller on the steps that actually write a log row.
    """
    total = cfg.lambda_fm * l_fm
    if "well" in parts:
        total = total + cfg.lambda_well * parts["well"]
    return total
