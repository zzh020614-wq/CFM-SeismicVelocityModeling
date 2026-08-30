"""Result figures for stage 5."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..training.normalize import unit_to_vel  # noqa: E402


def _image(ax, value, title, cmap, vmin=None, vmax=None):
    im = ax.imshow(value, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def save_sample_figure(true_v, generated_v, cond, class_id, class_name,
                       sample_index, out_path, metrics=None,
                       v_min=1500.0, v_max=6000.0):
    """One panel per sample: label, generation and every condition side by side.

    ``cond`` is the *normalised* tensor the dataloader actually produced,
    ``[rms, well_val, well_mask, imaging]``, so the figure shows what the
    network really saw rather than a recomputed approximation of it.
    """
    true_v = np.asarray(true_v)
    generated_v = np.asarray(generated_v)
    cond = np.asarray(cond)
    if cond.shape[0] not in (3, 4):
        raise ValueError(f"expected cond of shape (3|4, H, W), got {cond.shape}")

    rms_vel = unit_to_vel(cond[0], v_min, v_max)
    well_vel = unit_to_vel(cond[1], v_min, v_max)
    well_mask = cond[2] > 0.5
    imaging = cond[3] if cond.shape[0] > 3 else np.zeros_like(cond[2])

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    velocity_image = _image(axes[0, 0], true_v, "label: true $V_{int}$", "jet",
                            v_min, v_max)
    _image(axes[0, 1], generated_v, "generated $V_{int}$", "jet", v_min, v_max)
    _image(axes[0, 2], rms_vel, "condition: RMS velocity", "jet", v_min, v_max)

    limit = float(np.max(np.abs(imaging))) or 1.0
    imaging_image = _image(axes[1, 0], imaging, "condition: migrated image",
                           "gray", -limit, limit)
    _image(axes[1, 1], well_vel, "condition: well velocity", "jet", v_min, v_max)
    _image(axes[1, 2], well_mask.astype(np.float32), "condition: well mask",
           "gray_r", 0.0, 1.0)

    fig.colorbar(velocity_image,
                 ax=[axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 1]],
                 shrink=0.86, pad=0.02, label="velocity (m/s)")
    fig.colorbar(imaging_image, ax=axes[1, 0], shrink=0.86, pad=0.02,
                 label="normalised amplitude")

    metric_text = ""
    if metrics:
        metric_text = (f" | MAE {metrics['mae']:.1f} m/s"
                       f" | RMSE {metrics['rmse']:.1f} m/s"
                       f" | corr {metrics['corr']:.4f}")
    fig.suptitle(f"sample {sample_index:05d} | class {class_id}: {class_name}"
                 f"{metric_text}", fontsize=13)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def save_closed_loop_figure(true, gen, tau_axis, dz, nz, out_path,
                            v_min=1500.0, v_max=6000.0):
    """Generated time-domain model, its depth-domain conversion, and the truth."""
    from ..timeconv.convert import time_to_depth

    k = min(4, len(gen))
    fig, ax = plt.subplots(k, 3, figsize=(9, 2.7 * k))
    ax = np.atleast_2d(ax)
    for i in range(k):
        depth_model = time_to_depth(gen[i], tau_axis, dz, nz)
        _image(ax[i, 0], gen[i], r"generated $V_{int}(\tau)$", "jet", v_min, v_max)
        _image(ax[i, 1], depth_model, "converted to depth $V(z)$", "jet", v_min, v_max)
        _image(ax[i, 2], true[i], r"true $V_{int}(\tau)$", "jet", v_min, v_max)
    fig.suptitle("closed loop: time domain -> depth domain", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def save_diversity_figure(samples, out_path, class_id=None,
                          v_min=1500.0, v_max=6000.0):
    """Several draws from the same condition, plus the per-pixel standard deviation.

    A near-zero spread means the model has collapsed to a deterministic mapping;
    a spread far larger than the error means the conditions constrain it too
    weakly. Both are informative failures.
    """
    samples = np.asarray(samples)
    n = len(samples)
    std = samples.std(0)
    fig, ax = plt.subplots(1, n + 1, figsize=(2.4 * (n + 1), 2.6))
    for j in range(n):
        _image(ax[j], samples[j], f"draw {j}", "jet", v_min, v_max)
    im = ax[-1].imshow(std, cmap="magma", aspect="auto")
    ax[-1].set_title("per-pixel std", fontsize=8)
    ax[-1].set_xticks([])
    ax[-1].set_yticks([])
    fig.colorbar(im, ax=ax[-1], shrink=0.8)
    title = "same condition, different starting noise"
    if class_id is not None:
        title += f" (class {class_id})"
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
