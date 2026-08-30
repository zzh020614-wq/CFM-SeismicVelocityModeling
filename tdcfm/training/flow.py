"""Flow-matching batch construction and target-side condition alignment.

Plain conditional flow matching leaves the batch order of ``x1`` untouched.
The optimal-transport variant does not: it resamples the ``(x0, x1)`` pairs
according to a minibatch OT plan, so ``x_t`` no longer corresponds row for row
to the ``cond`` and ``y`` that came out of the dataloader. The conditions must
be reordered by the *same* target-side indices, otherwise every sample is
trained against another sample's conditions -- an error that does not raise and
does not obviously show up in the loss curve. ``tests/test_flow_alignment.py``
covers both branches.
"""
from __future__ import annotations


def align_target_conditions(cond, y, target_indices):
    """Reorder the spatial conditions and the class label by the target indices."""
    return cond[target_indices], y[target_indices]


def _sample_otcfm_batch(fm, x0, x1, cond, y, target_indices):
    """Call the torchcfm guided interface and align ``cond``/``y`` with its output.

    ``target_indices`` starts as ``[0, 1, ..., B-1]``. torchcfm resamples this
    array with the same plan it applies to ``x1``, so the returned copy records
    exactly which original target each ``x_t``/``u_t`` row came from.
    """
    t, xt, ut, _, sampled_target_indices = (
        fm.guided_sample_location_and_conditional_flow(
            x0,
            x1,
            y0=None,
            y1=target_indices,
        )
    )
    sampled_target_indices = sampled_target_indices.to(
        device=cond.device,
        dtype=y.dtype,
    )
    cond_aligned, y_aligned = align_target_conditions(
        cond,
        y,
        sampled_target_indices,
    )
    return t, xt, ut, cond_aligned, y_aligned, sampled_target_indices


def sample_flow_batch(fm, model_name, x0, x1, cond, y):
    """Draw one flow-matching training state with conditions aligned to it.

    Returns ``(t, xt, ut, cond_aligned, y_aligned, sampled_target_indices)``.
    """
    if model_name == "otcfm":
        # Inherit dtype and device from y instead of relying on a global device.
        target_indices = y.new_tensor(list(range(x1.shape[0])))
        return _sample_otcfm_batch(fm, x0, x1, cond, y, target_indices)

    t, xt, ut = fm.sample_location_and_conditional_flow(x0, x1)
    target_indices = y.new_tensor(list(range(x1.shape[0])))
    return t, xt, ut, cond, y, target_indices
