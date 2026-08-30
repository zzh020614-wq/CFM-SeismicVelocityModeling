"""Evaluation metrics (NumPy only).

Two families:

*Accuracy* -- MAE, RMSE, correlation and SSIM against the ground-truth interval
velocity.

*Physical consistency* -- whether the generated model still obeys the relations
that produced its conditions. A model can have a decent MAE and still be
physically inconsistent, and under weak conditioning many models are equally
admissible, so consistency is the metric that keeps meaning when MAE stops
being decisive.
"""
from __future__ import annotations

import numpy as np

from ..conditions.dix import dix_forward


def mae(a, b) -> float:
    return float(np.abs(a - b).mean())


def rmse(a, b) -> float:
    return float(np.sqrt(((a - b) ** 2).mean()))


def corr(a, b) -> float:
    af = a.ravel() - a.mean()
    bf = b.ravel() - b.mean()
    d = np.sqrt((af ** 2).sum() * (bf ** 2).sum())
    return float((af * bf).sum() / d) if d > 0 else 0.0


def ssim(a, b, data_range=4500.0, sigma=1.5, k1=0.01, k2=0.03) -> float:
    """Structural similarity (Wang et al., 2004) with Gaussian weighting.

    ``a`` and ``b`` are ``(H, W)`` physical velocities in m/s; ``data_range``
    defaults to ``v_max - v_min = 4500``. Equivalent to
    ``skimage.metrics.structural_similarity(..., gaussian_weights=True,
    sigma=1.5, use_sample_covariance=False)``; implemented here so that the
    package does not depend on scikit-image. Returns the mean SSIM over the
    image, 1.0 being identical.
    """
    from scipy.ndimage import gaussian_filter

    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)

    def f(x):
        return gaussian_filter(x, sigma, mode="nearest")

    mu_a, mu_b = f(a), f(b)
    # Local variance/covariance in E[x^2] - E[x]^2 form, matching the Gaussian window.
    saa = f(a * a) - mu_a * mu_a
    sbb = f(b * b) - mu_b * mu_b
    sab = f(a * b) - mu_a * mu_b
    c1, c2 = (k1 * data_range) ** 2, (k2 * data_range) ** 2
    num = (2 * mu_a * mu_b + c1) * (2 * sab + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2)
    return float(np.mean(num / den))


def dix_rms_consistency(gen_vint, true_vint) -> float:
    """RMSE (m/s) between the Dix forward models of the generated and true fields."""
    return rmse(dix_forward(gen_vint), dix_forward(true_vint))


def well_consistency(gen_vint, true_vint, well_mask) -> float:
    """MAE (m/s) at the well pixels: does the model honour the logs it was given?"""
    m = well_mask > 0.5
    if m.sum() == 0:
        return float("nan")
    return mae(gen_vint[m], true_vint[m])


def aggregate_by_class(records: list[dict], class_names: dict) -> dict:
    """Average every metric overall and per geological class.

    ``records`` is a list of dicts each holding ``class_id`` plus scalar metrics.
    """
    keys = [k for k in records[0] if k != "class_id"]
    out = {"overall": {}, "per_class": {}}
    for k in keys:
        vals = [r[k] for r in records if not np.isnan(r[k])]
        out["overall"][k] = float(np.mean(vals)) if vals else float("nan")
    for c in sorted({r["class_id"] for r in records}):
        sub = [r for r in records if r["class_id"] == c]
        d = {"n": len(sub), "name": class_names.get(str(c), class_names.get(c, str(c)))}
        for k in keys:
            vals = [r[k] for r in sub if not np.isnan(r[k])]
            d[k] = float(np.mean(vals)) if vals else float("nan")
        out["per_class"][int(c)] = d
    return out
