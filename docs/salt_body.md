# Salt body specification (class 4)

Used by class 4 only; no other class places an anomalous body. Implemented in
`tdcfm/synth/operators.py` (`generate_salt`, `_random_polygon`,
`_dome_perturbed`, `_clean`, `_off_top_sides`) and configured by
`SaltCfg` in `tdcfm/synth/config.py`. One body per sample.

## Layouts

Two layouts of equal probability, each with its own normalised bounding region
`(z0, z1, x0, x1)` as fractions of the model:

| Layout | Region | Shape method |
|---|---|---|
| `mid_flat` — mid-section salt sheet | (0.40, 0.60, 0.10, 0.90) | rounded random polygon |
| `bottom` — bottom-rooted dome | (0.60, 1.00, 0.15, 0.85) | perturbed semi-elliptical dome |

The shape is constructed inside a box occupying a fraction of that region:
`mid_flat` (0.78, 0.78), `bottom` (width 0.90, height 0.80). The box is built
directly at the target aspect ratio rather than being stretched afterwards,
which would distort the outline.

## `mid_flat` — rounded random polygon

* 14–22 vertices at sorted random angles, radii drawn from [0.80, 0.98] of the
  half-box.
* The radius sequence is smoothed circularly with a width-5 box kernel, repeated
  2–3 times. This rounds the corners into the lobate outline of a salt sheet
  instead of leaving a star-shaped polygon.

## `bottom` — perturbed dome

* Base profile `sqrt(clip(1 − ((x − center)/width)², 0))`, with
  `center ∈ [0.40, 0.60] · box_width` and `width ∈ [0.42, 0.55] · box_width`.
* A low-frequency perturbation of harmonics 1–5 with amplitude ≈ 0.05–0.16 / k
  is added; the height is clipped to [0, 1].
* The crest is placed at `top = (bh − 1) − height · (bh − 1) · U(0.78, 0.92)`;
  everything at or below `top` is salt, giving a wide base and a rounded crown.
* A Gaussian of σ = 0.8 followed by a 0.5 threshold softens the outline
  without rounding it away.

## Placement

* Lateral position is random inside the region.
* `bottom` is pinned to the base of its region (`pz = z1 − h`); `mid_flat` is
  vertically centred in its region.
* The body is eroded (up to 6 iterations) until it no longer touches the top or
  side boundaries. The bottom edge is exempt for the `bottom` layout, where the
  salt is rooted at the base of the model by design.

## Cleanup and filling

* After every construction step the largest connected component is kept and
  holes are filled, so the body is always solid and simply connected.
* The body is filled with a constant 5000 m/s. The default hard edge is
  intentional: it is the sharp velocity contrast that makes this the hardest
  class. Optional Gaussian feathering of the boundary exists but is off by
  default.

Every quantity above is a configurable field of `SaltCfg`.

## A note on reproducibility

When the layout probabilities are uniform, `generate_salt` calls
`rng.choice(names)` without the `p` argument. Passing `p` consumes the random
stream differently, and the released dataset was generated on the no-`p` branch,
so the branch is kept to preserve seed-level reproducibility. `p` is only used
when genuinely non-uniform probabilities are configured.
