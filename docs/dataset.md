# Dataset

The dataset is fully synthetic. No field data is used, and every sample is
regenerable from its seed with the commands in
[`reproduce.md`](reproduce.md).

## Construction

The dataset follows the parametric-operator paradigm of OpenFWI (Deng et al.,
2022): rather than drawing on field surveys, velocity models are produced by
composing a small set of controllable structural operators with randomised
parameters. As in OpenFWI, samples are organised into families by structural
style and drawn in balanced proportions. The differences are that OpenFWI
targets full-waveform inversion, with depth-domain velocity models as labels and
recorded wavefields as input, whereas here the label is the **time-domain
interval velocity** `V_int(τ, x)` and the input is a set of conditions forward
modelled from that same label.

Construction proceeds in three stages.

**Stage 1 — depth-domain ground truth.** A layered background with a compaction
trend `v = v₀ + kz` is the base. The class recipe then applies, in order,
dip/wedge, folding (a vertical warp of superposed sine harmonics), Gaussian
smoothing, faulting (rigid throw along the fault plane by integer index shifts,
so the plane stays sharp) and a salt body (a rounded polygon or a perturbed
dome, filled at constant velocity with a hard edge). Velocities are finally
clipped to [1500, 6000] m/s. The Gaussian smoothing sits deliberately *before*
faulting and salt emplacement, so that it softens only the continuous
sedimentary background and leaves the structural discontinuities sharp.

**Stage 2 — vertical depth-to-time conversion.** Two-way traveltime is
accumulated per trace as `τ(z) = 2∫dz/v` and every trace is resampled onto a
single global time axis. Its upper limit `T_max` is the 90th percentile of the
total two-way time over all traces of the whole dataset — one fixed value shared
by every sample, with `n_t = 256`. Deriving `T_max` per sample would give each
sample its own time scale and the network would learn a systematically drifting
depth position. The conversion ignores ray bending, so `V_int` is an interval
velocity in the vertical two-way-time sense.

**Stage 3 — condition forward modelling.** From the time-domain truth: the RMS
velocity field by Dix forward modelling followed by Gaussian smoothing; the well
condition by extracting columns of the truth at the well positions; and the
migrated image by convolving the reflectivity
`r = (vᵢ₊₁ − vᵢ)/(vᵢ₊₁ + vᵢ)` with a Ricker wavelet of 25–30 Hz dominant
frequency plus band-limited noise.

## Splits

22,000 samples in total, split 10:1 into 20,000 for training and 2,000 for
validation. The two seed ranges are disjoint (training 20260630–20280629,
validation 90000000–90001999) and every random parameter of a sample is
determined solely by its seed, so there is no leakage of any kind and the split
is exactly reproducible. Both splits are class-balanced: 4,000 per class for
training and 400 per class for validation.

## Key parameters

| Item | Value |
|---|---|
| Grid | nz × nx = 256 × 256 |
| Depth sampling `dz` | 16 m (depth range ≈ 4096 m) |
| Trace spacing `dx` | 25 m (width ≈ 6400 m) |
| Velocity range | [1500, 6000] m/s |
| Time samples `n_t` | 256 |
| Time axis limit `T_max` | 90th percentile of all traces' `τ_max`, global and fixed |
| Training set | 20,000 (4,000 per class) |
| Validation set | 2,000 (400 per class) |
| Training seed range | 20260630 – 20280629 |
| Validation seed range | 90000000 – 90001999 |
| Storage | HDF5 shards (gzip level 4), 500 samples per shard |

## Geological classes

| id | Name | Operators |
|---|---|---|
| 0 | `near_horizontal` | layered background + compaction + Gaussian smoothing |
| 1 | `dip_wedge` | background + dip/wedge (apparent dip ±12°) |
| 2 | `fold` | background + folding (1–3 harmonics, amplitude 40–160 m) |
| 3 | `fault` | background + gentle dip + faulting (1–3 faults, throw 40–200 m) |
| 4 | `salt` | background + salt body (constant velocity, hard edge; hardest class) |

## Operator parameter ranges

| Operator | Main parameters |
|---|---|
| Layered background + compaction | 4–8 layers; surface velocity `v₀` 1600–2200 m/s; gradient `k` 0.3–0.9 (m/s)/m; interface contrast 150–600 m/s (85% positive) |
| Dip / wedge | apparent dip −12° to +12°; wedge probability 0.4 |
| Folding | 1–3 harmonics; amplitude 40–160 m; wavelength 0.3–1.0 × model width; depth decay 0.4 |
| Gaussian smoothing | applied before faulting and salt, so only the continuous background is softened |
| Faulting | 1–3 faults; throw 40–200 m; nearest-neighbour index shift, no interpolation |
| Salt body | mid-section rounded polygon or bottom-rooted dome; constant velocity, hard edge |

Assembly order:

```
layered background + compaction → dip/fold → Gaussian smoothing → faulting → salt → clip[1500, 6000]
```

The specific numerical ranges are engineering settings chosen to span a
plausible range of structural styles; the operator forms themselves follow
standard structural-geology kinematics and the linear velocity–depth relation
(Slotnick, 1936).

## Condition channels

Stage 3 writes four channels per sample, in this order:

| Channel | Content |
|---|---|
| 0 | RMS velocity field: Dix forward model of the truth, Gaussian smoothed with σ = (9, 4) samples |
| 1 | Well velocities: 3–8 single columns of the truth at random positions (placeholder, see below) |
| 2 | Well mask, binary |
| 3 | Migrated image: reflectivity ⊛ Ricker (25–30 Hz) plus band-limited noise |

The training dataloader rewrites two of these, so their strength can be changed
without regenerating the dataset:

* **RMS smoothing** is additive in variance, `σ_total² = σ_base² + σ_extra²`, so
  the loader only applies the difference between the target σ and the σ = (9, 4)
  already on disk. Smoothing cannot be undone, so a target below the baseline
  falls back to the baseline with a warning. The training configuration used
  here is σ = (59, 26) samples, i.e. 702 ms × 650 m.
* **The well condition** is rebuilt from the label and the stage-3 channels are
  discarded. Depth-to-time conversion is purely vertical and per trace, so the
  time-domain log at trace `c` is exactly column `c` of the label; the number and
  width of the wells are therefore free parameters at training time. The
  configuration used here is 4 wells of 3 columns at stratified random positions,
  without lateral interpolation between them.

Lateral interpolation between the wells is off deliberately. These velocity
models are laterally smooth, so linearly interpolating even 4 wells covering
4.7% of the section reconstructs the truth to about 44 m/s MAE — close enough to
the answer that any conclusion about how much the network relies on the wells
would be meaningless.

## Salt body specification

See [`salt_body.md`](salt_body.md).
