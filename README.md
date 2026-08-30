# tdcfm — time-domain conditional flow matching for seismic velocity model building

Reference implementation accompanying the manuscript submitted to *Computers &
Geosciences*.

The method generates a **time-domain interval velocity model** `V_int(τ, x)` on
a 256 × 256 grid from several conditions that are routinely available in a
seismic velocity-model-building workflow: a picked RMS velocity field, sparse
well logs, a migrated image, and a categorical geological-pattern label. The
generator is a conditional flow matching model with an optimal-transport
coupling, trained with an additional well consistency loss.

Everything needed to reproduce the results is here: the synthetic dataset
generator, the forward modelling of the conditions, the model definition, the
training loop, conditional sampling, and the evaluation metrics and baselines.

```
                     ┌──────────────────────────────────────────┐
  six structural     │  stage 1  depth-domain velocity model    │
  operators, five ──▶│           V(z, x), 256 × 256             │
  geological classes └────────────────────┬─────────────────────┘
                                          │ stage 2  vertical depth-to-time
                                          ▼          conversion, global τ axis
                          ┌───────────────────────────────┐
                          │  V_int(τ, x)   ← the label x₁  │
                          └───────────────┬───────────────┘
                                          │ stage 3  forward modelling
              ┌───────────────┬───────────┴────┬──────────────────┐
              ▼               ▼                ▼                  ▼
        RMS velocity    well logs +      migrated image     geological class
         (Dix, then      mask (sparse     (reflectivity        (label y)
          smoothed)       columns)         ⊛ Ricker)
              └───────────────┴────────────────┘                  │
                     concatenated as channels c                   │
                                          │                       │
                                          ▼                       ▼
                       stage 4  conditional UNet  vθ(x_t, t, c, y)
                              OT-CFM + well consistency loss
                                          │
                                          ▼  stage 5
                    metrics, baselines, closed loop back to depth
```

Every condition is derived from the *same* ground truth, so conditions and
label are self-consistent by construction and no forward-modelling mismatch has
to be accounted for.

## Installation

```bash
git clone <repository-url> && cd CFM
pip install -r requirements.txt          # or: pip install -e ".[train]"
```

Stages 1–3 need only NumPy, SciPy, h5py and Matplotlib and run on a CPU.
Stages 4–5 additionally need PyTorch, `POT` (imported as `ot`) and
`torchcfm==1.0.7`, which supplies the flow-matching objectives and the UNet
backbone. A CUDA GPU is required for training in a practical amount of time.

Verify the installation, generate a tiny dataset and exercise the whole
pipeline in about a minute:

```bash
bash scripts/smoke_test.sh
```

All commands are run from the repository root as `python -m <module>`.

## Reproducing the paper

Full commands, hardware notes and expected runtimes are in
[`docs/reproduce.md`](docs/reproduce.md). In outline:

```bash
# stage 1  synthesise depth-domain velocity models (disjoint seed ranges)
python -m tdcfm.synth.generate --n-samples 20000 --seed-base 20260630 \
    --out-dir data/stage1/train --shard-size 500
python -m tdcfm.synth.generate --n-samples 2000  --seed-base 90000000 \
    --out-dir data/stage1/val   --shard-size 500

# stage 2  depth-to-time conversion; the validation split reuses the training axis
python -m tdcfm.timeconv.run --source-dir data/stage1/train \
    --out-dir data/stage2/train --n-t 256 --shard-size 500
python -m tdcfm.timeconv.run --source-dir data/stage1/val \
    --out-dir data/stage2/val --shard-size 500 \
    --time-axis data/stage2/train/time_axis.json

# stage 3  forward-model the conditions (both splits use the training axis)
for split in train val; do
  python -m tdcfm.conditions.run --source-dir data/stage1/$split \
      --time-axis data/stage2/train/time_axis.json \
      --out-dir data/stage3/$split --shard-size 500
done

# stage 4  train
python -m tdcfm.training.train --data-dir data/stage3/train \
    --out-dir runs/train total_steps=80000 batch_size=48 amp=bf16

# stage 5  evaluate, and check against the conventional baselines
python -m tdcfm.evaluation.evaluate --ckpt runs/train/otcfm/ckpt_step_79999.pt \
    --data-dir data/stage3/val --time-axis data/stage2/train/time_axis.json \
    --n 500 --best-of 4 --ode-steps 100 --out runs/eval
python -m tdcfm.evaluation.baselines --data-dir data/stage3/val \
    --time-axis data/stage2/train/time_axis.json \
    --ckpt runs/train/otcfm/ckpt_step_79999.pt --n 64
```

`data/stage2/train/time_axis.json` is the reference file of the whole project:
every later stage and every evaluation script must be pointed at the *training*
axis, so all splits share one time scale.

## Repository layout

```
tdcfm/
  shards.py           sharded HDF5/npz dataset I/O, shared by all stages
  synth/              stage 1  six structural operators, five class recipes
  timeconv/           stage 2  depth-to-time conversion and its inverse
  conditions/         stage 3  Dix transforms, migrated image, well logs
  models/             stage 4  conditional UNet and the Euler ODE sampler
  training/           stage 4  dataloader, OT-CFM objective, well consistency loss
  inference/          stage 4  conditional sampling from a checkpoint
  evaluation/         stage 5  metrics, baselines, result figures
tests/                unit tests (57 tests, no GPU required)
docs/                 dataset description, reproduction guide, salt-body spec
scripts/smoke_test.sh end-to-end installation check on a tiny dataset
```

Data and training artefacts are written to `data/` and `runs/`, both of which
are excluded from version control.

## Design notes

Some choices are easy to misread as arbitrary; they are not, and each is
documented at the place it is implemented.

**One global time axis.** `T_max` is the 90th percentile of the total two-way
time of every trace in the dataset, fixed once and shared by all samples. A
per-sample `T_max` would give every sample its own time scale, and the network
would learn a systematically drifting depth position.

**The RMS condition, not a Dix-inverted initial model.** A Dix inversion of the
smoothed RMS field is already very close to `V_int`, so a network given it can
satisfy the flow-matching identity by copying its input — learning an identity
map rather than a generative model. Feeding the RMS field instead forces the
network to learn the inversion implicitly.

**Wells are widened to several columns.** The UNet downsamples six times, so a
single-pixel well vanishes from the deep feature maps and its constraint never
reaches them. Inside a widened band each column carries its own true value.

**Conditions enter in two different ways.** Anything with a per-pixel location
(RMS field, image, well values and mask) is concatenated as an input channel and
is deliberately *not* given a downsampling encoder, which would destroy the
pixel alignment. The geological class has no spatial extent and goes through a
label embedding added to the timestep embedding.

**The well constraint needs no ODE solve.** `L = L_FM + 0.35·L_well` acts on
`x̂₁ = x_t + (1−t)·v_θ`, which the forward pass of the flow-matching loss already
provides, so enforcing it costs essentially nothing. `L_well` is still computed
and logged when its weight is zero, so a control run stays comparable.

**The conditions are deliberately degraded.** The RMS field is smoothed to
σ = (59, 26) samples, far beyond the σ = (9, 4) written to disk, and only 4
wells of 3 columns each are given, without lateral interpolation. A lightly
smoothed RMS field is close enough to `V_int` that a network can copy it and
learn an identity map instead of a generative one.

**Baselines matter more than the metric.** Under weak conditioning many
velocity models are equally admissible and MAE stops discriminating. The model
must beat the conventional strong baselines in
`tdcfm.evaluation.baselines` — Dix inversion and lateral well interpolation —
not merely the trivial ones.

## Reproducibility

Every random parameter of a sample is determined by
`sample_seed = seed_base + global_index`, and the training and validation splits
use disjoint seed ranges, so the split is leak-free by construction and the
whole dataset can be regenerated from the commands above.

Within stage 3 each condition draws from its own seeded RNG sub-stream, so
changing the settings of one condition cannot perturb the others. Training fixes
the Python, NumPy and PyTorch seeds but does not enable CUDA's fully
deterministic mode, so GPU runs are reproducible only up to floating-point
non-determinism.

## Citation

If you use this code, please cite the accompanying paper. See
[`CITATION.cff`](CITATION.cff); update it with the final bibliographic details
once the paper is accepted.

## License

MIT — see [`LICENSE`](LICENSE).

## Acknowledgements

The flow-matching objectives and the UNet backbone come from
[`torchcfm`](https://github.com/atong01/conditional-flow-matching) (Tong et al.),
whose UNet in turn derives from OpenAI's guided-diffusion architecture; both are
MIT licensed. The synthetic-dataset paradigm follows the parametric operator
approach of OpenFWI (Deng et al., 2022, *NeurIPS Datasets and Benchmarks*);
OpenFWI released its data and benchmark training code but not its generator, so
the operators here are an independent implementation of the described method.
