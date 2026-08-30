# Reproduction guide

Run every command from the repository root, as `python -m <module>`.

## 0. Environment

```bash
pip install -r requirements.txt
python -c "import numpy, scipy, h5py, matplotlib; print('stages 1-3 OK')"
python -c "import torch, torchvision, tqdm, ot, torchcfm; print('stages 4-5 OK')"
python -m unittest discover -s tests            # expect: Ran 57 tests, OK
```

`ot` is the import name of the `POT` package and is required by the
optimal-transport coupling. `torchcfm` supplies both the flow-matching
objectives and the UNet backbone.

A quick end-to-end check on a tiny dataset, which also runs the unit tests:

```bash
bash scripts/smoke_test.sh
```

Approximate resources for the full pipeline: stages 1–3 take a few hours of CPU
time and produce roughly 60 GB of HDF5 shards for 22,000 samples; stage 4 takes
about a day on a single modern data-centre GPU at 80k steps.

## 1. Generate the dataset

The two splits must use **disjoint seed ranges** — that is what makes the split
leak-free.

```bash
python -m tdcfm.synth.generate --n-samples 20000 --seed-base 20260630 \
    --out-dir data/stage1/train --shard-size 500 --fmt hdf5
python -m tdcfm.synth.generate --n-samples 2000 --seed-base 90000000 \
    --out-dir data/stage1/val --shard-size 500 --fmt hdf5
```

Useful flags: `--classes 0,3,4` restricts the class subset; `--no-compress`
trades disk space for write speed.

## 2. Depth-to-time conversion

The training split derives the global time axis. **Every other split must reuse
it**, or the splits end up on different time scales.

```bash
python -m tdcfm.timeconv.run --source-dir data/stage1/train \
    --out-dir data/stage2/train --n-t 256 --shard-size 500 --fmt hdf5

python -m tdcfm.timeconv.run --source-dir data/stage1/val \
    --out-dir data/stage2/val --shard-size 500 --fmt hdf5 \
    --time-axis data/stage2/train/time_axis.json
```

`data/stage2/train/time_axis.json` is the reference file of the entire project.
Every later stage and evaluation script is pointed at it.

## 3. Forward-model the conditions

Both splits use the **training** time axis.

```bash
for split in train val; do
  python -m tdcfm.conditions.run --source-dir data/stage1/$split \
      --time-axis data/stage2/train/time_axis.json \
      --out-dir data/stage3/$split --shard-size 500 --fmt hdf5
done
```

Each shard holds `target (N, 256, 256)`, `cond (N, 4, 256, 256)`,
`class_id (N,)` and `seed (N,)`. The condition channels are
`[rms, well_val, well_mask, imaging]`.

Note that the dataloader can strengthen the RMS smoothing and change the well
count and width without regenerating anything (see
[`dataset.md`](dataset.md#condition-channels)). Only a change to the *baseline*
smoothing, the imaging parameters or the grid requires rerunning this stage.

## 4. Train

Single GPU:

```bash
python -m tdcfm.training.train \
    --data-dir data/stage3/train --out-dir runs/train \
    total_steps=80000 batch_size=48 amp=bf16
```

Multiple GPUs with DDP — `batch_size` is **per GPU**, so the global batch is
`batch_size × number of GPUs`:

```bash
torchrun --standalone --nproc_per_node=2 --module tdcfm.training.train \
    --data-dir data/stage3/train --out-dir runs/train \
    total_steps=80000 batch_size=24 amp=bf16
```

Resume with `--resume runs/train/otcfm/ckpt_step_30000.pt`.

Outputs land in `runs/train/otcfm/`:

* `ckpt_step_*.pt` — the newest `keep_last_n` checkpoints only; each stores the
  training weights, the EMA weights, the optimiser state and the full
  configuration
* `sample_step_*.png` — ground truth on the top row, generated below
* `loss_log.csv` — `step, loss, loss_ema, lr, ema_decay, l_well, l_total`,
  with the all-zero-prediction baseline recorded in the header comment

### Configuration overrides

Any `TrainCfg` field can be overridden as a trailing `key=value` argument:

```
lambda_well=0                          the unconstrained control run
rms_smooth_tau=18 rms_smooth_x=8       a less degraded RMS condition
n_wells=10 well_width=5                a richer well condition
well_interp=True                       fill laterally between the wells (see below)
use_imaging=True                       keep the migrated-image channel
overfit=8 total_steps=500              small-sample overfitting smoke test
```

Two points about the defaults are worth stating explicitly.

**The conditions are deliberately degraded.** `rms_smooth = (59, 26)` samples is
far beyond the σ = (9, 4) written to disk, and there are only 4 wells of 3
columns each, without lateral interpolation. A lightly smoothed RMS field is
close enough to the answer that a network can copy it rather than learn to
generate, and `well_interp=True` with many wide wells fills most of the section
with true velocities, which makes the well channel close to the answer too. Any
experiment that measures how much the model actually relies on the wells — a
sweep over `lambda_well` in particular — must keep `well_interp=False`,
otherwise the result is a foregone conclusion.

**`lambda_well` defaults to 0.35**, so the well constraint is active out of the
box. It is still computed and written to `loss_log.csv` when set to zero, so the
control run (`lambda_well=0`) stays directly comparable with the constrained one
during training rather than only after a full evaluation.

### Reading the loss curve

The header of `loss_log.csv` records the `L_FM` a network that always predicts
zero would achieve. Two failure modes bracket the useful range: a loss stuck at
that baseline means the network outputs zero and has learned nothing; a loss
approaching zero means a condition is leaking the answer and the network has
found a shortcut rather than learned to generate.

## 5. Evaluate

```bash
python -m tdcfm.evaluation.evaluate \
    --ckpt runs/train/otcfm/ckpt_step_79999.pt \
    --data-dir data/stage3/val --time-axis data/stage2/train/time_axis.json \
    --n 500 --best-of 4 --ode-steps 100 --batch-size 8 --out runs/eval
```

Produces `report.json` (MAE, RMSE, correlation, SSIM, Dix/RMS consistency and
well consistency, overall and per geological class), per-sample figures, the
closed-loop time-to-depth figure, and a diversity figure showing several draws
from one condition.

Then the baselines the model has to beat:

```bash
python -m tdcfm.evaluation.baselines --data-dir data/stage3/val \
    --time-axis data/stage2/train/time_axis.json \
    --ckpt runs/train/otcfm/ckpt_step_79999.pt --n 64 --model-mae <model MAE>
```

The table separates the conditions used unchanged (group A, a lower reference
only) from the conditions after the conventional processing a practitioner would
apply — Dix inversion, lateral well interpolation, and their average (group B).
Only an improvement over the strongest group-B baseline is a real contribution.

To draw samples without computing metrics:

```bash
python -m tdcfm.inference.sample --ckpt runs/train/otcfm/ckpt_step_79999.pt \
    --data-dir data/stage3/val --n 8 --ode-steps 100 --out runs/samples
```

## Conventions worth remembering

1. Run everything from the repository root with `python -m`.
2. Scripts taking `--ckpt` restore the conditioning setup from the checkpoint.
   Never re-specify it by hand: evaluating a model under settings other than the
   ones it was trained with fails silently and produces plausible, wrong numbers.
3. `--weights` selects `net` (the training weights, the default) or `ema`. The
   EMA lags badly before it has warmed up, so an early checkpoint evaluated with
   `ema` understates the model.
4. `--time-axis` is always `data/stage2/train/time_axis.json`, for every split.
5. Controlled comparisons should fix both the sample set and the starting noise
   `x0`, so the only difference between two runs is the variable under study.
