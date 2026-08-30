#!/usr/bin/env bash
# End-to-end smoke test of the data pipeline on a tiny dataset.
#
# Stages 1-3 are NumPy/SciPy only and run on a CPU in about a minute. If
# PyTorch and torchcfm are installed, a very short overfitting training run and
# one sampling pass are appended, so the whole pipeline is exercised.
#
#   bash scripts/smoke_test.sh [output_directory]
#
# Nothing here reproduces the paper; it verifies that the installation works.
set -euo pipefail

OUT="${1:-runs/smoke}"
N_TRAIN=40
N_VAL=20
SHARD=20

cd "$(dirname "$0")/.."
echo "==> smoke test output: ${OUT}"

echo "==> unit tests"
python -m unittest discover -s tests

echo "==> stage 1: synthetic depth-domain velocity models"
python -m tdcfm.synth.generate --n-samples "${N_TRAIN}" --seed-base 20260630 \
    --out-dir "${OUT}/data/stage1/train" --shard-size "${SHARD}"
python -m tdcfm.synth.generate --n-samples "${N_VAL}" --seed-base 90000000 \
    --out-dir "${OUT}/data/stage1/val" --shard-size "${SHARD}"

echo "==> stage 2: depth-to-time conversion"
python -m tdcfm.timeconv.run --source-dir "${OUT}/data/stage1/train" \
    --out-dir "${OUT}/data/stage2/train" --n-t 256 --shard-size "${SHARD}"
# The validation split must reuse the training time axis, never recompute it.
python -m tdcfm.timeconv.run --source-dir "${OUT}/data/stage1/val" \
    --out-dir "${OUT}/data/stage2/val" --shard-size "${SHARD}" \
    --time-axis "${OUT}/data/stage2/train/time_axis.json"

echo "==> stage 3: condition forward modelling"
for split in train val; do
    python -m tdcfm.conditions.run --source-dir "${OUT}/data/stage1/${split}" \
        --time-axis "${OUT}/data/stage2/train/time_axis.json" \
        --out-dir "${OUT}/data/stage3/${split}" --shard-size "${SHARD}"
done

if ! python -c "import torch, torchcfm" 2>/dev/null; then
    echo "==> PyTorch/torchcfm not installed: stopping after stage 3."
    echo "    Install them with: pip install -r requirements.txt"
    exit 0
fi

echo "==> stage 4: short overfitting training run"
python -m tdcfm.training.train \
    --data-dir "${OUT}/data/stage3/train" --out-dir "${OUT}/train" \
    overfit=8 batch_size=2 total_steps=20 save_step=10 num_workers=0 \
    ode_steps=5 sample_n=2 amp=off

CKPT=$(ls -t "${OUT}"/train/otcfm/ckpt_step_*.pt | head -1)
echo "==> stage 4: sampling from ${CKPT}"
python -m tdcfm.inference.sample --ckpt "${CKPT}" \
    --data-dir "${OUT}/data/stage3/val" --n 2 --ode-steps 5 \
    --out "${OUT}/samples"

echo "==> stage 5: evaluation and baselines"
python -m tdcfm.evaluation.evaluate --ckpt "${CKPT}" \
    --data-dir "${OUT}/data/stage3/val" \
    --time-axis "${OUT}/data/stage2/train/time_axis.json" \
    --n 4 --best-of 2 --ode-steps 5 --batch-size 2 --out "${OUT}/eval"
python -m tdcfm.evaluation.baselines --data-dir "${OUT}/data/stage3/val" \
    --time-axis "${OUT}/data/stage2/train/time_axis.json" \
    --ckpt "${CKPT}" --n 4

echo "==> smoke test finished; artefacts in ${OUT}"
