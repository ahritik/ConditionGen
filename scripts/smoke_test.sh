#!/usr/bin/env bash
set -e

echo "=== ConditionGen mock smoke test ==="
ROOT=$(pwd)

# Make a tiny mock dataset (NPZ) with random signals
python scripts/make_mock_dataset.py --out_npz out/mock/train.npz --N 256 --C 8 --T 800

# Train for a few steps
python train.py --npz out/mock/train.npz --out_dir out/mock_run --steps 200 --ckpt_every 100 --log_tb

# Sample
CKPT=$(ls out/mock_run/ckpt_*.pt | tail -n1)
python sample.py --ckpt "$CKPT" --n 8 --length 800 --channels 8 --steps 10 --sampler heun2 --out_npz out/mock/samples.npz --use_ema

# Eval PSD (mock vs samples just for sanity)
python - <<'PY'
import numpy as np
from eval.psd import psd_band_errors
r = np.load("out/mock/train.npz")
s = np.load("out/mock/samples.npz")
real = r["x"][:8]
synth = s["x"][:8]
print(psd_band_errors(real, synth, fs=200))
PY

echo "Smoke test complete."
