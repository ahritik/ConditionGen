#!/usr/bin/env bash
set -e

TUAR_ROOT=${1:-"/Users/hritikarasu/Developer/TUAR"}
echo "=== ConditionGen real TUAR smoke ==="
echo "Using TUAR_ROOT=$TUAR_ROOT"

mkdir -p out/real

# EDA of TUAR
python scripts/eda_tuar.py --tuar_root "$TUAR_ROOT" --out_json out/eda/tuar_summary.json

# Windows
python data/make_windows.py --tuar_root "$TUAR_ROOT" --out_npz out/real/train.npz --fs 200 --win_sec 4.0 --overlap 0.5

# NPZ EDA
python scripts/eda_npz.py --npz out/real/train.npz --out_dir out/eda

# Report
python scripts/eda_report.py --tuar_json out/eda/tuar_summary.json --npz_json out/eda/npz_summary.json --out_md out/eda/EDA_Report.md --to_html

# Train (short)
python train.py --npz out/real/train.npz --out_dir out/real_run --steps 1000 --ckpt_every 500 --log_tb

# Sample
CKPT=$(ls out/real_run/ckpt_*.pt | tail -n1)
python sample.py --ckpt "$CKPT" --n 8 --length 800 --channels 8 --steps 20 --sampler heun2 --out_npz out/real/samples.npz --use_ema

echo "Real smoke complete."
