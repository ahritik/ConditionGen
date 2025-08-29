#!/usr/bin/env bash
set -e
TUAR_ROOT=${1:-/Users/hritikarasu/Developer/TUAR}
python data/make_windows.py --tuar_root "$TUAR_ROOT" --out_dir out/npz --fs 200 --win_sec 4 --overlap 0.5 --bandpass 0.5 45 --notch 60 --montage_id 0
python scripts/eda_npz.py --npz_dir out/npz --out_dir out/eda
python train.py --npz_dir out/npz --log_dir out/condgen --batch 32 --steps 10000 --ckpt_every 2000 --log_tb
python sample.py --ckpt out/condgen/checkpoints/step_008000_ema.pt --n 64 --steps 20 --use_ema --artifact eye --intensity 0.8 --seizure 0 --age_bin 1 --montage_id 0 --out_dir out/samples_eye08
python eval/psd.py --real_dir out/npz --fake_dir out/samples_eye08 --out out/metrics_psd.json
python eval/cov_acf.py --real_dir out/npz --fake_dir out/samples_eye08 --out out/metrics_cov_acf.json
python eval/classifier_eval.py --real_dir out/npz --task artifact --augment_with out/samples_eye08 --out out/augment_gain_artifact.json
