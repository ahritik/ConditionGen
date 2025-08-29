#!/usr/bin/env bash
set -e
python -V
python scripts/make_mock_dataset.py --out_dir out/mock_npz --n 2048
python train.py --npz_dir out/mock_npz --log_dir out/mock_run --batch 16 --steps 200 --ckpt_every 100
python sample.py --ckpt out/mock_run/checkpoints/step_000100_ema.pt --n 16 --steps 10 --use_ema --out_dir out/mock_samples
python eval/psd.py --real_dir out/mock_npz --fake_dir out/mock_samples --out out/mock_psd.json
python eval/cov_acf.py --real_dir out/mock_npz --fake_dir out/mock_samples --out out/mock_covacf.json
python eval/classifier_eval.py --real_dir out/mock_npz --task artifact --augment_with out/mock_samples --out out/mock_util.json
