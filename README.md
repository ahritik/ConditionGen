# ConditionGen (TUAR) — Controllable Diffusion for Clinical EEG

Small, fast 1D diffusion for multi‑channel EEG with explicit controls (artifact type/intensity, seizure flag, age bin, montage).

## Features
- Depthwise‑separable 1D UNet with FiLM conditioning
- Cosine schedule, **v‑prediction**, SNR‑weighted loss
- **Heun2** and **DDIM** samplers (15–20 steps fast path)
- TUAR → canonical 8‑ch windows (4s @ 200Hz, 50% overlap), z‑score, channel mask
- Centralized constants/mappings to avoid drift
- Training with AMP (CUDA/MPS), EMA, CSV + optional TensorBoard
- EDA + Markdown/HTML report
- Fidelity + specificity eval (PSD bands, covariance/ACF, tiny artifact classifier)
- Smoke tests (mock + real TUAR)

## Install
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Data prep (TUAR → NPZ)
```bash
python data/make_windows.py --tuar_root /path/to/TUAR --out_npz out/real/train.npz --fs 200 --win_sec 4.0 --overlap 0.5
python scripts/eda_npz.py --npz out/real/train.npz --out_dir out/eda
python scripts/eda_report.py --tuar_json out/eda/tuar_summary.json --npz_json out/eda/npz_summary.json --out_md out/eda/EDA_Report.md --to_html
```

> **Note:** `make_windows.py` expects a per‑EDF CSV next to each EDF with at least `start_time, stop_time, label`. Adjust columns as needed.

## Train
```bash
python train.py --npz out/real/train.npz --out_dir out/run1 --steps 100000 --ckpt_every 5000 --log_tb
```

## Sample
```bash
CKPT=out/run1/ckpt_100000.pt
python sample.py --ckpt "$CKPT" --n 16 --length 800 --channels 8 --steps 20 --sampler heun2 --out_npz out/samples.npz --use_ema \
  --artifact muscle --intensity 0.8 --seizure 0 --age_bin 2 --montage canon8
```

## Evaluate
- **Fidelity (PSD bands):**
```bash
python -c "import numpy as np; from eval.psd import psd_band_errors; d=np.load('out/real/train.npz'); s=np.load('out/samples.npz'); print(psd_band_errors(d['x'][:16], s['x'][:16], fs=200))"
```
- **Covariance/ACF distance:**
```bash
python -c "import numpy as np; from eval.cov_acf import channel_covariance_distance, acf_distance; d=np.load('out/real/train.npz'); s=np.load('out/samples.npz'); print(channel_covariance_distance(d['x'][:16], s['x'][:16])); print(acf_distance(d['x'][:16], s['x'][:16]))"
```
- **Condition recovery (tiny CNN):**
```bash
python eval/clf_artifact.py --real_npz out/real/train.npz --synth_npz out/samples.npz
```

## Figures to include (paper)
- Fig 1: model + conditioning diagram (export from `models/*` description or your own schematic)
- Fig 2: PSD overlays (use `scripts/eda_npz.py` as a starting point)
- Fig 3: artifact‑strength ladder (vary `--intensity` in `sample.py`)
- Tbl 1: fidelity metrics. Tbl 2: condition recovery accuracy. Tbl 3: augmentation gains.

## Smoke tests
- **Mock:** `bash scripts/smoke_test.sh`
- **Real TUAR:** `bash scripts/smoke_test_real.sh /path/to/TUAR`

## Notes
- If training is slow: shrink widths to `48,96,192`, keep steps modest, rely on utility improvements + spectral metrics.
- EMA is applied correctly in validation/sampling via `--use_ema` flag.
- MPS autocast is supported on Apple Silicon; use `python scripts/check_mps.py` to verify.

## Repo
```
conditiongen_tueg/
  data/ loaders_tuar_tusz.py  make_windows.py
  models/ unet1d_film.py  diffusion.py  conditioning.py
  eval/ psd.py  cov_acf.py  clf_artifact.py
  train.py  sample.py
  scripts/ eda_tuar.py  eda_npz.py  eda_report.py  plot_training.py  list_checkpoints.py  smoke_test.sh  smoke_test_real.sh  check_mps.py  make_mock_dataset.py
  utils/ constants.py
  figs/ paper/
```
