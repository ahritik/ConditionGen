# ConditionGen: Controllable Diffusion for Clinical EEG (TUAR ecosystem)

**Working title:** ConditionGen: Controllable Diffusion for Clinical EEG (TUAR ecosystem) — GenAI4Health @ NeurIPS

A compact, fast diffusion model that generates multi‑channel EEG (canonical 8‑ch: `Fp1,Fp2,C3,C4,P3,P4,O1,O2`) with explicit controls:
- Artifact type & intensity
- Seizure flag
- Age bin
- Montage id

We evaluate fidelity (PSD/ACF/cov), condition specificity (classifier recovery), and utility (augmentation boosts for downstream classifiers on rare/noisy slices).

## Repo layout

```
conditiongen_tueg/
  data/
    loaders_tuar_tusz.py
    make_windows.py
  models/
    unet1d_film.py
    diffusion.py
    conditioning.py
  eval/
    psd.py
    cov_acf.py
    classifier_eval.py
  utils/
    constants.py
  scripts/
    eda_tuar.py
    eda_npz.py
    eda_report.py
    plot_training.py
    list_checkpoints.py
    smoke_test.sh
    smoke_test_real.sh
    check_mps.py
    make_mock_dataset.py
  figs/
  paper/
  train.py
  sample.py
  requirements.txt
```

## Quickstart

> Tested on Apple Silicon (M-series) with PyTorch MPS; also supports CUDA/CPU.

### 0) Install
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 1) (Option A) Prepare NPZ windows from TUAR
The script expects a TUAR folder where EDFs sit next to per‑record CSV label files.
It canonicalizes channels to 8‑ch, does 4s@200Hz windows with 50% overlap, z‑scores, and writes `.npz` shards.

```bash
python data/make_windows.py \
  --tuar_root /path/to/TUAR \
  --out_dir out/npz \
  --fs 200 --win_sec 4 --overlap 0.5 \
  --bandpass 0.5 45 --notch 60 \
  --montage_id 0
```

**Output structure** (example):
```
out/npz/
  train_000.npz
  train_001.npz
  val_000.npz
  test_000.npz
  meta.json         # split seeds, channel map, normalization stats
```

### 1b) (Option B) Mock dataset for smoke tests
```bash
python scripts/make_mock_dataset.py --out_dir out/mock_npz --n 1024
```

### 2) EDA
```bash
python scripts/eda_npz.py --npz_dir out/npz --out_dir out/eda
python scripts/eda_report.py --eda_json out/eda/npz_summary.json --to_html
```

### 3) Train ConditionGen
```bash
python train.py \
  --npz_dir out/npz \
  --log_dir out/condgen \
  --batch 32 --steps 150000 \
  --stft_win 128 --stft_hop 64 \
  --lr 1e-4 --lambda_stft 0.1 \
  --log_tb \
  --ckpt_every 5000
```

Resume:
```bash
python train.py --npz_dir out/npz --log_dir out/condgen --resume out/condgen/checkpoints/step_050000_ema.pt
```

### 4) Sample
```bash
python sample.py \
  --ckpt out/condgen/checkpoints/step_050000_ema.pt \
  --n 32 --steps 20 --use_ema \
  --artifact eye --intensity 0.8 \
  --seizure 0 --age_bin 1 --montage_id 0 \
  --out_dir out/samples_eye08
```

### 5) Evaluation (fidelity + specificity + utility)
Fidelity:
```bash
python eval/psd.py --real_dir out/npz --fake_dir out/samples_eye08 --out out/metrics_psd.json
python eval/cov_acf.py --real_dir out/npz --fake_dir out/samples_eye08 --out out/metrics_cov_acf.json
```

Specificity (does generated data reflect intended labels?):
```bash
python eval/classifier_eval.py \
  --real_dir out/npz \
  --fake_dir out/samples_eye08 \
  --task artifact \
  --out out/clf_recovery_artifact.json
```

Utility (augmentation):
```bash
python eval/classifier_eval.py \
  --real_dir out/npz \
  --task artifact \
  --augment_with out/samples_eye08 \
  --out out/augment_gain_artifact.json
```

### 6) Figures & Reporting
```bash
python scripts/plot_training.py --log_csv out/condgen/train_log.csv --out_png figs/training_curves.png
python scripts/eda_report.py --eda_json out/eda/npz_summary.json --to_html --out_html out/eda/EDA_Report.html
```

## Notes

- Mixed precision is enabled automatically on Apple MPS (`--amp_mps` is on by default).
- All constants/label maps are centralized in `utils/constants.py`.
- Checkpoints contain model, optimizer, EMA state, conditioning embedding state, and step.
- Baselines (VAE, WGAN-GP, TimeGAN) are suggested for completeness; this repo focuses on the diffusion baseline. You can plug your baselines into the same NPZ pipeline.

## License
Apache-2.0 (replace if needed).
