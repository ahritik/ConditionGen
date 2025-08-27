# ConditionGen (TUAR-only)
Controllable diffusion for multi-channel clinical EEG artifact synthesis (TUAR).

## Quick start
1) Create a Python env (3.10+ recommended) and install deps:
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

2) Preprocess TUAR into windows (4s @ 200 Hz, 50% overlap). You need your TUAR root path and optionally a CSV of artifact segments.
- `--tuar_root`: folder containing EDF files (recursively).
- Optional `--ann_csv` format (header required):
```
edf_path,start_sec,end_sec,artifact
/path/to/rec1.edf,12.3,20.7,eye
/path/to/rec1.edf,45.0,60.0,muscle
...
```
If `--ann_csv` is omitted, the whole recording is labeled `none` (useful for smoke tests).

```bash
python data/make_windows.py   --tuar_root /PATH/TO/TUAR   --out data/tuar_4s_200hz.npz   --fs 200 --win_s 4 --overlap 0.5   --bandpass 0.5 45 --notch 60   --canon_ch Fp1 Fp2 C3 C4 P3 P4 O1 O2   --subject_regex "(?P<subject>.+?)/"   --ann_csv /PATH/TO/annotations.csv
```

3) Train diffusion:
```bash
python train.py   --data data/tuar_4s_200hz.npz   --batch 32 --lr 1e-4 --steps 120000   --widths 64 128 256 --resblocks 2   --stft_win 128 --stft_hop 64 --stft_lambda 0.1   --cfg_pdrop 0.2 --mps --amp
```

4) Sample conditioned EEG:
```bash
python sample.py   --ckpt ckpts/best.pt --n 2000   --ddim_steps 50 --cfg_scale 2.0   --artifact eye --intensity high --out samples/eye_high.npz
```

5) Evaluate specificity (condition recovery):
```bash
python eval/clf_artifact.py --data data/tuar_4s_200hz.npz --train_real
python eval/clf_artifact.py --data samples/ --eval_generated --report out/cond_recovery.json
```

6) Evaluate augmentation utility:
```bash
python eval/eval_util.py   --real data/tuar_4s_200hz.npz   --synthetic samples/ --augment_rare --report out/utility.json
```

## Notes
- MPS support on Apple Silicon is enabled with `--mps`. Use `--amp` for mixed-precision autocast on MPS.
- If your TUAR metadata format differs, adapt `data/make_windows.py` to parse your annotation files (see `parse_annotations()`).
