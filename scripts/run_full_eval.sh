#!/usr/bin/env bash
# Full pipeline: sample -> fidelity -> recovery -> extras -> summary
# TUAR taxonomy: none, eye, muscle, chewing, shiver, electrode (6 classes)

set -euo pipefail

# ---- Config (env overrides welcome) ----
export PYTHONPATH="${PYTHONPATH:-}:$PWD"

# Real data base (expects {train,val,test} under here)
REAL_BASE=${REAL_BASE:-out/tuar_npz}
REAL_TRAIN=${REAL_TRAIN:-"$REAL_BASE/train"}
REAL_VAL=${REAL_VAL:-"$REAL_BASE/val"}
REAL_TEST=${REAL_TEST:-"$REAL_BASE/test"}

RUN_DIR=${RUN_DIR:-out/eval_run}
EVAL_DIR=${EVAL_DIR:-out/clf_eval}

CKPT=${CKPT:?CKPT is required (path to .pt checkpoint)}
USE_EMA=${USE_EMA:-0}            # kept for compatibility (not used by sample.py)
STEPS=${STEPS:-80}
GUIDANCE=${GUIDANCE:-1.0}
ETA=${ETA:-0.0}
N=${N:-3000}
BATCH_SAMP=${BATCH_SAMP:-256}

# Optional sample shape (defaults are 8x800 inside sample.py)
SHAPE_C=${SHAPE_C:-8}
SHAPE_T=${SHAPE_T:-800}

FORCE_RESAMPLE=${FORCE_RESAMPLE:-0} # 1: re-sample all
FORCE_REEVAL=${FORCE_REEVAL:-0}     # 1: re-run PSD/Cov-ACF
DO_EXTRA=${DO_EXTRA:-1}             # 1: extra metrics

# Simple classifier settings (matches classifier_eval.py we shipped)
CLF_EPOCHS=${CLF_EPOCHS:-8}
CLF_BATCH=${CLF_BATCH:-256}
CLF_LR=${CLF_LR:-1e-3}

ARTS="none eye muscle chewing shiver electrode"

mkdir -p "$RUN_DIR" "$EVAL_DIR"

echo "[run] CKPT=$CKPT"
echo "[run] REAL_BASE=$REAL_BASE"
echo "[run] RUN_DIR=$RUN_DIR"
echo "[run] EVAL_DIR=$EVAL_DIR"

# ---- Step 0: Sampling ----
if [[ "$FORCE_RESAMPLE" == "1" ]]; then
  for A in $ARTS; do
    OUT_A="$RUN_DIR/synth_${A}"
    mkdir -p "$OUT_A"
    echo "[sample] $A"
    python sample.py \
      --ckpt "$CKPT" \
      --artifact "$A" \
      --n "$N" --steps "$STEPS" --guidance "$GUIDANCE" --eta "$ETA" \
      --seizure 0 --age_bin 1 --montage_id 0 \
      --shape "$SHAPE_C" "$SHAPE_T" \
      --out_dir "$OUT_A" --save_npz
  done
else
  echo "[sample] Skipped (FORCE_RESAMPLE=0)"
fi

# ---- Step 1: Fidelity (PSD + Cov/ACF) ----
# Uses REAL_TEST by default (unbiased against training set)
if [[ "$FORCE_REEVAL" == "1" ]]; then
  for A in $ARTS; do
    echo "[fidelity] PSD | $A"
    python psd.py \
      --real_dir "$REAL_TEST" \
      --fake_dir "$RUN_DIR/synth_${A}" \
      --out "$EVAL_DIR/psd_${A}.json"

    echo "[fidelity] Cov/ACF | $A"
    python cov_acf.py \
      --real_dir "$REAL_TEST" \
      --fake_dir "$RUN_DIR/synth_${A}" \
      --out "$EVAL_DIR/covacf_${A}.json"
  done
else
  echo "[fidelity] Skipped (FORCE_REEVAL=0)"
fi

# ---- Step 2: Recovery (single reference classifier) ----
# Our classifier_eval.py expects explicit train/val/test dirs (+ optional fake_dir)
for A in $ARTS; do
  echo "[recovery] $A"
  python classifier_eval.py \
    --real_train "$REAL_TRAIN" \
    --real_val   "$REAL_VAL" \
    --real_test  "$REAL_TEST" \
    --fake_dir   "$RUN_DIR/synth_${A}" \
    --out        "$EVAL_DIR/recovery_${A}.json"
done

# ---- Step 3: Extra metrics (optional) ----
if [[ "$DO_EXTRA" == "1" ]]; then
  for A in $ARTS; do
    echo "[extra] stat92 | $A"
    python extra_metrics.py \
      --real_dir "$REAL_TEST" --fake_dir "$RUN_DIR/synth_${A}" \
      --feature_kind stat92 --out "$EVAL_DIR/extra_${A}_stat92.json"

    echo "[extra] spec32 | $A"
    python extra_metrics.py \
      --real_dir "$REAL_TEST" --fake_dir "$RUN_DIR/synth_${A}" \
      --feature_kind spec32 --out "$EVAL_DIR/extra_${A}_spec32.json"
  done
fi

# ---- Step 4: Summary ----
python - <<'PY'
import os, json, glob, numpy as np
E=os.environ.get("EVAL_DIR","out/clf_eval")
R=os.environ.get("RUN_DIR","out/eval_run")
arts=["none","eye","muscle","chewing","shiver","electrode"]

def J(p):
  try:
    with open(p) as f: return json.load(f)
  except: return None

def count_fakes(folder: str) -> int:
  # Prefer the meta JSON written by sample.py; fallback to inspecting arrays
  metas = glob.glob(os.path.join(folder, "*.json"))
  for mf in metas:
    try:
      j=json.load(open(mf))
      if "n" in j: return int(j["n"])
    except: pass
  # Otherwise, look for any npz/npy and count rows
  for npz in glob.glob(os.path.join(folder, "*.npz")):
    try:
      with np.load(npz, allow_pickle=True) as z:
        # our sample saves 'x' in the npz
        if "x" in z: return int(z["x"].shape[0])
    except: pass
  for npy in glob.glob(os.path.join(folder, "*.npy")):
    try:
      arr=np.load(npy, mmap_mode="r"); return int(arr.shape[0])
    except: pass
  return 0

lines=[]
lines+=["# Classifier + Fidelity Summary",
         f"- RUN_DIR: {R}",
         f"- EVAL_DIR: {E}",
         "",
         "# Table 1 — Fidelity",
         "| Artifact | Δδ | Δθ | Δα | Δβ | Cov Fro ↓ | ACF L2 ↓ | n_fake |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]

for A in arts:
  psd = J(f"{E}/psd_{A}.json") or {}
  cov = J(f"{E}/covacf_{A}.json") or {}
  dd  = psd.get("delta_delta",0.0)
  dt  = psd.get("delta_theta",0.0)
  da  = psd.get("delta_alpha",0.0)
  db  = psd.get("delta_beta",0.0)
  fro = float(cov.get("cov_fro",0.0))
  acf = float(cov.get("acf_l2",0.0))
  n_fake = count_fakes(os.path.join(R, f"synth_{A}"))
  lines.append(f"| {A} | {dd:.3f} | {dt:.3f} | {da:.3f} | {db:.3f} | {fro:.3f} | {acf:.0f} | {n_fake} |")

lines+=["", "## Table 2 — Specificity (recovery)"]
for A in arts:
  j = J(f"{E}/recovery_{A}.json") or {}
  base = j.get("baseline", {})
  rec  = j.get("recovery", {})
  if base:
    f1  = base.get("macro_f1", 0.0)
    acc = base.get("acc", 0.0)
    lines.append(f"- **{A}**: baseline F1={f1:.3f}, Acc={acc:.3f}")
  if rec:
    im  = rec.get("intended_match", 0.0) or 0.0
    nfk = rec.get("n_fake", 0)
    lines.append(f"  - Recovery: IM={im:.3f}, n_fake={nfk}")

with open(f"{E}/summary.md","w") as f:
  f.write("\n".join(lines))
print(f"Wrote {E}/summary.md")
PY

echo "[done] All stages complete. See $EVAL_DIR"
