#!/usr/bin/env bash
# Full pipeline: sample -> fidelity -> recovery+augmentation -> extras -> summary
# Works with 6 artifacts: none, eye, muscle, chewing, shiver, electrode

set -euo pipefail

# ---- Config (env overrides welcome) ----
export PYTHONPATH="${PYTHONPATH:-}:$PWD"

REAL_DIR=${REAL_DIR:-out/npz}
RUN_DIR=${RUN_DIR:-out/eval_run}
EVAL_DIR=${EVAL_DIR:-out/clf_eval}

CKPT=${CKPT:?CKPT is required (path to .pt)}
USE_EMA=${USE_EMA:-1}               # 1 or 0
STEPS=${STEPS:-150}
GUIDANCE=${GUIDANCE:-1.0}
N=${N:-3000}
BATCH_SAMP=${BATCH_SAMP:-256}
COND_DIM=${COND_DIM:-13}
# widths for the UNet (must match training)
WIDTHS_STR=${WIDTHS_STR:-"64 128 256"}

FORCE_RESAMPLE=${FORCE_RESAMPLE:-0} # 1: re-sample all
FORCE_REEVAL=${FORCE_REEVAL:-0}     # 1: re-run PSD/Cov-ACF
DO_EXTRA=${DO_EXTRA:-1}             # 1: extra metrics

CLF_EPOCHS=${CLF_EPOCHS:-8}
CLF_BATCH=${CLF_BATCH:-256}
CLF_LR=${CLF_LR:-1e-3}
LABEL_KEY=${LABEL_KEY:-y_artifact}

ARTS="none eye muscle chewing shiver electrode"

mkdir -p "$RUN_DIR" "$EVAL_DIR"

echo "[run] CKPT=$CKPT"
echo "[run] RUN_DIR=$RUN_DIR"
echo "[run] EVAL_DIR=$EVAL_DIR"

# ---- Step 0: Sampling ----
if [[ "$FORCE_RESAMPLE" == "1" ]]; then
  for A in $ARTS; do
    OUT_A="$RUN_DIR/synth_${A}"
    mkdir -p "$OUT_A"
    echo "[sample] $A"
    python sample.py \
      --ckpt "$CKPT" $( [[ "$USE_EMA" == "1" ]] && echo --use_ema ) \
      --n "$N" --steps "$STEPS" --guidance "$GUIDANCE" \
      --artifact "$A" --intensity 0.6 --seizure 0 --age_bin 1 --montage_id 0 \
      --out_dir "$OUT_A" --save_npy --batch "$BATCH_SAMP" \
      --cond_dim "$COND_DIM" --widths $WIDTHS_STR
  done
else
  echo "[sample] Skipped (FORCE_RESAMPLE=0)"
fi

# ---- Step 1: Fidelity (PSD + Cov/ACF) ----
if [[ "$FORCE_REEVAL" == "1" ]]; then
  for A in $ARTS; do
    echo "[fidelity] PSD | $A"
    python -m eval.psd --real_dir "$REAL_DIR" --fake_dir "$RUN_DIR/synth_${A}" \
      --out "$EVAL_DIR/psd_${A}.json"

    echo "[fidelity] Cov/ACF | $A"
    python -m eval.cov_acf --real_dir "$REAL_DIR" --fake_dir "$RUN_DIR/synth_${A}" \
      --out "$EVAL_DIR/covacf_${A}.json"
  done
else
  echo "[fidelity] Skipped (FORCE_REEVAL=0)"
fi

# ---- Step 2: Recovery + Augmentation (writes one JSON per arch/artifact) ----
for arch in tiny resnet1d eegnet; do
  for A in $ARTS; do
    echo "[recovery+aug] $arch | $A"
    OUT_REC="$EVAL_DIR/recovery_${A}_${arch}.json"
    python -m eval.classifier_eval \
      --real_dir "$REAL_DIR" \
      --fake_dir "$RUN_DIR/synth_${A}" \
      --augment_artifact "$A" \
      --task artifact --arch "$arch" \
      --epochs "$CLF_EPOCHS" --batch "$CLF_BATCH" --lr "$CLF_LR" \
      --label_key "$LABEL_KEY" --tqdm \
      --out "$OUT_REC"

    # Optional compatibility file for older summary scripts:
    cp -f "$OUT_REC" "$EVAL_DIR/augment_gain_${A}_${arch}.json" || true
  done
done

# ---- Step 3: Extra metrics (optional) ----
if [[ "$DO_EXTRA" == "1" ]]; then
  for A in $ARTS; do
    echo "[extra] stat92 | $A"
    python -m eval.extra_metrics \
      --real_dir "$REAL_DIR" --fake_dir "$RUN_DIR/synth_${A}" \
      --feature_kind stat92 --out "$EVAL_DIR/extra_${A}_stat92.json"

    echo "[extra] spec32 | $A"
    python -m eval.extra_metrics \
      --real_dir "$REAL_DIR" --fake_dir "$RUN_DIR/synth_${A}" \
      --feature_kind spec32 --out "$EVAL_DIR/extra_${A}_spec32.json"
  done
fi

# ---- Step 4: Summary ----
python - <<'PY'
import os, json, glob, numpy as np
E=os.environ.get("EVAL_DIR","out/clf_eval")
R=os.environ.get("RUN_DIR","out/eval_run")
arts=["none","eye","muscle","chewing","shiver","electrode"]
archs=["tiny","resnet1d","eegnet"]

def J(p):
  try:
    with open(p) as f: return json.load(f)
  except: return None

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
  n_fake = 0
  for cand in ("samples.npy","samples_post.npy","x.npy"):
    fp=f"{R}/synth_{A}/{cand}"
    if os.path.exists(fp):
      n_fake = int(np.load(fp, mmap_mode="r").shape[0]); break
  lines.append(f"| {A} | {dd:.3f} | {dt:.3f} | {da:.3f} | {db:.3f} | {fro:.3f} | {acf:.0f} | {n_fake} |")

lines+=["", "## Table 2 — Specificity (recovery)"]
for A in arts:
  lines.append(f"### {A}")
  for arch in archs:
    j = J(f"{E}/recovery_{A}_{arch}.json") or {}
    base = j.get("baseline", {})
    rec  = j.get("recovery", {})
    if base and rec:
      f1  = base.get("macro_f1", 0.0)
      acc = base.get("acc", 0.0)
      im  = rec.get("intended_match", 0.0) or 0.0
      lines.append(f"- **{arch}**: F1={f1:.3f}, Acc={acc:.3f}, IM={im:.3f}, n_fake={rec.get('n_fake',0)}")
  lines.append("")

lines+=["", "## Table 3 — Utility (augmentation gains)"]
for A in arts:
  for arch in archs:
    j = J(f"{E}/recovery_{A}_{arch}.json") or {}
    aug = j.get("augmentation", {})
    base= j.get("baseline", {})
    if aug and base:
      df1  = aug.get("delta_macro_f1", 0.0)
      dacc = aug.get("delta_acc", 0.0)
      ntr  = aug.get("n_train_aug", 0)
      lines.append(f"- **{A}** ({arch}): ΔF1={df1:+.3f}, ΔAcc={dacc:+.3f} (n_train_aug={ntr})")

with open(f"{E}/summary.md","w") as f:
  f.write("\n".join(lines))
print(f"Wrote {E}/summary.md")
PY

echo "[done] All stages complete. See $EVAL_DIR"