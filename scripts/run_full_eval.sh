#!/usr/bin/env bash
set -euo pipefail

# ---------- Config (override via env) ----------
REAL_DIR="${REAL_DIR:-out/npz}"
CKPT="${CKPT:-out/condgen/checkpoints/step_145000_ema.pt}"
RUN_DIR="${RUN_DIR:-out/eval_run_$(date +%Y%m%d_%H%M%S)}"
EVAL_DIR="${EVAL_DIR:-out/clf_eval_$(date +%Y%m%d_%H%M%S)}"

USE_EMA="${USE_EMA:-1}"
STEPS="${STEPS:-80}"
GUIDANCE="${GUIDANCE:-1.5}"
BATCH="${BATCH:-256}"
N_PER="${N_PER:-3000}"

# fixed artifact set for TUAR (no 'movement')
ARTS=("none" "eye" "muscle" "chewing" "shiver" "electrode")

# knobs
FORCE_RESAMPLE="${FORCE_RESAMPLE:-0}"
FORCE_REEVAL="${FORCE_REEVAL:-0}"
DO_EXTRA="${DO_EXTRA:-1}"

echo "[run] CKPT=${CKPT}"
echo "[run] RUN_DIR=${RUN_DIR}"
echo "[run] EVAL_DIR=${EVAL_DIR}"

mkdir -p "${RUN_DIR}" "${EVAL_DIR}"

# ---------- 0) Verify dataset ----------
if [ ! -d "${REAL_DIR}" ]; then
  echo "ERROR: REAL_DIR not found: ${REAL_DIR}" >&2
  exit 2
fi

# ---------- 1) Sample ----------
for A in "${ARTS[@]}"; do
  OUTD="${RUN_DIR}/synth_${A}"
  if [[ "${FORCE_RESAMPLE}" -eq 1 ]] || [[ ! -f "${OUTD}/samples.npy" ]]; then
    echo "[sample] ${A}"
    python sample.py \
      --ckpt "${CKPT}" $([ "${USE_EMA}" = "1" ] && echo --use_ema ) \
      --n "${N_PER}" --steps "${STEPS}" --guidance "${GUIDANCE}" \
      --artifact "${A}" --intensity 0.6 --seizure 0 --age_bin 1 --montage_id 0 \
      --out_dir "${OUTD}" --save_npy --batch "${BATCH}" --cond_dim 13 --widths 64 128 256
  else
    echo "[sample] ${A} exists -> skip"
  fi
done

# ---------- 2) Fidelity (PSD + Cov/ACF) ----------
for A in "${ARTS[@]}"; do
  FAKED="${RUN_DIR}/synth_${A}"
  # PSD
  OP="${EVAL_DIR}/psd_${A}.json"
  if [[ "${FORCE_REEVAL}" -eq 1 ]] || [[ ! -f "${OP}" ]]; then
    python -m eval.psd_covacf --mode psd --real_dir "${REAL_DIR}" --fake_dir "${FAKED}" --out "${OP}" || true
  fi
  # Cov-ACF
  OP="${EVAL_DIR}/covacf_${A}.json"
  if [[ "${FORCE_REEVAL}" -eq 1 ]] || [[ ! -f "${OP}" ]]; then
    python -m eval.psd_covacf --mode covacf --real_dir "${REAL_DIR}" --fake_dir "${FAKED}" --out "${OP}" || true
  fi
done

# ---------- 3) Recovery + Aug gains ----------
for arch in tiny resnet1d eegnet; do
  for A in "${ARTS[@]}"; do
    OUTP="${EVAL_DIR}/recovery_${A}_${arch}.json"
    if [[ "${FORCE_REEVAL}" -eq 1 ]] || [[ ! -f "${OUTP}" ]]; then
      echo "[recovery] ${arch} | ${A}"
      python -m eval.classifier_eval \
        --real_dir "${REAL_DIR}" \
        --fake_dir "${RUN_DIR}/synth_${A}" \
        --augment_with "${A}" \
        --task artifact --arch "${arch}" \
        --epochs 8 --batch 256 --lr 1e-3 \
        --label_key y_artifact --tqdm \
        --out "${OUTP}"
    fi
  done
done

# also write augmentation deltas explicitly for (electrode, shiver)
for A in electrode shiver; do
  for arch in tiny resnet1d eegnet; do
    # The previous step already wrote augmentation block into recovery_*; nothing more needed.
    :
  done
done

# ---------- 4) Extra metrics ----------
if [[ "${DO_EXTRA}" -eq 1 ]]; then
  for A in "${ARTS[@]}"; do
    OP="${EVAL_DIR}/extra_${A}.json"
    if [[ "${FORCE_REEVAL}" -eq 1 ]] || [[ ! -f "${OP}" ]]; then
      python -m eval.extra_metrics \
        --real_dir "${REAL_DIR}" \
        --fake_dir "${RUN_DIR}/synth_${A}" \
        --feature_kind stat92 \
        --out "${OP}" || true
    fi
  done
fi

# ---------- 5) Summary (markdown) ----------
python - <<'PY'
import os, json, glob
E = os.environ["EVAL_DIR"]
R = os.environ["RUN_DIR"]
ARTS = ["none","eye","muscle","chewing","shiver","electrode"]
def J(p):
    try:
        with open(p,"r") as f: return json.load(f)
    except Exception: return None

lines = [ "# Classifier + Fidelity Summary", f"- RUN_DIR: {R}", f"- EVAL_DIR: {E}", "", "# Table 1 — Fidelity", "| Artifact | Δδ | Δθ | Δα | Δβ | Cov Fro ↓ | ACF L2 ↓ | n_fake |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
for a in ARTS:
    psd = J(os.path.join(E, f"psd_{a}.json")) or {}
    cov = J(os.path.join(E, f"covacf_{a}.json")) or {}
    n_fake = cov.get("n_fake", psd.get("n_fake", 0))
    dd, dt, da, db = (psd.get("delta_delta",0.0), psd.get("delta_theta",0.0), psd.get("delta_alpha",0.0), psd.get("delta_beta",0.0))
    lines.append(f"| {a} | {dd:.3f} | {dt:.3f} | {da:.3f} | {db:.3f} | {cov.get('cov_fro',0.0):.3f} | {cov.get('acf_l2',0):.0f} | {n_fake} |")

lines += ["", "## Table 2 — Specificity (recovery)"]
for a in ARTS:
    lines += [f"### {a}"]
    for arch in ["tiny","resnet1d","eegnet"]:
        p = os.path.join(E, f"recovery_{a}_{arch}.json")
        j = J(p) or {}
        rec = j.get("recovery", {})
        base = j.get("baseline", {})
        im  = rec.get("intended_match", 0.0) if rec else 0.0
        nfk = rec.get("n_fake", 0) if rec else 0
        lines.append(f"- **{arch}**: F1={base.get('macro_f1',0.0):.3f}, Acc={base.get('acc',0.0):.3f}, IM={im:.3f}, n_fake={nfk}")
    lines.append("")

lines += ["", "## Table 3 — Utility (augmentation gains)"]
for a in ["electrode","shiver"]:
    for arch in ["tiny","resnet1d","eegnet"]:
        p = os.path.join(E, f"recovery_{a}_{arch}.json")
        j = J(p) or {}
        aug = j.get("augmentation", None)
        if aug:
            lines.append(f"- **{a}** ({arch}): ΔF1={aug.get('delta_macro_f1',0.0):+0.3f}, ΔAcc={aug.get('delta_acc',0.0):+0.3f} (n_train_aug={aug.get('n_train_aug',0)})")

# Extra metrics table (optional)
extra = []
for a in ARTS:
    p = os.path.join(E, f"extra_{a}.json")
    j = J(p)
    if j:
        extra.append((a, j))
if extra:
    lines += ["", "## Extra Metrics (features) — per artifact", "| Artifact | FFD ↓ | MMD (RBF) ↓ | kNN-Prec ↑ | kNN-Rec ↑ | 1-NN Acc → 0.5 |", "|---|---:|---:|---:|---:|---:|"]
    for a,j in extra:
        lines.append(f"| {a} | {j.get('ffd',0.0):.3f} | {j.get('mmd_rbf',0.0):.4f} | {j.get('knn_precision',0.0):.3f} | {j.get('knn_recall',0.0):.3f} | {j.get('nn_two_sample_acc',0.0):.3f} |")

outp = os.path.join(E, "summary.md")
open(outp, "w").write("\n".join(lines))
print("Wrote", outp)
PY

echo "[run] Done."