#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=$PWD:$PYTHONPATH

# 0) choose the most recent checkpoint (prefers last.pt)
CKPT=""
if ls -t out/*/checkpoints/last.pt >/dev/null 2>&1; then
  CKPT="$(ls -t out/*/checkpoints/last.pt | head -n1)"
else
  CKPT="$(ls -t out/*/checkpoints/step_*.pt 2>/dev/null | head -n1 || true)"
fi
if [[ -z "${CKPT}" ]]; then
  echo "[error] No checkpoints found under out/*/checkpoints"; exit 1
fi
echo "[ckpt] ${CKPT}"

STAMP="$(date +"%Y%m%d_%H%M%S")"
RUN_DIR="out/eval_run_${STAMP}"
EVAL_DIR="out/clf_eval_${STAMP}"
mkdir -p "${RUN_DIR}" "${EVAL_DIR}"

# Make these visible to the summary step
export RUN_DIR EVAL_DIR

ARTS=(none eye muscle chewing shiver electrode movement)
ARCHS=(tiny resnet1d eegnet)

# 1) sampling (EMA from inside ckpt)
for A in "${ARTS[@]}"; do
  echo "[sample] ${A}"
  python sample.py --ckpt "${CKPT}" --use_ema \
    --n 3000 --steps 80 --guidance 1.5 \
    --artifact "${A}" --intensity 0.6 --seizure 0 --age_bin 1 --montage_id 0 \
    --out_dir "${RUN_DIR}/synth_${A}" --save_npy --batch 256 --cond_dim 13
done

# 2) fidelity: PSD + Cov/ACF
for A in "${ARTS[@]}"; do
  FDIR="${RUN_DIR}/synth_${A}"
  echo "[eval:psd/cov_acf] ${A}"
  python -m eval.psd --real_dir out/npz --fake_dir "${FDIR}" --split test \
    --out "${EVAL_DIR}/psd_${A}.json"
  python -m eval.cov_acf --real_dir out/npz --fake_dir "${FDIR}" --split test \
    --out "${EVAL_DIR}/covacf_${A}.json"
done

# 3) specificity: classifier recovery (3 backbones)
for A in "${ARTS[@]}"; do
  FDIR="${RUN_DIR}/synth_${A}"
  for ARCH in "${ARCHS[@]}"; do
    echo "[eval:recovery] ${A} | ${ARCH}"
    python -m eval.classifier_eval --real_dir out/npz --fake_dir "${FDIR}" \
      --task artifact --arch "${ARCH}" \
      --out "${EVAL_DIR}/recovery_${A}_${ARCH}.json" --recovery_only
  done
done

# 4) utility: augmentation gains (two harder classes)
for A in shiver electrode; do
  FDIR="${RUN_DIR}/synth_${A}"
  for ARCH in "${ARCHS[@]}"; do
    echo "[eval:augment] ${A} | ${ARCH}"
    python -m eval.classifier_eval --real_dir out/npz --fake_dir "${FDIR}" \
      --task artifact --arch "${ARCH}" --augment "${A}" \
      --out "${EVAL_DIR}/augment_gain_${A}_${ARCH}.json"
  done
done

# 5) extra metrics (FFD/MMD/kNN/1-NN)
for A in "${ARTS[@]}"; do
  FDIR="${RUN_DIR}/synth_${A}"
  echo "[eval:extra] ${A}"
  python -m eval.extra_metrics --real_dir out/npz --fake_dir "${FDIR}" --split test \
    --out "${EVAL_DIR}/extra_${A}.json"
done

# 6) summary tables for paper
python - <<'PY'
import os, json
RUN_DIR=os.environ["RUN_DIR"]; EVAL_DIR=os.environ["EVAL_DIR"]
arts=["none","eye","muscle","chewing","shiver","electrode","movement"]
archs=["tiny","resnet1d","eegnet"]
def J(p):
  try:
    with open(p,"r") as f: return json.load(f)
  except: return {}
lines=[]
lines+=["# Table 1 — Fidelity",
        "| Artifact | Δδ | Δθ | Δα | Δβ | Cov Fro ↓ | ACF L2 ↓ | n_fake |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
for A in arts:
  psd=J(os.path.join(EVAL_DIR,f"psd_{A}.json"))
  cov=J(os.path.join(EVAL_DIR,f"covacf_{A}.json"))
  bre=psd.get("band_rel_err",{})
  fmt=lambda v: ("nan" if v is None else f"{float(v):.3f}")
  lines.append(f"| {A} | {fmt(bre.get('delta',0))} | {fmt(bre.get('theta',0))} | {fmt(bre.get('alpha',0))} | {fmt(bre.get('beta',0))} | {fmt(cov.get('cov_fro',0))} | {int(cov.get('acf_l2',0))} | {psd.get('n_fake',0)} |")
lines+=["","## Table 2 — Specificity (recovery)"]
for A in arts:
  lines.append(f"### {A}")
  for arch in archs:
    r=J(os.path.join(EVAL_DIR,f"recovery_{A}_{arch}.json")).get("recovery",{})
    lines.append(f"- **{arch}**: F1={r.get('macro_f1',0):.3f}, Acc={r.get('acc',0):.3f}, IM={(r.get('intended_match',0) or 0):.3f}, n_fake={r.get('n_fake',0)}")
  lines.append("")
lines+=["## Table 3 — Utility (augmentation gains)"]
for A in ["electrode","shiver"]:
  for arch in archs:
    j=J(os.path.join(EVAL_DIR,f"augment_gain_{A}_{arch}.json"))
    aug=j.get("augmentation",{})
    if aug:
      lines.append(f"- **{A}** ({arch}): ΔF1={aug.get('delta_macro_f1',0):+.3f}, ΔAcc={aug.get('delta_acc',0):+.3f} (n_train_aug={aug.get('n_train_aug',0)})")
lines+=["","## Extra Metrics (features) — per artifact",
        "| Artifact | FFD ↓ | MMD (RBF) ↓ | kNN-Prec ↑ | kNN-Rec ↑ | 1-NN Acc → 0.5 |",
        "|---|---:|---:|---:|---:|---:|"]
for A in arts:
  e=J(os.path.join(EVAL_DIR,f"extra_${A}.json".replace("${A}",A)))
  lines.append(f"| {A} | {e.get('ffd',e.get('FID_like',0.0)):.3f} | {e.get('mmd_rbf',e.get('mmd',0.0)):.4f} | {e.get('knn_prec',0.0):.3f} | {e.get('knn_rec',0.0):.3f} | {e.get('one_nn_acc',e.get('nn1_acc',0.0)):.3f} |")
outp=os.path.join(EVAL_DIR,"summary_tables.md")
open(outp,"w").write("\n".join(lines))
print("Wrote", outp)
PY

echo
echo "============================"
echo "DONE."
echo "Samples:   ${RUN_DIR}"
echo "Eval out:  ${EVAL_DIR}"
echo "Summary:   ${EVAL_DIR}/summary_tables.md"
echo "============================"
