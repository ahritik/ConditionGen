#!/usr/bin/env bash
# Convert TUAR -> NPZ (6-class), then run EDA tables + visualizations.
# Produces: out/tuar_npz/{train,val,test} + per-split summary.json files.

set -euo pipefail

# -------- Config (override by exporting env vars before running) -------------
export PYTHONPATH="${PYTHONPATH:-}:$PWD"

TUAR_ROOT=${TUAR_ROOT:-"/Users/hritikarasu/Developer/TUAR"}
OUT_DIR=${OUT_DIR:-"out/tuar_npz"}

FS=${FS:-200}
WIN_SEC=${WIN_SEC:-4.0}
OVERLAP=${OVERLAP:-0.5}
BANDPASS_LO=${BANDPASS_LO:-0.5}
BANDPASS_HI=${BANDPASS_HI:-45.0}
NOTCH=${NOTCH:-60}
MONTAGE_ID=${MONTAGE_ID:-0}

# -----------------------------------------------------------------------------
echo "[split] TUAR_ROOT=$TUAR_ROOT"
echo "[split] OUT_DIR=$OUT_DIR"
mkdir -p "$OUT_DIR"

# 1) Convert TUAR -> NPZ (writes label_map.json w/ 6 classes)
python data/make_windows.py \
  --tuar_root "$TUAR_ROOT" \
  --out_dir   "$OUT_DIR" \
  --fs "$FS" --win_sec "$WIN_SEC" --overlap "$OVERLAP" \
  --bandpass "$BANDPASS_LO" "$BANDPASS_HI" --notch "$NOTCH" \
  --montage_id "$MONTAGE_ID"

# 2) EDA tables + per-split summary.json
if [ -d "$OUT_DIR" ]; then
  python utils/eda.py --base "$OUT_DIR"
else
  echo "[split] ERROR: OUT_DIR missing after conversion"; exit 1
fi

# 3) Visualizations
python utils/eda_viz.py --base "$OUT_DIR" || true

# 4) Quick sanity check
python - <<'PY'
import os, glob, json, numpy as np
base = os.environ.get("OUT_DIR","out/tuar_npz")
print("[sanity] base:", base)
lm = os.path.join(base, "label_map.json")
if os.path.exists(lm):
    names = json.load(open(lm)).get("artifact_names", [])
    print("[sanity] label_map:", names)
else:
    print("[sanity] WARNING: label_map.json not found")
for split in ("train","val","test"):
    d = os.path.join(base, split)
    if not os.path.isdir(d): 
        print(f"[sanity] skip {split} (missing)")
        continue
    uniq = []
    for fp in glob.glob(os.path.join(d, "*.npz")):
        with np.load(fp, allow_pickle=True) as z:
            a = z["artifact"] if "artifact" in z else z["y_artifact"]
            uniq.append(np.unique(a))
    if uniq:
        u = np.unique(np.concatenate(uniq))
        print(f"[sanity] {split}: unique artifact ids {u}, min={u.min() if u.size else None}, max={u.max() if u.size else None}")
        assert u.size == 0 or (u.min() >= 0 and u.max() <= 5), "artifact id outside 0..5"
PY

echo "[split] Done. NPZ at $OUT_DIR"
echo "[split] EDA tables in $OUT_DIR/eda/<split>/* and summary.json in each split dir."
