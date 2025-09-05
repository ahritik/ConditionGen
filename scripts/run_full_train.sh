#!/usr/bin/env bash
# Train ConditionGen on out/tuar_npz (6-class). Optionally launch TensorBoard.

set -euo pipefail

# -------- Config (override by exporting env vars before running) -------------
export PYTHONPATH="${PYTHONPATH:-}:$PWD"

DATA_BASE=${DATA_BASE:-"out/tuar_npz"}
TRAIN_DIR=${TRAIN_DIR:-"$DATA_BASE/train"}
VAL_DIR=${VAL_DIR:-"$DATA_BASE/val"}   # optional; used only if exists

LOG_DIR=${LOG_DIR:-"out/runs/condgen_tuar_6cls"}
EPOCHS=${EPOCHS:-50}
BATCH=${BATCH:-256}
LR=${LR:-2e-4}
TIMESTEPS=${TIMESTEPS:-1000}
NO_AMP=${NO_AMP:-0}          # set to 1 to disable AMP

# TensorBoard helper
RUN_TB=${RUN_TB:-0}          # set to 1 to auto-start TB (if installed)
TB_PORT=${TB_PORT:-6006}

# -----------------------------------------------------------------------------
echo "[train] TRAIN_DIR=$TRAIN_DIR"
[ -d "$TRAIN_DIR" ] || { echo "[train] ERROR: $TRAIN_DIR not found"; exit 1; }

VAL_ARGS=()
if [ -d "$VAL_DIR" ]; then
  VAL_ARGS+=(--val_npz_dir "$VAL_DIR")
  echo "[train] VAL_DIR=$VAL_DIR"
else
  echo "[train] VAL_DIR missing — training without validation."
fi

mkdir -p "$LOG_DIR"

# 1) Launch TensorBoard (optional) — only if requested and installed
if [ "$RUN_TB" = "1" ] && command -v tensorboard >/dev/null 2>&1; then
  echo "[train] Starting TensorBoard on port $TB_PORT (will use $LOG_DIR/tb if present)"
  # Run in background; suppress noisy output
  ( tensorboard --logdir "$LOG_DIR/tb" --port "$TB_PORT" >/dev/null 2>&1 & echo $! > "$LOG_DIR/tb.pid" ) || true
  echo "[train] → http://localhost:$TB_PORT"
fi

# 2) Train
CMD=(python train.py
  --npz_dir "$TRAIN_DIR"
  --log_dir "$LOG_DIR"
  --epochs "$EPOCHS"
  --batch "$BATCH"
  --lr "$LR"
  --timesteps "$TIMESTEPS"
)

# AMP toggle
if [ "$NO_AMP" = "1" ]; then
  CMD+=(--no_amp)
fi

# Add validation args if present
CMD+=("${VAL_ARGS[@]}")

echo "[train] Running: ${CMD[*]}"
"${CMD[@]}"

echo "[train] Done."
echo "[train] Checkpoints: $LOG_DIR/checkpoints/"
echo "[train] CSV log:     $LOG_DIR/train_log.csv"
echo "[train] TensorBoard: tensorboard --logdir \"$LOG_DIR/tb\" --port $TB_PORT"
