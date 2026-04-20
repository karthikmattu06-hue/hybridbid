#!/bin/bash
# Template: experiments/run_experiment.sh
# Copy to experiments/<experiment_name>.sh, edit the CONFIG block.
# The LOCKED section must not be changed.
#
# Usage:
#   cp experiments/run_experiment.sh experiments/v592_base.sh
#   # Edit CONFIG block below
#   bash experiments/v592_base.sh

set -euo pipefail

# ============================================================
# CONFIG — agent-editable
# ============================================================
EXPERIMENT_NAME="v592_base"
TOTAL_STEPS=500000
CHECKPOINT_DIR="checkpoints/${EXPERIMENT_NAME}"
LOG_FILE="logs/${EXPERIMENT_NAME}.log"

# Extra args passed to train_stage1. Add any hyperparam overrides here.
# Example: "--lr_actor 1e-4 --hidden_dim 512"
TRAIN_EXTRA_ARGS=""

# Use --v60 flag if this experiment uses Stage1V60Config (enriched obs).
# Leave empty for standard Stage1Config.
EVAL_CONFIG_FLAG=""
# ============================================================

# ============================================================
# LOCKED — do not edit below this line
# ============================================================
cd "$(dirname "$0")/.."

mkdir -p "${CHECKPOINT_DIR}" logs

echo "============================================================"
echo "Experiment : ${EXPERIMENT_NAME}"
echo "Steps      : ${TOTAL_STEPS}"
echo "Checkpoint : ${CHECKPOINT_DIR}"
echo "Log        : ${LOG_FILE}"
echo "Started    : $(date)"
echo "============================================================"

# Launch training
python -u -m src.training.train_stage1 \
    --experiment_name "${EXPERIMENT_NAME}" \
    --total_steps "${TOTAL_STEPS}" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    ${TRAIN_EXTRA_ARGS} \
    2>&1 | tee "${LOG_FILE}"

# Evaluate final checkpoint
FINAL_CKPT=$(ls -t "${CHECKPOINT_DIR}"/*.pt | head -1)
echo ""
echo "--- Evaluating ${FINAL_CKPT} ---"

python -u -m experiments.prepare \
    --checkpoint "${FINAL_CKPT}" \
    --experiment "${EXPERIMENT_NAME}" \
    ${EVAL_CONFIG_FLAG} \
    2>&1 | tee -a "${LOG_FILE}"

echo "Finished: $(date)"
