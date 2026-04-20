#!/bin/bash
# Smoke test for Stage 1 Tier 1 v1 — runs 5k steps on M4, validates all 8 changes.
set -e
cd ~/hybridbid

LOG=logs/smoke_tier1.log
mkdir -p logs checkpoints/stage1_tier1_v1

echo "=== Stage 1 Tier 1 v1 Smoke Test (5k steps) ===" | tee $LOG
date | tee -a $LOG
echo "" | tee -a $LOG

python -u -m src.training.train_stage1 \
    --tier1 \
    --steps 5000 \
    --log-interval 1000 \
    2>&1 | tee -a $LOG

echo "" | tee -a $LOG
echo "=== Smoke test complete ===" | tee -a $LOG

# Success criteria summary — note: grep on macOS does not support -P; use -oE + sed
echo "" | tee -a $LOG
echo "--- Criterion 1: No NaN ---" | tee -a $LOG
NAN_COUNT=$(grep "^Step" $LOG | grep -c "NaN" || true)
if [ "$NAN_COUNT" -gt 0 ]; then
    echo "  FAIL: NaN detected in $NAN_COUNT Step lines" | tee -a $LOG
else
    echo "  PASS: No NaN in Step lines" | tee -a $LOG
fi

echo "" | tee -a $LOG
echo "--- Criterion 2: bin_ent trajectory (should trend down from ~4.6) ---" | tee -a $LOG
grep "^Step" $LOG | grep -oE 'bin_ent=[0-9.]+' | sed 's/bin_ent=//' | head -10 | tee -a $LOG

echo "" | tee -a $LOG
echo "--- Criterion 3: q_exp_maxabs (must stay below 20) ---" | tee -a $LOG
grep "^Step" $LOG | grep -oE 'q_exp_maxabs=[0-9.]+' | sed 's/q_exp_maxabs=//' | head -10 | tee -a $LOG

echo "" | tee -a $LOG
echo "--- Criterion 4: critic_loss range ---" | tee -a $LOG
grep "^Step" $LOG | grep -oE 'critic=[0-9.]+' | sed 's/critic=//' | head -10 | tee -a $LOG

echo "" | tee -a $LOG
echo "--- Criterion 5: No alpha= in log (fixed alpha confirmed) ---" | tee -a $LOG
ALPHA_COUNT=$(grep "^Step" $LOG | grep -c "alpha=" || true)
if [ "$ALPHA_COUNT" -gt 0 ]; then
    echo "  FAIL: alpha= found in $ALPHA_COUNT Step lines" | tee -a $LOG
else
    echo "  PASS: No alpha= in log" | tee -a $LOG
fi

echo "" | tee -a $LOG
echo "--- Criterion 6: mode distribution at final step ---" | tee -a $LOG
grep "^Step" $LOG | tail -1 | grep -oE 'mode_batch=\[[^]]+\]' | tee -a $LOG

echo "" | tee -a $LOG
echo "--- Criterion 7: training speed (steps/s) ---" | tee -a $LOG
grep "^Step" $LOG | grep -oE '[0-9.]+ steps/s' | tail -5 | tee -a $LOG
