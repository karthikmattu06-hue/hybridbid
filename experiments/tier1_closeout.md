# Tier 1 Closeout — 2026-04-20

Tier 1 is declared complete with **v1 @ 75k** as the record holder. Two adaptation attempts (v2, v2.1) failed to improve on the v1 peak; both were killed before 500k per the pre-declared stopping criteria. Day 2 pivots to Tier 2 (discrete action space) and Tier 2c (IQL on MILP trajectories).

## Tier 1 record

| Run | Step | gross $/day | net $/day | capture vs TBx | violations (/64) | status |
|-----|-----:|------------:|----------:|---------------:|-----------------:|--------|
| **v1** (tier1_v1_peak_75k.pt) | **75,000** | **$469.77** | **$119.77** | **54.0%** | **7** | **RECORD** |
| v1 | 100,000 | — | — | — | — | collapsed (NaN at 148k) |
| v2 (stop-grad fix, warm-start from v1-50k) | killed @ ~150k | — | — | — | — | spikes > 50 persisted; FLAT trajectory |
| v2.1 (stop-grad + fresh Adam, resumed from v1-50k) | killed @ 75k | $90.90 | **−$1909.10** | 10.5% | 40 | 3/5 kill criteria tripped |

Canonical record checkpoint: `checkpoints/records/tier1_v1_peak_75k.pt` (copied from `checkpoints/tier1_seed42_preserved/checkpoint_step75000.pt`).

Eval numbers produced by `experiments/prepare.py` (locked yardstick), seeds [10,11,12,13,14], test period 2025-10-01 → 2025-12-04, TBx=$870/day, PF=$1519/day.

## What happened

### v1 (baseline): BroNet + HL-Gauss + reward prescale ÷100
- Ran cleanly from 0 → 75k, actor gradient well-behaved, critic loss stable.
- Peaked at step 75k (gross $469.77/day, 54% capture, 7 SoC violations).
- Grad spikes began at step 85k (`grad_a` > 50), Q-surface `q_exp_maxabs` climbed, critic quality degraded.
- Catastrophic collapse at step 148k: NaN parameters, training killed.

### v2: add stop-gradient on actor → TTFE path, warm-start from v1-50k
- Hypothesis: actor-driven TTFE feedback was the amplification path.
- Diagnostic (`scripts/diagnose_gumbel_spike_correlation.py`): TTFE attn gradient pinned at 0.00 at all spike events — the stop-grad fix **did** sever the feedback loop as intended.
- But spikes > 50 persisted with comparable frequency. Post-resume trajectory regression (`scripts/diagnose_post_resume_trajectory.py`, slope = −2.59e-05, p=0.55, n=6) returned verdict **FLAT** — spike magnitude not decaying.
- Optimizer-state inspection showed v2 carried stale Adam 2nd-moment statistics from v1, hypothesized as a secondary amplifier.

### v2.1: stop-gradient + fresh Adam moments, resumed from v1-50k
- First 5k post-resume was pristine (max grad_a = 3.82 vs v2's 412.3 over the same window) — confirmed stale Adam was a contributor.
- But spike trajectory turned **GROWING** (slope +1.42e-04, R=+0.55, n=4 spikes > 50 between step 50k and 72k).
- Kill at step 72k: `grad_a_pre`=1291 (> 1000 hard ceiling), `net_return`=−$259 (< $50 floor), `violations`=40 (> 25 ceiling). 3 of 5 relaxed kill criteria tripped.
- Confirming eval of v2.1 @ 75k: gross $90.90/day, net −$1909/day, 40 violations — far below v1's record.

## Root cause (current understanding)

The actor→TTFE feedback loop and stale Adam momentum were **two amplifiers of a deeper instability**, not the cause. The surviving amplification path is:

- `∂Q/∂action` through the symexp gradient amplification on the HL-Gauss critic head.
- `q_exp_maxabs` showed +0.67 correlation with spike events; `|bin_argmax|` +0.56. Price correlation was partial (step 62k/71k spikes coincided with $4,479 RT-LMP extremes), but step 72k spike (1291) occurred at a normal $921 RT-LMP — so spikes are not purely price-driven.
- The Gumbel-STE mode head was **not** the driver (weak correlation of τ_g and mode-shift with spikes).

Implication: Tier 1's architectural stack (BroNet + HL-Gauss + symlog + Gumbel-Softmax) has a residual instability that closing the actor→TTFE loop and resetting Adam cannot fix. A structural action-space change is required.

## Decision: pivot to Tier 2

- **Tier 2a** — replace Gumbel-Softmax + continuous magnitude with a discrete N=7 categorical over `{−P_max, −2P_max/3, −P_max/3, 0, +P_max/3, +2P_max/3, +P_max}`. Discrete SAC removes the reparameterized action that flows through symexp. Retain all other Tier 1 architectural choices (BroNet+HL-Gauss, reward prescale ÷100, stop-gradient, fresh Adam). 500k steps from scratch, seed 42.
- **Tier 2c** — IQL on MILP-labeled offline data. No bootstrap targets → no symexp-amplification path. 200k gradient steps, expectile τ=0.9, AWR β=5. Uses `data/expert_trajectories/receding_horizon_train.npz` (pending MILP re-run after NaN fix).

## Preserved artifacts

- `checkpoints/records/tier1_v1_peak_75k.pt` — canonical record
- `checkpoints/tier1_seed42_preserved/` — v1 lineage (25k/50k/75k/100k on Narnia; 75k + log on M4)
- `checkpoints/tier1_seed42_v21/checkpoint_step75000.pt` — v2.1 record (pulled from Narnia)
- `experiments/logs/tier1_seed42_v2.log`, `tier1_seed42_v21.log` — training logs
- `experiments/diagnose_gumbel_spike_correlation.png`
- `experiments/diagnose_post_resume_trajectory.png`
- `experiments/diagnose_qvalue_saturation.png`

## Diagnostic scripts written for Tier 1

- `scripts/diagnose_gumbel_spike_correlation.py` — ruled out Gumbel-STE as the spike driver.
- `scripts/diagnose_post_resume_trajectory.py` — 5k-bucketed stats + log-regression of spikes > 50; verdicts FLAT (v2) / GROWING (v2.1).
- `scripts/inspect_ttfe_weights.py` — confirmed 50k TTFE weights clean (no post-spike corruption); rules out weight-state as the persistence mechanism.
