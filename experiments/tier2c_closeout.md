# Tier 2c closeout — April 21, 2026

## Result: clean negative, peak at 10k = −$2,375 net

Six evaluated checkpoints, all net-negative on the locked test set
(2025-10-01 → 2025-12-04, 64 days). Best: step 10k at +$275.04 gross /
−$2,374.96 net / 53 violations / 18.1% capture of PF. v1-75k record stands.

## Full sweep table

Locked test set: 2025-10-01 → 2025-12-04 (64 days). Net = gross − 50 × violations.

| Step  | Gross $/day | Net $/day  | Viol/64 | TBx capture | PF capture | Charge % | Discharge % | Idle % | Avg SoC (MWh) |
|------:|------------:|-----------:|--------:|------------:|-----------:|---------:|------------:|-------:|--------------:|
|  5k   |    +117.84  |  −2,682.16 |   56    |   +13.5%    |    +7.8%   |   10.2   |      8.5    |  81.3  |     9.98      |
| **10k** | **+275.04** | **−2,374.96** | **53** | **+31.6%** | **+18.1%** | **15.5** | **17.2**   | **67.3** |  **5.28**   |
| 15k   |    −127.84  |  −3,277.84 |   63    |   −14.7%    |    −8.4%   |   58.7   |     23.2    |  18.2  |    10.81      |
| 20k   |     −69.82  |  −2,969.82 |   58    |    −8.0%    |    −4.6%   |   44.3   |     21.2    |  34.5  |    10.08      |
| 25k¹  |    +140.11  |  −2,659.89 |   56    |   +16.1%    |    +9.2%   |   36.6   |     23.1    |  40.3  |     7.27      |
| 50k¹  |     −33.85  |  −3,233.85 |   64    |    −3.9%    |    −2.2%   |   44.8   |     25.9    |  29.3  |    10.94      |

¹ original run1 numbers (not re-evaluated). 5k/10k/15k/20k from run2 reproduce
run1 at 25k byte-for-byte on val-side metrics — training is deterministic on
seed 42 / fixed GPU kernel.

Expert (MILP train split) action distribution for reference:
20.6 / 0.2 / 0.9 / **59.0** / 0.4 / 0.5 / 18.5 on atoms [0..6], i.e.
~21% full-charge, ~59% idle, ~18% full-discharge, <2% on mid-atoms.

## Hypothesis tested

IQL with expectile τ=0.9 and advantage temperature β=5 on MILP-expert
trajectories (Option D format, 420k transitions, 7-atom discretization).
If offline RL without bootstrap-through-critic could produce a stable
policy on ERCOT's heavy-tailed reward distribution, Tier 2c should
beat v1-75k.

Implementation details:
- Expert policy: receding-horizon MIP (24h lookahead, 1h commit),
  GUROBI solver, 35,035 solves over 1460 days of train split
- Dataset format (Option D): MILP rewards used directly, no env replay,
  standalone numpy observation reconstruction, per-day episode truncation
- Action space: 7-atom discrete {−P, −2P/3, −P/3, 0, +P/3, +2P/3, +P}
  (matches Tier 2a for architectural isolation)
- Network: BroNet body + HL-Gauss critic head (101 bins, support [−20, 20])
  + 7-way softmax policy head
- Paper-standard IQL hyperparameters (Kostrikov et al. 2021):
  τ_expectile=0.9, β_advantage=5.0, γ=0.97, AdamW lr=3e-4 wd=0.01,
  polyak τ_V=0.005, batch 256, AWR weight clip=100

## What the data actually shows

1. **Onset of degradation is sharp, not gradual.** 10k peak ($275 gross,
   53 viol, mode mix 15/17/67 close to expert 21/18/59) collapses by
   15k to −$128 gross, 63 viol, mode mix 59/23/18. Policy committed to
   charging dominance within 5k gradient steps.

2. **Mid-atoms collapsed to zero as predicted.** Action distribution at
   all checkpoints: 98%+ mass on atoms {0, 3, 6} (full-charge, idle,
   full-discharge). Thin AWR supervision on atoms {1, 2, 4, 5}
   eliminated them. Effective policy class is 3-atom bang-bang,
   matching MILP expert structure.

3. **Charge bias emerged as AWR distilled.** Early checkpoints (5k)
   were idle-dominated (81%) — policy barely acted. 10k was closest
   to expert mix. 15k onward: charge share 44–59%, far above MILP's
   ~22%. AWR amplified MILP's charge-heavy action frequency without
   learning the timing logic that made those charges appropriate.

4. **All violations are genuine policy failures.** Env is deterministic
   given fixed test data; violations reflect actual SoC limit breaches
   during rollout. The −$50 × violations penalty dominates revenue at
   every checkpoint.

5. **V-function trained stably throughout.** V_symlog overshot to +11.3
   at step 5k, descended smoothly to +3.4 by 50k, remained within
   HL-Gauss support [−20, +20] at all times. Q and V stayed within
   |Δ| ≤ 0.15 in symlog space throughout. No NaN, no divergence.
   Training losses: v_loss ≈ 0.010, q_loss ≈ 4.19 (HL-Gauss CE plateau),
   AWR mean-weight 0.85–1.30, max-weight ≤ 30 (never hit clip=100).
   The failure is in the policy head, not the value estimation.

## Refined interpretation

IQL faithfully learns the marginal action distribution of an expert
policy but fails to reproduce temporal structure when:
(a) the expert's state distribution is strongly self-consistent
(MILP always acts from SoC trajectories consistent with its own
optimal past actions), and
(b) the deployed policy induces meaningfully different state
distributions (deployed Tier 2c policy's SoC trajectory diverges from
MILP's after a few steps due to action quantization).

This is a distribution-shift failure mode specific to offline RL
from model-based experts whose trajectories are tightly coupled to
their own decision structure. Pure imitation (BC) would exhibit the
same failure; IQL's value function does not help here because the
value is learned under the expert's state distribution, which the
deployed policy does not produce.

## Design decisions added to the lineage

| # | Decision | Outcome |
|---|---|---|
| 16 | Option D (obs reconstruction without env replay, MILP rewards used directly) | 420k clean train + 183k val transitions, zero reward mismatch by construction |
| 17 | IQL paper-standard hyperparameters (τ=0.9, β=5, γ=0.97) | Trained stably through 50k, policy degraded past 10k |
| 18 | Per-day truncation (truncated=True at day boundaries, done=False everywhere) | Clean episode segmentation, no ill effects on V-regression |
| 19 | 7-atom discrete action space via snap-to-nearest | 98% of MILP actions snap to {0, 3, 6}; effective policy is 3-atom bang-bang |

## Sprint-wide implication

All three Tier variants tested fail to beat v1-75k:

| Variant | Method family | Best net $/day | Violations |
|---|---|---|---|
| v1-75k (Tier 1 peak) | Online SAC, continuous actor, reward pre-scaling | +$119 | 7 |
| Tier 1 v2.x | Online SAC + stop-gradient on actor→TTFE | kill criteria triggered | n/a |
| Tier 2a | Online SAC, discrete N=7 actor | all net-negative, best −$291 | 5–64 |
| Tier 2c | Offline IQL on MILP expert | all net-negative, best −$2,375 | 53–64 |

v1-75k remains the first and only valid policy in the v5/v6/sprint
lineage. Online and offline RL variants on ERCOT's heavy-tailed
reward distribution are structurally difficult to train to convergence
under the current problem formulation.

## Key artifacts

- `checkpoints/tier2c_seed42_run1/` on Narnia (original 25k + 50k checkpoints, preserved)
- `checkpoints/tier2c_seed42/` on M4 (5k, 10k, 15k, 20k, 25k from rerun, for onset-of-degradation evidence)
- `experiments/logs/tier2c_seed42.log` (training log, full 0→50k trajectory)
- `experiments/logs/tier2c_eval_sweep.log` (6-row test-set eval sweep)
- `data/expert_trajectories/receding_horizon_{train,val}_option_d.npz` (reusable for future offline RL work)
- `scripts/preprocess_milp_option_d.py` (standalone obs encoder, no env.step dependency)
- `src/agents/iql.py` (IQL agent implementation)
- `src/training/train_iql.py` (training loop with process-level mandatory pause at 50k)
