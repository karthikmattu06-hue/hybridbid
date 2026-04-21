# Tier 2a Closeout — 2026-04-21

Tier 2a is declared complete as a **clean negative**. All 21 evaluated checkpoints net-negative; best checkpoint (step 50k) nets −$291/day. The v1-75k record stands. Day 2 afternoon pivots to Tier 2c (IQL on MILP trajectories).

## Result summary

| Rank | Step | gross $/day | net $/day | viol/64 | capture vs TBx |
|-----:|-----:|------------:|----------:|--------:|---------------:|
| best | 50k | −$41.39 | **−$291.39** | 5 | −4.8% |
| 2 | 275k | −$30.73 | −$980.73 | 19 | −3.5% |
| 3 | 225k | +$14.74 | −$1,035.26 | 21 | +1.7% |
| worst | 25k | −$554.63 | −$3,754.63 | 64 | −63.8% |

Reference row: **Tier 1 v1 @ 75k** — gross $469.77 / **net +$119.77** / 7 violations / +54.0% capture.

Tier 2a peak net (−$291) is **$411 below** the Tier 1 v1 record. No checkpoint clears the net ≥ $50 threshold from the Day 2 kill criteria.

## Full 21-checkpoint eval sweep

Eval run on Narnia 2026-04-21 06:35–06:45 EDT via `experiments/prepare.py --tier2a`. Five-seed IQM (deterministic env + deterministic actor → min = max across seeds), test period 2025-10-01 → 2025-12-04, TBx=$870/day, PF=$1,519/day.

| Step | gross $/day | net $/day | viol/64 | capture % (TBx) | mode dist ch/dc/id | avg_soc MWh |
|-----:|------------:|----------:|--------:|----------------:|--------------------|------------:|
| 25k | −554.63 | −3,754.63 | 64 | −63.8 | 57.1 / 37.2 / 5.7 | 12.43 |
| 50k | −41.39 | **−291.39** | 5 | −4.8 | 23.2 / 24.3 / 52.5 | 10.09 |
| 75k | −208.65 | −2,458.65 | 45 | −24.0 | 43.8 / 25.2 / 31.0 | 10.14 |
| 100k | −192.55 | −1,442.55 | 25 | −22.1 | 31.5 / 27.9 / 40.7 | 8.78 |
| 125k | −431.55 | −2,831.55 | 48 | −49.6 | 48.3 / 45.4 / 6.2 | 10.98 |
| 150k | −283.40 | −3,133.40 | 57 | −32.6 | 32.8 / 28.5 / 38.7 | 11.46 |
| 175k | −160.76 | −2,410.76 | 45 | −18.5 | 34.7 / 28.4 / 37.0 | 10.15 |
| 200k | −30.66 | −2,280.66 | 45 | −3.5 | 39.0 / 41.5 / 19.6 | 8.63 |
| 225k | +14.74 | −1,035.26 | 21 | +1.7 | 25.7 / 29.0 / 45.3 | 7.58 |
| 250k | −45.19 | −1,095.19 | 21 | −5.2 | 36.0 / 32.3 / 31.7 | 6.70 |
| 275k | −30.73 | −980.73 | 19 | −3.5 | 20.9 / 21.1 / 58.0 | 4.98 |
| 300k | −50.55 | −1,750.55 | 34 | −5.8 | 38.5 / 35.6 / 25.9 | 5.71 |
| 325k | +193.74 | −1,156.26 | 27 | +22.3 | 28.4 / 40.0 / 31.6 | 4.79 |
| 350k | +21.20 | −2,378.80 | 48 | +2.4 | 31.1 / 29.3 / 39.6 | 5.79 |
| 375k | +29.16 | −1,820.84 | 37 | +3.4 | 21.1 / 20.7 / 58.2 | 5.28 |
| 400k | +67.93 | −2,782.07 | 57 | +7.8 | 27.1 / 25.5 / 47.3 | 5.55 |
| 425k | −215.03 | −2,965.03 | 55 | −24.7 | 42.4 / 32.7 / 24.9 | 7.59 |
| 450k | −294.71 | −1,944.71 | 33 | −33.9 | 37.7 / 25.1 / 37.2 | 7.15 |
| 475k | −176.74 | −2,376.74 | 44 | −20.3 | 34.4 / 28.7 / 36.9 | 7.56 |
| 500k | −101.85 | −1,201.85 | 22 | −11.7 | 30.6 / 34.0 / 35.4 | 5.74 |
| final | −101.85 | −1,201.85 | 22 | −11.7 | 30.6 / 34.0 / 35.4 | 5.74 |
| ─── | ─── | ─── | ─── | ─── | ─── | ─── |
| **v1-75k ref** | **+469.77** | **+119.77** | **7** | **+54.0** | (ref, Tier 1) | (ref) |

Metric definitions:
- `gross` = IQM of daily revenue over seeds [10,11,12,13,14]
- `net` = gross − 50 × total_violations (absolute count, 64 eval days); per `experiments/prepare.py` line 67
- `capture %` = gross / 870 (TBx baseline) from `prepare.py` RESULT line
- `mode dist` from eval-rollout step-count fractions; `avg_soc` averaged across 64 eval days

## Hypothesis tested

Discrete N=7 action space removes the ∂Q/∂action_magnitude × symexp_derivative amplification path identified as the residual spike driver in Tier 1 v2.1. If that amplification path was the root cause of Tier 1 failure modes, the discrete actor should train stably and produce a valid policy.

Design fixes relative to Tier 1 v2.1 (all retained from Tier 1 stack, plus discrete action):

1. BroNet critic + HL-Gauss (101 bins, [−20, 20], σ=0.75)
2. Reward pre-scale ÷100 before symlog
3. Stop-gradient on actor→TTFE path (critic owns TTFE updates)
4. AdamW lr=1e-4, wd=0.1 for critic
5. Fixed α=0.1 (no entropy auto-tune)
6. γ=0.97, τ=0.001
7. **New**: Categorical(7) over `{−P, −2P/3, −P/3, 0, +P/3, +2P/3, +P}`; critic is `Q(s) → (B, N, n_atoms)` with no action input; closed-form actor loss `E_a[α log π − Q]`

## What the data actually shows

1. **Amplification path was closed.** No mega-spikes comparable to Tier 1 v2.1 (which hit `grad_a_pre` = 1,291). Tier 2a maximum `grad_a_pre` across 500k was **13.93 at step 400k**, bounded by the unified clip to 1.00. Grad-a was sub-unity for 90%+ of training.
2. **A different failure replaced it: policy non-convergence.** Mode distribution oscillates without settling on an arbitrage strategy:
   - 25k env-mode split 57 / 37 / 6 (charge-heavy)
   - 50k 23 / 24 / 52 (idle-dominated)
   - 125k 48 / 45 / 6 (balance collapsed)
   - 300k 38 / 36 / 26 (balanced)
   - 500k 30 / 34 / 35 (balanced-idle)
   Batch-mode idle% was pinned at 3–7% for most of training while env-mode idle% ranged 5–58%, indicating the policy's idle commitment in deployment is inconsistent with its replay-buffer sampling.
3. **No monotonic improvement across training.** Gross $/day across milestones: step 25k −$555, 50k −$41, 125k −$432, 325k +$194, 500k −$102. Actor loss oscillates −44 → −907 → −2,435 → −25 → −53 across the same milestones. This is not a plateau or a clean divergence — it is wandering.
4. **Peak checkpoint (50k) is a dominant-idle policy.** The best net ($−291) comes from the earliest post-warmup checkpoint where idle% hit 52.5%. This is the "do almost nothing" policy — the only Tier 2a policy with ≤ 5 violations. Every checkpoint that attempts meaningful cycling (≥ 30% charge + ≥ 30% discharge) incurs 20–65 violations.
5. **Per-action distribution shows action-6 (+P, full discharge) dominance.** Across steps 225k–500k, `p6` stays in 0.42–0.54 while `p0` (−P, full charge) drifts between 0.11–0.60. The middle actions (`p1` through `p5` = ±2P/3, ±P/3) are jointly < 30% of probability mass. The discrete head uses the endpoints, not the interior — effectively a 3-way choice degenerate to `{−P, 0, +P}`, losing the benefit of the finer discretization.

## Refined interpretation of the amplification problem

Tier 1 failure was diagnosed as symexp amplification through the continuous action magnitude path. Tier 2a results suggest the diagnosis was correct but incomplete: closing that specific path exposes a second failure mode in the same class.

- **Continuous actor + symexp path (Tier 1 v2.1):** tail gradients amplified via `∂Q/∂a`, TTFE features corrupted, training diverges.
- **Discrete actor without symexp path (Tier 2a):** actor loss is a closed-form expectation over 7 atoms of `Q_raw(s, a)`. No action gradient, no symexp amplification. But the `Q_raw` values themselves come from symexp of the HL-Gauss expected symlog, and a single heavy-tailed return that lands at `q_symlog ≈ 3` maps to `q_raw ≈ 19`. When such a transition appears in a minibatch, the `exp(βA)`-like weighting in the actor expectation shifts mass toward whichever action produced it — often the action at the tail distribution's endpoint, not the optimal one.

Both failure modes share a root-cause class: **bootstrap-target updates against heavy-tailed symlog-space Q-estimates propagate tail shocks into the policy.** The amplification path differs (gradient vs. expectation), but the sensitivity to ERCOT's reward distribution is the same.

## Implication for Tier 2c

Tier 2c (IQL on MILP trajectories) avoids the shared root cause:
- **No bootstrap from Q:** IQL's Q loss targets `r + γ V(s')`, where `V` is trained by expectile regression — `V` does not chase `Q` spikes.
- **No online actor update against Q:** policy is recovered via advantage-weighted behavioral cloning on an offline dataset, `L_π ∝ exp(β A(s,a)) · log π(a|s)`. The weighting clips with β, and the BC signal dominates early.
- **Offline dataset bounds the policy class:** MILP rollouts under perfect foresight generate a dataset where the action distribution is already near-optimal, so π is pulled toward validated behavior, not toward Q-estimate tails.

If IQL trains stably and clears net ≥ $100 at 50k gradient steps, the root cause is specifically online SAC-family bootstrap against heavy-tailed Q, not the problem formulation itself. If IQL also fails (V diverges, or net < $50 at 50k), the reward formulation or observation space may need revision before any further RL attempts.

## Design decisions added to the lineage today

| # | Decision | Outcome |
|---|---|---|
| 15 | Discrete N=7 categorical action space; `Q(s) → (B, N, n_atoms)` critic; closed-form actor expectation | Closes ∂Q/∂a amplification path; replaces with policy non-convergence via mode oscillation. Clean negative. |

## Key artifacts

- `checkpoints/tier2a_seed42/` — 21 periodic checkpoints (25k–500k at 25k cadence) + `checkpoint_final.pt`, all preserved
- `experiments/logs/tier2a_seed42.log` — full training log (500k steps, 5.76h wall on Narnia GPU 1 A16)
- `experiments/logs/tier2a_eval_sweep.log` — 21-checkpoint eval sweep, produced this morning
- `experiments/logs/tier2a_seed42_attempt1_crashed.log` — first launch attempt (crashed at warmup exit on `agent.tau_gumbel` AttributeError; preserved for provenance)
- Code: `src/models/networks.py` (`ActorDiscrete7`, `DiscreteBroNetCritic`, `TwinDiscreteBroNetCritic`), `src/models/sac.py` (`SACAgentTier2a`), `src/training/config.py` (`Stage1Tier2aConfig`), `tests/test_tier2a.py` (11 tests)
- Commits: `feadd7f` (implementation), `503b5c4` (`tau_gumbel` getattr guard)
