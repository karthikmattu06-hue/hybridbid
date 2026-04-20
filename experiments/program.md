# HybridBid Stage 1 Sprint: Research Agenda

**DO NOT MODIFY. Locked on day 1.**

## Problem

Maximize pre-RTC+B ERCOT battery arbitrage capture rate vs TBx baseline ($870/day).

Confirmed baselines (all measured with `experiments/prepare.py`, 2026-04-20):

| Checkpoint | Steps | gross $/day | violations (/64) | net $/day | charge/discharge/idle | Policy class |
|------------|-------|-------------|------------------|-----------|----------------------|--------------|
| v5.9.1 `stage1/175k` | 175k | $308.76 | 31 | **−$1,241** | 12.5 / 70.8 / 16.7 | Partial dump |
| v5.9.1 `stage1/250k` | 250k | $292.00 | 16 | **−$508** | 30.6 / 59.2 / 10.2 | Partial dump |
| v5.9.1 `stage1/600k` | 600k | $371.73 | 64 | **−$2,828** | 0.8 / 84.3 / 14.9 | Full dump |
| v5.9.2 `stage1_v592/250k` | 250k | $245.37 | 1 | **+$195.37** | 45.6 / 53.7 / 0.7 | Near-valid |
| v5.9.2 final | — | $258.74 | 55 | −$2,441 | 0.0 / 99.8 / 0.2 | Full dump |
| v6.0 final | — | $383.11 | 64 | −$2,817 | 6.5 / 90.6 / 2.9 | Full dump |

**Primary metric: `net_return = gross − 50 × violations`.** Gross $/day comparisons across policies with different violation counts are not meaningful — the eval harness excludes the −50 SoC penalty from `info["energy_revenue"]`, so a dump-to-floor policy that terminates early shows inflated gross revenue.

**Starting point for the sprint: v5.9.2 250k at net $195/day (1 violation).** This is the only near-valid policy in the v5.x lineage. The v5.9.2 band-aids (idle_logit_bonus, alpha_max) — which Tier 1 explicitly disables — were the only intervention producing a policy that respected SoC constraints.

**Tier 1 progress criterion:**
- Net $/day ≥ $195 (match v5.9.2 250k)
- Violations ≤ 5 / 64 days
- No dump-and-terminate degeneracy (discharge% < 70%)

If Tier 1 achieves <5 violations via architectural means (BroNet LN + HL-Gauss + fixed α + AdamW WD), the v5.9.2 band-aids are structurally unnecessary. If violations climb back to v5.9.1 levels, we'll need "Tier 1.5" re-introducing the band-aids — not a failure, just additional evidence the constraint-respecting policy requires both architectural stability AND explicit mode biasing.

**Historical note:** The pre-sprint "$309/day at 175k" number aligns with v5.9.1 175k gross ($308.76 measured by prepare.py). The old and new harnesses agree on gross revenue. Neither harness reported net revenue, which is why the dump-and-terminate failure mode went undiagnosed across 4 checkpoints.

Ceiling: Perfect Foresight MIP at $1,519/day (gross, assumed compliant). 57% of PF = $870 = TBx.

## ⚠️ Critical: Training reward ≠ test revenue

Historical performance numbers cited from training logs (`avg_reward`) are **NOT
comparable** to `prepare.py` output (test-set daily revenue in dollars). Do not
use them as baselines.

Three structural reasons they differ for identical policy behavior:

1. **Timing bonus included in training reward, excluded from evaluation.**
   The env step reward is `energy_term + timing_bonus` where timing_bonus =
   `β_S=10 × energy_mag × |ρ−ρ̄| × ...`. `prepare.py` uses only
   `info["energy_revenue"]` (= energy_term), omitting the timing bonus entirely.

2. **Units differ.** Training `avg_reward` is episode-total reward in p.u.
   (energy_mag ∈ [0,1], NOT multiplied by P_max). `prepare.py` multiplies by
   P_max=10 MW and sums over 288 steps to get actual $/day.

3. **Test period vs train distribution.** Training samples from 2020–2023;
   `prepare.py` runs on the held-out 2025-10-01 → 2025-12-04 test set.

**Consequence:** `avg_reward=309` in a training log has no predictable
relationship to `iqm_return=X` from `prepare.py`. The only valid baseline is a
number produced by `prepare.py` on a real checkpoint. Establish this before
comparing experiments.

---

## Measurement

All experiments measured by `experiments/prepare.py`:
- Test period: 2025-10-01 → 2025-12-04 (65 days, pre-RTC+B, held out)
- Seeds: [10, 11, 12, 13, 14] — IQM of middle 3
- Primary metric: `iqm_return` ($/day)
- Capture: `iqm_return / 870` (fraction of TBx baseline)

## Allowed modifications

- SAC hyperparameters (lr, gamma, tau, batch size, buffer)
- Network width and depth (hidden_dim, n_layers, d_model)
- Critic architecture (LayerNorm, residual blocks, HL-Gauss head, spectral norm)
- Critic optimizer (AdamW, weight decay, schedulers)
- Reward transforms (symlog, clipping) — applied consistently train and eval
- Entropy temperature (fixed vs learned, bounds)
- Gradient clipping schemes
- Offline RL / BC integration with MILP expert trajectories
- Observation space additions (engineered features, not raw TTFE input changes)
- Potential-based reward shaping (must provably preserve policy optimality)
- Policy EMA / SWA at evaluation time

## Forbidden modifications

- `experiments/prepare.py` (any change invalidates all prior results)
- Training data range: train=2020-2023, val=2024-2025-09, test=2025-10-01:12-04
- SoC penalty value: -50 (Li et al.)
- Eval seeds: [10, 11, 12, 13, 14]
- TBx baseline: $870/day
- Total training steps: 500k per experiment (for fair comparison)

## Experiment naming discipline

Each experiment name encodes its parent and the single variable changed:

```
v592_base            — v5.9.2 architecture, paper-spec hyperparams (parent: none)
v592_hl_gauss        — parent: v592_base,  changed: HL-Gauss critic head
v592_ln_critic       — parent: v592_base,  changed: LayerNorm in critic
v592_adww            — parent: v592_base,  changed: AdamW + weight decay
v592_bc_warmstart    — parent: v592_base,  changed: BC pre-train from MILP trajectories
v592_symlog          — parent: v592_base,  changed: symlog reward transform
v592_ternary         — parent: v592_base,  changed: discrete 3-action space (no magnitude)
tier1_best_plus_bc   — parent: best tier1, changed: BC integration
```

## Target

| Milestone | net $/day | gross $/day | violations (/64) |
|-----------|-----------|-------------|------------------|
| Current (v5.9.2 250k, baseline to beat) | $195    | $245.37     | 1                |
| Day 2 goal | ≥$300     | —           | ≤5               |
| Stretch   | ≥$500     | —           | ≤5               |
| Ceiling   | $870      | $870        | 0                |

Report net_return, gross iqm_return, AND violations for every experiment. Gross
improvements with violation counts above the starting point are not progress — they
are harness artifacts from the eval excluding the −50 SoC penalty.

## Ablation discipline

- ONE variable changed per experiment relative to its named parent
- Log the parent name and changed variable in the experiment name
- Never cherry-pick: report ALL experiments run, including failures
- A "failure" (regression) is still informative — log it

## MILP Expert Trajectories (Day 1 output)

Available in `data/expert_trajectories/`:
- `receding_horizon_train.npz` — 2020-2023, 24h horizon, 1h commit
- `receding_horizon_val.npz`   — 2024-2025-09, 24h horizon, 1h commit
- `clairvoyant_train.npz`      — 2020-2023, full PF-MIP (optional comparison)

Trajectory format (per .npz):
- `modes`:       int8 array, 0=charge 1=discharge 2=idle
- `magnitudes`:  float32 [0,1], normalized bid power
- `socs`:        float32 MWh, state of charge before action
- `rewards_env`: float32, Li et al. Eq.26 reward (matches env exactly)
- `rewards_raw`: float32, plain energy revenue (p_dch - p_ch) * rt_lmp * dt
- `rt_lmp`:      float32, realized RT LMP at each step
- `timestamps`:  str array, UTC timestamps

Use `rewards_env` for IQL value learning (matches env reward signal).
Use `rewards_raw` for revenue-based analysis.
