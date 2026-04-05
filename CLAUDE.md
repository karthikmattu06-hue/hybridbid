# CLAUDE.md — TempDRL for ERCOT RTC+B
**Date:** April 5, 2026
**Project:** Temporal-Aware Deep Reinforcement Learning for Battery Storage Bidding in ERCOT's Post-RTC+B Market
**Status:** All modules built and tested (62 tests passing) → Resolve data issues → Launch Stage 1 training

---

## CRITICAL INSTRUCTIONS FOR CLAUDE CODE

1. **Work incrementally.** Do NOT attempt to build the full system at once. Build one module, test it, confirm, then proceed.
2. **Two-stage training is the core architectural decision.** Do not mix pre-RTC+B and post-RTC+B data in training. Stage 1 uses pre-RTC+B only, Stage 2 uses post-RTC+B only.
3. **Ask before building.** When encountering ambiguity, stop and report rather than guessing.
4. **This CLAUDE.md is the single source of truth.** All prior handoff documents (`HybridBid_Project_Handoff_v2.md`, previous `CLAUDE.md` versions) are obsolete.
5. **Era-aware features (is_post_rtcb, days_since_rtcb, rt_as_available) are REMOVED from the observation space.** The structural break is handled by the two-stage training schedule, not observation flags. The `is_post_rtcb` column may still exist in processed Parquet files — it must NOT be used as an input feature.
6. **Explore and report first.** Surface findings before building. Never attempt to build the full system at once. Ask before proceeding on ambiguity.

---

## WHAT WE'RE BUILDING

Adapting the TempDRL approach from Li et al. (2024) for ERCOT's post-RTC+B market, with a **two-stage pretrain → finetune** training strategy to handle the RTC+B structural break.

> **Li, J., Wang, C., Zhang, Y., & Wang, H.** "Temporal-Aware Deep Reinforcement Learning for Energy Storage Bidding in Energy and Contingency Reserve Markets." *IEEE Transactions on Energy Markets, Policy and Regulation*, Sept 2024. (arXiv:2402.19110)

**Target users:** Small battery storage operators (5-20 MW) in ERCOT.
**Primary deliverable:** A working system with initial results within 6-8 weeks.

---

## WHY TWO-STAGE TRAINING (NOT ERA-AWARE FEATURES)

### The Problem
- Post-RTC+B data: ~30,000 five-minute intervals (~3.5 months). Insufficient for SAC convergence alone.
- Pre-RTC+B data: ~525,000 intervals (2020–2025). Abundant but from a fundamentally different MDP (no real-time AS market).
- Mixing both with era flags creates problems: non-stationary MDP, 95/5 sample ratio imbalance, Q-function must represent two different value landscapes in shared weights.

### The Solution: Pretrain → Finetune
- **Stage 1 (Pre-RTC+B, energy-only):** Train the TTFE + 1D energy actor + twin critics on 525k transitions. The agent learns temporal price patterns, SoC management, load/solar cycle awareness. These are transferable skills.
- **Stage 2 (Post-RTC+B, co-optimize):** Initialize TTFE from Stage 1. Replace actor with 6D head (energy dim from Stage 1, AS dims near-zero). Fresh twin critics. Finetune on 30k post-RTC+B transitions with a progressive unfreezing schedule.

This preserves the Li et al. architecture while respecting the structural break through the training schedule rather than observation engineering.

---

## PRE-TRAINING CHECKLIST

These issues must be resolved before launching Stage 1 training:

- [ ] **`is_post_rtcb` column in processed data:** Confirm that `ercot_env.py` and training code do NOT use this column as an observation feature. It may remain in Parquet for data-splitting convenience, but must never enter the observation space.
- [ ] **`load_forecast` raw data gap:** Raw files end at Nov 2024 (~16 months stale), but processed files cover the full range. Verify: did the pipeline fill this from another source, or is load_forecast data missing/zero-filled for Dec 2024–present?
- [ ] **Wind/solar raw data:** Raw directories appear thin. Same question: verify processed wind/solar data is real, not zero-filled.
- [ ] **Data transfer from Air:** `~/ercot_backfill/` does not exist on M4. The SCP transfer from MacBook Air (`karthikmattu@100.99.63.48`) has not happened. Transfer the backfill data, then re-run preprocessing to produce the final canonical dataset.
- [ ] **Run baselines on test periods:** TBx and Perfect Foresight MIP are implemented but have not been run. Generate baseline results in `data/results/` before training so we have comparison targets.
- [ ] **Commit and push:** 2 unpushed commits + uncommitted changes to CLAUDE.md and config.py. Clean up and push before starting training runs.

---

## DATA STRATEGY

### Data Access (Confirmed via Exploration)
| Product | Method | Granularity | History |
|---------|--------|-------------|---------|
| RT LMP | ErcotAPI `get_lmp_by_settlement_point` | 5-min | 2020+ |
| DAM SPP | Scraper `get_dam_spp(year=)` | Hourly | 2020+ |
| DAM AS | ErcotAPI `get_as_prices` | Hourly | 2020+ |
| RT SCED MCPC | ErcotAPI data endpoint (NP6-332-CD) | 5-min | Dec 5, 2025+ |
| Load Actual | Scraper `get_hourly_load_post_settlements` | Hourly | 2020+ |
| Load Forecast | ErcotAPI `get_load_forecast_by_model` | Hourly | 2020+ |
| Wind | ErcotAPI `get_wind_actual_and_forecast_hourly` | Hourly | 2020+ |
| Solar | ErcotAPI `get_solar_actual_and_forecast_hourly` | Hourly | 2020+ |

### Raw Data Status (as of April 5, 2026)
| Directory | Files | Date Range | Notes |
|-----------|-------|------------|-------|
| dam_as | 2,274 | 2020-01-01 → 2026-03-23 | Current |
| rt_lmp | 1,180 | 2020-01-01 → 2026-03-23 | Current |
| rt_lmp_5min | 2,192 | 2020-01-01 → 2025-12-31 | Ends Dec 2025 |
| dam_spp | 88 (yearly) | 2020 → 2026 | Current |
| load_actual | 7 (yearly) | 2020 → 2026 | Current |
| load_forecast | 1,793 | 2020-01-01 → 2024-11-27 | **STALE — 16 months behind** |
| sced_mcpc | 109 | ~Dec 2025 → ~Mar 2026 | As expected |
| rt_spp | 7 (yearly) | 2020 → 2025 | Missing 2026 |
| solar, wind | Minimal | — | Likely stale |

### Canonical Schema (5-min intervals, UTC)
Three Parquet tables partitioned by month. Hourly data forward-filled to 5-min. See `src/data/schema.py` for exact column mappings.

- **energy_prices:** timestamp_utc, rt_lmp, dam_spp
- **as_prices:** timestamp_utc, rt_mcpc_{regup,regdn,rrs,ecrs,nsrs} (zero pre-RTC+B), dam_as_{regup,regdn,rrs,ecrs,nsrs}
- **system_conditions:** timestamp_utc, total_load_mw, load_forecast_mw, wind_actual_mw, wind_forecast_mw, solar_actual_mw, solar_forecast_mw, net_load_mw

**Note:** Processed Parquet files contain an `is_post_rtcb` column. This is for data-splitting convenience ONLY — it must not be used as an observation feature.

---

## TEMPDRL ARCHITECTURE

### Observation Space (78 dimensions total)

**TTFE Input** — Rolling window of L=32 price vectors, each 12-dim:
- RT LMP (1)
- RT MCPC × 5: RegUp, RegDn, RRS, ECRS, NSRS (zeros pre-RTC+B)
- DAM SPP (1)
- DAM AS × 5: RegUp, RegDn, RRS, ECRS, NSRS (ECRS zero pre-June 2023)

**TTFE Output:** 64-dimensional compressed temporal feature vector.

**Concatenated with TTFE output (bypasses transformer):**
- System conditions (7): total_load_mw, load_forecast_mw, wind_actual_mw, wind_forecast_mw, solar_actual_mw, solar_forecast_mw, net_load_mw
- Cyclical time features (6): hour_of_day sin/cos, day_of_week sin/cos, month sin/cos
- Battery state (1): SoC_t

**Full actor/critic input: 64 + 7 + 6 + 1 = 78 dimensions.**

Rationale: TTFE processes temporal price dynamics at 5-min resolution. System conditions are hourly forward-filled — putting them through the transformer would dilute attention. Concatenation keeps the TTFE focused on price structure while giving the policy access to system state.

### Action Space

**Stage 1 (energy_only):** a_t ∈ ℝ¹
- a_0: Net energy power p_net ∈ [-P_max, P_max]

**Stage 2 (co_optimize):** a_t ∈ ℝ⁶
- a_0: Net energy power p_net ∈ [-P_max, P_max]
- a_1–a_5: AS capacity offers [MW] for RegUp, RegDown, RRS, ECRS, NSRS (each ≥ 0)

### Feasibility Projection

**Stage 1:** Simple SoC/power clipping:
- |p_net| ≤ P_max
- SoC_min ≤ SoC_{t+1} ≤ SoC_max

**Stage 2:** Full AS constraint projection (applied after actor, before environment):
- p_discharge + RegUp + RRS + ECRS ≤ P_max (joint upward capacity)
- p_charge + RegDown ≤ P_max (joint downward capacity)
- SoC duration requirements:
  - RegUp/RegDown: 0.5 MWh per 1 MW (30-min sustain)
  - RRS: 0.5 MWh per 1 MW (30-min sustain)
  - ECRS: 1.0 MWh per 1 MW (1-hour sustain)
  - Non-Spin: 4.0 MWh per 1 MW (4-hour sustain)
- SoC_t ≥ SoC_min + Σ(c_k × duration_k) for all awarded AS products
- Critics see projected (feasible) actions, not raw outputs.

### Reward Function

**Stage 1:**
```
r_t = Revenue_Energy - Cost_Degradation
```

**Stage 2:**
```
r_t = Revenue_Energy + Σ Revenue_AS,i - Cost_Degradation - Penalty_SPD - Penalty_Imbalance
```

Where:
- Revenue_Energy = Δt × (p_dch × η_dch - p_ch / η_ch) × LMP_t
- Revenue_AS,i = Δt × c_i × MCPC_i,t
- Cost_Degradation = C_deg × (p_ch + p_dch) × Δt
- Penalty_SPD = -β_SPD × max(0, |a_t - p_actual| - δ) where δ = max(0.03 × |avg_set_point|, 3 MW)
- Penalty_Imbalance = cost of failing to sustain AS capacity if SoC drops below required level

### TTFE Architecture

- Input: S_t = [ρ_{t-L+1}, ..., ρ_t] — shape (L=32, 12)
- Multi-Head Self-Attention: 4 heads
- d_model: 64
- n_layers: 2 transformer layers
- Output: f_t — 64-dimensional temporal feature vector

### SAC Hyperparameters

- Automatic entropy coefficient tuning (α)
- Twin Q-networks (clipped double Q-learning)
- Target network soft update: τ = 0.005
- Discount factor γ: 0.99
- Actor: 2 hidden layers × 256 units, ReLU
- Critic: 2 hidden layers × 256 units, ReLU
- Learning rate: 3e-4 (Stage 1), adjustable for Stage 2

**Stage 1:**
- Replay buffer: 1,000,000 transitions
- Batch size: 256

**Stage 2:**
- Replay buffer: 30,000–50,000 transitions
- Batch size: 128
- Fewer gradient steps per env step to avoid overfitting

---

## TWO-STAGE TRAINING PROTOCOL

### Stage 1: Energy-Only Pretraining (Pre-RTC+B)

**Data:** 2020-01-01 → 2025-12-04 (~525k transitions)
**Environment mode:** `energy_only` (1D action)
**Goal:** Learn temporal price patterns, SoC management, energy arbitrage

Train until convergence:
- TTFE + 1D actor + twin critics
- Standard SAC with Li et al. hyperparameters
- Replay buffer: 1M, batch: 256

**Checkpoint:** Save TTFE weights + actor weights.

### Stage 2: Co-optimization Finetuning (Post-RTC+B)

**Data:** 2025-12-05 → 2026-03-20 (~30k transitions)
**Environment mode:** `co_optimize` (6D action)
**Goal:** Learn AS capacity allocation and joint energy+AS optimization

**Initialization:**
- TTFE: Load Stage 1 weights
- 6D Actor: Energy output dimension initialized from Stage 1 actor's final layer. AS output dimensions initialized near zero (small Gaussian weights, zero bias).
- Twin Critics: Fresh random initialization.
- Replay buffer: Empty, post-RTC+B only.

**Progressive Unfreezing Schedule:**

*Phase 1 — Frozen encoder warm-up:*
- Freeze all TTFE parameters
- Train new 6D actor + fresh critics
- LR: 1e-3 for heads (or whatever worked in Stage 1)
- Until: losses stabilize, policy stops making insane bids

*Phase 2 — Partial unfreeze:*
- Unfreeze top 1–2 TTFE transformer layers
- Keep lower layers frozen
- TTFE LR: 10× smaller than heads
- Continue on post-RTC+B only

*Phase 3 (optional) — Full unfreeze:*
- If validation backtests show underfitting to AS patterns
- Unfreeze all TTFE with very low LR
- Short fine-tune pass

---

## BATTERY PARAMETERS (Reference: 10 MW / 20 MWh)

| Parameter | Value |
|-----------|-------|
| P_max | 10 MW |
| E_max | 20 MWh (2-hour duration) |
| SoC_min | 10% (2 MWh) |
| SoC_max | 90% (18 MWh) |
| SoC_initial | 50% (10 MWh) |
| η_charge | 0.92 |
| η_discharge | 0.92 |
| C_degradation | $2/MWh throughput |
| Ramp rate | 10 MW/min (full capacity in 1 min) |

---

## EVALUATION PLAN (Three Dimensions)

### 1. Baseline Comparison
Compare the trained two-stage agent against:
- **TBx (Time-Based Arbitrage):** Charge cheapest 4 hours, discharge most expensive 4 hours daily. No AS. Already implemented in `src/baselines/tbx.py`.
- **Perfect Foresight MIP:** CVXPY + HiGHS. Theoretical energy-only ceiling. Already implemented in `src/baselines/perfect_foresight.py`.
- **Vanilla SAC (no TTFE):** Standard SAC with flat observation vector (no transformer temporal encoding). Tests the value of the TTFE component.

### 2. Train-from-Scratch Comparison
Train a fresh agent on post-RTC+B data only (no pretraining, no Stage 1 initialization) to test whether the two-stage approach is justified. This replaces the old "Stage 1 vs Stage 2 direct comparison."

### 3. Ablation Study
Test Stage 2 adaptation components in isolation:
- Progressive unfreezing vs full fine-tune vs full freeze of TTFE
- Fresh critics vs warm-started critics
- Near-zero AS init vs random AS init

### Explicitly Removed from Committed Evaluation
These can be added back if results warrant, but are not committed deliverables:
- Revenue decomposition (energy vs AS breakdown)
- Consistency and risk metrics (daily returns, drawdowns)
- Sample efficiency curve
- TTFE attention weight visualization
- Stage 1 vs Stage 2 representation analysis

---

## EXISTING CODE

```
hybridbid/
├── configs/
│   ├── battery.yaml
│   └── data_products.yaml
├── data/
│   ├── raw/                      # Daily Parquet files per product
│   │   ├── rt_lmp/
│   │   ├── rt_lmp_5min/
│   │   ├── dam_spp/
│   │   ├── dam_as/
│   │   ├── load_actual/
│   │   ├── load_forecast/         # ⚠️ Stale — ends Nov 2024
│   │   ├── wind/                  # ⚠️ Thin — verify processed data
│   │   ├── solar/                 # ⚠️ Thin — verify processed data
│   │   ├── sced_mcpc/             # 109 daily files (Dec 5, 2025 – Mar 2026)
│   │   └── rt_spp/
│   ├── processed/                 # Canonical schema Parquet (5-min, UTC)
│   │   ├── energy_prices/         # 75 monthly files, ~8,928 rows/month
│   │   ├── as_prices/             # 75 monthly files
│   │   └── system_conditions/     # 75 monthly files
│   └── results/
│       └── eda/                   # EDA output only — no baseline results yet
├── src/
│   ├── data/
│   │   ├── pipeline.py            # Orchestrator with rate limiting
│   │   ├── ercot_fetcher.py       # Confirmed access methods per product
│   │   ├── schema.py              # Validated column mappings
│   │   └── preprocessing.py       # Cleaning, alignment, resampling
│   ├── baselines/
│   │   ├── tbx.py                 # Time-Based Arbitrage
│   │   ├── perfect_foresight.py   # Perfect foresight MIP (CVXPY/HiGHS)
│   │   └── run_baselines.py       # CLI runner
│   ├── models/
│   │   ├── ttfe.py                # Transformer Temporal Feature Extractor (L=32, d=64)
│   │   ├── sac.py                 # SAC agent with two-stage architecture
│   │   ├── networks.py            # Actor (squashed Gaussian) and TwinCritic
│   │   ├── feasibility.py         # Differentiable feasibility projection
│   │   └── replay_buffer.py       # Fixed-capacity replay buffer for dict observations
│   ├── env/
│   │   └── ercot_env.py           # Gymnasium env with energy_only / co_optimize modes
│   ├── training/
│   │   ├── train_stage1.py        # Stage 1 energy-only pretraining loop
│   │   ├── train_stage2.py        # Stage 2 finetuning with progressive unfreezing
│   │   └── config.py              # Hyperparameter configuration dataclass
│   ├── evaluation/
│   │   ├── metrics.py             # Revenue, capture rate, compliance metrics
│   │   └── visualization.py       # Publication-quality plots
│   └── utils/
│       ├── time_utils.py          # ERCOT CPT/UTC time conversion
│       └── battery_sim.py         # Battery state simulator (13 tests)
├── tests/
│   ├── test_battery_sim.py        # 13 tests
│   ├── test_feasibility.py        # 14 tests
│   ├── test_networks.py           # 6 tests
│   ├── test_ttfe.py               # 5 tests
│   ├── test_sac.py                # 8 tests
│   ├── test_env.py                # 12 tests
│   └── test_replay_buffer.py      # 4 tests
├── checkpoints/                   # (untracked) — empty, no training runs yet
├── scripts/                       # (untracked)
├── .env                           # ERCOT API credentials (gitignored)
├── .gitignore
├── requirements.txt               # Stale — lists ~15 packages, ~250+ installed
└── README.md
```

**All 62 tests passing** (as of April 5, 2026). No training runs have been executed yet.

---

## TRAIN / VALIDATION / TEST SPLITS

| Split | Period | Stage |
|-------|--------|-------|
| Train Stage 1 | 2020-01-01 → 2023-12-31 | Stage 1 (energy-only) |
| Validation Stage 1 | 2024-01-01 → 2025-09-30 | Stage 1 hyperparameter tuning |
| Test Pre-RTC+B | 2025-10-01 → 2025-12-04 | Stage 1 evaluation |
| Train Stage 2 | 2025-12-05 → 2026-01-31 | Stage 2 (co-optimize) |
| Validation Stage 2 | 2026-02-01 → 2026-02-28 | Stage 2 hyperparameter tuning |
| Test Post-RTC+B | 2026-03-01 → present | Stage 2 out-of-sample evaluation |

---

## WEEK PLAN

### Weeks 1-2: Data Pipeline + Baselines ✅
- [x] Project scaffold
- [x] Battery simulator (13 tests pass)
- [x] Data exploration (Phase 1 + 1b)
- [x] Pipeline rebuild with confirmed mappings
- [x] MCPC data downloaded (109 days)
- [x] Pipeline validated on Jan 6-12, 2026
- [x] Git repo initialized
- [x] Processed Parquet files generated (75 months × 3 tables)

### Weeks 3-4: TempDRL Implementation ✅
- [x] Implement TTFE (transformer temporal feature extractor) — 5 tests
- [x] Implement SAC agent (networks, replay buffer) — 8 + 6 + 4 tests
- [x] Implement feasibility projection (Stage 1 simple + Stage 2 full) — 14 tests
- [x] Build Gymnasium environment (energy_only + co_optimize modes) — 12 tests
- [x] Implement Stage 1 training loop
- [x] Implement Stage 2 training loop with progressive unfreezing
- [ ] Verify Stage 1 trains end-to-end on small data slice
- [ ] **Go/No-Go:** Stage 1 loss curves decreasing, policy beats random

### Current Phase: Pre-Training Validation
- [ ] Resolve pre-training checklist (see above)
- [ ] Transfer backfill data from Air to M4
- [ ] Re-preprocess with complete raw data
- [ ] Run baselines on test periods (TBx + MIP)
- [ ] Verify Stage 1 end-to-end on small data slice
- [ ] Commit, push, clean up git state

### Weeks 5-6: Training + Evaluation
- [ ] Stage 1 full training on 2020-2023 data
- [ ] Stage 1 validation on 2024-Sep 2025
- [ ] Stage 2 Phase 1: frozen encoder warm-up
- [ ] Stage 2 Phase 2: partial unfreeze
- [ ] Stage 2 Phase 3 (if needed): full unfreeze
- [ ] Run baselines on post-RTC+B test period
- [ ] Train-from-scratch agent on post-RTC+B data
- [ ] Run ablation study on Stage 2 components
- [ ] **Go/No-Go:** Two-stage agent outperforms TBx baseline AND train-from-scratch

### Weeks 7-8: Analysis + Write-up
- [ ] Compare across all three evaluation dimensions
- [ ] Draft results section and figures
- [ ] Sensitivity analysis across battery configurations (if time permits)

---

## EXPLICITLY DEFERRED

- 10-point EB/OC bid curve output
- DreamerV3 / model-based RL comparison
- Cross-market transfer learning
- MIP with full AS co-optimization
- LLM context module
- Meta-controller / hybrid routing
- Predict-and-Optimize benchmark (implement if time permits)
- Era-aware features (replaced by two-stage training)
- Revenue decomposition analysis (can add back if results warrant)
- Consistency/risk metrics (can add back if results warrant)
- Sample efficiency curve (can add back if results warrant)
- TTFE attention weight visualization (can add back if results warrant)

---

## KEY REFERENCES

1. **Li et al. (2024)** — arXiv:2402.19110. **Primary implementation reference.**
2. **ERCOT RTC+B Battery Overview** — ercot.com
3. **gridstatus library** — github.com/gridstatus/gridstatus (v0.35.0)
4. **Ascend Analytics RTC+B blog** — ascendanalytics.com (market context)
5. **SYSO Technologies RTC+B overview** — sysotechnologies.com (operational context)
