# CLAUDE.md — TempDRL for ERCOT RTC+B
**Date:** March 20, 2026
**Project:** Temporal-Aware Deep Reinforcement Learning for Battery Storage Bidding in ERCOT's Post-RTC+B Market
**Status:** Week 1 scaffold built → Begin data exploration and iterative implementation

---

## CRITICAL INSTRUCTIONS FOR CLAUDE CODE

1. **Work incrementally.** Do NOT attempt to build the full system at once. Explore data first, report findings, wait for confirmation, then proceed to the next step.
2. **Data governs strategy.** Every design decision (column mappings, feature engineering, action space dimensions) should be informed by what we actually find in the data, not assumptions.
3. **Ask before building.** When encountering ambiguity (e.g., unexpected column names, missing data products, API failures), stop and report rather than guessing.
4. **The old `HybridBid_Project_Handoff_v2.md` is OBSOLETE.** It describes a 3-tier hybrid architecture (forecasting ensemble + DreamerV3 + LLM context module + meta-controller) that has been dropped. Ignore it entirely. This CLAUDE.md is the single source of truth.

---

## WHAT WE'RE BUILDING

We are implementing the approach from this paper for the ERCOT market:

> **Li, J., Wang, C., Zhang, Y., & Wang, H.** "Temporal-Aware Deep Reinforcement Learning for Energy Storage Bidding in Energy and Contingency Reserve Markets." *IEEE Transactions on Energy Markets, Policy and Regulation*, Sept 2024. (arXiv:2402.19110)

The paper uses a **Soft Actor-Critic (SAC)** algorithm paired with a **Transformer-based Temporal Feature Extractor (TTFE)** for battery storage bidding in Australia's National Electricity Market (NEM), jointly optimizing energy and contingency reserve (FCAS) markets.

We are adapting this to **ERCOT's post-RTC+B market** (live since December 5, 2025), where energy and 5 ancillary service products are now co-optimized every 5 minutes in SCED.

**Target users:** Small battery storage operators (5-20 MW) who currently achieve ~56% of optimal revenue using simple time-based strategies.

**Primary deliverable:** A working system with initial results within 6-8 weeks. Publication is a secondary objective.

---

## WHY ERCOT POST-RTC+B

ERCOT was the last major US ISO to implement real-time co-optimization of energy and ancillary services. The RTC+B redesign (December 5, 2025) introduced:

- **Real-time co-optimization:** Energy and AS are simultaneously dispatched in SCED every 5 minutes (previously AS was procured only in the Day-Ahead Market).
- **Single-model ESR:** Batteries are now a single unified resource with a continuous operating range from max charge (negative MW) to max discharge (positive MW). Replaces the old "combo model" with separate generator and load resources.
- **ISO-managed SoC:** ERCOT's SCED engine directly considers telemetered State of Charge every 5 minutes to ensure dispatch feasibility. Unique among US ISOs.
- **Ancillary Service Demand Curves (ASDCs):** Replace the former Operating Reserve Demand Curve (ORDC). Scarcity is now priced inside the dispatch optimization, affecting both LMPs and MCPCs simultaneously.
- **5 AS products co-optimized:** Regulation Up, Regulation Down, Responsive Reserve (RRS), ERCOT Contingency Reserve Service (ECRS), Non-Spinning Reserve.
- **10-point monotonic Energy Bid/Offer Curve (EB/OC):** Required bidding format. Deferred for initial implementation — we use scalar actions first.
- **Set Point Deviation (SPD) compliance:** Replaces Base Point Deviation. Tolerance: max(3% of average set point, 3 MW). Penalties for deviation.

**The structural similarity to Australia's NEM** (where TempDRL was validated) makes this adaptation feasible: both markets co-optimize energy and reserves at 5-minute intervals with battery participation. Key differences: ERCOT uses nodal pricing (NEM uses regional), ERCOT has ASDCs (NEM uses different scarcity mechanisms), and ERCOT has stricter SPD compliance requirements.

---

## DATA STRATEGY

### The Data Constraint
- **Post-RTC+B data:** Dec 5, 2025 → present ≈ ~3.5 months ≈ ~30,000 five-minute intervals.
- **Pre-RTC+B data:** 2020 → Dec 4, 2025 ≈ ~5 years ≈ ~525,000 intervals per product.
- **SAC is model-free** and typically needs 100k-500k transitions to converge. Post-RTC+B data alone is insufficient.

### Chosen Approach: Era-Aware Full-History Training
Train on ALL 2020–2026 ERCOT data, with explicit features that tell the agent which market regime it's operating in:

- `is_post_rtcb` — binary flag (True after Dec 5, 2025)
- `rt_as_available` — binary flag (True when real-time AS clearing prices exist)
- `days_since_rtcb` — continuous feature (0 for pre-RTC+B, increasing after Dec 5)

**Pre-RTC+B intervals:** RT AS price columns are NaN/zero (no RT AS market existed). The agent learns that AS offers had no value before Dec 5. Energy price patterns, load cycles, solar ramps, SoC management fundamentals still transfer.

**Post-RTC+B intervals:** RT AS prices (MCPCs) become available. The agent discovers co-optimization revenue opportunities.

This is honest (we don't hide the structural break), practical (one training pipeline), and produces an analyzable result (attention weights can show how the agent adapts across the regime boundary).

### Data Access
- **Primary tool:** `gridstatus` Python library (v0.34.0+)
  - `gridstatus.Ercot` class — web scraping, works for full historical range
  - `gridstatus.ErcotAPI` class — REST API, data from Dec 2023 onward
- **ERCOT Public API:** Free registration at apiexplorer.ercot.com
- **Key data products:**
  - RT Settlement Point Prices (NP6-905-CD) — 15-min intervals
  - RT LMPs by SCED interval (NP6-788-CD) — 5-min intervals
  - DAM Settlement Point Prices (NP4-190-CD) — hourly
  - **RT AS Clearing Prices / MCPCs (NP6-332-CD)** — 5-min, post-RTC+B only (NEW)
  - DAM AS Clearing Prices (NP4-188-CD) — hourly, full history
  - Load, wind/solar actuals and forecasts — various granularities

### Canonical Schema (Parquet, UTC timestamps, 5-min intervals)
Three tables, partitioned by month:
- `energy_prices` — RT SPP (hub + zones), DAM SPP, is_post_rtcb flag
- `as_prices` — RT MCPCs (5 products, NaN pre-RTC+B), DAM AS prices (5 products), is_post_rtcb flag
- `system_conditions` — load, wind/solar actuals+forecasts, net load, is_post_rtcb flag

Column name mappings from gridstatus → canonical schema need to be determined during data exploration. The preprocessing module has TODO placeholders for this.

---

## TEMPDRL ARCHITECTURE (Li et al. adapted for ERCOT)

### MDP Formulation

**State space** s_t = (SoC_t, ρ_{t-1}, f_{t-1})
- SoC_t: Current state of charge [MWh] (fraction of E_max)
- ρ_{t-1}: Latest price vector = [LMP, MCPC_RegUp, MCPC_RegDn, MCPC_RRS, MCPC_ECRS, MCPC_NSRS] (6 values)
  - Pre-RTC+B: MCPCs are zero; use DAM AS prices as partial proxy or leave as zero
- f_{t-1}: Temporal feature vector extracted by the TTFE from the last L price observations

**ERCOT-specific state augmentation:**
- `is_post_rtcb` (binary)
- `days_since_rtcb` (continuous)
- `hour_of_day`, `day_of_week`, `month` (cyclical encoding)
- `total_load_mw`, `net_load_mw` (system conditions)
- `wind_actual_mw`, `solar_actual_mw` (renewable generation)

**Action space** a_t ∈ ℝ^6 (continuous, simplified for initial implementation):
- a_0: Net energy power p_net ∈ [-P_max, P_max] (negative = charge, positive = discharge)
- a_1-a_5: AS capacity offers [MW] for RegUp, RegDown, RRS, ECRS, NSRS (each ≥ 0)

**Constraints on actions (enforced by feasibility projection layer):**
- p_discharge + Σ(RegUp, RRS, ECRS) ≤ P_max (joint upward capacity)
- p_charge + RegDown ≤ P_max (joint downward capacity)
- SoC must support AS duration requirements:
  - RegUp/RegDown: 0.5 MWh per 1 MW award (30-min sustain)
  - RRS (PFR/UFR): 0.5 MWh per 1 MW (30-min sustain)
  - RRS (FFR): 0.25 MWh per 1 MW (15-min sustain)
  - ECRS: 1.0 MWh per 1 MW (1-hour sustain)
  - Non-Spin: 4.0 MWh per 1 MW (4-hour sustain)
- Total SoC reservation: SoC_t ≥ SoC_min + Σ(c_k × duration_k) for all awarded AS products

**Note:** The full 10-point EB/OC bid curve output is DEFERRED. Initial implementation uses scalar power outputs. The bid curve parameterization is a future refinement.

**Reward function** r_t:
```
r_t = Revenue_Energy + Σ Revenue_AS,i - Cost_Degradation - Penalty_SPD - Penalty_Imbalance
```

Where:
- Revenue_Energy = Δt × (p_dch × η_dch - p_ch / η_ch) × LMP_t
- Revenue_AS,i = Δt × c_i × MCPC_i,t (for each AS product i)
- Cost_Degradation = C_deg × (p_ch + p_dch) × Δt
- Penalty_SPD = -β_SPD × max(0, |a_t - p_actual| - δ) where δ = max(0.03 × |avg_set_point|, 3 MW)
- Penalty_Imbalance = cost of failing to sustain AS capacity if SoC drops below required level

### Transformer Temporal Feature Extractor (TTFE)

Processes a rolling window of L historical price vectors:
- Input: S_t = [ρ_{t-L+1}, ..., ρ_t] — shape (L, n_prices)
- Multi-Head Attention: S_j^SA = softmax(Q_j K_j^T / √F') × V_j
- Output: f_t — compressed temporal feature vector

**Key hyperparameters:**
- L (segment length): Start with 32 intervals (~2.7 hours). This is what Li et al. used.
- n_heads: 4 (Li et al. default)
- d_model: 64 (embedding dimension)
- n_layers: 2 transformer layers

### SAC Algorithm

Standard Soft Actor-Critic with:
- Automatic entropy coefficient tuning (α)
- Twin Q-networks (clipped double Q-learning)
- Target network with soft update (τ = 0.005)
- Replay buffer size: 1,000,000 transitions
- Batch size: 256
- Learning rate: 3e-4 (actor and critic)
- Discount factor γ: 0.99
- Actor: 2 hidden layers × 256 units, ReLU
- Critic: 2 hidden layers × 256 units, ReLU
- TTFE feeds into both actor and critic as the observation encoder

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

Configured in `configs/battery.yaml`.

---

## BASELINES FOR COMPARISON

1. **TBx (Time-Based Arbitrage):** Charge cheapest 4 hours, discharge most expensive 4 hours daily. No AS. Expected: ~40-50% of perfect foresight. Already implemented in `src/baselines/tbx.py`.

2. **Energy-Only Perfect Foresight MIP:** Solve energy-only optimization with actual future prices using CVXPY + HiGHS. This is the theoretical ceiling for energy-only strategies. Already implemented in `src/baselines/perfect_foresight.py`.

3. **Predict-and-Optimize (P&O) benchmark** (to implement later): XGBoost price forecast → energy-only MIP. This is the benchmark Li et al. compared TempDRL against (~23-24% improvement).

---

## EXISTING CODE (Week 1 Scaffold)

The project already has a working scaffold in the `hybridbid/` directory:

```
hybridbid/
├── configs/
│   ├── battery.yaml              # Battery parameters (10MW/20MWh reference)
│   └── data_products.yaml        # ERCOT data product IDs and access config
├── data/
│   ├── raw/                      # Downloaded ERCOT files
│   ├── processed/                # Clean Parquet files (canonical schema)
│   ├── mappings/                 # ESR combo→single model mapping
│   └── results/                  # Baseline and evaluation outputs
├── notebooks/
│   └── 01_data_exploration.ipynb # Day 1 exploration notebook
├── src/
│   ├── data/
│   │   ├── pipeline.py           # Main ingestion orchestrator
│   │   ├── ercot_fetcher.py      # gridstatus wrapper (API + scraping)
│   │   ├── schema.py             # Canonical Parquet schema definitions
│   │   └── preprocessing.py      # Cleaning, alignment, resampling
│   ├── baselines/
│   │   ├── tbx.py                # Time-based arbitrage baseline
│   │   ├── perfect_foresight.py  # Energy-only MIP (CVXPY + HiGHS)
│   │   └── run_baselines.py      # CLI runner
│   ├── evaluation/
│   │   ├── metrics.py            # Revenue, TB2 capture, constraint compliance
│   │   └── visualization.py      # Revenue curves, SoC trajectories, plots
│   └── utils/
│       ├── time_utils.py         # CPT/UTC conversion, ERCOT hour-ending
│       └── battery_sim.py        # Battery state simulator (13 tests pass)
├── tests/
│   └── test_battery_sim.py       # Battery simulator unit tests (all pass)
├── requirements.txt
└── README.md
```

**Status:** All 13 battery simulator tests pass. Preprocessing module has TODO placeholders for column mappings that must be filled after data exploration.

**What needs to be added (Weeks 3-8):**
```
├── src/
│   ├── models/
│   │   ├── ttfe.py               # Transformer Temporal Feature Extractor
│   │   ├── sac.py                # Soft Actor-Critic agent
│   │   ├── networks.py           # Actor, Critic, and shared network architectures
│   │   └── replay_buffer.py      # Experience replay buffer
│   ├── env/
│   │   ├── ercot_env.py          # Gymnasium-compatible ERCOT battery environment
│   │   └── feasibility.py        # Action feasibility projection (AS constraints)
│   └── training/
│       ├── train.py              # Training loop
│       └── config.py             # Hyperparameter configuration
```

---

## 8-WEEK PLAN

### Weeks 1-2: Data Pipeline + Baselines
- [x] Project scaffold built
- [x] Battery simulator implemented and tested
- [ ] Run data exploration notebook — discover actual gridstatus column names
- [ ] Update preprocessing.py column mappings based on exploration findings
- [ ] Run data pipeline for 2024-2026 (priority) then backfill 2020-2023
- [ ] Run TBx and perfect foresight baselines
- [ ] Confirm post-RTC+B AS data (NP6-332-CD MCPCs) is accessible
- [ ] **Go/No-Go #1:** Data pipeline working, baselines producing numbers

### Weeks 3-4: TempDRL Implementation
- [ ] Implement TTFE (Transformer Temporal Feature Extractor)
- [ ] Implement SAC agent with TTFE as observation encoder
- [ ] Build Gymnasium-compatible ERCOT environment wrapper
- [ ] Implement reward function with ERCOT-specific penalties (SPD, AS imbalance)
- [ ] Implement action feasibility projection (AS duration/SoC constraints)
- [ ] Verify training loop runs end-to-end on a small data slice
- [ ] **Go/No-Go #2:** Agent training loop working, loss curves decreasing

### Weeks 5-6: Training + Evaluation
- [ ] Train on full 2020-2026 dataset with era-aware features
- [ ] Hyperparameter sweep: L (segment length), learning rate, replay buffer size
- [ ] Evaluate on held-out test sets:
  - Test Set 1 (pre-RTC+B): Oct-Nov 2025
  - Test Set 2 (post-RTC+B): Dec 2025-Jan 2026
  - Test Set 3 (ongoing): Feb 2026+
- [ ] Compare against baselines (TBx, perfect foresight)
- [ ] **Go/No-Go #3:** Agent outperforms TBx baseline

### Weeks 7-8: Analysis + Write-up
- [ ] Attention weight visualization (what temporal patterns does the TTFE learn?)
- [ ] Pre vs. post RTC+B performance analysis (natural experiment)
- [ ] Revenue decomposition (energy vs. each AS product)
- [ ] Sensitivity analysis across battery configurations (5 MW, 10 MW, 20 MW)
- [ ] Draft results section and figures

---

## TRAIN / VALIDATION / TEST SPLITS

| Split | Period | Purpose |
|-------|--------|---------|
| Train | 2020-01-01 → 2023-12-31 | Main training data (pre-RTC+B only) |
| Validation | 2024-01-01 → 2025-09-30 | Hyperparameter tuning |
| Test Pre-RTC+B | 2025-10-01 → 2025-12-04 | Natural experiment control |
| Test Post-RTC+B | 2025-12-05 → 2026-01-31 | Natural experiment treatment |
| Test Ongoing | 2026-02-01 → present | Out-of-sample evaluation |

---

## EXPLICITLY DEFERRED (Future Work)

- 10-point EB/OC bid curve output (parameterized action space)
- DreamerV3 / model-based RL comparison
- Cross-market transfer learning (pre-train on Australia NEM or NYISO data)
- MIP with full AS co-optimization (energy-only MIP is sufficient as baseline)
- LLM context module
- Meta-controller / hybrid MIP+RL routing
- Predict-and-Optimize (P&O) benchmark (implement if time permits in Week 6)

---

## KEY REFERENCES

1. **Li et al. (2024)** — "Temporal-Aware Deep Reinforcement Learning for Energy Storage Bidding in Energy and Contingency Reserve Markets." IEEE TEMPR. arXiv:2402.19110. **This is the paper we're implementing.**
2. **ERCOT RTC+B Battery Overview** — https://www.ercot.com/files/docs/2025/07/15/RTC-B-Battery-Overview.pdf
3. **ERCOT RTC+B Go-Live Announcement** — https://www.ercot.com/news/release/12052025-ercot-goes-live
4. **Modo Energy: RTC+B AS Duration Requirements** — https://modoenergy.com/research/en/rtcb-real-time-cooptimization-rtc-ercot-ancillary-service-duration-soc-management
5. **Tyba Energy: Guide to ERCOT RTC+B** — https://www.tyba.ai/resources/guides/guide-to-ercot-rtcb/
6. **gridstatus library** — https://github.com/gridstatus/gridstatus (v0.34.0+, BSD-3 license)

---

## IMMEDIATE NEXT STEP

**Start with data exploration.** Run the notebook `notebooks/01_data_exploration.ipynb` or equivalent exploration to:
1. Install gridstatus, inspect available methods for ERCOT
2. Pull 1 week of sample data for each key product (pre and post RTC+B)
3. Document actual column names, data types, granularity
4. Confirm RT AS MCPCs (NP6-332-CD) are accessible post-RTC+B
5. Check how far back historical data is accessible via gridstatus

Report findings before proceeding to pipeline implementation. Do not build further until column mappings are confirmed.
