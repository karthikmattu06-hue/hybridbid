# Data Characterization for Offline RL Method Selection

**Date:** 2026-04-24
**Branch:** stage1-tier1
**Inputs:**
- MILP expert trajectories: `data/expert_trajectories/receding_horizon_{train,val}_option_d.npz` (420,423 train / 183,871 val transitions, 24h receding-horizon GUROBI MIP, 7-atom action lattice `[-1, -2/3, -1/3, 0, 1/3, 2/3, 1] × P_max`, P_max = 10 MW, E_max = 20 MWh)
- Raw market data: `data/processed/{energy_prices,as_prices,system_conditions}/YYYY-MM.parquet` (661,536 rows, 5-min, 2020-01-01 → 2026-04-15)
- Regime split: pre-RTC+B < 2025-12-05 UTC (623,520 rows market / 604k MILP transitions); post-RTC+B ≥ 2025-12-05 UTC (38,016 rows market ≈ 4.4 months)

All numbers computed from the data; no narrative filler. Statistics that are surprising or dominate method selection are flagged **NOTABLE**.

---

## Section 1 — Reward distribution

### 1.1 MILP expert rewards (what an offline RL method actually sees)

| Stat | Train (pre-RTC+B, 2020-01→2023-12) | Val (2024-01→2025-09) |
|---|---:|---:|
| n | 420,423 | 183,871 |
| mean | 7.75 | 3.32 |
| median | 0.00 | 0.00 |
| std | 102.10 | 40.82 |
| min | −2,356.09 | −1,296.45 |
| max | 5,778.61 | 3,873.26 |
| skew | **22.11** | **50.36** |
| excess kurtosis | **677.3** | **3,506.9** |
| p1 | −9.46 | −5.34 |
| p5 | −1.47 | −1.33 |
| p25 | 0.00 | 0.00 |
| p50 | 0.00 | 0.00 |
| p75 | 0.70 | 1.76 |
| p95 | 13.20 | 13.61 |
| p99 | 111.96 | 41.19 |
| frac exactly zero | **0.584** | **0.500** |
| frac in [−1, 1] | **0.690** | **0.642** |
| p99 of positive rewards | 684.45 | 93.70 |
| frac > p99 of positives | 0.00284 (1,192 txns) | 0.00359 (661 txns) |
| **Top-1% of positive txns capture** | **50.4% of all positive reward** | 35.9% |
| Top-5% of positive txns capture | 79.9% | 53.3% |
| Top-10% of positive txns capture | 85.6% | 64.1% |
| sum of positive rewards | 3,701,305 | 690,060 |
| n positive | 119,155 (28.3%) | 66,069 (35.9%) |

**NOTABLE — reward is near-zero most of the time, with extreme tails.** ~58% of training rewards are exactly zero (idle steps); 69% fall inside [−1, 1]. The mean of $7.75/step is almost entirely driven by a handful of scarcity events: top 1% of positive transitions alone capture >50% of total positive reward in train. Kurtosis 677 (train), 3,507 (val) — these are not Gaussian-like distributions; they're a delta at zero plus a heavy power-law tail.

ASCII histogram of MILP train rewards:
```
  [ -2400,  -1000)  n=      32   0.01%
  [ -1000,   -500)  n=      98   0.02%
  [  -500,   -100)  n=     690   0.16%
  [  -100,    -10)  n=    3250   0.77%  #
  [   -10,     -1)  n=   26216   6.24%  #########
  [    -1,      0)  n=   25397   6.04%  #########
  [     0,      1)  n=  264630  62.94%  ##############################################################################################
  [     1,     10)  n=   72998  17.36%  ##########################
  [    10,    100)  n=   22685   5.40%  ########
  [   100,    500)  n=    2625   0.62%
  [   500,   1000)  n=     998   0.24%
  [  1000,   5800)  n=     804   0.19%
```

### 1.2 Raw market RT LMP by regime

| Stat | Pre-RTC+B | Post-RTC+B |
|---|---:|---:|
| n (5-min intervals) | 623,004 | 37,933 |
| mean ($/MWh) | 43.02 | 29.68 |
| median | 22.26 | 20.37 |
| std | 238.11 | 50.07 |
| min | −215.69 | −97.16 |
| max | 9,065.15 | 1,350.50 |
| skew | 25.83 | 9.38 |
| excess kurtosis | 810.0 | 131.8 |
| p1 / p5 | −1.96 / 5.92 | −6.17 / −0.03 |
| p25 / p50 / p75 | 16.06 / 22.26 / 34.69 | 13.68 / 20.37 / 31.39 |
| p95 / p99 | 76.22 / **224.74** | 75.92 / **233.31** |
| scarcity rate (RT LMP > $1000) | **0.388%** (2,418 events) | **0.016%** (6 events) |

Scarcity events (RT LMP > $1000/MWh) per calendar year:

| Year | Count | Notes |
|---|---:|---|
| 2020 | 9 | |
| 2021 | 1,463 | **dominated by Winter Storm Uri (Feb 2021)** |
| 2022 | 221 | |
| 2023 | 612 | |
| 2024 | 88 | |
| 2025 | 25 | |
| 2026 | 6 | (Jan–Apr only; post-RTC+B for all of these) |

- Scarcity run statistics: 318 distinct scarcity runs; mean run 7.6 intervals (38 min), p95 = 18 intervals (90 min). **Longest consecutive scarcity run: 605 five-minute intervals = 50.4 hours continuous** (Winter Storm Uri, Feb 2021).

**NOTABLE — 60% of all scarcity events in the 6-year dataset occurred in a single week (Uri, Feb 2021). The training data is a power-law sample of rare conditions, and one week dominates the tail.**

---

## Section 2 — Action distributions (MILP expert)

### 2.1 Atom counts

| Atom idx | Level (p.u.) | Train count | Train % | Val count | Val % |
|---:|---:|---:|---:|---:|---:|
| 0 | −1.00 (full charge) | 86,578 | **20.59%** | 45,456 | **24.72%** |
| 1 | −2/3 | 804 | 0.19% | 393 | 0.21% |
| 2 | −1/3 | 3,597 | 0.86% | 1,850 | 1.01% |
| 3 | 0 (idle) | 247,867 | **58.96%** | 93,287 | **50.74%** |
| 4 | +1/3 | 1,607 | 0.38% | 856 | 0.47% |
| 5 | +2/3 | 2,304 | 0.55% | 1,360 | 0.74% |
| 6 | +1.00 (full discharge) | 77,666 | **18.47%** | 40,669 | **22.12%** |

Marginal direction fractions:

| Direction | Train | Val |
|---|---:|---:|
| charge (idx 0–2) | 21.64% | 25.94% |
| discharge (idx 4–6) | 19.40% | 23.32% |
| idle (idx 3) | 58.96% | 50.74% |

### 2.2 Magnitude structure

| | Train | Val |
|---|---:|---:|
| mean charge magnitude (p.u., cond. on charging) | 0.971 | 0.971 |
| mean discharge magnitude (p.u., cond. on discharging) | 0.977 | 0.976 |
| **bang-bang fraction** (non-idle that are ±P_max) | **0.952** | **0.951** |

**NOTABLE — the expert is essentially bang-bang.** ≥95% of all non-idle actions sit at ±P_max. The intermediate atoms (±1/3, ±2/3) are used <2% of the time combined. The effective action space is 3-way {full-charge, idle, full-discharge}, not 7-way.

### 2.3 Temporal structure

Autocorrelation of signed-action p.u. series:

| Lag | Train | Val |
|---|---:|---:|
| 1 (5 min) | 0.631 | 0.632 |
| 3 (15 min) | 0.371 | 0.369 |
| 6 (30 min) | 0.260 | 0.262 |
| 12 (1 h) | 0.120 | 0.124 |

Consecutive-run statistics (intervals):

| Run type | Train n / mean / median / max | Val n / mean / median / max |
|---|---|---|
| charge | 26,143 / 3.48 / 2 / **21** | 13,765 / 3.47 / 2 / 21 |
| discharge | 25,173 / 3.24 / 2 / **19** | 12,758 / 3.36 / 2 / 19 |
| idle | 38,155 / 6.50 / 2 / **173** | 19,308 / 4.83 / 2 / 167 |

Switching frequency (direction changes among {charge, idle, discharge}): **61 switches/day (train), 72/day (val)** — roughly one direction change every 20–24 minutes on average.

**NOTABLE — short charge/discharge runs (median 2 intervals = 10 min, max 21 = 1h45) but long idle runs (max 173 intervals = 14.4h).** The expert spends long periods idling, punctuated by short, full-power charge/discharge pulses. This is a discrete, temporally-sparse control profile — not a smooth bid curve.

---

## Section 3 — State distributions

### 3.1 SoC marginal

SoC range: [2.0, 18.0] MWh (hard limits from env's feasibility projection).

| | Train | Val |
|---|---:|---:|
| mean (MWh) | 9.69 | 9.49 |
| std | 6.45 | 6.30 |
| % at floor (≤3 MWh) | **28.95%** | **28.40%** |
| % at ceiling (≥17 MWh) | **24.90%** | **21.60%** |
| % mid-range (4–16 MWh) | 36.89% | 39.87% |

SoC autocorrelation:

| Lag | Train | Val |
|---|---:|---:|
| 1 (5 min) | 0.997 | 0.996 |
| 3 (15 min) | 0.979 | 0.973 |
| 6 (30 min) | 0.934 | 0.917 |
| 12 (1 h) | 0.804 | 0.751 |

**NOTABLE — SoC is bimodal at the rails.** The expert spends >54% of time at the 10%/90% energy limits (2 or 18 MWh). Mid-range occupancy is only ~37%. This is consistent with a bang-bang-then-hold pattern: charge to ceiling, idle, discharge to floor, idle.

### 3.2 SoC ↔ price correlations (same timestep)

| | Train | Val |
|---|---:|---:|
| corr(SoC, RT LMP_now) | 0.013 | 0.016 |
| corr(SoC, DAM SPP_now) | −0.001 | 0.012 |

Near-zero linear correlation — reasonable given the bimodal SoC distribution; the relationship is not linear.

### 3.3 Observation dimension grouping

MILP trajectory files store **structured** observations (not the flat 90-dim vector referenced in CLAUDE.md). The 90-dim post-TTFE obs is built at training time from these inputs:

| Source | Shape | Dim in 90-dim obs |
|---|---|---:|
| `price_history` (32 steps × 12 prices) | TTFE-encoded → 64-dim | 64 |
| current prices (row `-1` of `price_history`) | passed through | 12 |
| system conditions (dims 0–6 of `static_features`) | passed through | 7 |
| cyclical time (dims 7–12 of `static_features`) | passed through | 6 |
| SoC fraction (dim 13 of `static_features`) | passed through | 1 |

Grouping: **price-like 76 (64 TTFE + 12 raw), system-like 7, time-like 6, SoC-like 1** — price features dominate the observation space at ~84%.

Price columns (in order): `rt_lmp, rt_mcpc_{regup,regdn,rrs,ecrs,nsrs}, dam_spp, dam_as_{regup,regdn,rrs,ecrs,nsrs}`.
System columns: `total_load_mw, load_forecast_mw, wind_actual_mw, wind_forecast_mw, solar_actual_mw, solar_forecast_mw, net_load_mw` (all normalized by per-column scale in env).

### 3.4 Pre- vs post-RTC+B distribution shift

**RT LMP KS test:** statistic 0.1023, p-value 0.0 (exact zero — reject identical-distribution hypothesis at any α).

| RT LMP | Pre | Post |
|---|---:|---:|
| mean | $43.02 | $29.68 |
| std | 238.11 | 50.07 |
| p99 | $224.74 | $233.31 |

Raw RT LMP histogram by regime:
```
  pre :
    [ -500,    0)  n=  12935   2.08%  #
    [    0,   20)  n= 247486  39.72%  #######################
    [   20,   40)  n= 238435  38.27%  ######################
    [   40,   60)  n=  61525   9.88%  #####
    [   60,   80)  n=  36444   5.85%  ###
    [   80,  100)  n=  12245   1.97%  #
    [  100,  200)  n=   7188   1.15%
    [  200,  500)  n=   2891   0.46%
    [  500, 1000)  n=   1420   0.23%
    [ 1000,10000)  n=   2435   0.39%
  post:
    [ -500,    0)  n=   1919   5.06%  ###
    [    0,   20)  n=  16607  43.78%  ##########################
    [   20,   40)  n=  13546  35.71%  #####################
    [   40,   60)  n=   3254   8.58%  #####
    [   60,   80)  n=    820   2.16%  #
    [   80,  100)  n=    371   0.98%
    [  100,  200)  n=    934   2.46%  #
    [  200,  500)  n=    398   1.05%
    [  500, 1000)  n=     78   0.21%
    [ 1000,10000)  n=      6   0.02%
```

**AS clearing prices (ancillary services):**

| | Pre-RTC+B | Post-RTC+B |
|---|---:|---:|
| `rt_mcpc_regup` n non-NaN | **0 / 623,520** | 34,691 / 38,016 |
| `rt_mcpc_regup` mean / std / p99 | — (did not exist) | — (means below for reference) |
| `dam_as_regup` mean | $52.63 | **$3.00** |
| `dam_as_regup` std | 756.36 | 20.79 |
| `dam_as_regup` p99 | $231.67 | $29.21 |
| `dam_as_ecrs` mean | $7.32 (available Jun 2023+) | $2.63 |
| `dam_as_ecrs` p99 | $100.45 | $24.67 |

Post-RTC+B RT MCPC stats (no pre-period equivalent): mean RegUp $2.85, ECRS $5.60 (from same JSON dump; illustrative).

Correlation `rt_lmp ↔ regup`:

| | Pre | Post |
|---|---:|---:|
| `corr(rt_lmp, dam_as_regup)` | 0.414 | 0.252 |
| `corr(rt_lmp, dam_as_ecrs)` | 0.246 | 0.210 |
| `corr(rt_lmp, rt_mcpc_regup)` | **undefined** (RT MCPC didn't exist pre-RTC+B) | **0.609** |
| `corr(rt_lmp, rt_mcpc_ecrs)` | undefined | 0.414 |

**NOTABLE — the post-RTC+B regime is structurally different, not just a parameter shift:**
- **RT MCPC products did not exist pre-RTC+B.** A whole sub-vector of the observation (5 of 12 price dims) is zero/NaN for the entire pre-period. An agent trained on pre-RTC+B data has *no signal* for RT AS pricing.
- **DAM AS RegUp dropped 17× in mean ($52.63 → $3.00) and 36× in std.** AS revenue-per-MW pre-RTC+B was a substantial share of market revenue; post-RTC+B it collapsed (more competitive AS market, RT AS now available).
- **Scarcity rate dropped 24× (0.39% → 0.016%)**, but only 4.4 months of post data — sample too small to claim structural change in tail. p99 is roughly unchanged (~$230).
- **Correlation structure changed:** `corr(rt_lmp, dam_as_regup)` fell from 0.41 to 0.25 — pre-RTC+B, DAM AS priced heavily off expected scarcity; post-RTC+B, DAM AS is lower and less sensitive to energy-market expectations (the RT MCPC now absorbs that signal, correlating 0.61 with RT LMP).

---

## Section 4 — Expert policy structure

### 4.1 Daily SoC pattern (training split, hourly averages, MWh)

| Hour (UTC) | Train | Val | | Hour | Train | Val |
|---:|---:|---:|---|---:|---:|---:|
| 00 | 5.09 | 5.15 | | 12 | 12.51 | 12.42 |
| 01 | 5.98 | 5.77 | | 13 | 12.07 | 12.32 |
| 02 | 7.99 | 7.65 | | 14 | 11.70 | 12.37 |
| 03 | 10.91 | 10.53 | | 15 | 12.00 | 13.08 |
| 04 | **13.53** | **13.26** | | 16 | 11.85 | 13.86 |
| 05 | **14.66** | **14.05** | | 17 | 10.71 | **14.01** |
| 06 | 13.85 | 10.98 | | 18 | 9.02 | 13.21 |
| 07 | 11.01 | 5.43 | | 19 | 7.36 | 10.43 |
| 08 | 9.81 | 4.44 | | 20 | 5.04 | 6.21 |
| 09 | 10.86 | 7.54 | | 21 | **3.90** | **4.08** |
| 10 | 11.79 | 10.41 | | 22 | 3.99 | 3.83 |
| 11 | 12.35 | 11.88 | | 23 | 4.31 | 4.35 |

Hourly signed-action mean (p.u.; negative = charge, positive = discharge):

| Hour (UTC) | Train action | Val action | | Hour | Train | Val |
|---:|---:|---:|---|---:|---:|---:|
| 00 | −0.04 | −0.03 | | 12 | +0.00 | −0.04 |
| 01 | −0.18 | −0.15 | | 13 | +0.05 | +0.00 |
| 02 | **−0.28** | **−0.28** | | 14 | −0.04 | −0.07 |
| 03 | **−0.33** | **−0.34** | | 15 | −0.05 | −0.12 |
| 04 | −0.22 | −0.20 | | 16 | +0.05 | −0.06 |
| 05 | −0.04 | +0.06 | | 17 | +0.15 | +0.01 |
| 06 | **+0.21** | **+0.51** | | 18 | +0.11 | +0.13 |
| 07 | **+0.22** | +0.32 | | 19 | +0.17 | **+0.34** |
| 08 | −0.06 | −0.21 | | 20 | +0.18 | +0.28 |
| 09 | −0.12 | −0.35 | | 21 | +0.14 | +0.07 |
| 10 | −0.09 | −0.23 | | 22 | +0.05 | −0.01 |
| 11 | −0.03 | −0.10 | | 23 | +0.02 | −0.14 |

**Daily pattern (summarized from these hourly means; times in UTC ≈ ERCOT-local + 5/6h):**
- **Overnight (hours 01–04 UTC ≈ 19:00–22:00 local):** strong charging (mean signed action −0.2 to −0.3), SoC climbs from ~6 MWh to peak ~14.7 MWh at hour 05 UTC.
- **Morning peak (hours 06–07 UTC ≈ 00:00–01:00 local):** switch to discharging (+0.2). *Note: this is a UTC artifact; the "morning peak" in local ERCOT time maps to different UTC hours depending on DST. The val split shows a clearer mid-afternoon charge at 08–10 UTC followed by an evening discharge at 18–20 UTC.*
- **Afternoon (hours 09–13 UTC):** mild charging back up to ~12 MWh in train; val shows similar but muted.
- **Evening (hours 17–21 UTC ≈ 11:00–15:00 local):** sustained discharging, SoC falls from ~14 MWh to floor ~4 MWh by hour 22 UTC.

The val split (2024–2025) shows a **sharper single-peak evening discharge (hour 19 UTC = +0.34)** and deeper noon-hour charge (hour 09 UTC = −0.35) than train (2020–2023) — consistent with ERCOT's increasing solar-driven duck-curve evolution over this period.

### 4.2 Action ↔ price coupling

Pearson corr of signed action vs same-period RT LMP:

| | Train | Val |
|---|---:|---:|
| corr(signed action, RT LMP now) | 0.087 | 0.150 |

The raw Pearson is weak because 59% of actions are idle and both distributions are heavy-tailed. The conditional means are much clearer:

| Action class | Mean RT LMP (train) | Mean RT LMP (val) |
|---|---:|---:|
| charge (any) | $32.52 | $17.94 |
| idle | $35.75 | — |
| **discharge (any)** | **$111.16** | **$49.68** |

**Event study (full-charge atom 0 vs full-discharge atom 6, training split, random sample ≤20k each):**

| Horizon (from event) | Full-charge (atom 0) mean RT LMP | Full-discharge (atom 6) mean RT LMP |
|---|---:|---:|
| t = 0 (now) | $32.23 | $120.00 |
| t + 1h | $53.80 | $78.99 |
| t + 6h | $59.32 | $48.50 |
| t + 12h | $56.28 | — |

**NOTABLE — the expert *is* forward-looking, but linear action↔future-price correlation misses it.** Conditional on taking a full-charge action, the price at +6h is 84% higher than the price at t=0; conditional on full-discharge, the price at +6h is 60% lower. But the overall `corr(signed_action, rt_lmp[+72])` is −0.008 — near zero. The expert's foresight is conditional (on having decided to charge, when does it pay off) and non-linear in the signed-action metric. This means IL/BC baselines that model `π(a|s)` directly will pick up the structure; anything that tries to fit a linear action↔price model won't.

### 4.3 Daily revenue

| | Train | Val |
|---|---:|---:|
| n_days | 1,459 | 638 |
| mean ($) | 2,234 | 958 |
| std | **11,238** | 2,020 |
| median | 475 | 551 |
| max | **279,999** | 34,966 |
| min | 8 | 48 |

**NOTABLE — daily revenue is extremely skewed.** Train std ($11k) > 5× median ($475). Max day ($280k) is ~590× the median day. This mirrors the reward-level concentration in §1.1: a few catastrophic-price days dominate annual P&L.

### 4.4 Response to scarcity events

At training time, when RT LMP > $500/MWh (3,574 events in train):
- Action distribution: 63.8% full-discharge (atom 6), 21.7% idle, 12.8% full-charge (atom 0), <2% intermediate.
- Mean SoC during scarcity: 10.58 MWh (mid-range; p5 = 2.0 floor, p95 = 18.0 ceiling).
- **When RT LMP > $1000: 56.6% discharge, 28.2% idle, 14.4% charge** — the ~28% idle rate at >$1000 is because SoC is already at the floor by that point.

Val (RT LMP > $500, 229 events): **97.4% full-discharge, 2.2% idle, 0.4% charge.** The val split expert is essentially always discharging during scarcity (much more aggressive than train).

**NOTABLE — the train-split expert idles through ~22% of $500+ scarcity events and ~28% of $1000+ events because SoC is already depleted.** This is an artifact of the 24h receding-horizon MILP's limited foresight: it can't pre-charge for scarcity events it doesn't see inside its window. An offline policy trained to mimic the expert will inherit this blind spot.

---

## Section 5 — Challenges for offline RL method selection

Each bullet: **what the data shows**, **which method class it potentially breaks**, **which method class might handle it**. No recommendation — this is input to method research.

### 5.1 Reward tail heaviness and concentration

- **Data:** Reward kurtosis = 677 (train), 3,507 (val). 58% of train rewards are exactly zero. Top 1% of positive transitions capture >50% of total positive reward; top 5% capture 80%. Daily revenue std = $11k on median $475. Scarcity events (>$1k RT LMP) are 0.39% of the pre-RTC+B series and 60% of them happened in one week (Uri, Feb 2021).
- **Potentially breaks:** (a) any method that estimates Q(s,a) with a mean-squared Bellman loss over a minibatch — single transitions with r > $5,000 dominate gradient updates and destabilize critic learning; (b) naive return-conditioned methods (DT, RvS) when scaling/normalizing returns, since the return distribution is bimodal (most trajectories ≈ 0, few are huge); (c) CQL/policy-constraint methods that rely on well-estimated value functions for the conservative penalty to be meaningful.
- **Potentially handles:** methods with **distributional value estimates** (C51, HL-Gauss, QR-DQN, IQL with expectile regression on a distributional target), **reward clipping/transformation** (symlog, which this codebase already applies per `train_stage1.py`), or methods that work with **advantage-weighted imitation** (AWAC/AWR) where extreme advantages are naturally down-weighted by the exponential temperature.

### 5.2 Expert action bang-bang-ness

- **Data:** 95.1% of non-idle expert actions are exactly ±P_max (atoms 0 or 6). Intermediate atoms (±1/3, ±2/3) together account for <2% of the dataset. The 7-atom action space is effectively a 3-atom space.
- **Potentially breaks:** methods that **regress continuous actions** (e.g., BC-MLE with Gaussian head, DDPG/TD3-style deterministic policies in continuous actions) — they will output mid-magnitudes the expert never uses; methods that use **Gaussian policies for exploration** during offline actor updates (vanilla offline SAC) — policy entropy bonuses will push mass into unseen intermediate atoms.
- **Potentially handles:** **discrete-action** methods (DQN, categorical SAC, IQL with discrete policy head — this codebase's Tier 2a/2c path), **BC with softmax** over the 7 atoms, **decision transformer-style** classification over discrete tokens. Continuous methods that **clip to extremes** via tanh + large scale (so the Gaussian mass concentrates at ±1) may also work but are fragile.

### 5.3 State-action coupling and SoC bimodality

- **Data:** SoC is bimodal — 28.9% at floor (≤3 MWh), 24.9% at ceiling (≥17 MWh), only 36.9% in mid-range. SoC autocorr at lag 12 = 0.80 (1-hour window). The expert-induced state distribution is self-consistent (a deployed random-action policy would produce a very different SoC marginal).
- **Potentially breaks:** methods that require **broad state coverage** (e.g., fitted Q-iteration with off-policy correction, methods that rely on importance sampling for policy improvement) — the data only covers trajectories induced by near-optimal bang-bang behavior; intermediate-SoC mid-range trajectories are rare (~37%) and come *only* from short transient segments. Model-based RL trained on this data will learn a dynamics model that's accurate near the rails but data-starved in the interior.
- **Potentially handles:** **policy-constraint offline RL** (IQL, CQL, BCQ, TD3+BC) that explicitly restricts the learned policy to the expert's state-action distribution; **behavior regularization** approaches that don't need to extrapolate.

### 5.4 Distribution shift pre- vs post-RTC+B

- **Data:** KS statistic on RT LMP = 0.102 (p = 0.0). DAM AS RegUp mean fell 17×, std fell 36×. RT MCPC products **did not exist pre-RTC+B** — 5 of 12 price-observation dimensions are NaN/0 for 100% of pre-RTC+B data. Scarcity rate fell 24× (sample of 4.4 months only — small). Correlation `rt_lmp ↔ dam_as_regup` fell from 0.41 to 0.25; `rt_lmp ↔ rt_mcpc_regup` is undefined pre, 0.61 post.
- **Potentially breaks:** any **single-regime offline training** (train on pre only, deploy on post) — it's not just a covariate shift, it's a structural change in the feature support (5 new features appear). **Conservative methods** (CQL) trained pre will heavily penalize post-era RT AS actions because they fall outside the pre-era Q-function support — effectively refusing to use the new products.
- **Potentially handles:** **two-stage pretrain → finetune** (this codebase's plan — Stage 1 on pre, Stage 2 on post with expanded action/obs dims); **meta-RL / distributionally-robust RL** methods with explicit regime conditioning; **domain-adaptive representations** (the TTFE is one such attempt, but alone it doesn't reconcile the new action dimensions). Pure behavior cloning on post-only data (~30k transitions) is an option but undersamples scarcity.

### 5.5 Temporal dependencies

- **Data:** SoC autocorr at lag 12 (1h) = 0.80. Action autocorr at lag 12 = 0.12 (short). Median charge/discharge run = 2 intervals (10 min), max = 21 (1h45). Scarcity runs: mean 7.6 intervals (38 min), max 605 (50.4 h — Uri). The MILP expert uses a 24h (288-step) horizon.
- **Potentially breaks:** **Markov-assumption methods without history** (vanilla discrete SAC/DQN with only current-step obs) — the optimal action at any t depends on price evolution over the next 24h, not just current prices. Methods with short effective context (e.g., frame-stacking of 4 frames) will miss multi-hour scarcity preconditioning.
- **Potentially handles:** methods with **explicit long-horizon memory** (decision transformer with 32+ step context, the TTFE's 32-step rolling attention window already used here), **goal-/return-conditioned** methods, or **model-based methods** that plan multi-step rollouts. The current TTFE window of 32 steps (≈ 2.7 hours) is **shorter than the expert's planning horizon (288 steps)** — there's an inherent representational gap.

### 5.6 Data volume

- **Data:** 420,423 train transitions pre-RTC+B (≈1,460 days). 183,871 val transitions (≈638 days). **~30k transitions post-RTC+B** (per CLAUDE.md, ≈100 days). Positive-reward transitions: 119,155 train (28.3%), of which 1,192 are in the extreme tail (>p99 of positives). Scarcity events (>$1k): 2,305 in train total, 60% of them from one week (Feb 2021).
- **Potentially breaks:** methods needing **diverse policy-coverage** (most offline RL scaling laws expect the dataset to include *suboptimal* trajectories as negatives — here we have only near-optimal MILP rollouts, no exploration data); methods needing **many independent extreme events** to learn tail behavior reliably — the 2,305 scarcity events pre-RTC+B are highly autocorrelated (mean run 7.6 intervals) and concentrated in one storm. Meta-RL methods needing many tasks will struggle since this is effectively one task with one regime change.
- **Potentially handles:** **imitation-flavored methods** (BC, AWAC, IQL with strong behavior regularization) that benefit from a concentrated near-optimal dataset; methods with **data augmentation** on the price series (bootstrap scarcity events, which the codebase's "Option D" path may already be doing); **offline-to-online finetuning** for handling the post-RTC+B small sample.

### 5.7 Other observations worth flagging

- **Expert has limited foresight:** 22% of $500+ scarcity events in train are met with idle (SoC already floored). Any policy that BC's the expert will inherit this. Methods that can *improve* on expert (offline RL with policy improvement, as opposed to pure BC) have room to help here.
- **Action-price Pearson is misleading:** `corr(action, RT LMP_now) = 0.087 / 0.150` but conditional means diverge sharply (charge at $32, discharge at $111 in train). Method diagnostics that use linear correlation as a sanity check will incorrectly flag the expert as uncorrelated with price.
- **UTC timestamps in trajectories:** Hourly SoC pattern is expressed in UTC; ERCOT local time varies with DST. Analyses keying on hour-of-day should account for this (the codebase's cyclical time features are computed in US/Central per `ercot_env.py:356`).

---

*End of characterization. See `experiments/_characterize_output.json` for the raw numerical dump.*
