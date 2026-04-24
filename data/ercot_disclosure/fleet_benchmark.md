# ERCOT ESR Fleet Benchmark — T-60 Panel
**Eval window:** 2026-01-01 → 2026-02-23 (54 operating days, all post-RTC+B)
**Data source:** ERCOT 60-Day SCED + DAM Disclosure (NP3-965-ER / NP3-966-ER) via gridstatus 0.35.0
**Computed:** 2026-04-24

---

## Fleet Distribution ($/kW-year equivalent, 269 ESRs)

| Metric | Full fleet (n=269) | 5–20 MW cohort (n=120) |
|---|---:|---:|
| Median | **$24.93** | **$25.15** |
| P25 | $14.76 | $14.78 |
| P75 | $32.23 | $28.17 |
| P90 | $42.45 | $36.32 |
| Max | $159.38 | $159.38 |

*$/kW-year equivalent = (mean daily revenue / nameplate MW / 1000) × 365*

---

## Top 10 ESRs by $/kW (window period)

| Rank | Resource | Nameplate MW | $/kW-yr equiv |
|---:|---|---:|---:|
| 1 | PALACIOS_ESR1 | 9.9 | $159.38 |
| 2 | ANDMDSLR_ESR2 | 79.4 | $109.21 |
| 3 | CNLY_ESS_ESR1 | 125.1 | $92.24 |
| 4 | SHAMROCK_ESR1 | 99.5 | $73.03 |
| 5 | CROSSTRL_ESR1 | 57.3 | $72.94 |
| 6 | ENDPARKS_ESR1 | 52.0 | $64.25 |
| 7 | GIGA_ESS_ESR1 | 125.2 | $62.49 |
| 8 | CHIL_SLR_ESR1 | 151.6 | $58.67 |
| 9 | MUSTNGCK_ESR1 | 71.0 | $52.67 |
| 10 | SWOOSEII_ESR1 | 101.1 | $52.65 |

---

## Revenue Composition (fleet-wide, $65.8M total)

| Stream | Amount | Share |
|---|---:|---:|
| DAM Energy | $20,875,709 | 31.7% |
| RT Energy (imbalance) | $14,318,389 | 21.8% |
| DAM AS (RegUp/Down/RRS/ECRS/NonSpin) | $25,836,392 | 39.3% |
| RT AS (post-RTC+B new product) | $4,725,427 | 7.2% |

**DAM AS dominates** (39.3%). RT AS is nonzero (confirms post-RTC+B co-optimization active), contributing 7.2%.

---

## Scarcity Event Impact

Winter Storm Fern (Jan 26 2026) contributed disproportionately to window revenue:

| | Median fleet $/kW-day |
|---|---:|
| Jan 26 only | **$1.042** ($380/kW-yr equiv) |
| All other days (53 days) | $0.044 ($16.00/kW-yr equiv) |
| Full 54-day window | $0.068 ($24.93/kW-yr equiv) |

The gap between $24.93/kW-yr (54-day annualized) and $16.00/kW-yr (excluding Jan 26) quantifies how much of the window annualized is driven by one day. Compared to Modo Energy's 2025 full-year fleet median of ~$29/kW, this window is modestly below on a steady-state basis but boosted by Fern.
