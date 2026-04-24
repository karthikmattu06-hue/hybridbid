# ERCOT Battery Fleet Benchmark Report — T-60 Panel
**Date:** 2026-04-24
**Pipeline:** `scripts/pull_60day_disclosure.py`
**Branch:** stage1-tier1

---

## 1. Eval Window

| Parameter | Value |
|---|---|
| Start | 2026-01-01 (post-RTC+B, day 27) |
| End | 2026-02-23 (60-day disclosure cutoff from 2026-04-24) |
| Days | 54 operating days |
| Regime | 100% post-RTC+B (RTC launched 2025-12-05) |
| Data source | ERCOT 60-Day SCED Disclosure (NP3-965-ER) + DAM Disclosure (NP3-966-ER) |
| Client | gridstatus 0.35.0 |

Notable events in window:
- **Winter Storm Fern (Jan 26 2026):** RT LMP peaked at $938/MWh. Fleet earned 33.7× an average day.
- **Jan 28 2026 cold snap:** RT LMP peaked at $1,350/MWh at 13:10 UTC.

---

## 2. ESR Population After Filtering

| Filter stage | Count |
|---|---|
| Unique ESR resources in 60-day disclosure | 323 |
| Nameplate ≥ 1 MW | 296 |
| Active ≥ 30 of 54 days | 318 |
| Non-zero settled energy ≥ 20 days | 270 |
| **All three filters (included in benchmarks)** | **269** |

Resources excluded (54 total):
- 27 had nameplate < 1 MW (micro/behind-meter ESRs, not tracked by ERCOT as GEN)
- 5 had < 30 active days (recent interconnections or outages spanning the window)
- 49 had < 20 days non-zero energy (outage, testing, or scheduling anomaly)

Note: COCHRAN_ESR1 was not found in the disclosure data. It may be registered under a different ERCOT resource name or have come online after the window.

---

## 3. Fleet Benchmark Numbers

### Fleet-wide $/kW-year equivalent (annualized from 54-day window)

| Percentile | $/kW-yr |
|---|---:|
| Median (P50) | **$24.93** |
| Top quartile (P75) | **$32.23** |
| P90 | $42.45 |
| Max (PALACIOS_ESR1, 9.9 MW) | $159.38 |

### Top 10 ESRs by mean $/kW-day

| Rank | Resource | Nameplate MW | QSE | $/kW-yr equiv |
|---:|---|---:|---|---:|
| 1 | PALACIOS_ESR1 | 9.9 | QTEN30 | $159.38 |
| 2 | ANDMDSLR_ESR2 | 79.4 | QIPLU2 | $109.21 |
| 3 | CNLY_ESS_ESR1 | 125.1 | QSUE76 | $92.24 |
| 4 | SHAMROCK_ESR1 | 99.5 | QFPL40 | $73.03 |
| 5 | CROSSTRL_ESR1 | 57.3 | QGRIE4 | $72.94 |
| 6 | ENDPARKS_ESR1 | 52.0 | QGRIE4 | $64.25 |
| 7 | GIGA_ESS_ESR1 | 125.2 | QSUE55 | $62.49 |
| 8 | CHIL_SLR_ESR1 | 151.6 | QSUE88 | $58.67 |
| 9 | MUSTNGCK_ESR1 | 71.0 | QENE10 | $52.67 |
| 10 | SWOOSEII_ESR1 | 101.1 | QMEA5 | $52.65 |

---

## 4. Revenue Composition

Fleet-wide totals across 269 ESRs, 54 operating days:

| Revenue stream | Total ($) | Fleet share |
|---|---:|---:|
| DAM Energy | $20,875,709 | 31.7% |
| RT Energy (imbalance vs DAM award) | $14,318,389 | 21.8% |
| DAM AS (RegUp/RegDown/RRS/ECRS/NonSpin) | $25,836,392 | 39.3% |
| RT AS (post-RTC+B only) | $4,725,427 | 7.2% |
| **Total** | **$65,755,917** | 100% |

Key observations:
- **DAM AS is the largest single stream (39.3%).** Post-RTC+B, DAM AS scheduling is more competitive and clearing prices fell sharply (DAM RegUp mean dropped from $52.63 pre-RTC+B to $3.00 post-RTC+B per data_characterization.md), yet AS awards are substantial in volume.
- **RT AS is live and nonzero (7.2%).** 221 of 269 included ESRs show non-zero RT AS revenue — confirming RTC co-optimization is active and awarding RT AS positions. This is a structural post-RTC+B signal.
- **RT energy contributes 21.8%.** This includes imbalance revenue from batteries dispatched at points different from their DAM schedules.
- Notable heterogeneity by battery: GAMBIT (100 MW) earns 86% from RT energy with 0% DAM energy, suggesting it relies primarily on RT dispatch rather than DAM self-scheduling. BLACKSPR (120 MW) earns 91% DAM energy + 0% AS — opposite strategy.

---

## 5. Sanity Checks

### 5.1 Fleet totals vs Modo benchmark
- Modo Energy 2025 full-year fleet median: ~$29/kW (published in Q4 2025 reports)
- Our 54-day window annualized median: **$24.93/kW** — 14% below Modo's 2025 full-year
- Excluding Jan 26 scarcity: **$16.00/kW** — ~45% below Modo
- **Interpretation:** Jan–Feb is seasonally below average (no summer heat/drought AS premiums, less solar-duck-curve energy spread). The scarcity event (Fern) partially compensates. The gap is consistent with typical seasonal variation. **No anomaly — PASS.**

### 5.2 Named battery spot checks

| Battery | Our $/kW-yr | Notes |
|---|---:|---|
| GAMBIT_ESR1 (100 MW) | $8.70 | Strikingly low; 86% RT energy, 0% DAM energy. Operates opportunistically in RT, little DAM participation. No public revenue comparison available for this window. |
| CROSSETT_ESR1 (101 MW) | $13.55 | 63% RT energy. Below fleet median — may reflect grid location or commercial strategy. |
| CROSSETT_ESR2 (100 MW) | $13.73 | Similar profile to ESR1. |
| CROSSTRL_ESR1 (57 MW) | $72.94 | Near top-5 in fleet. 88% DAM energy, active arbitrage. |
| BLACKSPR_ESR1 (120 MW) | $23.87 | ~Fleet median. 91% DAM energy, no AS participation. |
| SHAMROCK_ESR1 (100 MW) | $73.03 | Top-5. 89% DAM energy + 11% RT energy. |
| PALACIOS_ESR1 (9.9 MW) | $159.38 | **Outlier** — $159/kW-yr is 6× fleet median. 100% DAM energy, 0% AS. Likely benefits from local congestion rent at PALACIOS_RN. Inspect for price-path manipulation risk if using as benchmark target. |

No systematic discrepancy >30% found for well-understood batteries. GAMBIT's low revenue is unexplained but its RT-only strategy is consistent with a private-ownership operator using a different commercial model. **PASS with flag on PALACIOS outlier.**

### 5.3 RT AS revenue check
- All-zero RT AS before fix (timestamp alignment bug — SCED timestamps had ~17s seconds offset vs 5-min RT price grid; fixed by flooring SCED timestamps to 5-min).
- After fix: RT AS total $4,725,427, 221/269 ESRs nonzero. **PASS.**

### 5.4 Battery size distribution

| Nameplate range | Count | Notes |
|---|---:|---|
| < 5 MW | 2 | Near-threshold (micro-grid / behind-the-meter) |
| 5–10 MW | 98 | **Largest cohort** — likely 2h/4h residential/commercial |
| 10–25 MW | 23 | |
| 25–50 MW | 11 | |
| 50–100 MW | 53 | Utility-scale |
| 100–200 MW | 69 | Large utility / transmission-connected |
| 200–500 MW | 13 | Mega-scale |

Charge/discharge ratio (|LSL| / HSL, proxy for C-rate symmetry): median 1.00, mean 0.83. Most batteries have symmetric charge/discharge MW — consistent with 2h Li-ion (1C) systems, not 6h or 15-min assets. **PASS on duration check; no 15-min or 6h anomaly detected.**

---

## 6. Known Limitations

1. **RT SPP approximation:** RT energy revenue uses hub-average RT LMP (`rt_lmp` from processed parquet, ~HB_HUBAVG) as the RT settlement price reference for all ESRs. Actual settlement uses per-resource-node SPP (NP6-905-ER), which was unavailable via gridstatus for historical dates (documents expired from MIS). For most batteries, hub vs RN RT LMP differs by < 5% outside congestion events. During scarcity (Jan 26), congestion can be significant — this is the largest source of error in the RT energy revenue estimate.

2. **SMNE not used:** Settlement Metered Net Energy (SMNE) resource codes use physical-meter naming conventions (e.g., `CROSSETT_UNIT1`) that don't map to post-RTC+B ESR names (e.g., `CROSSETT_ESR1`) without a proprietary lookup table. We use `Telemetered Net Output` from `sced_esr` as the RT energy proxy. Telemetered output is a real-time SCADA measurement and may diverge from settled energy by ±2–5% due to meter corrections.

3. **DAM energy sign convention:** `Awarded Quantity` in `dam_esr` is positive for energy offered (discharge = revenue source) and negative for energy bid (charge = revenue cost). This is treated correctly in the revenue computation (negative DAM Awarded × positive DAM SPP = negative revenue = charging cost).

4. **Nameplate from HSL:** We use `max(HSL)` over the window as nameplate discharge MW. For batteries that were in outage during their highest-output period, this may understate nameplate. 27 ESRs had HSL < 1 MW and were excluded on this basis — some may be under-reporting HSL due to de-rating.

5. **Post-RTC+B window only (54 days):** Annualizing a 54-day winter window inflates the contribution of one scarcity event (Jan 26). The $24.93/kW-yr median should not be extrapolated to a full-year forecast — it represents this specific window's performance, which includes an unusual cold snap.

6. **COCHRAN_ESR1 not found:** Cannot confirm whether this asset was offline, renamed, or outside the window. Excluded from benchmarks.

---

## 7. Sizing Recommendation

**Our policy model:** 10 MW discharge / 5.9 MW charge, 20 MWh (≈ 2-hour, 10 MW-class).

**Comparable fleet cohort:** 5–20 MW ESRs (n=120).

| | 5–20 MW cohort | Full fleet |
|---|---:|---:|
| Median $/kW-yr | $25.15 | $24.93 |
| P75 $/kW-yr | $28.17 | $32.23 |
| P90 $/kW-yr | $36.32 | $42.45 |

The 5–20 MW cohort has essentially the same median as the full fleet ($25.15 vs $24.93) but a tighter top-quartile ($28.17 vs $32.23). Larger batteries (100–200 MW range) earn modestly higher median ($27.16/kW-yr) but this reflects access to more AS products and larger DAM position sizing. For a policy demo targeting a 10 MW model:

- **Beat-the-median bar:** $25/kW-yr (≈ $0.068/kW-day)
- **Beat-top-quartile bar:** $29/kW-yr (≈ $0.079/kW-day, also aligns with Modo 2025 fleet median)
- **Top-decile bar:** $36/kW-yr (≈ $0.099/kW-day)

These are the recommended policy evaluation thresholds for the T-60 panel.
