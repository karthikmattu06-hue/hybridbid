"""Pull ERCOT 60-day SCED+DAM disclosure for 2026-01-01 → 2026-02-23 (T-60 eval window).

Stage 1: download per-day dicts via gridstatus, filter to ESR-relevant rows, persist
         slim parquet files to data/ercot_disclosure/{sced,dam}/YYYY-MM-DD/.
Stage 2: concatenate across days into master tables.
Stage 3: compute per-ESR daily revenue using:
   - DAM energy rev:  dam_esr["Awarded Quantity"] × "Energy Settlement Point Price"
   - DAM AS rev:      Σ_product (Awarded × MCPC) per hour, from dam_esr
   - RT energy rev:   (integrated telemetered output − DAM award per hour) × RT LMP (hub proxy)
   - RT AS rev (post-RTC+B): Σ_product (sced_esr "AS Awards" − dam_esr award for the hour)
                     × RT MCPC per 5-min (from data/processed/as_prices/)

Outputs:
   data/ercot_disclosure/esr_population.parquet
   data/ercot_disclosure/esr_daily_revenue.parquet

Raw per-day archives are NOT committed to git (too large).
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

import gridstatus
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "data" / "ercot_disclosure"
SCED_DIR = OUTDIR / "sced"
DAM_DIR = OUTDIR / "dam"
PROC = ROOT / "data" / "processed"

WINDOW_START = pd.Timestamp("2026-01-01")
WINDOW_END = pd.Timestamp("2026-02-23")

# ---- ESR-relevant columns we keep (drop bid curves to save space) ---------

DAM_ESR_COLS = [
    "Time", "Interval Start", "Interval End",
    "QSE", "DME", "Resource Name", "Resource Type",
    "HSL", "LSL", "Resource Status",
    "Awarded Quantity", "Settlement Point Name", "Energy Settlement Point Price",
    "RegUp Awarded", "RegUp MCPC",
    "RegDown Awarded", "RegDown MCPC",
    "RRSPFR Awarded", "RRSFFR Awarded", "RRSUFR Awarded", "RRS MCPC",
    "ECRSSD Awarded", "ECRS MCPC",
    "NonSpin Awarded", "NonSpin MCPC",
]

SCED_ESR_COLS = [
    "SCED Timestamp", "Repeated Hour Flag",
    "QSE", "DME", "Resource Name", "Resource Type",
    "HSL", "HDL", "LSL", "LDL",
    "Telemetered Resource Status", "Base Point", "Telemetered Net Output",
    "Ramp Rate Up", "Ramp Rate Down",
    "AS Capability REGUP", "AS Capability REGDN", "AS Capability ECRS",
    "AS Capability NSPIN", "AS Capability RRSPF", "AS Capability RRSFF",
    "State of Charge", "Minimum SOC", "Maximum SOC",
    "AS Awards NSPIN", "AS Awards RRSFFR", "AS Awards RRSPFR", "AS Awards RRSUFR",
    "AS Awards ECRS", "AS Awards REGUP", "AS Awards REGDN",
]

# ---------------------------------------------------------------------------
# Stage 1: download
# ---------------------------------------------------------------------------

def fetch_day(date: pd.Timestamp, ercot, verbose=False) -> dict:
    """Fetch SCED + DAM for one operating day. Returns slim dict."""
    date_str = date.strftime("%Y-%m-%d")
    t0 = time.time()
    sced = ercot.get_60_day_sced_disclosure(date=date_str, verbose=verbose)
    t1 = time.time()
    dam = ercot.get_60_day_dam_disclosure(date=date_str, verbose=verbose)
    t2 = time.time()

    # Slim down sced_esr
    sced_esr_full = sced["sced_esr"]
    avail_cols = [c for c in SCED_ESR_COLS if c in sced_esr_full.columns]
    missing = set(SCED_ESR_COLS) - set(avail_cols)
    if missing:
        print(f"  [warn] missing SCED cols for {date_str}: {missing}")
    sced_esr_slim = sced_esr_full[avail_cols].copy()

    # Slim down dam_esr
    dam_esr_full = dam["dam_esr"]
    avail_cols = [c for c in DAM_ESR_COLS if c in dam_esr_full.columns]
    missing = set(DAM_ESR_COLS) - set(avail_cols)
    if missing:
        print(f"  [warn] missing DAM cols for {date_str}: {missing}")
    dam_esr_slim = dam_esr_full[avail_cols].copy()

    print(f"  {date_str}: SCED {t1-t0:.1f}s, DAM {t2-t1:.1f}s, "
          f"sced_esr={sced_esr_slim.shape}, dam_esr={dam_esr_slim.shape}")
    return {
        "date": date_str,
        "sced_esr": sced_esr_slim,
        "dam_esr": dam_esr_slim,
    }


def stage1_download(start: pd.Timestamp, end: pd.Timestamp, force=False) -> None:
    """Pull each day and persist to parquet."""
    ercot = gridstatus.Ercot()
    dates = pd.date_range(start, end, freq="D")
    print(f"Stage 1: downloading {len(dates)} days from {start.date()} → {end.date()}")

    for i, d in enumerate(dates):
        date_str = d.strftime("%Y-%m-%d")
        sced_dir = SCED_DIR / date_str
        dam_dir = DAM_DIR / date_str
        sced_path = sced_dir / "sced_esr.parquet"
        dam_path = dam_dir / "dam_esr.parquet"
        if not force and sced_path.exists() and dam_path.exists():
            continue
        sced_dir.mkdir(parents=True, exist_ok=True)
        dam_dir.mkdir(parents=True, exist_ok=True)
        try:
            data = fetch_day(d, ercot)
        except Exception as e:
            print(f"  [FAIL] {date_str}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=2)
            continue
        # Save parquet (tz-aware timestamps fine)
        data["sced_esr"].to_parquet(sced_path, index=False)
        data["dam_esr"].to_parquet(dam_path, index=False)
    print("Stage 1 complete.")


# ---------------------------------------------------------------------------
# Stage 2: concat master tables
# ---------------------------------------------------------------------------

def load_master() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Concat per-day parquet files into a single DataFrame."""
    sced_files = sorted(glob.glob(str(SCED_DIR / "*" / "sced_esr.parquet")))
    dam_files = sorted(glob.glob(str(DAM_DIR / "*" / "dam_esr.parquet")))
    print(f"  loading {len(sced_files)} SCED days, {len(dam_files)} DAM days")
    sced = pd.concat([pd.read_parquet(f) for f in sced_files], ignore_index=True)
    dam = pd.concat([pd.read_parquet(f) for f in dam_files], ignore_index=True)
    print(f"  sced_esr master: {sced.shape}")
    print(f"  dam_esr master:  {dam.shape}")
    return sced, dam


# ---------------------------------------------------------------------------
# Stage 3: compute revenue
# ---------------------------------------------------------------------------

AS_PRODUCTS_DAM = [
    # (dam col Awarded, dam col MCPC, friendly name)
    ("RegUp Awarded", "RegUp MCPC", "RegUp"),
    ("RegDown Awarded", "RegDown MCPC", "RegDown"),
    ("RRSPFR Awarded", "RRS MCPC", "RRSPFR"),
    ("RRSFFR Awarded", "RRS MCPC", "RRSFFR"),
    ("RRSUFR Awarded", "RRS MCPC", "RRSUFR"),
    ("ECRSSD Awarded", "ECRS MCPC", "ECRS"),
    ("NonSpin Awarded", "NonSpin MCPC", "NonSpin"),
]

AS_PRODUCTS_SCED = [
    # (sced col AS Award, our rt_mcpc col, dam col Awarded to subtract per hour)
    ("AS Awards REGUP",  "rt_mcpc_regup", "RegUp Awarded"),
    ("AS Awards REGDN",  "rt_mcpc_regdn", "RegDown Awarded"),
    ("AS Awards RRSPFR", "rt_mcpc_rrs",   "RRSPFR Awarded"),
    ("AS Awards RRSFFR", "rt_mcpc_rrs",   "RRSFFR Awarded"),
    ("AS Awards RRSUFR", "rt_mcpc_rrs",   "RRSUFR Awarded"),
    ("AS Awards ECRS",   "rt_mcpc_ecrs",  "ECRSSD Awarded"),
    ("AS Awards NSPIN",  "rt_mcpc_nsrs",  "NonSpin Awarded"),
]


def _load_rt_prices() -> pd.DataFrame:
    """Load 5-min RT LMP (hub) + RT MCPC from processed parquet."""
    en_files = sorted(glob.glob(str(PROC / "energy_prices" / "*.parquet")))
    as_files = sorted(glob.glob(str(PROC / "as_prices" / "*.parquet")))
    en = pd.concat([pd.read_parquet(f) for f in en_files]).sort_index()
    as_p = pd.concat([pd.read_parquet(f) for f in as_files]).sort_index()
    mcpc_cols = sorted({c for _, c, _ in AS_PRODUCTS_SCED})
    df = en[["rt_lmp"]].join(as_p[mcpc_cols])
    # Window mask
    df = df.loc[(df.index >= WINDOW_START.tz_localize("UTC")) &
                (df.index <  (WINDOW_END + pd.Timedelta(days=1)).tz_localize("UTC"))]
    return df


def compute_daily_revenue(sced: pd.DataFrame, dam: pd.DataFrame) -> pd.DataFrame:
    """Per-ESR, per-operating-day revenue decomposition."""
    print("Stage 3: computing revenue ...")

    # --- timestamp hygiene ---
    dam = dam.copy()
    sced = sced.copy()

    # DAM intervals are hourly (Interval Start)
    dam["hour_start_utc"] = pd.to_datetime(dam["Interval Start"], utc=True)
    dam["date"] = dam["hour_start_utc"].dt.tz_convert("US/Central").dt.date
    # Numeric hygiene
    for c in dam.columns:
        if c in ("Awarded Quantity", "Energy Settlement Point Price",
                "RegUp Awarded", "RegUp MCPC",
                "RegDown Awarded", "RegDown MCPC",
                "RRSPFR Awarded", "RRSFFR Awarded", "RRSUFR Awarded", "RRS MCPC",
                "ECRSSD Awarded", "ECRS MCPC",
                "NonSpin Awarded", "NonSpin MCPC",
                "HSL", "LSL"):
            dam[c] = pd.to_numeric(dam[c], errors="coerce").fillna(0.0)

    # SCED intervals are 5-min (SCED Timestamp). Convert to UTC-aware, then floor to
    # nearest 5-min to match the processed RT price grid (SCED has ~17s of seconds offset).
    sced["sced_ts_utc"] = pd.to_datetime(sced["SCED Timestamp"], utc=True).dt.floor("5min")
    sced["hour_start_utc"] = sced["sced_ts_utc"].dt.floor("h")
    sced["date"] = sced["sced_ts_utc"].dt.tz_convert("US/Central").dt.date
    for c in SCED_ESR_COLS:
        if c in sced.columns and c not in ("SCED Timestamp", "QSE", "DME", "Resource Name",
                                           "Resource Type", "Telemetered Resource Status",
                                           "Repeated Hour Flag"):
            sced[c] = pd.to_numeric(sced[c], errors="coerce").fillna(0.0)

    # --- DAM energy revenue ($/hour) ---
    dam["dam_energy_rev"] = dam["Awarded Quantity"] * dam["Energy Settlement Point Price"]

    # --- DAM AS revenue ($/hour) ---
    dam["dam_as_rev"] = 0.0
    for aw, mcpc, _ in AS_PRODUCTS_DAM:
        if aw in dam.columns and mcpc in dam.columns:
            dam["dam_as_rev"] += dam[aw] * dam[mcpc]

    dam_daily = (dam.groupby(["Resource Name", "date"], as_index=False)
                     .agg(dam_energy_rev=("dam_energy_rev", "sum"),
                          dam_as_rev=("dam_as_rev", "sum"),
                          settlement_point=("Settlement Point Name", "first"),
                          qse=("QSE", "first"),
                          hsl_max_dam=("HSL", "max"),
                          lsl_min_dam=("LSL", "min"),
                          dam_hours_online=("Resource Status",
                                            lambda s: (s == "ON").sum())))

    # --- SCED: aggregate to hourly for imbalance, and join with DAM ---
    # RT price table (5-min index, UTC tz-aware)
    rt = _load_rt_prices()

    # Join SCED 5-min telemetry with RT prices
    rt_reset = rt.reset_index().rename(columns={"index": "ts_utc"})
    # The processed parquet index column name is "timestamp" typically
    if "ts_utc" not in rt_reset.columns:
        rt_reset = rt.reset_index()
        rt_reset = rt_reset.rename(columns={rt_reset.columns[0]: "ts_utc"})
    rt_reset["ts_utc"] = pd.to_datetime(rt_reset["ts_utc"], utc=True)

    sced = sced.merge(rt_reset, left_on="sced_ts_utc", right_on="ts_utc", how="left")
    sced = sced.reset_index(drop=True)

    # Coerce joined RT price columns to numeric (they arrive as Float64 from parquet)
    for _, m, _ in AS_PRODUCTS_SCED:
        if m in sced.columns:
            sced[m] = pd.to_numeric(sced[m], errors="coerce").fillna(0.0)
    sced["rt_lmp"] = pd.to_numeric(sced["rt_lmp"], errors="coerce").fillna(0.0)

    # Compute telemetered MWh per 5-min = MW × (5/60)
    sced["telem_mwh"] = sced["Telemetered Net Output"] * (5.0 / 60.0)
    # DAM-award MWh allocated per 5-min = (DAM_MW × 1h) / 12 intervals
    # We'll join DAM award per hour onto SCED after aggregating by hour.

    # Compute per-row RT AS revenue for each product; aggregate later.
    rt_as_cols = []
    for aw_col, mcpc_col, _ in AS_PRODUCTS_SCED:
        col = f"rt_as_rev_{aw_col.replace('AS Awards ', '')}"
        sced[col] = sced[aw_col] * sced[mcpc_col] * (5.0 / 60.0)
        rt_as_cols.append(col)
    sced_hr = (sced.groupby(["Resource Name", "hour_start_utc"], as_index=False)
                     .agg(telem_mwh_hr=("telem_mwh", "sum"),
                          rt_lmp_mean=("rt_lmp", "mean"),
                          **{c: (c, "sum") for c in rt_as_cols},
                          hsl_max_sced=("HSL", "max"),
                          lsl_min_sced=("LSL", "min"),
                          soc_max=("State of Charge", "max"),
                          soc_min=("State of Charge", "min"),
                          n_sced_intervals=("Resource Name", "size")))

    # Join DAM hourly awards onto SCED hourly for RT energy imbalance
    dam_hr = dam[["Resource Name", "hour_start_utc", "Awarded Quantity",
                  "Settlement Point Name"]].copy()
    dam_hr = dam_hr.rename(columns={"Awarded Quantity": "dam_award_mw_hr",
                                    "Settlement Point Name": "settlement_point"})
    merged = sced_hr.merge(dam_hr, on=["Resource Name", "hour_start_utc"], how="left")
    merged["dam_award_mwh_hr"] = merged["dam_award_mw_hr"].fillna(0.0)
    merged["rt_imbalance_mwh"] = merged["telem_mwh_hr"] - merged["dam_award_mwh_hr"]
    merged["rt_energy_rev"] = merged["rt_imbalance_mwh"] * merged["rt_lmp_mean"]

    # Sum rt_as rev per hour
    merged["rt_as_rev"] = merged[rt_as_cols].sum(axis=1)

    merged["date"] = (pd.to_datetime(merged["hour_start_utc"], utc=True)
                        .dt.tz_convert("US/Central").dt.date)

    rt_daily = (merged.groupby(["Resource Name", "date"], as_index=False)
                        .agg(rt_energy_rev=("rt_energy_rev", "sum"),
                             rt_as_rev=("rt_as_rev", "sum"),
                             telem_mwh_day=("telem_mwh_hr", "sum"),
                             rt_hours_online=("n_sced_intervals", lambda s: (s > 0).sum()),
                             hsl_max_sced=("hsl_max_sced", "max"),
                             lsl_min_sced=("lsl_min_sced", "min"),
                             soc_max_day=("soc_max", "max"),
                             soc_min_day=("soc_min", "min")))

    # Merge DAM + RT daily
    out = dam_daily.merge(rt_daily, on=["Resource Name", "date"], how="outer")
    out["dam_energy_rev"] = out["dam_energy_rev"].fillna(0.0)
    out["dam_as_rev"] = out["dam_as_rev"].fillna(0.0)
    out["rt_energy_rev"] = out["rt_energy_rev"].fillna(0.0)
    out["rt_as_rev"] = out["rt_as_rev"].fillna(0.0)
    out["total_rev"] = out[["dam_energy_rev", "rt_energy_rev",
                             "dam_as_rev", "rt_as_rev"]].sum(axis=1)

    # Nameplate = max HSL over the window (discharge capacity in MW)
    nameplate = (out.groupby("Resource Name")
                   .agg(nameplate_mw=("hsl_max_sced", "max"),
                        charge_mw_max=("lsl_min_sced", lambda s: -s.min())).reset_index())
    out = out.merge(nameplate, on="Resource Name", how="left")
    # Fall back to dam HSL
    out["nameplate_mw"] = np.where(out["nameplate_mw"].isna() | (out["nameplate_mw"] <= 0),
                                    out["hsl_max_dam"], out["nameplate_mw"])
    out["rev_per_kw_day"] = out["total_rev"] / (out["nameplate_mw"] * 1000.0)
    out["rev_per_kw_year_equiv"] = out["rev_per_kw_day"] * 365.0

    # Rename Resource Name → resource_name for output convention
    out = out.rename(columns={"Resource Name": "resource_name"})

    return out


# ---------------------------------------------------------------------------
# Stage 4: population & benchmarks
# ---------------------------------------------------------------------------

def build_population(daily: pd.DataFrame) -> pd.DataFrame:
    """Apply population filters and produce esr_population.parquet."""
    g = daily.groupby("resource_name")
    pop = g.agg(
        qse=("qse", "first"),
        settlement_point=("settlement_point", "first"),
        nameplate_mw=("nameplate_mw", "max"),
        charge_mw_max=("charge_mw_max", "max"),
        first_date=("date", "min"),
        last_date=("date", "max"),
        n_days_active=("date", "nunique"),
        n_days_nonzero_energy=("telem_mwh_day",
                               lambda s: (s.abs() > 0.1).sum()),
        total_rev_window=("total_rev", "sum"),
        mean_daily_rev=("total_rev", "mean"),
        max_daily_rev=("total_rev", "max"),
        min_daily_rev=("total_rev", "min"),
        mean_rev_per_kw_day=("rev_per_kw_day", "mean"),
    ).reset_index()

    # Filters
    pop["passes_nameplate"] = pop["nameplate_mw"] >= 1.0
    pop["passes_active_days"] = pop["n_days_active"] >= 30
    pop["passes_nonzero_days"] = pop["n_days_nonzero_energy"] >= 20
    pop["included"] = (pop["passes_nameplate"] & pop["passes_active_days"] &
                       pop["passes_nonzero_days"])

    return pop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["download", "process", "all"], default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.stage in ("download", "all"):
        stage1_download(WINDOW_START, WINDOW_END, force=args.force)

    if args.stage in ("process", "all"):
        sced, dam = load_master()
        daily = compute_daily_revenue(sced, dam)
        daily.to_parquet(OUTDIR / "esr_daily_revenue.parquet", index=False)
        print(f"Wrote {OUTDIR / 'esr_daily_revenue.parquet'}: {daily.shape}")

        pop = build_population(daily)
        pop.to_parquet(OUTDIR / "esr_population.parquet", index=False)
        print(f"Wrote {OUTDIR / 'esr_population.parquet'}: {pop.shape}")


if __name__ == "__main__":
    main()
