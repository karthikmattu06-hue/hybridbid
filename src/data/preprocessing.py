"""
Data preprocessing — transforms raw gridstatus DataFrames into
canonical Parquet schema (5-min UTC intervals).

Each process_* function takes raw DataFrames (as returned by the fetcher)
and produces a DataFrame indexed by 5-min UTC timestamps with canonical
column names.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import DAM_AS_COLUMN_MAP, RT_MCPC_ASTYPE_MAP, RTCB_BOUNDARY_UTC

logger = logging.getLogger(__name__)

RTCB_UTC = pd.Timestamp(RTCB_BOUNDARY_UTC, tz="UTC")


def _to_utc_index(df: pd.DataFrame, col: str = "Interval Start") -> pd.Series:
    """Convert a timestamp column to UTC, return as Series.

    Uses utc=True to handle mixed-offset columns that arise when old-schema
    (CPT fixed-offset) and new-schema (UTC-aware, normalized in fetcher) frames
    are concatenated. Naive timestamps are localized as US/Central first.
    """
    raw = df[col]
    # If all values are naive, localize as US/Central before converting
    if pd.api.types.is_datetime64_any_dtype(raw) and raw.dt.tz is None:
        raw = raw.dt.tz_localize("US/Central", ambiguous="NaT", nonexistent="shift_forward")
        return raw.dt.tz_convert("UTC")
    # Mixed or uniform tz-aware: utc=True converts everything to UTC in one pass
    return pd.to_datetime(raw, utc=True)


def _floor_5min(ts: pd.Series) -> pd.Series:
    """Floor timestamps to the nearest 5-minute boundary."""
    return ts.dt.floor("5min")


def _make_5min_index(start: str, end: str) -> pd.DatetimeIndex:
    """Create a canonical 5-min UTC DatetimeIndex for [start, end)."""
    return pd.date_range(
        start=pd.Timestamp(start, tz="UTC"),
        end=pd.Timestamp(end, tz="UTC"),
        freq="5min",
        inclusive="left",
    )


def _add_rtcb_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add is_post_rtcb column based on the index."""
    df["is_post_rtcb"] = df.index >= RTCB_UTC
    return df


# ──────────────────────────────────────────────
# Energy Prices
# ──────────────────────────────────────────────

def process_energy_prices(
    rt_lmp_df: pd.DataFrame,
    dam_spp_df: pd.DataFrame,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Process RT LMP + DAM SPP into canonical energy_prices table.

    rt_lmp_df: from fetch_rt_lmp() — columns include Interval Start, LMP
    dam_spp_df: from fetch_dam_spp() — columns include Interval Start, SPP
    """
    idx = _make_5min_index(start, end)
    result = pd.DataFrame(index=idx)
    result.index.name = "timestamp_utc"

    # --- RT LMP (already 5-min, just align) ---
    if not rt_lmp_df.empty:
        rt = rt_lmp_df.copy()
        rt["ts_utc"] = _floor_5min(_to_utc_index(rt))
        # If multiple SCED runs per 5-min interval, take the last one
        rt = rt.sort_values("SCED Timestamp" if "SCED Timestamp" in rt.columns else "ts_utc")
        rt = rt.drop_duplicates(subset=["ts_utc"], keep="last")
        rt = rt.set_index("ts_utc")["LMP"]
        result["rt_lmp"] = rt
        logger.info(f"  RT LMP: {rt.notna().sum()} values mapped")
    else:
        result["rt_lmp"] = np.nan

    # --- DAM SPP (hourly → ffill to 5-min) ---
    if not dam_spp_df.empty:
        dam = dam_spp_df.copy()
        dam["ts_utc"] = _to_utc_index(dam)
        dam = dam.drop_duplicates(subset=["ts_utc"], keep="last")
        dam = dam.set_index("ts_utc")["SPP"]
        # Reindex to 5-min then forward-fill
        dam_aligned = dam.reindex(idx)
        dam_aligned = dam_aligned.ffill()
        result["dam_spp"] = dam_aligned
        logger.info(f"  DAM SPP: {dam_aligned.notna().sum()} values after ffill")
    else:
        result["dam_spp"] = np.nan

    result = _add_rtcb_flag(result)
    return result


# ──────────────────────────────────────────────
# AS Prices
# ──────────────────────────────────────────────

def process_as_prices(
    dam_as_df: pd.DataFrame,
    rt_mcpc_df: pd.DataFrame,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Process DAM AS + RT MCPC into canonical as_prices table.

    dam_as_df: from fetch_dam_as() — wide format with named AS columns
    rt_mcpc_df: from load_rt_mcpc() — long format: SCEDTimestamp, ASType, MCPC
    """
    idx = _make_5min_index(start, end)
    result = pd.DataFrame(index=idx)
    result.index.name = "timestamp_utc"

    # --- DAM AS (hourly → ffill to 5-min) ---
    if not dam_as_df.empty:
        dam = dam_as_df.copy()
        dam["ts_utc"] = _to_utc_index(dam)
        dam = dam.drop_duplicates(subset=["ts_utc"], keep="last")
        dam = dam.set_index("ts_utc")

        for src_col, dst_col in DAM_AS_COLUMN_MAP.items():
            if src_col in dam.columns:
                series = dam[src_col].reindex(idx).ffill()
                result[dst_col] = series
            else:
                result[dst_col] = np.nan
        logger.info(f"  DAM AS: mapped {len(DAM_AS_COLUMN_MAP)} products")
    else:
        for dst_col in DAM_AS_COLUMN_MAP.values():
            result[dst_col] = np.nan

    # Fill ECRS NaN with 0 for pre-June 2023 (product didn't exist)
    ecrs_start = pd.Timestamp("2023-06-01", tz="UTC")
    pre_ecrs = result.index < ecrs_start
    if "dam_as_ecrs" in result.columns:
        result.loc[pre_ecrs, "dam_as_ecrs"] = result.loc[pre_ecrs, "dam_as_ecrs"].fillna(0.0)

    # --- RT MCPC (5-min, long → pivot to wide) ---
    rt_cols = list(RT_MCPC_ASTYPE_MAP.values())

    # Detect column names: support both PascalCase (legacy) and lowercase (current files)
    has_mcpc = not rt_mcpc_df.empty and (
        "sced_timestamp" in rt_mcpc_df.columns or "SCEDTimestamp" in rt_mcpc_df.columns
    )
    if has_mcpc:
        rt = rt_mcpc_df.copy()

        # Normalize column names to lowercase
        col_map = {
            "SCEDTimestamp": "sced_timestamp",
            "RepeatedHourFlag": "repeated_hour_flag",
            "ASType": "as_type",
            "MCPC": "mcpc",
        }
        rt.rename(columns={k: v for k, v in col_map.items() if k in rt.columns}, inplace=True)

        # Convert timestamp to UTC
        ts = pd.to_datetime(rt["sced_timestamp"])
        if ts.dt.tz is None:
            # Legacy string format: parse then localize to CPT → UTC
            ts = ts.dt.tz_localize("US/Central", ambiguous="NaT", nonexistent="shift_forward")
            ts = ts.dt.tz_convert("UTC")
        else:
            ts = ts.dt.tz_convert("UTC")
        rt["ts_utc"] = _floor_5min(ts)

        # Pivot: one column per AS type
        pivoted = rt.pivot_table(
            index="ts_utc", columns="as_type", values="mcpc", aggfunc="last"
        )
        for as_type, dst_col in RT_MCPC_ASTYPE_MAP.items():
            if as_type in pivoted.columns:
                result[dst_col] = pivoted[as_type].reindex(idx)
            else:
                result[dst_col] = np.nan
        logger.info(f"  RT MCPC: {pivoted.shape[0]} intervals mapped")
    else:
        for col in rt_cols:
            if col not in result.columns:
                result[col] = np.nan

    # Pre-RTC+B: RT MCPC columns should be NaN
    pre_rtcb = result.index < RTCB_UTC
    for col in rt_cols:
        if col in result.columns:
            result.loc[pre_rtcb, col] = np.nan

    result = _add_rtcb_flag(result)
    return result


# ──────────────────────────────────────────────
# System Conditions
# ──────────────────────────────────────────────

def _dedup_renewable(df: pd.DataFrame, gen_col: str, forecast_col: str):
    """
    Deduplicate wind/solar data which has multiple Publish Times per interval.

    For actuals: take the latest Publish Time per interval (most complete data).
    For forecasts: take the latest Publish Time that is before the interval start
                   (operational forecast available at decision time).
    """
    if df.empty:
        return pd.Series(dtype="float64"), pd.Series(dtype="float64")

    df = df.copy()
    df["ts_utc"] = _to_utc_index(df)

    # Actuals: latest publish time per interval, only where GEN is not NaN
    if gen_col in df.columns:
        actuals = df.dropna(subset=[gen_col])
        if "Publish Time" in actuals.columns:
            actuals = actuals.sort_values("Publish Time")
        actuals = actuals.drop_duplicates(subset=["ts_utc"], keep="last")
        actual_series = actuals.set_index("ts_utc")[gen_col]
    else:
        actual_series = pd.Series(dtype="float64")

    # Forecasts: latest publish time before interval start
    if forecast_col in df.columns:
        fcast = df.copy()
        if "Publish Time" in fcast.columns:
            pub = pd.to_datetime(fcast["Publish Time"])
            if pub.dt.tz is None:
                pub = pub.dt.tz_localize("US/Central", ambiguous="NaT").dt.tz_convert("UTC")
            else:
                pub = pub.dt.tz_convert("UTC")
            fcast["pub_utc"] = pub
            # Only keep forecasts published before the interval
            fcast = fcast[fcast["pub_utc"] < fcast["ts_utc"]]
            fcast = fcast.sort_values("pub_utc")
        fcast = fcast.drop_duplicates(subset=["ts_utc"], keep="last")
        forecast_series = fcast.set_index("ts_utc")[forecast_col]
    else:
        forecast_series = pd.Series(dtype="float64")

    return actual_series, forecast_series


def process_system_conditions(
    load_actual_df: pd.DataFrame,
    load_forecast_df: pd.DataFrame,
    wind_df: pd.DataFrame,
    solar_df: pd.DataFrame,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Process load, wind, solar into canonical system_conditions table.

    load_actual_df: from fetch_load_actual() — columns include Interval Start, ERCOT
    load_forecast_df: from fetch_load_forecast() — columns include Interval Start, System Total, In Use Flag
    wind_df: from fetch_wind() — columns include Interval Start, GEN SYSTEM WIDE, STWPF SYSTEM WIDE
    solar_df: from fetch_solar() — columns include Interval Start, GEN SYSTEM WIDE, STPPF SYSTEM WIDE
    """
    idx = _make_5min_index(start, end)
    result = pd.DataFrame(index=idx)
    result.index.name = "timestamp_utc"

    # --- Load actual (hourly → ffill) ---
    if not load_actual_df.empty:
        load = load_actual_df.copy()
        load["ts_utc"] = _to_utc_index(load)
        load = load.drop_duplicates(subset=["ts_utc"], keep="last")
        load = load.set_index("ts_utc")
        if "ERCOT" in load.columns:
            result["total_load_mw"] = load["ERCOT"].reindex(idx).ffill()
        logger.info(f"  Load actual: {result['total_load_mw'].notna().sum()} values")
    else:
        result["total_load_mw"] = np.nan

    # --- Load forecast (hourly → ffill, filter to In Use Flag == True) ---
    if not load_forecast_df.empty:
        fcast = load_forecast_df.copy()
        if "In Use Flag" in fcast.columns:
            fcast = fcast[fcast["In Use Flag"] == True]
        fcast["ts_utc"] = _to_utc_index(fcast)
        if "Publish Time" in fcast.columns:
            fcast = fcast.sort_values("Publish Time")
        fcast = fcast.drop_duplicates(subset=["ts_utc"], keep="last")
        fcast = fcast.set_index("ts_utc")
        if "System Total" in fcast.columns:
            result["load_forecast_mw"] = fcast["System Total"].reindex(idx).ffill()
        logger.info(f"  Load forecast: {result.get('load_forecast_mw', pd.Series()).notna().sum()} values")
    else:
        result["load_forecast_mw"] = np.nan

    # --- Wind ---
    if not wind_df.empty:
        wind_actual, wind_fcast = _dedup_renewable(
            wind_df, "GEN SYSTEM WIDE", "STWPF SYSTEM WIDE"
        )
        if not wind_actual.empty:
            result["wind_actual_mw"] = wind_actual.reindex(idx).ffill()
        else:
            result["wind_actual_mw"] = np.nan
        if not wind_fcast.empty:
            result["wind_forecast_mw"] = wind_fcast.reindex(idx).ffill()
        else:
            result["wind_forecast_mw"] = np.nan
        logger.info(f"  Wind: actual={result['wind_actual_mw'].notna().sum()}, forecast={result['wind_forecast_mw'].notna().sum()}")
    else:
        result["wind_actual_mw"] = np.nan
        result["wind_forecast_mw"] = np.nan

    # --- Solar ---
    if not solar_df.empty:
        solar_actual, solar_fcast = _dedup_renewable(
            solar_df, "GEN SYSTEM WIDE", "STPPF SYSTEM WIDE"
        )
        if not solar_actual.empty:
            result["solar_actual_mw"] = solar_actual.reindex(idx).ffill()
        else:
            result["solar_actual_mw"] = np.nan
        if not solar_fcast.empty:
            result["solar_forecast_mw"] = solar_fcast.reindex(idx).ffill()
        else:
            result["solar_forecast_mw"] = np.nan
        logger.info(f"  Solar: actual={result['solar_actual_mw'].notna().sum()}, forecast={result['solar_forecast_mw'].notna().sum()}")
    else:
        result["solar_actual_mw"] = np.nan
        result["solar_forecast_mw"] = np.nan

    # --- Derived: net load ---
    wind_gen = result["wind_actual_mw"].fillna(0)
    solar_gen = result["solar_actual_mw"].fillna(0)
    if result["total_load_mw"].notna().any():
        result["net_load_mw"] = result["total_load_mw"] - wind_gen - solar_gen
    else:
        result["net_load_mw"] = np.nan

    result = _add_rtcb_flag(result)
    return result


# ──────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────

def validate_dataframe(df: pd.DataFrame, table_name: str) -> dict:
    """Run quality checks on a processed DataFrame."""
    checks = {
        "table": table_name,
        "rows": len(df),
        "columns": list(df.columns),
    }

    if len(df) > 0:
        checks["date_range"] = f"{df.index.min()} → {df.index.max()}"
        checks["null_counts"] = df.isnull().sum().to_dict()

        # Check for gaps
        expected = pd.date_range(df.index.min(), df.index.max(), freq="5min")
        missing = expected.difference(df.index)
        checks["missing_intervals"] = len(missing)
        checks["coverage_pct"] = (1 - len(missing) / max(len(expected), 1)) * 100

        # Basic stats for numeric columns
        numeric = df.select_dtypes(include=[np.number])
        if not numeric.empty:
            checks["stats"] = {
                col: {
                    "min": float(numeric[col].min()) if numeric[col].notna().any() else None,
                    "max": float(numeric[col].max()) if numeric[col].notna().any() else None,
                    "mean": float(numeric[col].mean()) if numeric[col].notna().any() else None,
                }
                for col in numeric.columns
            }
    else:
        checks["date_range"] = "empty"
        checks["missing_intervals"] = 0
        checks["coverage_pct"] = 0

    return checks


# ──────────────────────────────────────────────
# Parquet I/O
# ──────────────────────────────────────────────

def write_parquet(
    df: pd.DataFrame,
    output_dir: Path,
    table_name: str,
):
    """Write processed DataFrame to monthly-partitioned Parquet files."""
    table_dir = output_dir / table_name
    table_dir.mkdir(parents=True, exist_ok=True)

    if len(df) == 0:
        logger.warning(f"Empty DataFrame for {table_name}, skipping write")
        return

    df["_year_month"] = df.index.to_period("M").astype(str)
    for ym, group in df.groupby("_year_month"):
        group = group.drop(columns=["_year_month"])
        path = table_dir / f"{ym}.parquet"
        group.to_parquet(path, compression="snappy")
        logger.info(f"  Wrote {len(group)} rows → {path}")
    df.drop(columns=["_year_month"], inplace=True)


def read_parquet(
    data_dir: Path,
    table_name: str,
    start: str = None,
    end: str = None,
) -> pd.DataFrame:
    """Read processed Parquet files with optional date filtering."""
    table_dir = data_dir / table_name
    if not table_dir.exists():
        raise FileNotFoundError(f"No data at {table_dir}")

    dfs = []
    for f in sorted(table_dir.glob("*.parquet")):
        dfs.append(pd.read_parquet(f))

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs).sort_index()
    df = df[~df.index.duplicated(keep="last")]

    if start:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df.index < pd.Timestamp(end, tz="UTC")]
    return df
