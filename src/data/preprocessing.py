"""
Data preprocessing — transforms raw gridstatus DataFrames into
canonical Parquet schema.

Key operations:
  - Timestamp normalization (CPT → UTC)
  - Resampling to 5-minute intervals
  - Column renaming to canonical names
  - Adding is_post_rtcb flag
  - NaN insertion for pre-RTC+B AS columns
  - Data quality validation
"""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..utils.time_utils import (
    RTCB_GO_LIVE,
    add_post_rtcb_flag,
    make_5min_index,
    resample_to_5min,
)

logger = logging.getLogger(__name__)


def _norm_col_name(col: str) -> str:
    """Normalize a column name for fuzzy matching."""
    return re.sub(r"[^a-z0-9]", "", str(col).lower())


def _find_first_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first matching column from candidates (exact or normalized)."""
    if df.empty:
        return None

    by_norm = {_norm_col_name(c): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        norm = _norm_col_name(cand)
        if norm in by_norm:
            return by_norm[norm]

    for col in df.columns:
        ncol = _norm_col_name(col)
        if any(_norm_col_name(c) in ncol for c in candidates):
            return col

    return None


def _find_price_column(df: pd.DataFrame) -> str | None:
    """Find likely price column in ERCOT market datasets."""
    return _find_first_column(
        df,
        [
            "SPP",
            "LMP",
            "MCPC",
            "Settlement Point Price",
            "Price",
            "Clearing Price",
            "Market Clearing Price",
        ],
    )


def _extract_hub_and_zones(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Extract hub and load-zone prices from long-format SPP/LMP data.

    Expected long format columns (version-dependent):
      - location column (Location/Settlement Point/...)
      - location type column (Location Type/...)
      - price column (SPP/LMP/Price/...)
    """
    if df.empty:
        return pd.DataFrame(index=df.index)

    price_col = _find_price_column(df)
    loc_col = _find_first_column(df, ["Location", "Settlement Point", "Point", "Node"])
    loc_type_col = _find_first_column(df, ["Location Type", "Settlement Point Type", "Type"])

    if price_col is None:
        return pd.DataFrame(index=df.index)

    out = pd.DataFrame(index=df.index.unique().sort_values())

    if loc_col is None:
        # Wide/single-series fallback
        out[f"{prefix}_hub"] = df.groupby(df.index)[price_col].mean()
        return out

    work = df[[price_col, loc_col] + ([loc_type_col] if loc_type_col else [])].copy()
    work[loc_col] = work[loc_col].astype(str)
    ts_col = work.index.name or "timestamp"
    work = work.reset_index().rename(columns={work.index.name or "index": ts_col})

    # Aggregate repeated rows at same timestamp/location.
    long = (
        work.groupby([ts_col, loc_col], as_index=False)[price_col]
        .mean()
        .rename(columns={price_col: "price", loc_col: "location"})
    )

    pivot = long.pivot(index=ts_col, columns="location", values="price")

    zone_aliases = {
        f"{prefix}_north": ["LZ_NORTH", "NORTH"],
        f"{prefix}_south": ["LZ_SOUTH", "SOUTH"],
        f"{prefix}_west": ["LZ_WEST", "WEST"],
        f"{prefix}_houston": ["LZ_HOUSTON", "HOUSTON"],
    }

    # Zone-level columns where available.
    for out_col, aliases in zone_aliases.items():
        matched = [c for c in pivot.columns if any(a in str(c).upper() for a in aliases)]
        if matched:
            out[out_col] = pivot[matched].mean(axis=1)

    # Hub: prefer explicit HUB location-type rows if present, otherwise hub-like names.
    hub_series = None
    if loc_type_col is not None:
        with_type = df[[price_col, loc_type_col]].copy()
        hub_rows = with_type[with_type[loc_type_col].astype(str).str.upper().str.contains("HUB", na=False)]
        if not hub_rows.empty:
            hub_series = hub_rows.groupby(hub_rows.index)[price_col].mean()

    if hub_series is None:
        hub_like_cols = [c for c in pivot.columns if "HUB" in str(c).upper()]
        if hub_like_cols:
            hub_series = pivot[hub_like_cols].mean(axis=1)

    if hub_series is None:
        # Last-resort fallback: average all available locations.
        hub_series = df.groupby(df.index)[price_col].mean()

    out[f"{prefix}_hub"] = hub_series
    return out


def _extract_as_wide(df: pd.DataFrame, out_prefix: str) -> pd.DataFrame:
    """Extract AS products from wide-format columns."""
    out = pd.DataFrame(index=df.index.unique().sort_values())

    product_map = {
        "regup": ["regup", "regupmw", "regupprice", "regupmcpc", "regupclearingprice", "regulationup"],
        "regdown": ["regdn", "regdown", "regdownmw", "regdownprice", "regdownmcpc", "regulationdown"],
        "rrs": ["rrs", "responsivereserve", "rrsmw", "rrsprice", "rrsmcpc"],
        "ecrs": ["ecrs", "ercotcontingencyreserve", "ercotcontingencyreserveservice", "ecrsmw", "ecrsprice", "ecrsmcpc"],
        "nsrs": ["nsrs", "nonspin", "nonspinningreserve", "nsrsmw", "nsrsprice", "nsrsmcpc"],
    }

    grouped_mean = df.groupby(df.index).mean(numeric_only=True)

    for product, aliases in product_map.items():
        matched = []
        for col in grouped_mean.columns:
            ncol = _norm_col_name(col)
            if any(alias in ncol for alias in aliases):
                matched.append(col)

        out_col = f"{out_prefix}_{product}"
        if matched:
            out[out_col] = grouped_mean[matched].mean(axis=1)
        else:
            out[out_col] = np.nan

    return out


def _extract_as_long(df: pd.DataFrame, out_prefix: str) -> pd.DataFrame | None:
    """Extract AS products from long-format rows with a product/service column."""
    prod_col = _find_first_column(
        df,
        [
            "Ancillary Type",
            "AS Type",
            "Service",
            "Product",
            "AS Product",
            "Service Type",
            "Ancillary Service",
        ],
    )
    price_col = _find_price_column(df)

    if prod_col is None or price_col is None:
        return None

    work = df[[prod_col, price_col]].copy()
    work[prod_col] = work[prod_col].astype(str).str.upper()

    product_aliases = {
        "regup": ["REGUP", "REG-UP", "REG_UP", "REGULATION UP"],
        "regdown": ["REGDN", "REGDOWN", "REG-DOWN", "REG_DOWN", "REGULATION DOWN"],
        "rrs": ["RRS", "RESPONSIVE"],
        "ecrs": ["ECRS", "CONTINGENCY"],
        "nsrs": ["NSRS", "NONSPIN", "NON-SPIN"],
    }

    out = pd.DataFrame(index=df.index.unique().sort_values())
    for product, aliases in product_aliases.items():
        mask = work[prod_col].apply(lambda v: any(a in v for a in aliases))
        series = work.loc[mask].groupby(work.loc[mask].index)[price_col].mean()
        out[f"{out_prefix}_{product}"] = series

    return out


def normalize_timestamp_index(
    df: pd.DataFrame,
    time_col: str = "Time",
) -> pd.DataFrame:
    """
    Ensure DataFrame has a UTC DatetimeIndex.

    gridstatus typically returns data with a 'Time' column or
    datetime index already localized. This normalizes to UTC.

    Parameters
    ----------
    df : pd.DataFrame
        Raw gridstatus output.
    time_col : str
        Name of the time column if not already the index.
    """
    df = df.copy()

    # If time is a column, set it as index
    if time_col in df.columns:
        df = df.set_index(time_col)
    elif "Interval Start" in df.columns:
        df = df.set_index("Interval Start")
    elif "Interval End" in df.columns:
        df = df.set_index("Interval End")
        # Shift back by interval duration to get start time
        if hasattr(df.index, 'freq') and df.index.freq:
            df.index = df.index - df.index.freq

    # Ensure datetime type
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Localize to UTC
    if df.index.tz is None:
        # Assume CPT if no timezone
        df.index = df.index.tz_localize("US/Central", ambiguous="NaT").tz_convert("UTC")
    elif str(df.index.tz) != "UTC":
        df.index = df.index.tz_convert("UTC")

    df.index.name = "timestamp"
    return df.sort_index()


def align_to_5min(
    df: pd.DataFrame,
    method: str = "ffill",
) -> pd.DataFrame:
    """
    Resample any DataFrame to 5-minute canonical intervals.

    For data at coarser granularity (15-min, hourly), forward-fills
    to maintain the last known value at each 5-min tick.
    """
    df = normalize_timestamp_index(df)

    # Detect source frequency
    if len(df) > 1:
        median_diff = df.index.to_series().diff().median()
        logger.info(f"Detected source frequency: {median_diff}")

    return resample_to_5min(df, method=method)


def process_energy_prices(
    rt_spp_df: pd.DataFrame,
    dam_spp_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Process raw energy price DataFrames into canonical schema.

    Handles both long-format (location + price rows) and wide-format
    (already split columns) gridstatus outputs.
    """
    rt = normalize_timestamp_index(rt_spp_df)
    dam = normalize_timestamp_index(dam_spp_df)

    rt_extracted = _extract_hub_and_zones(rt, prefix="rt_spp")
    dam_extracted = _extract_hub_and_zones(dam, prefix="dam_spp")

    rt_5min = resample_to_5min(rt_extracted)
    dam_5min = resample_to_5min(dam_extracted)

    result = rt_5min.join(dam_5min[["dam_spp_hub"]], how="outer")

    # Ensure schema columns are present even if unavailable in source data.
    for col in ["rt_spp_hub", "rt_spp_north", "rt_spp_south", "rt_spp_west", "rt_spp_houston", "dam_spp_hub"]:
        if col not in result.columns:
            result[col] = np.nan

    result = result.sort_index()
    result = add_post_rtcb_flag(result)
    return result


def process_as_prices(
    dam_as_df: pd.DataFrame,
    rt_as_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Process AS price DataFrames into canonical schema.

    Supports both wide and long DataFrame shapes for DAM and RT inputs.
    """
    dam = normalize_timestamp_index(dam_as_df)
    dam_long = _extract_as_long(dam, out_prefix="dam_as")
    dam_extracted = dam_long if dam_long is not None else _extract_as_wide(dam, out_prefix="dam_as")
    dam_5min = resample_to_5min(dam_extracted)

    result = dam_5min.copy()

    rt_cols = ["rt_mcpc_regup", "rt_mcpc_regdown", "rt_mcpc_rrs", "rt_mcpc_ecrs", "rt_mcpc_nsrs"]

    if rt_as_df is not None and not rt_as_df.empty:
        rt = normalize_timestamp_index(rt_as_df)
        rt_long = _extract_as_long(rt, out_prefix="rt_mcpc")
        rt_extracted = rt_long if rt_long is not None else _extract_as_wide(rt, out_prefix="rt_mcpc")
        rt_5min = resample_to_5min(rt_extracted)
        result = result.join(rt_5min[rt_cols], how="outer")

    for col in rt_cols:
        if col not in result.columns:
            result[col] = np.nan

    # Pre-RTC+B should have NaN RT MCPC columns.
    pre_mask = result.index < RTCB_GO_LIVE
    result.loc[pre_mask, rt_cols] = np.nan

    # Ensure all canonical DAM columns exist.
    for col in ["dam_as_regup", "dam_as_regdown", "dam_as_rrs", "dam_as_ecrs", "dam_as_nsrs"]:
        if col not in result.columns:
            result[col] = np.nan

    result = result.sort_index()
    result = add_post_rtcb_flag(result)
    return result


def process_system_conditions(
    load_df: pd.DataFrame,
    fuel_mix_df: pd.DataFrame = None,
    wind_df: pd.DataFrame = None,
    solar_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Process system condition data into canonical schema.
    """
    load = normalize_timestamp_index(load_df)
    load_aligned = resample_to_5min(load)

    result = pd.DataFrame(index=load_aligned.index)

    # Load data
    total_load_col = _find_first_column(
        load_aligned,
        ["Load", "Total Load", "System Total", "Actual Load", "Demand"],
    )
    if total_load_col:
        result["total_load_mw"] = load_aligned[total_load_col]

    forecast_col = _find_first_column(
        load_aligned,
        ["Forecast", "Load Forecast", "Forecast Load", "System Forecast"],
    )
    if forecast_col:
        result["load_forecast_mw"] = load_aligned[forecast_col]

    # Wind/Solar
    if wind_df is not None and not wind_df.empty:
        wind = normalize_timestamp_index(wind_df)
        wind_aligned = resample_to_5min(wind)
        wind_col = _find_first_column(wind_aligned, ["Wind", "Actual", "Generation", "Output"])
        if wind_col is None and len(wind_aligned.columns) > 0:
            wind_col = wind_aligned.select_dtypes(include=[np.number]).columns[0]
        if wind_col is not None:
            result["wind_actual_mw"] = wind_aligned[wind_col]

    if solar_df is not None and not solar_df.empty:
        solar = normalize_timestamp_index(solar_df)
        solar_aligned = resample_to_5min(solar)
        solar_col = _find_first_column(solar_aligned, ["Solar", "Actual", "Generation", "Output"])
        if solar_col is None and len(solar_aligned.columns) > 0:
            solar_col = solar_aligned.select_dtypes(include=[np.number]).columns[0]
        if solar_col is not None:
            result["solar_actual_mw"] = solar_aligned[solar_col]

    # Derived: net load
    if "total_load_mw" in result.columns:
        wind_gen = result.get("wind_actual_mw", 0)
        solar_gen = result.get("solar_actual_mw", 0)
        result["net_load_mw"] = result["total_load_mw"] - wind_gen - solar_gen

    result = add_post_rtcb_flag(result)
    return result


def validate_dataframe(
    df: pd.DataFrame,
    table_name: str,
) -> dict:
    """
    Run quality checks on a processed DataFrame.

    Returns a dict with validation results.
    """
    checks = {
        "table": table_name,
        "rows": len(df),
        "date_range": f"{df.index.min()} to {df.index.max()}" if len(df) > 0 else "empty",
        "columns": list(df.columns),
        "null_fractions": df.isnull().mean().to_dict(),
        "duplicated_timestamps": df.index.duplicated().sum(),
    }

    # Check for gaps in 5-minute index
    if len(df) > 1:
        expected_intervals = pd.date_range(
            df.index.min(), df.index.max(), freq="5min"
        )
        missing = expected_intervals.difference(df.index)
        checks["missing_intervals"] = len(missing)
        checks["coverage_pct"] = (1 - len(missing) / len(expected_intervals)) * 100
    else:
        checks["missing_intervals"] = 0
        checks["coverage_pct"] = 0

    return checks


def write_parquet(
    df: pd.DataFrame,
    output_dir: Path,
    table_name: str,
    partition_by_month: bool = True,
):
    """
    Write processed DataFrame to Parquet files.

    Parameters
    ----------
    df : pd.DataFrame
        Processed data with UTC DatetimeIndex.
    output_dir : Path
        Base output directory (e.g., data/processed/).
    table_name : str
        Table name (e.g., 'energy_prices').
    partition_by_month : bool
        If True, write separate files per month for efficient queries.
    """
    table_dir = output_dir / table_name
    table_dir.mkdir(parents=True, exist_ok=True)

    if partition_by_month and len(df) > 0:
        # Group by year-month and write separate files
        df["_year_month"] = df.index.to_period("M").astype(str)
        for ym, group in df.groupby("_year_month"):
            group = group.drop(columns=["_year_month"])
            path = table_dir / f"{ym}.parquet"
            group.to_parquet(path, compression="snappy")
            logger.info(f"Wrote {len(group)} rows to {path}")
    else:
        path = table_dir / f"{table_name}.parquet"
        df.to_parquet(path, compression="snappy")
        logger.info(f"Wrote {len(df)} rows to {path}")


def read_parquet(
    data_dir: Path,
    table_name: str,
    start: str = None,
    end: str = None,
) -> pd.DataFrame:
    """
    Read processed Parquet files for a given table and date range.

    Parameters
    ----------
    data_dir : Path
        Base data directory (e.g., data/processed/).
    table_name : str
        Table name.
    start, end : str, optional
        Date range filter.

    Returns
    -------
    pd.DataFrame
        Combined data from matching Parquet files.
    """
    table_dir = data_dir / table_name

    if not table_dir.exists():
        raise FileNotFoundError(f"No data found at {table_dir}")

    # Read all parquet files
    dfs = []
    for f in sorted(table_dir.glob("*.parquet")):
        dfs.append(pd.read_parquet(f))

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs).sort_index()

    # Remove any duplicates
    df = df[~df.index.duplicated(keep="last")]

    # Filter by date range
    if start:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df.index < pd.Timestamp(end, tz="UTC")]

    return df
