"""
ERCOT data fetcher — wraps the gridstatus library.

Two ingestion paths:
  Path A: gridstatus.ErcotAPI — for data from Dec 2023 onward (ERCOT REST API)
  Path B: gridstatus.Ercot    — web scraping for historical data and real-time feeds

Both paths produce pandas DataFrames that are then cleaned and written
to canonical Parquet schema by the preprocessing module.
"""

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _call_method_if_exists(obj, method_name: str, **kwargs):
    """Call a method if present; return None when not available."""
    if not hasattr(obj, method_name):
        return None
    method = getattr(obj, method_name)
    return method(**kwargs)


def _filter_by_datetime_window(
    df: pd.DataFrame,
    start: str,
    end: str,
    time_col_candidates: list[str] | None = None,
) -> pd.DataFrame:
    """Filter a DataFrame to [start, end) using index or common time columns."""
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df

    if time_col_candidates is None:
        time_col_candidates = ["Time", "Interval Start", "SCED Timestamp", "Timestamp"]

    out = df.copy()
    ts_col = next((c for c in time_col_candidates if c in out.columns), None)

    if ts_col is not None:
        ts = pd.to_datetime(out[ts_col], utc=True, errors="coerce")
        mask = (ts >= pd.Timestamp(start, tz="UTC")) & (ts < pd.Timestamp(end, tz="UTC"))
        return out.loc[mask].copy()

    if isinstance(out.index, pd.DatetimeIndex):
        idx = out.index
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
        mask = (idx >= pd.Timestamp(start, tz="UTC")) & (idx < pd.Timestamp(end, tz="UTC"))
        return out.loc[mask].copy()

    return out


def get_ercot_api():
    """
    Initialize gridstatus ErcotAPI client.
    Requires ERCOT API credentials (free registration at apiexplorer.ercot.com).
    """
    try:
        # Older/newer gridstatus versions may expose ErcotAPI differently.
        from gridstatus import ErcotAPI  # type: ignore
    except ImportError:
        try:
            from gridstatus.ercot_api.ercot_api import ErcotAPI  # type: ignore
        except ImportError as e:
            raise ImportError(
                "ErcotAPI not available in this gridstatus install. "
                "Ensure gridstatus>=0.34.0 is installed."
            ) from e

    # ErcotAPI will look for credentials in environment or allow anonymous
    # for public endpoints. Check gridstatus docs for auth setup.
    return ErcotAPI()


def get_ercot_scraper():
    """
    Initialize gridstatus Ercot web scraper client.
    No credentials needed — scrapes public ERCOT website.
    """
    try:
        from gridstatus import Ercot
    except ImportError:
        raise ImportError(
            "gridstatus not installed. Run: pip install gridstatus>=0.34.0"
        )

    return Ercot()


# ──────────────────────────────────────────────
# Energy Price Fetchers
# ──────────────────────────────────────────────

def fetch_rt_spp(
    start: str,
    end: str,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Fetch real-time Settlement Point Prices.

    Uses gridstatus Ercot scraper which provides SPP data across the
    full historical range via ERCOT's public website.

    Parameters
    ----------
    start, end : str
        Date range (e.g., '2024-01-01', '2024-02-01').

    Returns
    -------
    pd.DataFrame
        Columns vary by gridstatus version. Inspect output during
        Day 1 exploration and update column mappings in schema.py.
    """
    ercot = get_ercot_scraper()

    if verbose:
        logger.info(f"Fetching RT SPP: {start} to {end}")

    try:
        df = ercot.get_spp(
            date=start,
            end=end,
            market="REAL_TIME_15_MIN",
            verbose=verbose,
        )
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    except Exception as e:
        logger.warning(f"get_spp REAL_TIME_15_MIN failed: {e}")

    # Fallback for gridstatus versions where get_spp has flaky behavior.
    try:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        years = range(start_ts.year, end_ts.year + 1)
        frames = []
        for y in years:
            try:
                yearly = ercot.get_rtm_spp(year=y, verbose=verbose)
                frames.append(_filter_by_datetime_window(yearly, start, end))
            except Exception as year_err:
                logger.warning(f"get_rtm_spp(year={y}) failed: {year_err}")
        if frames:
            return pd.concat(frames, ignore_index=True)
    except Exception as e:
        logger.warning(f"Failed to fetch RT SPP via fallback get_rtm_spp: {e}")

    return pd.DataFrame()


def fetch_dam_spp(
    start: str,
    end: str,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fetch day-ahead Settlement Point Prices."""
    ercot = get_ercot_scraper()

    if verbose:
        logger.info(f"Fetching DAM SPP: {start} to {end}")

    try:
        df = ercot.get_spp(
            date=start,
            end=end,
            market="DAY_AHEAD_HOURLY",
            verbose=verbose,
        )
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    except Exception as e:
        logger.warning(f"get_spp DAY_AHEAD_HOURLY failed: {e}")

    try:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        years = range(start_ts.year, end_ts.year + 1)
        frames = []
        for y in years:
            try:
                yearly = ercot.get_dam_spp(year=y, verbose=verbose)
                frames.append(_filter_by_datetime_window(yearly, start, end))
            except Exception as year_err:
                logger.warning(f"get_dam_spp(year={y}) failed: {year_err}")
        if frames:
            return pd.concat(frames, ignore_index=True)
    except Exception as e:
        logger.warning(f"Failed to fetch DAM SPP via fallback get_dam_spp: {e}")

    return pd.DataFrame()


# ──────────────────────────────────────────────
# Ancillary Service Price Fetchers
# ──────────────────────────────────────────────

def fetch_dam_as_prices(
    start: str,
    end: str,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Fetch day-ahead AS clearing prices.
    Available for full historical period (2010+).
    """
    ercot = get_ercot_scraper()

    if verbose:
        logger.info(f"Fetching DAM AS prices: {start} to {end}")

    try:
        df = ercot.get_as_prices(
            date=start,
            end=end,
            verbose=verbose,
        )
        return df
    except Exception as e:
        logger.error(f"Failed to fetch DAM AS prices: {e}")
        raise


def fetch_rt_as_prices(
    start: str,
    end: str,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Fetch real-time AS MCPCs (post-RTC+B only, from Dec 5, 2025).

    This data product (NP6-332-CD) only exists after RTC+B go-live.
    Calling this for dates before Dec 5, 2025 will return empty results.

    The exact gridstatus method name varies by version. This function
    tries known candidates before falling back to report-type fetch.
    """
    if verbose:
        logger.info(f"Fetching RT AS MCPCs: {start} to {end}")

    # Try scraper methods first (some gridstatus versions expose these).
    try:
        ercot = get_ercot_scraper()
        scraper_candidates = [
            "get_as_mcpc",
            "get_rt_as_clearing_prices",
            "get_as_clearing_prices",
            "get_mcpc",
            "get_mcpc_sced",
            "get_mcpc_real_time_15_min",
            "get_indicative_mcpc_rtd",
        ]
        for method_name in scraper_candidates:
            try:
                df = _call_method_if_exists(
                    ercot,
                    method_name,
                    date=start,
                    end=end,
                    verbose=verbose,
                )
            except TypeError:
                # Some methods may not accept verbose/date kwarg names.
                df = _call_method_if_exists(ercot, method_name, start=start, end=end)
            if isinstance(df, pd.DataFrame):
                logger.info(f"Fetched RT AS via Ercot.{method_name}()")
                return df
    except Exception as e:
        logger.warning(f"Scraper RT AS method attempts failed: {e}")

    # API path: named methods first, then generic report_type_id fallback.
    try:
        api = get_ercot_api()
    except Exception as e:
        msg = str(e)
        if "Username, password, and subscription key must be provided" in msg:
            logger.warning("ERCOT API credentials not configured; returning empty RT AS DataFrame.")
            return pd.DataFrame()
        raise
    api_candidates = [
        "get_as_mcpc",
        "get_rt_as_clearing_prices",
        "get_as_clearing_prices",
    ]
    for method_name in api_candidates:
        try:
            df = _call_method_if_exists(api, method_name, start=start, end=end)
            if isinstance(df, pd.DataFrame):
                logger.info(f"Fetched RT AS via ErcotAPI.{method_name}()")
                return df
        except Exception as e:
            logger.debug(f"ErcotAPI.{method_name} failed: {e}")

    try:
        df = api.get_data(
            report_type_id="NP6-332-CD",
            start=start,
            end=end,
        )
        if isinstance(df, pd.DataFrame):
            logger.info("Fetched RT AS via ErcotAPI.get_data(report_type_id='NP6-332-CD')")
            return df
    except Exception as e:
        msg = str(e)
        if "Username, password, and subscription key must be provided" in msg:
            logger.warning("ERCOT API credentials not configured; returning empty RT AS DataFrame.")
            return pd.DataFrame()
        logger.error(f"Failed RT AS fallback fetch (NP6-332-CD): {e}")
        raise

    logger.warning("No RT AS MCPC method succeeded; returning empty DataFrame.")
    return pd.DataFrame()


# ──────────────────────────────────────────────
# System Conditions Fetchers
# ──────────────────────────────────────────────

def fetch_load(
    start: str,
    end: str,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fetch system load data (actual + forecast)."""
    ercot = get_ercot_scraper()

    if verbose:
        logger.info(f"Fetching load data: {start} to {end}")

    try:
        df = ercot.get_load(
            date=start,
            end=end,
            verbose=verbose,
        )
        return df
    except Exception as e:
        logger.error(f"Failed to fetch load: {e}")
        raise


def fetch_fuel_mix(
    start: str,
    end: str,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fetch generation by fuel type."""
    ercot = get_ercot_scraper()

    if verbose:
        logger.info(f"Fetching fuel mix: {start} to {end}")

    try:
        # get_fuel_mix currently works best with single-date calls.
        frames = []
        for day in pd.date_range(start=start, end=end, inclusive="left", freq="1D"):
            try:
                day_df = ercot.get_fuel_mix(
                    date=day.date().isoformat(),
                    verbose=verbose,
                )
                if isinstance(day_df, pd.DataFrame) and not day_df.empty:
                    frames.append(day_df)
            except Exception as day_err:
                logger.warning(f"Fuel mix fetch failed for {day.date()}: {day_err}")

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)
    except Exception as e:
        logger.error(f"Failed to fetch fuel mix: {e}")
        raise


def fetch_wind_solar(
    start: str,
    end: str,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Fetch wind and solar generation data (actuals + forecasts).

    Returns
    -------
    dict with keys 'wind' and 'solar', each a DataFrame.
    """
    ercot = get_ercot_scraper()
    results = {}

    if verbose:
        logger.info(f"Fetching wind/solar: {start} to {end}")

    results["wind"] = pd.DataFrame()
    results["solar"] = pd.DataFrame()

    wind_methods = [
        "get_wind_actual_and_forecast_hourly",
        "get_wind_actual_and_forecast_by_geographical_region_hourly",
    ]
    solar_methods = [
        "get_solar_actual_and_forecast_hourly",
        "get_solar_actual_and_forecast_by_geographical_region_hourly",
    ]

    for method_name in wind_methods:
        try:
            df = _call_method_if_exists(ercot, method_name, date=start, end=end, verbose=verbose)
            if isinstance(df, pd.DataFrame) and not df.empty:
                results["wind"] = df
                break
        except Exception as e:
            logger.warning(f"{method_name} failed (wind): {e}")

    for method_name in solar_methods:
        try:
            df = _call_method_if_exists(ercot, method_name, date=start, end=end, verbose=verbose)
            if isinstance(df, pd.DataFrame) and not df.empty:
                results["solar"] = df
                break
        except Exception as e:
            logger.warning(f"{method_name} failed (solar): {e}")

    return results


# ──────────────────────────────────────────────
# Exploration Helper
# ──────────────────────────────────────────────

def explore_available_methods():
    """
    Print all available gridstatus methods for ERCOT.
    Run this on Day 1 to discover what data is accessible.
    """
    ercot = get_ercot_scraper()

    print("=" * 60)
    print("gridstatus.Ercot — Available methods:")
    print("=" * 60)
    methods = [m for m in dir(ercot) if m.startswith("get_")]
    for m in sorted(methods):
        doc = getattr(ercot, m).__doc__
        first_line = doc.strip().split("\n")[0] if doc else "(no docstring)"
        print(f"  .{m}() — {first_line}")

    try:
        api = get_ercot_api()
        print("\n" + "=" * 60)
        print("gridstatus.ErcotAPI — Available methods:")
        print("=" * 60)
        methods = [m for m in dir(api) if m.startswith("get_")]
        for m in sorted(methods):
            doc = getattr(api, m).__doc__
            first_line = doc.strip().split("\n")[0] if doc else "(no docstring)"
            print(f"  .{m}() — {first_line}")
    except Exception as e:
        print(f"\nErcotAPI not available (credentials needed?): {e}")


if __name__ == "__main__":
    explore_available_methods()
