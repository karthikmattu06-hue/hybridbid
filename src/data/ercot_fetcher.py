"""
ERCOT data fetcher — confirmed access methods from Phase 1/1b exploration.

Access methods per product (all confirmed working):
  - RT LMP (5-min):    ErcotAPI.get_lmp_by_settlement_point()  — back to 2020
  - DAM SPP (hourly):  Ercot.get_dam_spp(year=)                — back to 2020
  - DAM AS (hourly):   ErcotAPI.get_as_prices()                 — back to 2020
  - Load actual:       Ercot.get_hourly_load_post_settlements() — back to 2020
  - Load forecast:     ErcotAPI.get_load_forecast_by_model()    — back to 2020
  - Wind:              ErcotAPI.get_wind_actual_and_forecast_hourly() — back to 2020
  - Solar:             ErcotAPI.get_solar_actual_and_forecast_hourly() — back to 2020
  - RT SCED MCPC:      File loader (pre-downloaded Parquet in data/raw/sced_mcpc/)

Credentials via environment variables:
  ERCOT_API_USERNAME, ERCOT_API_PASSWORD, ERCOT_PUBLIC_API_SUBSCRIPTION_KEY
"""

import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd

# Load .env from project root (no-op if vars already set)
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"

# Singleton clients (lazy-initialized)
_api_client = None
_scraper_client = None


def _get_api():
    """Get or create the ErcotAPI client (singleton)."""
    global _api_client
    if _api_client is not None:
        return _api_client

    try:
        from gridstatus.ercot_api.ercot_api import ErcotAPI
    except ImportError:
        raise ImportError("gridstatus>=0.35.0 required. Run: pip install gridstatus>=0.35.0")

    _api_client = ErcotAPI(sleep_seconds=2.0, max_retries=5)
    # Pre-acquire auth token to avoid intermittent failures on first request
    _api_client.get_token()
    logger.info("ErcotAPI client initialized (token acquired)")
    return _api_client


def _get_scraper():
    """Get or create the Ercot scraper client (singleton)."""
    global _scraper_client
    if _scraper_client is not None:
        return _scraper_client

    from gridstatus import Ercot
    _scraper_client = Ercot()
    logger.info("Ercot scraper client initialized")
    return _scraper_client


def _cache_path(product: str, date_str: str) -> Path:
    """Path for a cached raw daily Parquet file."""
    product_dir = DATA_RAW / product
    product_dir.mkdir(parents=True, exist_ok=True)
    return product_dir / f"{date_str}.parquet"


def _load_cached(product: str, date_str: str) -> pd.DataFrame | None:
    """Load cached raw file if it exists."""
    path = _cache_path(product, date_str)
    if path.exists():
        return pd.read_parquet(path)
    return None


def _save_cache(df: pd.DataFrame, product: str, date_str: str):
    """Save raw DataFrame to cache."""
    if df is not None and not df.empty:
        path = _cache_path(product, date_str)
        df.to_parquet(path, compression="snappy")


# ──────────────────────────────────────────────
# RT LMP (5-min SCED) via ErcotAPI
# ──────────────────────────────────────────────

def fetch_rt_lmp(start: str, end: str, location: str = "HB_HUBAVG") -> pd.DataFrame:
    """
    Fetch RT LMPs from ErcotAPI (NP6-788-CD, 5-min SCED intervals).

    Returns DataFrame with columns:
        Interval Start, Interval End, SCED Timestamp, Market, Location, Location Type, LMP

    Filtered to the specified location (default: HB_HUBAVG).
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    all_frames = []
    current = start_ts
    while current < end_ts:
        next_day = current + pd.Timedelta(days=1)
        if next_day > end_ts:
            next_day = end_ts
        date_str = current.strftime("%Y-%m-%d")

        cached = _load_cached("rt_lmp", date_str)
        if cached is not None:
            logger.info(f"RT LMP {date_str}: loaded from cache ({len(cached)} rows)")
            all_frames.append(cached)
            current = next_day
            continue

        logger.info(f"RT LMP {date_str}: fetching from ErcotAPI...")
        try:
            api = _get_api()
            df = api.get_lmp_by_settlement_point(
                date=date_str,
                end=next_day.strftime("%Y-%m-%d"),
                verbose=False,
            )
            # Filter to requested location before caching
            if not df.empty and "Location" in df.columns:
                df = df[df["Location"] == location].reset_index(drop=True)
            logger.info(f"RT LMP {date_str}: got {len(df)} rows")
            _save_cache(df, "rt_lmp", date_str)
            all_frames.append(df)
            # Rate limit protection between day requests
            logger.debug(f"RT LMP: sleeping 2s before next day...")
            time.sleep(2)
        except Exception as e:
            logger.warning(f"RT LMP {date_str}: fetch failed ({e}), skipping")
        current = next_day

    if not all_frames:
        return pd.DataFrame()
    return pd.concat(all_frames, ignore_index=True)


# ──────────────────────────────────────────────
# DAM SPP (hourly) via scraper yearly bulk
# ──────────────────────────────────────────────

def fetch_dam_spp(start: str, end: str, location: str = "HB_HUBAVG") -> pd.DataFrame:
    """
    Fetch DAM SPPs via scraper yearly bulk download.

    Returns long-format DataFrame filtered to specified location.
    Columns: Time, Interval Start, Interval End, Location, Location Type, Market, SPP
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    years = range(start_ts.year, end_ts.year + 1)

    scraper = _get_scraper()
    frames = []
    for year in years:
        cache_key = f"dam_spp_year_{year}"
        cached = _load_cached("dam_spp", str(year))
        if cached is not None:
            logger.info(f"DAM SPP {year}: loaded from cache ({len(cached)} rows)")
            frames.append(cached)
            continue

        logger.info(f"DAM SPP {year}: fetching yearly bulk...")
        df = scraper.get_dam_spp(year=year, verbose=False)
        # Filter to location before caching
        if not df.empty and "Location" in df.columns:
            df = df[df["Location"] == location].reset_index(drop=True)
        _save_cache(df, "dam_spp", str(year))
        logger.info(f"DAM SPP {year}: got {len(df)} rows")
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    # Filter to date range
    if "Interval Start" in combined.columns:
        combined["Interval Start"] = pd.to_datetime(combined["Interval Start"], utc=True)
        start_utc = pd.Timestamp(start, tz="UTC")
        end_utc = pd.Timestamp(end, tz="UTC")
        combined = combined[
            (combined["Interval Start"] >= start_utc)
            & (combined["Interval Start"] < end_utc)
        ]
    return combined.reset_index(drop=True)


# ──────────────────────────────────────────────
# DAM AS Prices (hourly) via ErcotAPI
# ──────────────────────────────────────────────

def fetch_dam_as(start: str, end: str) -> pd.DataFrame:
    """
    Fetch DAM AS clearing prices from ErcotAPI (NP4-188-CD).

    Returns wide-format DataFrame with columns:
        Interval Start, Interval End, Market,
        Non-Spinning Reserves, Regulation Down, Regulation Up,
        Responsive Reserves, ERCOT Contingency Reserve Service
    """
    logger.info(f"DAM AS: fetching {start} → {end}...")
    try:
        api = _get_api()
        df = api.get_as_prices(date=start, end=end, verbose=False)
        logger.info(f"DAM AS: got {len(df)} rows")
        return df
    except Exception as e:
        logger.warning(f"DAM AS: fetch failed ({e}), returning empty")
        return pd.DataFrame()


# ──────────────────────────────────────────────
# Load Actual (hourly) via scraper
# ──────────────────────────────────────────────

def fetch_load_actual(start: str, end: str) -> pd.DataFrame:
    """
    Fetch hourly actual load via scraper's post-settlements archive.

    Returns DataFrame with columns:
        Interval Start, Interval End, Coast, East, Far West, North,
        North Central, South, South Central, West, ERCOT
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    years = range(start_ts.year, end_ts.year + 1)

    scraper = _get_scraper()
    frames = []
    for year in years:
        cached = _load_cached("load_actual", str(year))
        if cached is not None:
            logger.info(f"Load actual {year}: loaded from cache ({len(cached)} rows)")
            frames.append(cached)
            continue

        logger.info(f"Load actual {year}: fetching yearly archive...")
        df = scraper.get_hourly_load_post_settlements(
            date=f"{year}-01-01", verbose=False
        )
        _save_cache(df, "load_actual", str(year))
        logger.info(f"Load actual {year}: got {len(df)} rows")
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    if "Interval Start" in combined.columns:
        combined["Interval Start"] = pd.to_datetime(combined["Interval Start"], utc=True)
        start_utc = pd.Timestamp(start, tz="UTC")
        end_utc = pd.Timestamp(end, tz="UTC")
        combined = combined[
            (combined["Interval Start"] >= start_utc)
            & (combined["Interval Start"] < end_utc)
        ]
    return combined.reset_index(drop=True)


# ──────────────────────────────────────────────
# Load Forecast (hourly) via ErcotAPI
# ──────────────────────────────────────────────

def fetch_load_forecast(start: str, end: str) -> pd.DataFrame:
    """
    Fetch hourly load forecast from ErcotAPI (NP3-565-CD).

    Returns DataFrame with columns:
        Interval Start, Interval End, Publish Time, Model,
        Coast, East, Far West, North, North Central,
        South Central, Southern, West, System Total, In Use Flag
    """
    logger.info(f"Load forecast: fetching {start} → {end}...")
    try:
        api = _get_api()
        df = api.get_load_forecast_by_model(date=start, end=end, verbose=False)
        logger.info(f"Load forecast: got {len(df)} rows")
        return df
    except Exception as e:
        logger.warning(f"Load forecast: fetch failed ({e}), returning empty")
        return pd.DataFrame()


# ──────────────────────────────────────────────
# Wind (hourly) via ErcotAPI
# ──────────────────────────────────────────────

def fetch_wind(start: str, end: str) -> pd.DataFrame:
    """
    Fetch wind actual + forecast from ErcotAPI (NP4-732-CD).

    Returns DataFrame with columns including:
        Interval Start, Interval End, Publish Time,
        GEN SYSTEM WIDE, STWPF SYSTEM WIDE, ...
    """
    logger.info(f"Wind: fetching {start} → {end}...")
    try:
        api = _get_api()
        df = api.get_wind_actual_and_forecast_hourly(date=start, end=end, verbose=False)
        logger.info(f"Wind: got {len(df)} rows")
        return df
    except Exception as e:
        logger.warning(f"Wind: fetch failed ({e}), returning empty")
        return pd.DataFrame()


# ──────────────────────────────────────────────
# Solar (hourly) via ErcotAPI
# ──────────────────────────────────────────────

def fetch_solar(start: str, end: str) -> pd.DataFrame:
    """
    Fetch solar actual + forecast from ErcotAPI (NP4-737-CD).

    Returns DataFrame with columns including:
        Interval Start, Interval End, Publish Time,
        GEN SYSTEM WIDE, STPPF SYSTEM WIDE, ...
    """
    logger.info(f"Solar: fetching {start} → {end}...")
    try:
        api = _get_api()
        df = api.get_solar_actual_and_forecast_hourly(date=start, end=end, verbose=False)
        logger.info(f"Solar: got {len(df)} rows")
        return df
    except Exception as e:
        logger.warning(f"Solar: fetch failed ({e}), returning empty")
        return pd.DataFrame()


# ──────────────────────────────────────────────
# RT SCED MCPC — file loader (pre-downloaded)
# ──────────────────────────────────────────────

def load_rt_mcpc(start: str, end: str) -> pd.DataFrame:
    """
    Load RT SCED MCPCs from pre-downloaded Parquet files.

    Files expected at: data/raw/sced_mcpc/YYYY-MM-DD.parquet
    Long format columns: sced_timestamp, repeated_hour_flag, as_type, mcpc
    (sced_timestamp is datetime64[ns, UTC])

    Returns empty DataFrame if no files found (pre-RTC+B dates).
    """
    mcpc_dir = DATA_RAW / "sced_mcpc"
    if not mcpc_dir.exists():
        logger.info("RT MCPC: no sced_mcpc directory found, returning empty")
        return pd.DataFrame()

    start_date = pd.Timestamp(start).date()
    end_date = pd.Timestamp(end).date()

    frames = []
    current = start_date
    while current < end_date:
        path = mcpc_dir / f"{current.isoformat()}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            frames.append(df)
        current += pd.Timedelta(days=1)

    if not frames:
        logger.info(f"RT MCPC: no files found for {start} → {end}")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"RT MCPC: loaded {len(combined)} rows from {len(frames)} files")
    return combined
