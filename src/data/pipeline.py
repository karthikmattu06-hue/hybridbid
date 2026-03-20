"""
HybridBid Data Pipeline — Main orchestrator.

Downloads ERCOT data, processes it to canonical schema, and writes Parquet files.

Usage:
    python -m src.data.pipeline --start 2024-01-01 --end 2026-02-01
    python -m src.data.pipeline --start 2024-01-01 --end 2026-02-01 --tables energy_prices
    python -m src.data.pipeline --backfill  # Full 2020-2026 backfill (slow)
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import ercot_fetcher as fetcher
from . import preprocessing as pp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Project root detection
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def ensure_dirs():
    """Create data directories if they don't exist."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


def run_energy_prices_pipeline(start: str, end: str):
    """Fetch and process energy prices."""
    logger.info(f"{'='*60}")
    logger.info(f"Energy Prices Pipeline: {start} → {end}")
    logger.info(f"{'='*60}")

    # Fetch
    logger.info("Fetching RT SPP...")
    rt_spp = fetcher.fetch_rt_spp(start, end)
    logger.info(f"  Got {len(rt_spp)} rows")

    logger.info("Fetching DAM SPP...")
    dam_spp = fetcher.fetch_dam_spp(start, end)
    logger.info(f"  Got {len(dam_spp)} rows")

    # Process
    logger.info("Processing energy prices...")
    df = pp.process_energy_prices(rt_spp, dam_spp)

    # Validate
    checks = pp.validate_dataframe(df, "energy_prices")
    logger.info(f"  Validation: {checks['rows']} rows, {checks['coverage_pct']:.1f}% coverage")
    if checks["duplicated_timestamps"] > 0:
        logger.warning(f"  {checks['duplicated_timestamps']} duplicated timestamps")

    # Write
    pp.write_parquet(df, DATA_PROCESSED, "energy_prices")
    logger.info("Energy prices pipeline complete.")
    return df


def run_as_prices_pipeline(start: str, end: str):
    """Fetch and process ancillary service prices."""
    logger.info(f"{'='*60}")
    logger.info(f"AS Prices Pipeline: {start} → {end}")
    logger.info(f"{'='*60}")

    # DAM AS — available for full history
    logger.info("Fetching DAM AS prices...")
    dam_as = fetcher.fetch_dam_as_prices(start, end)
    logger.info(f"  Got {len(dam_as)} rows")

    # RT AS MCPCs — only post-RTC+B
    rt_as = pd.DataFrame()
    if pd.Timestamp(end) > pd.Timestamp("2025-12-05"):
        rt_start = max(start, "2025-12-05")
        logger.info(f"Fetching RT AS MCPCs ({rt_start} → {end})...")
        try:
            rt_as = fetcher.fetch_rt_as_prices(rt_start, end)
            logger.info(f"  Got {len(rt_as)} rows")
        except Exception as e:
            logger.warning(f"  RT AS fetch failed (will use NaN): {e}")

    # Process
    logger.info("Processing AS prices...")
    df = pp.process_as_prices(dam_as, rt_as if not rt_as.empty else None)

    # Validate
    checks = pp.validate_dataframe(df, "as_prices")
    logger.info(f"  Validation: {checks['rows']} rows")

    # Write
    pp.write_parquet(df, DATA_PROCESSED, "as_prices")
    logger.info("AS prices pipeline complete.")
    return df


def run_system_conditions_pipeline(start: str, end: str):
    """Fetch and process system conditions (load, renewables, fuel mix)."""
    logger.info(f"{'='*60}")
    logger.info(f"System Conditions Pipeline: {start} → {end}")
    logger.info(f"{'='*60}")

    # Load
    logger.info("Fetching load data...")
    load_df = fetcher.fetch_load(start, end)
    logger.info(f"  Got {len(load_df)} rows")

    # Fuel mix (optional, for enrichment)
    fuel_mix_df = None
    try:
        logger.info("Fetching fuel mix...")
        fuel_mix_df = fetcher.fetch_fuel_mix(start, end)
        logger.info(f"  Got {len(fuel_mix_df)} rows")
    except Exception as e:
        logger.warning(f"  Fuel mix fetch failed (non-critical): {e}")

    # Wind/Solar
    wind_df, solar_df = None, None
    try:
        logger.info("Fetching wind/solar...")
        ws = fetcher.fetch_wind_solar(start, end)
        wind_df = ws.get("wind")
        solar_df = ws.get("solar")
    except Exception as e:
        logger.warning(f"  Wind/solar fetch failed (non-critical): {e}")

    # Process
    logger.info("Processing system conditions...")
    df = pp.process_system_conditions(load_df, fuel_mix_df, wind_df, solar_df)

    # Validate
    checks = pp.validate_dataframe(df, "system_conditions")
    logger.info(f"  Validation: {checks['rows']} rows")

    # Write
    pp.write_parquet(df, DATA_PROCESSED, "system_conditions")
    logger.info("System conditions pipeline complete.")
    return df


def run_pipeline(
    start: str,
    end: str,
    tables: list[str] = None,
):
    """
    Run the full data pipeline or selected tables.

    Parameters
    ----------
    start, end : str
        Date range.
    tables : list[str], optional
        Specific tables to process. Default: all.
    """
    ensure_dirs()

    all_tables = ["energy_prices", "as_prices", "system_conditions"]
    if tables is None:
        tables = all_tables

    results = {}

    if "energy_prices" in tables:
        results["energy_prices"] = run_energy_prices_pipeline(start, end)

    if "as_prices" in tables:
        results["as_prices"] = run_as_prices_pipeline(start, end)

    if "system_conditions" in tables:
        results["system_conditions"] = run_system_conditions_pipeline(start, end)

    logger.info(f"\n{'='*60}")
    logger.info("Pipeline Summary:")
    for name, df in results.items():
        logger.info(f"  {name}: {len(df)} rows, {df.index.min()} → {df.index.max()}")
    logger.info(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(description="HybridBid Data Pipeline")
    parser.add_argument("--start", type=str, default="2024-01-01",
                        help="Start date (default: 2024-01-01)")
    parser.add_argument("--end", type=str, default="2026-02-01",
                        help="End date (default: 2026-02-01)")
    parser.add_argument("--tables", nargs="+", default=None,
                        help="Specific tables (energy_prices, as_prices, system_conditions)")
    parser.add_argument("--backfill", action="store_true",
                        help="Full backfill from 2020 (slow)")

    args = parser.parse_args()

    if args.backfill:
        logger.info("Starting full backfill 2020-2026...")
        # Process in yearly chunks to manage memory
        for year in range(2020, 2027):
            year_start = f"{year}-01-01"
            year_end = f"{year+1}-01-01" if year < 2026 else args.end
            logger.info(f"\n>>> Processing {year}...")
            try:
                run_pipeline(year_start, year_end, args.tables)
            except Exception as e:
                logger.error(f"Failed for {year}: {e}")
                continue
    else:
        run_pipeline(args.start, args.end, args.tables)


if __name__ == "__main__":
    main()
