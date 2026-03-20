"""
HybridBid Data Pipeline — Main orchestrator.

Downloads ERCOT data, processes it to canonical schema, and writes Parquet files.

Usage:
    python -m src.data.pipeline --start 2026-01-06 --end 2026-01-13
    python -m src.data.pipeline --start 2024-01-01 --end 2026-02-01 --tables energy_prices
    python -m src.data.pipeline --backfill
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from . import ercot_fetcher as fetcher
from . import preprocessing as pp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def run_energy_prices(start: str, end: str) -> pd.DataFrame:
    """Fetch and process energy prices."""
    logger.info(f"{'='*60}")
    logger.info(f"Energy Prices: {start} → {end}")
    logger.info(f"{'='*60}")

    rt_lmp = fetcher.fetch_rt_lmp(start, end)
    dam_spp = fetcher.fetch_dam_spp(start, end)

    df = pp.process_energy_prices(rt_lmp, dam_spp, start, end)

    checks = pp.validate_dataframe(df, "energy_prices")
    logger.info(f"  {checks['rows']} rows, {checks['coverage_pct']:.1f}% coverage")
    _log_nulls(checks)

    pp.write_parquet(df, DATA_PROCESSED, "energy_prices")
    return df


def run_as_prices(start: str, end: str) -> pd.DataFrame:
    """Fetch and process ancillary service prices."""
    logger.info(f"{'='*60}")
    logger.info(f"AS Prices: {start} → {end}")
    logger.info(f"{'='*60}")

    dam_as = fetcher.fetch_dam_as(start, end)
    rt_mcpc = fetcher.load_rt_mcpc(start, end)

    df = pp.process_as_prices(dam_as, rt_mcpc, start, end)

    checks = pp.validate_dataframe(df, "as_prices")
    logger.info(f"  {checks['rows']} rows, {checks['coverage_pct']:.1f}% coverage")
    _log_nulls(checks)

    pp.write_parquet(df, DATA_PROCESSED, "as_prices")
    return df


def run_system_conditions(start: str, end: str) -> pd.DataFrame:
    """Fetch and process system conditions."""
    logger.info(f"{'='*60}")
    logger.info(f"System Conditions: {start} → {end}")
    logger.info(f"{'='*60}")

    load_actual = fetcher.fetch_load_actual(start, end)
    load_forecast = fetcher.fetch_load_forecast(start, end)
    wind = fetcher.fetch_wind(start, end)
    solar = fetcher.fetch_solar(start, end)

    df = pp.process_system_conditions(
        load_actual, load_forecast, wind, solar, start, end
    )

    checks = pp.validate_dataframe(df, "system_conditions")
    logger.info(f"  {checks['rows']} rows, {checks['coverage_pct']:.1f}% coverage")
    _log_nulls(checks)

    pp.write_parquet(df, DATA_PROCESSED, "system_conditions")
    return df


def _log_nulls(checks: dict):
    """Log null counts from validation checks."""
    nulls = checks.get("null_counts", {})
    for col, count in nulls.items():
        if count > 0:
            total = checks["rows"]
            pct = count / total * 100 if total > 0 else 0
            logger.info(f"    {col}: {count} nulls ({pct:.1f}%)")


def run_pipeline(start: str, end: str, tables: list[str] = None):
    """Run the full pipeline or selected tables."""
    all_tables = ["energy_prices", "as_prices", "system_conditions"]
    if tables is None:
        tables = all_tables

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    results = {}
    if "energy_prices" in tables:
        results["energy_prices"] = run_energy_prices(start, end)
    if "as_prices" in tables:
        results["as_prices"] = run_as_prices(start, end)
    if "system_conditions" in tables:
        results["system_conditions"] = run_system_conditions(start, end)

    logger.info(f"\n{'='*60}")
    logger.info("Pipeline Summary:")
    for name, df in results.items():
        logger.info(f"  {name}: {len(df)} rows, {df.index.min()} → {df.index.max()}")
    logger.info(f"{'='*60}")
    return results


def main():
    parser = argparse.ArgumentParser(description="HybridBid Data Pipeline")
    parser.add_argument("--start", type=str, default="2026-01-06")
    parser.add_argument("--end", type=str, default="2026-01-13")
    parser.add_argument("--tables", nargs="+", default=None)
    parser.add_argument("--backfill", action="store_true",
                        help="Full backfill from 2020")

    args = parser.parse_args()

    if args.backfill:
        for year in range(2020, 2027):
            year_start = f"{year}-01-01"
            year_end = f"{year + 1}-01-01" if year < 2026 else args.end
            logger.info(f"\n>>> Processing {year}...")
            try:
                run_pipeline(year_start, year_end, args.tables)
            except Exception as e:
                logger.error(f"Failed for {year}: {e}")
    else:
        run_pipeline(args.start, args.end, args.tables)


if __name__ == "__main__":
    main()
