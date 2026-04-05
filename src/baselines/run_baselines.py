"""
Baseline runner — runs TBx and perfect foresight across test periods.

Usage:
    python -m src.baselines.run_baselines
    python -m src.baselines.run_baselines --test-start 2025-10-01 --test-end 2026-02-01
    python -m src.baselines.run_baselines --energy-only  # Skip AS (Week 1 default)
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from ..data.preprocessing import read_parquet
from ..evaluation.metrics import compute_all_metrics, print_metrics_comparison
from ..evaluation.visualization import plot_baseline_comparison
from ..utils.battery_sim import BatteryParams
from .perfect_foresight import run_perfect_foresight_daily
from .tbx import run_tbx_daily

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
CONFIG_DIR = PROJECT_ROOT / "configs"
OUTPUT_DIR = PROJECT_ROOT / "data" / "results"


def load_battery_params() -> BatteryParams:
    """Load battery parameters from config."""
    config_path = CONFIG_DIR / "battery.yaml"
    return BatteryParams.from_yaml(str(config_path))


def load_prices(start: str, end: str) -> pd.Series:
    """Load energy prices from processed Parquet files."""
    df = read_parquet(DATA_DIR, "energy_prices", start=start, end=end)

    if df.empty:
        raise ValueError(
            f"No energy price data found for {start} to {end}. "
            "Run the data pipeline first: python -m src.data.pipeline"
        )

    # Use RT LMP as the primary price signal, fall back to DAM SPP
    price_col = None
    for col in ["rt_lmp", "dam_spp"]:
        if col in df.columns and df[col].notna().any():
            price_col = col
            break

    if price_col is None:
        raise ValueError(f"No price column found. Available: {df.columns.tolist()}")

    prices = df[price_col].dropna()
    logger.info(f"Loaded {len(prices)} price observations ({price_col})")
    return prices


def run_baselines(
    test_start: str = "2025-10-01",
    test_end: str = "2026-02-01",
    solver: str = None,
):
    """
    Run all baselines and compute comparison metrics.

    Parameters
    ----------
    test_start, test_end : str
        Test period date range.
    solver : str, optional
        MIP solver for perfect foresight.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load config and data
    params = load_battery_params()
    logger.info(f"Battery: {params.power_max_mw}MW / {params.energy_max_mwh}MWh")

    prices = load_prices(test_start, test_end)

    # Split into pre/post RTC+B
    rtcb_date = pd.Timestamp("2025-12-05", tz="UTC")
    prices_pre = prices[prices.index < rtcb_date]
    prices_post = prices[prices.index >= rtcb_date]

    results = {}

    # ── Run TBx ──
    logger.info("\n" + "=" * 60)
    logger.info("Running TBx baseline...")
    logger.info("=" * 60)

    if not prices_pre.empty:
        logger.info(f"  Pre-RTC+B: {len(prices_pre)} intervals")
        results["tbx_pre"] = run_tbx_daily(prices_pre, params)

    if not prices_post.empty:
        logger.info(f"  Post-RTC+B: {len(prices_post)} intervals")
        results["tbx_post"] = run_tbx_daily(prices_post, params)

    results["tbx_full"] = run_tbx_daily(prices, params)

    # ── Run Perfect Foresight ──
    logger.info("\n" + "=" * 60)
    logger.info("Running Perfect Foresight (energy-only)...")
    logger.info("=" * 60)

    if not prices_pre.empty:
        logger.info(f"  Pre-RTC+B: {len(prices_pre)} intervals")
        results["pf_pre"] = run_perfect_foresight_daily(prices_pre, params, solver=solver)

    if not prices_post.empty:
        logger.info(f"  Post-RTC+B: {len(prices_post)} intervals")
        results["pf_post"] = run_perfect_foresight_daily(prices_post, params, solver=solver)

    results["pf_full"] = run_perfect_foresight_daily(prices, params, solver=solver)

    # ── Compute Metrics ──
    logger.info("\n" + "=" * 60)
    logger.info("Computing metrics...")
    logger.info("=" * 60)

    metrics = {}
    for name, history_df in results.items():
        if not history_df.empty:
            m = compute_all_metrics(history_df, params)
            metrics[name] = m

    # Compute TB2-equivalent capture rates
    if "pf_full" in metrics and "tbx_full" in metrics:
        pf_rev = metrics["pf_full"]["total_revenue_usd"]
        tbx_rev = metrics["tbx_full"]["total_revenue_usd"]
        if pf_rev > 0:
            metrics["tbx_full"]["capture_rate_pct"] = (tbx_rev / pf_rev) * 100

    if "pf_pre" in metrics and "tbx_pre" in metrics:
        pf_rev = metrics["pf_pre"]["total_revenue_usd"]
        tbx_rev = metrics["tbx_pre"]["total_revenue_usd"]
        if pf_rev > 0:
            metrics["tbx_pre"]["capture_rate_pct"] = (tbx_rev / pf_rev) * 100

    if "pf_post" in metrics and "tbx_post" in metrics:
        pf_rev = metrics["pf_post"]["total_revenue_usd"]
        tbx_rev = metrics["tbx_post"]["total_revenue_usd"]
        if pf_rev > 0:
            metrics["tbx_post"]["capture_rate_pct"] = (tbx_rev / pf_rev) * 100

    # ── Print Results ──
    print_metrics_comparison(metrics)

    # ── Save Results ──
    for name, history_df in results.items():
        if not history_df.empty:
            path = OUTPUT_DIR / f"baseline_{name}.parquet"
            history_df.to_parquet(path)
            logger.info(f"Saved {name} → {path}")

    # Save metrics summary
    metrics_df = pd.DataFrame(metrics).T
    metrics_path = OUTPUT_DIR / "baseline_metrics.csv"
    metrics_df.to_csv(metrics_path)
    logger.info(f"Saved metrics → {metrics_path}")

    # ── Plot ──
    try:
        plot_baseline_comparison(results, metrics, OUTPUT_DIR)
        logger.info(f"Saved plots → {OUTPUT_DIR}")
    except Exception as e:
        logger.warning(f"Plotting failed (non-critical): {e}")

    return results, metrics


def main():
    parser = argparse.ArgumentParser(description="Run HybridBid baselines")
    parser.add_argument("--test-start", default="2025-10-01")
    parser.add_argument("--test-end", default="2026-02-01")
    parser.add_argument("--solver", default=None,
                        help="MIP solver (GUROBI, HIGHS, etc.)")
    args = parser.parse_args()

    run_baselines(args.test_start, args.test_end, args.solver)


if __name__ == "__main__":
    main()
