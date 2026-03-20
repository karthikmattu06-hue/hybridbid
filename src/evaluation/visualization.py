"""
Visualization utilities for HybridBid evaluation.

Generates publication-quality plots for:
  - Daily revenue time series (TBx vs Perfect Foresight)
  - Cumulative revenue curves
  - SoC trajectory plots
  - Pre vs. post RTC+B comparison
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CI
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Style
plt.rcParams.update({
    "figure.figsize": (12, 6),
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

COLORS = {
    "tbx": "#4A90D9",          # Blue
    "perfect_foresight": "#2ECC71",  # Green
    "hybridbid": "#E74C3C",    # Red (future)
    "pre_rtcb": "#95A5A6",     # Gray
    "post_rtcb": "#F39C12",    # Orange
}


def plot_daily_revenue(
    results: dict[str, pd.DataFrame],
    output_dir: Path,
    filename: str = "daily_revenue.png",
):
    """
    Plot daily revenue time series for each strategy.

    Parameters
    ----------
    results : dict
        Maps strategy name → simulation history DataFrame.
    output_dir : Path
        Directory to save the plot.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    for name, history in results.items():
        if history.empty:
            continue

        # Aggregate to daily revenue
        daily = history["net_revenue_usd"].resample("1D").sum()
        daily = daily[daily != 0]  # Remove days with no activity

        color = COLORS.get("tbx" if "tbx" in name else "perfect_foresight", "#333")
        linestyle = "--" if "pre" in name else "-"
        label = name.replace("_", " ").title()

        ax.plot(daily.index, daily.values, color=color, linestyle=linestyle,
                alpha=0.8, label=label, linewidth=1.5)

    # Mark RTC+B transition
    rtcb_date = pd.Timestamp("2025-12-05", tz="UTC")
    ax.axvline(rtcb_date, color="red", linestyle=":", alpha=0.7, label="RTC+B Go-Live")

    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Net Revenue ($)")
    ax.set_title("Daily Revenue: TBx vs Perfect Foresight")
    ax.legend(loc="upper left", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()

    path = output_dir / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")


def plot_cumulative_revenue(
    results: dict[str, pd.DataFrame],
    output_dir: Path,
    filename: str = "cumulative_revenue.png",
):
    """Plot cumulative revenue over time."""
    fig, ax = plt.subplots(figsize=(14, 6))

    for name, history in results.items():
        if history.empty or "full" not in name:
            continue

        cumrev = history["net_revenue_usd"].cumsum()

        color = COLORS.get("tbx" if "tbx" in name else "perfect_foresight", "#333")
        label = name.replace("_", " ").title()

        ax.plot(cumrev.index, cumrev.values, color=color,
                label=label, linewidth=2)

    rtcb_date = pd.Timestamp("2025-12-05", tz="UTC")
    ax.axvline(rtcb_date, color="red", linestyle=":", alpha=0.7, label="RTC+B Go-Live")

    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Revenue ($)")
    ax.set_title("Cumulative Revenue Comparison")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()

    path = output_dir / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")


def plot_soc_trajectory(
    history: pd.DataFrame,
    title: str = "SoC Trajectory",
    output_dir: Path = None,
    filename: str = "soc_trajectory.png",
    days: int = 3,
):
    """
    Plot SoC trajectory for a few days to visualize dispatch patterns.

    Parameters
    ----------
    history : pd.DataFrame
        Simulation history.
    days : int
        Number of days to show (first N days).
    """
    if history.empty:
        return

    # Select first N days
    if isinstance(history.index, pd.DatetimeIndex):
        start = history.index.min()
        end = start + pd.Timedelta(days=days)
        subset = history[start:end]
    else:
        subset = history.head(days * 288)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # SoC
    ax1.plot(subset.index, subset["soc_after_mwh"], color="#2ECC71", linewidth=1.5)
    ax1.set_ylabel("SoC (MWh)")
    ax1.set_title(title)
    ax1.axhline(y=subset["soc_after_mwh"].max(), color="gray", linestyle="--", alpha=0.5)
    ax1.axhline(y=subset["soc_after_mwh"].min(), color="gray", linestyle="--", alpha=0.5)

    # Power
    ax2.fill_between(subset.index, 0, subset["p_discharge_mw"],
                      alpha=0.6, color="#E74C3C", label="Discharge")
    ax2.fill_between(subset.index, 0, -subset["p_charge_mw"],
                      alpha=0.6, color="#4A90D9", label="Charge")
    ax2.set_ylabel("Power (MW)")
    ax2.set_xlabel("Time")
    ax2.legend()

    fig.tight_layout()

    if output_dir:
        path = output_dir / filename
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {path}")
    else:
        plt.close(fig)


def plot_capture_rate_bar(
    metrics: dict,
    output_dir: Path,
    filename: str = "capture_rates.png",
):
    """Bar chart of capture rates (% of perfect foresight)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    strategies = []
    rates = []

    for name, m in metrics.items():
        if "capture_rate_pct" in m:
            strategies.append(name.replace("_", " ").title())
            rates.append(m["capture_rate_pct"])

    if not strategies:
        plt.close(fig)
        return

    bars = ax.bar(strategies, rates, color=[COLORS["tbx"]] * len(strategies), alpha=0.8)

    # Reference lines
    ax.axhline(y=100, color="green", linestyle="--", alpha=0.5, label="Perfect Foresight")
    ax.axhline(y=75, color="red", linestyle="--", alpha=0.5, label="HybridBid Target (75%)")
    ax.axhline(y=56, color="orange", linestyle="--", alpha=0.5, label="Avg Operator (56%)")

    ax.set_ylabel("Capture Rate (% of Perfect Foresight)")
    ax.set_title("Baseline Capture Rates")
    ax.legend()
    ax.set_ylim(0, 110)

    # Add value labels on bars
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{rate:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    path = output_dir / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")


def plot_baseline_comparison(
    results: dict[str, pd.DataFrame],
    metrics: dict,
    output_dir: Path,
):
    """Generate all baseline comparison plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_daily_revenue(results, output_dir)
    plot_cumulative_revenue(results, output_dir)
    plot_capture_rate_bar(metrics, output_dir)

    # SoC trajectories for key strategies
    for name in ["tbx_full", "pf_full"]:
        if name in results and not results[name].empty:
            plot_soc_trajectory(
                results[name],
                title=f"SoC Trajectory: {name.replace('_', ' ').title()}",
                output_dir=output_dir,
                filename=f"soc_{name}.png",
            )
