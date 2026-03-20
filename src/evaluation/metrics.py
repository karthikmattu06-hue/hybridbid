"""
Evaluation metrics for HybridBid.

Core metrics:
  - Revenue ($/kW-month) — comparable to Tyba Energy benchmarks
  - TB2-equivalent capture rate — % of perfect foresight achieved
  - Constraint compliance — SoC violations, ramp rate violations
  - Pre vs. post RTC+B performance delta
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from ..utils.battery_sim import BatteryParams

logger = logging.getLogger(__name__)


def compute_revenue_metrics(
    history: pd.DataFrame,
    params: BatteryParams,
) -> dict:
    """
    Compute revenue metrics from simulation history.

    Parameters
    ----------
    history : pd.DataFrame
        Output from BatterySimulator.get_history_df().
    params : BatteryParams
        Battery configuration (for $/kW-month normalization).

    Returns
    -------
    dict with revenue metrics.
    """
    if history.empty:
        return {"total_revenue_usd": 0, "error": "empty history"}

    total_energy_rev = history["energy_revenue_usd"].sum()
    total_as_rev = history["as_revenue_usd"].sum()
    total_deg_cost = history["degradation_cost_usd"].sum()
    total_net_rev = history["net_revenue_usd"].sum()

    # Duration in months (for $/kW-month normalization)
    if isinstance(history.index, pd.DatetimeIndex):
        duration_days = (history.index.max() - history.index.min()).days
    else:
        duration_days = len(history) * 5 / (60 * 24)  # Estimate from interval count
    duration_months = max(duration_days / 30.44, 0.001)  # Avoid division by zero

    # $/kW-month (industry standard metric)
    rev_per_kw_month = (total_net_rev / params.power_max_mw) / (1000 * duration_months)
    # Note: power_max_mw * 1000 = kW

    # Actually, $/kW-month = total_revenue / (capacity_kw * months)
    capacity_kw = params.power_max_mw * 1000
    rev_per_kw_month = total_net_rev / (capacity_kw * duration_months) if capacity_kw > 0 else 0

    # Daily average revenue
    n_days = max(duration_days, 1)
    daily_avg_rev = total_net_rev / n_days

    return {
        "total_revenue_usd": total_net_rev,
        "energy_revenue_usd": total_energy_rev,
        "as_revenue_usd": total_as_rev,
        "degradation_cost_usd": total_deg_cost,
        "revenue_per_kw_month": rev_per_kw_month,
        "daily_avg_revenue_usd": daily_avg_rev,
        "duration_days": duration_days,
        "duration_months": duration_months,
        "n_intervals": len(history),
    }


def compute_constraint_compliance(history: pd.DataFrame) -> dict:
    """
    Check constraint compliance from simulation history.

    Returns
    -------
    dict with compliance metrics. Target: zero violations.
    """
    if history.empty:
        return {"n_violations": 0}

    total_violations = history["n_violations"].sum()
    violation_rate = total_violations / len(history) if len(history) > 0 else 0

    # Count by type
    violation_types = {}
    if "violations" in history.columns:
        all_violations = history["violations"].str.split("|").explode()
        all_violations = all_violations[all_violations != ""]
        if not all_violations.empty:
            for v in all_violations:
                vtype = v.split(":")[0] if ":" in v else v
                violation_types[vtype] = violation_types.get(vtype, 0) + 1

    return {
        "n_violations": int(total_violations),
        "violation_rate": violation_rate,
        "violation_types": violation_types,
        "is_compliant": total_violations == 0,
    }


def compute_soc_statistics(history: pd.DataFrame, params: BatteryParams) -> dict:
    """Compute SoC utilization statistics."""
    if history.empty:
        return {}

    soc = history["soc_after_mwh"]
    return {
        "soc_mean_mwh": soc.mean(),
        "soc_std_mwh": soc.std(),
        "soc_min_mwh": soc.min(),
        "soc_max_mwh": soc.max(),
        "soc_utilization": (soc.max() - soc.min()) / params.usable_energy_mwh,
        "n_full_cycles": history["p_discharge_mw"].sum() * (5 / 60) / params.usable_energy_mwh,
    }


def compute_all_metrics(
    history: pd.DataFrame,
    params: BatteryParams,
) -> dict:
    """
    Compute all evaluation metrics for a simulation run.

    Parameters
    ----------
    history : pd.DataFrame
        Output from BatterySimulator.get_history_df().
    params : BatteryParams
        Battery configuration.

    Returns
    -------
    dict with all metrics flattened.
    """
    revenue = compute_revenue_metrics(history, params)
    compliance = compute_constraint_compliance(history)
    soc_stats = compute_soc_statistics(history, params)

    # Flatten into a single dict
    metrics = {}
    metrics.update(revenue)
    metrics.update({f"compliance_{k}": v for k, v in compliance.items()
                    if k != "violation_types"})
    metrics.update(soc_stats)

    return metrics


def print_metrics_comparison(metrics: dict):
    """Pretty-print metrics comparison table."""
    print("\n" + "=" * 80)
    print("BASELINE COMPARISON")
    print("=" * 80)

    # Key metrics to display
    key_metrics = [
        ("total_revenue_usd", "Total Revenue ($)", "${:,.2f}"),
        ("revenue_per_kw_month", "Revenue ($/kW-month)", "${:,.2f}"),
        ("daily_avg_revenue_usd", "Daily Avg Revenue ($)", "${:,.2f}"),
        ("capture_rate_pct", "Capture Rate (%)", "{:.1f}%"),
        ("compliance_n_violations", "Violations", "{}"),
        ("n_full_cycles", "Full Cycles", "{:.1f}"),
        ("duration_days", "Duration (days)", "{:.0f}"),
    ]

    # Header
    strategies = list(metrics.keys())
    header = f"{'Metric':<30}" + "".join(f"{s:>20}" for s in strategies)
    print(header)
    print("-" * len(header))

    for key, label, fmt in key_metrics:
        row = f"{label:<30}"
        for strategy in strategies:
            val = metrics[strategy].get(key, "—")
            if val == "—" or val is None:
                row += f"{'—':>20}"
            else:
                try:
                    row += f"{fmt.format(val):>20}"
                except (ValueError, TypeError):
                    row += f"{str(val):>20}"
        print(row)

    print("=" * 80)

    # Highlight capture rate
    if "tbx_full" in metrics and "capture_rate_pct" in metrics.get("tbx_full", {}):
        rate = metrics["tbx_full"]["capture_rate_pct"]
        print(f"\n  TBx captures {rate:.1f}% of perfect foresight (target: 40-50%)")

    if "tbx_pre" in metrics and "tbx_post" in metrics:
        pre_rev = metrics["tbx_pre"].get("daily_avg_revenue_usd", 0)
        post_rev = metrics["tbx_post"].get("daily_avg_revenue_usd", 0)
        if pre_rev > 0:
            delta = ((post_rev - pre_rev) / pre_rev) * 100
            print(f"  TBx pre→post RTC+B daily revenue change: {delta:+.1f}%")

    print()
