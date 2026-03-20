"""
TBx — Time-Based Arbitrage Baseline.

The simplest reasonable strategy for battery storage:
  - Charge during the cheapest hours overnight
  - Discharge during the most expensive hours in the afternoon
  - No ancillary service participation
  - Full cycle each day

This establishes the performance floor (~40-50% of perfect foresight).
Any system we build must significantly beat this.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from ..utils.battery_sim import BatteryAction, BatteryParams, BatterySimulator

logger = logging.getLogger(__name__)


def identify_tbx_schedule(
    prices: pd.Series,
    n_charge_hours: int = 4,
    n_discharge_hours: int = 4,
) -> pd.DataFrame:
    """
    Identify charge/discharge windows for TBx strategy.

    For each day, finds the cheapest hours (charge) and most
    expensive hours (discharge).

    Parameters
    ----------
    prices : pd.Series
        Energy prices with DatetimeIndex (any frequency).
    n_charge_hours : int
        Number of hours to charge each day.
    n_discharge_hours : int
        Number of hours to discharge each day.

    Returns
    -------
    pd.DataFrame
        With columns: 'action' ('charge', 'discharge', 'idle'),
        'price', indexed by timestamp.
    """
    # Resample to hourly for schedule identification
    hourly_prices = prices.resample("1h").mean()

    schedules = []

    for date, day_prices in hourly_prices.groupby(hourly_prices.index.date):
        if len(day_prices) < n_charge_hours + n_discharge_hours:
            continue

        # Find cheapest hours → charge
        charge_hours = day_prices.nsmallest(n_charge_hours).index

        # Find most expensive hours (excluding charge hours) → discharge
        remaining = day_prices.drop(charge_hours)
        discharge_hours = remaining.nlargest(n_discharge_hours).index

        for ts in day_prices.index:
            if ts in charge_hours:
                action = "charge"
            elif ts in discharge_hours:
                action = "discharge"
            else:
                action = "idle"
            schedules.append({
                "timestamp": ts,
                "action": action,
                "price": day_prices[ts],
            })

    return pd.DataFrame(schedules).set_index("timestamp")


def run_tbx(
    prices: pd.Series,
    params: BatteryParams,
    n_charge_hours: int = 4,
    n_discharge_hours: int = 4,
) -> pd.DataFrame:
    """
    Run TBx strategy simulation.

    Parameters
    ----------
    prices : pd.Series
        Energy prices [$/MWh] at 5-minute intervals with DatetimeIndex.
    params : BatteryParams
        Battery configuration.
    n_charge_hours : int
        Hours to charge per day.
    n_discharge_hours : int
        Hours to discharge per day.

    Returns
    -------
    pd.DataFrame
        Simulation history from BatterySimulator.
    """
    # Get daily schedule
    schedule = identify_tbx_schedule(prices, n_charge_hours, n_discharge_hours)

    # Simulate at 5-minute resolution
    sim = BatterySimulator(params)

    for ts, price in prices.items():
        # Look up the hourly schedule for this 5-min interval
        hour_ts = ts.floor("h")
        if hour_ts in schedule.index:
            action_type = schedule.loc[hour_ts, "action"]
        else:
            action_type = "idle"

        # Create action based on schedule
        if action_type == "charge":
            # Check if we have room to charge
            headroom_mwh = params.soc_max_mwh - sim.soc_mwh
            max_charge_mw = min(
                params.power_max_mw,
                headroom_mwh / (params.eta_charge * sim.DELTA_T_HOURS),
            )
            action = BatteryAction(p_charge_mw=max(0, max_charge_mw))

        elif action_type == "discharge":
            # Check if we have energy to discharge
            available_mwh = sim.soc_mwh - params.soc_min_mwh
            max_discharge_mw = min(
                params.power_max_mw,
                available_mwh * params.eta_discharge / sim.DELTA_T_HOURS,
            )
            action = BatteryAction(p_discharge_mw=max(0, max_discharge_mw))

        else:
            action = BatteryAction()  # Idle

        sim.step(action, energy_price=price, timestamp=ts)

    return sim.get_history_df()


def run_tbx_daily(
    prices: pd.Series,
    params: BatteryParams,
    n_charge_hours: int = 4,
    n_discharge_hours: int = 4,
) -> pd.DataFrame:
    """
    Run TBx with daily SoC reset (cleaner for evaluation).

    Resets battery to initial SoC at the start of each operating day.
    This removes inter-day SoC coupling, making daily revenue
    comparisons cleaner for baselining.
    """
    results = []

    for date, day_prices in prices.groupby(prices.index.date):
        if len(day_prices) < 12:  # Skip incomplete days
            continue

        sim = BatterySimulator(params)  # Fresh sim each day
        schedule = identify_tbx_schedule(day_prices, n_charge_hours, n_discharge_hours)

        for ts, price in day_prices.items():
            hour_ts = ts.floor("h")
            if hour_ts in schedule.index:
                action_type = schedule.loc[hour_ts, "action"]
            else:
                action_type = "idle"

            if action_type == "charge":
                headroom_mwh = params.soc_max_mwh - sim.soc_mwh
                max_charge = min(
                    params.power_max_mw,
                    headroom_mwh / (params.eta_charge * sim.DELTA_T_HOURS),
                )
                action = BatteryAction(p_charge_mw=max(0, max_charge))
            elif action_type == "discharge":
                available_mwh = sim.soc_mwh - params.soc_min_mwh
                max_discharge = min(
                    params.power_max_mw,
                    available_mwh * params.eta_discharge / sim.DELTA_T_HOURS,
                )
                action = BatteryAction(p_discharge_mw=max(0, max_discharge))
            else:
                action = BatteryAction()

            sim.step(action, energy_price=price, timestamp=ts)

        day_history = sim.get_history_df()
        if not day_history.empty:
            results.append(day_history)

    if results:
        return pd.concat(results)
    return pd.DataFrame()
