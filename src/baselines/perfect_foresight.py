"""
Perfect Foresight Baseline — Energy-Only MIP.

Solves for the optimal battery schedule given perfect knowledge of
future energy prices. This is the theoretical upper bound on what any
energy-arbitrage-only strategy can achieve.

Week 1: Energy-only (no AS co-optimization).
Week 2+: Extend to joint energy+AS co-optimization (Tier 2A formulation).

Solver: HiGHS (default, open-source) or Gurobi (once academic license is set up).

Mathematical Formulation:
    maximize Σ_t [ λ(t) · (p_dch(t) - p_ch(t)) · Δt - C_deg · (p_ch(t) + p_dch(t)) · Δt ]

    subject to:
        SoC(t+1) = SoC(t) + η_ch · p_ch(t) · Δt - (p_dch(t) / η_dch) · Δt
        SoC_min ≤ SoC(t) ≤ SoC_max
        0 ≤ p_ch(t)  ≤ P_max · u(t)
        0 ≤ p_dch(t) ≤ P_max · (1 - u(t))
        u(t) ∈ {0, 1}  (mutual exclusivity)
"""

import logging
import time
from typing import Optional

import cvxpy as cp
import numpy as np
import pandas as pd

from ..utils.battery_sim import BatteryAction, BatteryParams, BatterySimulator

logger = logging.getLogger(__name__)

# Check available solvers
AVAILABLE_SOLVERS = cp.installed_solvers()
logger.info(f"Available CVXPY solvers: {AVAILABLE_SOLVERS}")


def select_solver() -> str:
    """Select the best available MIP solver."""
    if "GUROBI" in AVAILABLE_SOLVERS:
        return "GUROBI"
    elif "CPLEX" in AVAILABLE_SOLVERS:
        return "CPLEX"
    elif "SCIP" in AVAILABLE_SOLVERS:
        return "SCIP"
    elif "HIGHS" in AVAILABLE_SOLVERS:
        return "HIGHS"
    elif "GLPK_MI" in AVAILABLE_SOLVERS:
        return "GLPK_MI"
    else:
        raise RuntimeError(
            f"No MIP solver found. Available: {AVAILABLE_SOLVERS}. "
            "Install HiGHS: pip install highspy"
        )


def solve_energy_only_mip(
    prices: np.ndarray,
    params: BatteryParams,
    soc_initial: float = None,
    solver: str = None,
    verbose: bool = False,
) -> dict:
    """
    Solve energy-only perfect foresight MIP for a time horizon.

    Parameters
    ----------
    prices : np.ndarray
        Energy prices [$/MWh] for each 5-minute interval. Shape: (T,)
    params : BatteryParams
        Battery configuration.
    soc_initial : float, optional
        Initial SoC [MWh]. Defaults to params.soc_initial_mwh.
    solver : str, optional
        CVXPY solver name. Auto-selects if None.
    verbose : bool
        Print solver output.

    Returns
    -------
    dict with keys:
        'p_charge': np.ndarray of charge power [MW]
        'p_discharge': np.ndarray of discharge power [MW]
        'soc': np.ndarray of SoC [MWh] (T+1 values)
        'revenue': float total net revenue [$]
        'energy_revenue': float energy revenue [$]
        'degradation_cost': float total degradation cost [$]
        'solve_time': float seconds
        'status': str solver status
    """
    if solver is None:
        solver = select_solver()

    T = len(prices)
    dt = 5.0 / 60.0  # 5 minutes in hours

    if soc_initial is None:
        soc_initial = params.soc_initial_mwh

    # ── Decision Variables ──
    p_ch = cp.Variable(T, nonneg=True, name="p_charge")       # Charge power [MW]
    p_dch = cp.Variable(T, nonneg=True, name="p_discharge")   # Discharge power [MW]
    u = cp.Variable(T, boolean=True, name="mode")             # 1=charging, 0=discharging
    soc = cp.Variable(T + 1, name="soc")                      # State of charge [MWh]

    # ── Objective: Maximize net revenue ──
    # Energy revenue: (discharge - charge) * price * interval_duration
    energy_rev = cp.sum(cp.multiply(prices, (p_dch - p_ch)) * dt)

    # Degradation cost: throughput * cost_rate * interval_duration
    deg_cost = params.degradation_cost_per_mwh * cp.sum((p_ch + p_dch) * dt)

    objective = cp.Maximize(energy_rev - deg_cost)

    # ── Constraints ──
    constraints = []

    # Initial SoC
    constraints.append(soc[0] == soc_initial)

    # SoC dynamics: SoC(t+1) = SoC(t) + η_ch*p_ch*dt - (p_dch/η_dch)*dt
    for t in range(T):
        constraints.append(
            soc[t + 1] == soc[t]
            + params.eta_charge * p_ch[t] * dt
            - (p_dch[t] / params.eta_discharge) * dt
        )

    # SoC bounds
    constraints.append(soc >= params.soc_min_mwh)
    constraints.append(soc <= params.soc_max_mwh)

    # Power limits with mutual exclusivity (Big-M formulation)
    constraints.append(p_ch <= params.power_max_mw * u)
    constraints.append(p_dch <= params.power_max_mw * (1 - u))

    # ── Solve ──
    problem = cp.Problem(objective, constraints)

    t_start = time.time()
    try:
        problem.solve(solver=solver, verbose=verbose)
    except cp.error.SolverError as e:
        logger.error(f"Solver error: {e}")
        return {
            "status": "error",
            "solve_time": time.time() - t_start,
            "error": str(e),
        }
    solve_time = time.time() - t_start

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        logger.warning(f"MIP solve status: {problem.status}")
        return {
            "status": problem.status,
            "solve_time": solve_time,
        }

    # ── Extract results ──
    p_ch_val = np.array(p_ch.value).flatten()
    p_dch_val = np.array(p_dch.value).flatten()
    soc_val = np.array(soc.value).flatten()

    energy_rev_val = float(np.sum(prices * (p_dch_val - p_ch_val) * dt))
    deg_cost_val = float(params.degradation_cost_per_mwh * np.sum((p_ch_val + p_dch_val) * dt))

    return {
        "p_charge": p_ch_val,
        "p_discharge": p_dch_val,
        "soc": soc_val,
        "revenue": energy_rev_val - deg_cost_val,
        "energy_revenue": energy_rev_val,
        "degradation_cost": deg_cost_val,
        "solve_time": solve_time,
        "status": problem.status,
        "objective_value": float(problem.value),
    }


def run_perfect_foresight(
    prices: pd.Series,
    params: BatteryParams,
    horizon_hours: int = 24,
    step_hours: int = 24,
    solver: str = None,
) -> pd.DataFrame:
    """
    Run rolling-horizon perfect foresight optimization.

    Solves the MIP over rolling windows, using the optimal schedule
    for each window to compute realized revenue.

    Parameters
    ----------
    prices : pd.Series
        Energy prices [$/MWh] at 5-minute intervals.
    params : BatteryParams
        Battery configuration.
    horizon_hours : int
        MIP optimization horizon in hours (default: 24).
    step_hours : int
        Step size between windows in hours (default: 24 = daily).
    solver : str, optional
        CVXPY solver name.

    Returns
    -------
    pd.DataFrame
        Simulation history with optimal dispatch.
    """
    intervals_per_hour = 12  # 60/5
    horizon_intervals = horizon_hours * intervals_per_hour
    step_intervals = step_hours * intervals_per_hour

    sim = BatterySimulator(params)
    solve_times = []

    n_windows = max(1, (len(prices) - horizon_intervals) // step_intervals + 1)
    logger.info(
        f"Perfect foresight: {n_windows} windows, "
        f"{horizon_hours}h horizon, {step_hours}h step, "
        f"solver={solver or 'auto'}"
    )

    for window_idx in range(n_windows):
        start_idx = window_idx * step_intervals
        end_idx = min(start_idx + horizon_intervals, len(prices))

        if start_idx >= len(prices):
            break

        window_prices = prices.iloc[start_idx:end_idx]

        if len(window_prices) < intervals_per_hour:
            logger.warning(f"Window {window_idx}: too few intervals ({len(window_prices)}), skipping")
            continue

        # Solve MIP
        result = solve_energy_only_mip(
            prices=window_prices.values,
            params=params,
            soc_initial=sim.soc_mwh,
            solver=solver,
        )

        if result["status"] not in ["optimal", "optimal_inaccurate"]:
            logger.warning(f"Window {window_idx}: solver status={result['status']}")
            # Execute idle actions for this window
            for i in range(min(step_intervals, end_idx - start_idx)):
                ts = window_prices.index[i]
                sim.step(BatteryAction(), energy_price=window_prices.iloc[i], timestamp=ts)
            continue

        solve_times.append(result["solve_time"])

        # Execute the optimal schedule for step_intervals (not full horizon)
        execute_intervals = min(step_intervals, end_idx - start_idx)
        for i in range(execute_intervals):
            ts = window_prices.index[i]
            action = BatteryAction(
                p_charge_mw=float(result["p_charge"][i]),
                p_discharge_mw=float(result["p_discharge"][i]),
            )
            sim.step(action, energy_price=window_prices.iloc[i], timestamp=ts)

    if solve_times:
        logger.info(
            f"Solve times: mean={np.mean(solve_times):.2f}s, "
            f"max={np.max(solve_times):.2f}s"
        )

    return sim.get_history_df()


def run_perfect_foresight_daily(
    prices: pd.Series,
    params: BatteryParams,
    solver: str = None,
) -> pd.DataFrame:
    """
    Run perfect foresight with daily SoC reset (for clean comparisons with TBx).

    Each day is solved independently with fresh initial SoC.
    """
    results = []

    for date, day_prices in prices.groupby(prices.index.date):
        if len(day_prices) < 12:
            continue

        result = solve_energy_only_mip(
            prices=day_prices.values,
            params=params,
            solver=solver,
        )

        if result["status"] not in ["optimal", "optimal_inaccurate"]:
            logger.warning(f"{date}: solver status={result['status']}")
            continue

        # Simulate the optimal schedule
        sim = BatterySimulator(params)
        for i, (ts, price) in enumerate(day_prices.items()):
            action = BatteryAction(
                p_charge_mw=float(result["p_charge"][i]),
                p_discharge_mw=float(result["p_discharge"][i]),
            )
            sim.step(action, energy_price=price, timestamp=ts)

        day_history = sim.get_history_df()
        if not day_history.empty:
            results.append(day_history)

    if results:
        return pd.concat(results)
    return pd.DataFrame()
