"""
Receding-horizon MILP expert for BC/IQL training data generation.

At each decision point (every `commit_hours` hours):
  1. Load realized prices for the next `horizon_hours` hours (oracle window)
  2. Solve MILP for the full horizon with initial SoC = current SoC
  3. Commit the first `commit_hours` of actions; discard the rest
  4. Advance time, update SoC, re-solve

This is a weaker oracle than full perfect-foresight MIP: the expert sees
`horizon_hours` ahead clearly but cannot plan beyond that window. This gives
realistic expert trajectories without future-leakage concerns for the BC step.

Battery config matches CLAUDE.md / ercot_env.py exactly:
  - p_max=10 MW, e_max=20 MWh, eta_ch=eta_dch=0.95
  - soc_min=2 MWh (10%), soc_max=18 MWh (90%), soc_init=10 MWh (50%)
  - NO degradation cost (not in env step reward per Li et al. Eq. 30)

Reward saved = Li et al. Eq. 26 (energy term + EMA timing bonus), matching the
Stage 1 environment exactly. Both raw energy revenue and Eq.26 reward are saved.

Usage:
    python -m src.data.receding_horizon_milp --split train
    python -m src.data.receding_horizon_milp --split val
    python -m src.data.receding_horizon_milp --split train --pf  # clairvoyant comparison
"""

import argparse
import glob
import logging
import time
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd

from src.baselines.perfect_foresight import select_solver, solve_energy_only_mip
from src.utils.battery_sim import BatteryParams

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Battery config: must match ercot_env.py DEFAULT_BATTERY ──
BATTERY_PARAMS = BatteryParams(
    power_max_mw=10.0,
    energy_max_mwh=20.0,
    soc_min_frac=0.10,
    soc_max_frac=0.90,
    soc_initial_frac=0.50,
    eta_charge=0.95,
    eta_discharge=0.95,
    degradation_cost_per_mwh=0.0,   # not in env step reward
    ramp_rate_mw_per_min=10.0,
)

# ── Env reward constants (Li et al. Eq. 26) ──
EMA_TAU = 0.9
BETA_ARB = 10.0
DT = 5.0 / 60.0  # hours per 5-min interval
INTERVALS_PER_HOUR = 12

# ── Temporal splits ──
SPLITS = {
    "train": ("2020-01-01", "2023-12-31"),
    "val":   ("2024-01-01", "2025-09-30"),
}


def load_prices(start_date: str, end_date: str) -> pd.Series:
    """Load rt_lmp from monthly parquets, returning a timezone-aware Series."""
    files = sorted(glob.glob("data/processed/energy_prices/*.parquet"))
    if not files:
        raise FileNotFoundError("No energy price parquets found in data/processed/energy_prices/")

    chunks = []
    for f in files:
        df = pd.read_parquet(f, columns=["rt_lmp"])
        chunks.append(df)

    prices = pd.concat(chunks).sort_index()["rt_lmp"]

    # Slice to requested range (index is UTC-aware)
    start = pd.Timestamp(start_date, tz="UTC")
    end = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    prices = prices.loc[start:end]

    # Drop NaN (sparse at very start of dataset)
    n_before = len(prices)
    prices = prices.dropna()
    if len(prices) < n_before:
        logger.warning(f"Dropped {n_before - len(prices)} NaN price rows")

    logger.info(f"Loaded {len(prices):,} price intervals from {prices.index[0]} to {prices.index[-1]}")
    return prices


def compute_env_reward(
    energy_mag: float,
    mode: int,
    rt_lmp: float,
    ema_price: float,
    eta: float = 0.95,
    beta: float = BETA_ARB,
    dt: float = DT,
) -> float:
    """
    Li et al. Eq. 26 spot reward for Stage 1, energy-only.

    energy_mag: [0,1] normalized bid (p/p_max)
    mode: 0=charge, 1=discharge, 2=idle
    """
    v_dch = 1.0 if mode == 1 else 0.0
    v_ch  = 1.0 if mode == 0 else 0.0

    price_dev = abs(rt_lmp - ema_price)
    I_dch = np.sign(rt_lmp - ema_price)
    I_ch  = np.sign(ema_price - rt_lmp)

    energy_term = energy_mag * rt_lmp * (v_dch * eta - v_ch / eta) * dt
    timing_bonus = beta * energy_mag * price_dev * (I_dch * v_dch * eta + I_ch * v_ch / eta) * dt
    return float(energy_term + timing_bonus)


def milp_to_action(p_ch_i: float, p_dc_i: float, p_max: float, eps: float = 1e-3):
    """Convert MILP power values to (mode, magnitude) in env action space."""
    if p_dc_i > eps:
        return 1, float(p_dc_i / p_max)   # discharge
    elif p_ch_i > eps:
        return 0, float(p_ch_i / p_max)   # charge
    else:
        return 2, 0.0                      # idle


def generate_trajectory(
    prices: pd.Series,
    params: BatteryParams = BATTERY_PARAMS,
    horizon_hours: int = 24,
    commit_hours: int = 1,
    solver: str = None,
    split_name: str = "unknown",
    time_limit: float = 5.0,
) -> dict:
    """
    Roll out the receding-horizon MILP expert over a price series.

    Returns a dict ready to be saved with np.savez_compressed.
    """
    if solver is None:
        solver = select_solver()
    logger.info(f"Using solver: {solver}")

    horizon_steps = horizon_hours * INTERVALS_PER_HOUR   # 288 for 24h
    commit_steps  = commit_hours  * INTERVALS_PER_HOUR   # 12 for 1h

    price_arr = prices.values
    timestamps = prices.index
    n_total = len(price_arr)

    # Pre-allocate outputs
    modes      = np.empty(n_total, dtype=np.int8)
    magnitudes = np.empty(n_total, dtype=np.float32)
    socs       = np.empty(n_total, dtype=np.float32)
    rewards_env = np.empty(n_total, dtype=np.float32)
    rewards_raw = np.empty(n_total, dtype=np.float32)  # plain energy revenue

    soc_min = params.soc_min_mwh
    soc_max = params.soc_max_mwh
    p_max   = params.power_max_mw
    eta     = params.eta_charge    # symmetric by assumption

    current_soc = params.soc_initial_mwh
    ema_price   = price_arr[0] if len(price_arr) > 0 else 50.0

    solve_times = []
    n_failed = 0
    t = 0
    n_windows = 0
    t_start = time.time()

    while t < n_total:
        # How many steps until end of series
        remaining = n_total - t
        window_len = min(horizon_steps, remaining)

        if window_len < INTERVALS_PER_HOUR:
            # Not enough data for even 1 hour — pad rest as idle
            for i in range(remaining):
                modes[t + i] = 2
                magnitudes[t + i] = 0.0
                socs[t + i] = current_soc
                rt = float(price_arr[t + i])
                ema_price = EMA_TAU * ema_price + (1.0 - EMA_TAU) * rt
                rewards_env[t + i] = 0.0
                rewards_raw[t + i] = 0.0
            t += remaining
            break

        window_prices = price_arr[t:t + window_len]
        # ECRS pre-June 2023 + sparse early-dataset gaps can leak NaN into the
        # MIP; replace with 0.0 (idle-equivalent LMP) before solving.
        window_prices = np.nan_to_num(window_prices, nan=0.0)

        # Solve MILP for window
        t_solve = time.time()
        result = solve_energy_only_mip(
            prices=window_prices,
            params=params,
            soc_initial=current_soc,
            solver=solver,
            verbose=False,
            time_limit=time_limit,
        )
        solve_times.append(time.time() - t_solve)

        if result["status"] not in ["optimal", "optimal_inaccurate", "user_limit", "user_limit_inaccurate"]:
            n_failed += 1
            logger.warning(
                f"  window t={t} status={result['status']} — "
                f"no feasible solution within time_limit={time_limit}s, "
                f"idling {min(commit_steps, remaining)} steps"
            )
            # Fall back to idle for this commit window
            commit_this = min(commit_steps, remaining)
            for i in range(commit_this):
                modes[t + i] = 2
                magnitudes[t + i] = 0.0
                socs[t + i] = current_soc
                rt = float(price_arr[t + i])
                ema_price = EMA_TAU * ema_price + (1.0 - EMA_TAU) * rt
                rewards_env[t + i] = 0.0
                rewards_raw[t + i] = 0.0
            t += commit_this
            continue

        # If the result is a time-limit solve, the arrays are still present but
        # may be missing (None). Guard against that before committing actions.
        if result.get("p_charge") is None or result.get("p_discharge") is None:
            n_failed += 1
            logger.warning(
                f"  window t={t} status={result['status']} — "
                f"solver returned no solution arrays, idling"
            )
            commit_this = min(commit_steps, remaining)
            for i in range(commit_this):
                modes[t + i] = 2
                magnitudes[t + i] = 0.0
                socs[t + i] = current_soc
                rt = float(price_arr[t + i])
                ema_price = EMA_TAU * ema_price + (1.0 - EMA_TAU) * rt
                rewards_env[t + i] = 0.0
                rewards_raw[t + i] = 0.0
            t += commit_this
            continue

        p_ch_sol  = result["p_charge"]
        p_dch_sol = result["p_discharge"]

        # Commit first commit_steps actions
        commit_this = min(commit_steps, remaining)
        for i in range(commit_this):
            mode, mag = milp_to_action(p_ch_sol[i], p_dch_sol[i], p_max)

            modes[t + i] = mode
            magnitudes[t + i] = mag
            socs[t + i] = current_soc

            # Update SoC using env physics (matching ercot_env.py exactly)
            p_ch_i  = p_ch_sol[i]
            p_dch_i = p_dch_sol[i]
            if mode == 1:  # discharge
                current_soc -= (p_dch_i / eta) * DT
            elif mode == 0:  # charge
                current_soc += p_ch_i * eta * DT
            # idle: soc unchanged
            current_soc = np.clip(current_soc, soc_min, soc_max)

            # Reward
            rt_lmp = float(price_arr[t + i])
            ema_price = EMA_TAU * ema_price + (1.0 - EMA_TAU) * rt_lmp

            rewards_env[t + i] = compute_env_reward(mag, mode, rt_lmp, ema_price, eta)
            rewards_raw[t + i] = (p_dch_i - p_ch_i) * rt_lmp * DT

        t += commit_this
        n_windows += 1

        if n_windows % 100 == 0:
            elapsed_min = (time.time() - t_start) / 60.0
            elapsed_days = t * DT / 24.0
            total_rev = float(rewards_raw[:t].sum())
            throughput = n_windows / max(elapsed_min, 1e-6)
            logger.info(
                f"  windows={n_windows} t={t:,}/{n_total:,} ({100*t/n_total:.1f}%) "
                f"| elapsed {elapsed_min:.1f} min "
                f"| throughput {throughput:.1f} windows/min "
                f"| days={elapsed_days:.1f} rev=${total_rev:,.0f} "
                f"avg_solve={np.mean(solve_times[-100:]):.3f}s soc={current_soc:.2f} MWh"
            )

    # Trim if loop exited early
    valid = t
    modes      = modes[:valid]
    magnitudes = magnitudes[:valid]
    socs       = socs[:valid]
    rewards_env = rewards_env[:valid]
    rewards_raw = rewards_raw[:valid]

    # Compute summary stats
    n_days = valid * DT / 24.0
    total_raw  = float(rewards_raw.sum())
    total_env  = float(rewards_env.sum())
    mode_counts = np.bincount(modes.astype(int), minlength=3)
    mode_pcts  = mode_counts / valid * 100

    soc_min_obs = float(socs.min())
    soc_max_obs = float(socs.max())

    solve_arr = np.array(solve_times)

    logger.info(f"\n{'='*60}")
    logger.info(f"Trajectory summary ({split_name})")
    logger.info(f"  Steps:          {valid:,}  ({n_days:.1f} days)")
    logger.info(f"  Raw revenue:    ${total_raw:,.0f}  (${total_raw/n_days:.0f}/day)")
    logger.info(f"  Env reward sum: {total_env:.0f}")
    logger.info(f"  Mode dist:      charge={mode_pcts[0]:.1f}%  discharge={mode_pcts[1]:.1f}%  idle={mode_pcts[2]:.1f}%")
    logger.info(f"  SoC range:      [{soc_min_obs:.3f}, {soc_max_obs:.3f}] MWh  (limits [{soc_min:.1f}, {soc_max:.1f}])")
    logger.info(f"  Solver:         {valid//commit_steps:,} solves  failed={n_failed}")
    logger.info(f"  Solve time:     mean={solve_arr.mean():.3f}s  max={solve_arr.max():.3f}s  total={solve_arr.sum()/60:.1f}min")
    logger.info(f"{'='*60}\n")

    return {
        "timestamps":   np.array([str(ts) for ts in timestamps[:valid]]),
        "modes":        modes,
        "magnitudes":   magnitudes,
        "socs":         socs,
        "rewards_env":  rewards_env,
        "rewards_raw":  rewards_raw,
        "rt_lmp":       price_arr[:valid].astype(np.float32),
        "meta": {
            "split":          split_name,
            "horizon_hours":  horizon_hours,
            "commit_hours":   commit_hours,
            "solver":         solver,
            "n_steps":        valid,
            "n_days":         n_days,
            "total_raw_revenue": total_raw,
            "daily_avg_raw":  total_raw / n_days,
            "mode_pct_charge":    mode_pcts[0],
            "mode_pct_discharge": mode_pcts[1],
            "mode_pct_idle":      mode_pcts[2],
            "soc_min_observed":   soc_min_obs,
            "soc_max_observed":   soc_max_obs,
            "n_solver_failures":  n_failed,
            "mean_solve_s":   float(solve_arr.mean()),
            "max_solve_s":    float(solve_arr.max()),
        },
    }


def generate_clairvoyant_trajectory(
    prices: pd.Series,
    params: BatteryParams = BATTERY_PARAMS,
    solver: str = None,
    split_name: str = "unknown",
) -> dict:
    """
    Full perfect-foresight MIP: each day solved in one shot (horizon=24h, step=24h).
    Provides upper-bound revenue for comparison with receding-horizon.
    """
    logger.info("Generating clairvoyant (full PF-MIP) trajectory...")
    # Clairvoyant = receding horizon where commit == horizon (24h at once)
    return generate_trajectory(
        prices=prices,
        params=params,
        horizon_hours=24,
        commit_hours=24,
        solver=solver,
        split_name=f"{split_name}_clairvoyant",
    )


def save_trajectory(traj: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Pull meta out — np.savez doesn't handle dicts inside arrays
    meta = traj.pop("meta")
    np.savez_compressed(output_path, **traj)
    traj["meta"] = meta  # restore

    # Save meta as a small text summary alongside
    meta_path = output_path.with_suffix(".txt")
    with open(meta_path, "w") as f:
        f.write(f"Receding-horizon MILP trajectory\n")
        f.write(f"{'='*40}\n")
        for k, v in meta.items():
            f.write(f"{k}: {v}\n")

    logger.info(f"Saved trajectory → {output_path}  ({output_path.stat().st_size / 1e6:.1f} MB)")
    logger.info(f"Saved meta      → {meta_path}")


def sanity_check(traj: dict, prices: pd.Series) -> None:
    """Run post-generation sanity checks and print results."""
    logger.info("\nRunning sanity checks...")
    modes = traj["modes"]
    magnitudes = traj["magnitudes"]
    socs = traj["socs"]
    rt_lmp = traj["rt_lmp"]
    meta = traj["meta"]

    ok = True

    # 1. Mode distribution
    mode_pcts = [meta["mode_pct_charge"], meta["mode_pct_discharge"], meta["mode_pct_idle"]]
    for name, pct in zip(["charge", "discharge", "idle"], mode_pcts):
        flag = "OK" if 5 < pct < 70 else "WARN"
        logger.info(f"  [{flag}] Mode {name}: {pct:.1f}%")
        if flag == "WARN":
            ok = False

    # 2. SoC bounds
    soc_min = BATTERY_PARAMS.soc_min_mwh
    soc_max = BATTERY_PARAMS.soc_max_mwh
    soc_ok = socs.min() >= soc_min - 0.01 and socs.max() <= soc_max + 0.01
    flag = "OK" if soc_ok else "FAIL"
    logger.info(f"  [{flag}] SoC bounds: [{socs.min():.3f}, {socs.max():.3f}] vs [{soc_min}, {soc_max}]")
    if not soc_ok:
        ok = False

    # 3. Revenue
    daily_avg = meta["daily_avg_raw"]
    flag = "OK" if 200 < daily_avg < 5000 else "WARN"
    logger.info(f"  [{flag}] Daily avg revenue: ${daily_avg:.0f}/day")
    if flag == "WARN":
        ok = False

    # 4. Mutual exclusivity: check via modes (if mode=charge, p_dch must be 0 by construction)
    n_idle_nonzero = int(np.sum((modes == 2) & (magnitudes > 1e-3)))
    flag = "OK" if n_idle_nonzero == 0 else "WARN"
    logger.info(f"  [{flag}] Idle steps with nonzero magnitude: {n_idle_nonzero}")

    # 5. Feb 2021 Uri check (if in range)
    uri_start = pd.Timestamp("2021-02-15", tz="UTC")
    uri_end   = pd.Timestamp("2021-02-18", tz="UTC")
    uri_mask  = (prices.index[:len(modes)] >= uri_start) & (prices.index[:len(modes)] < uri_end)
    if uri_mask.sum() > 0:
        uri_modes = modes[uri_mask]
        uri_prices = rt_lmp[uri_mask]
        dch_pct = float((uri_modes == 1).mean() * 100)
        avg_price = float(uri_prices.mean())
        flag = "OK" if dch_pct > 40 else "WARN"
        logger.info(f"  [{flag}] Feb 15-17 2021 (Uri): {dch_pct:.0f}% discharge, avg price ${avg_price:.0f}/MWh")
        if flag == "WARN":
            ok = False

    logger.info(f"\n  Overall: {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED — review above'}")


def main():
    parser = argparse.ArgumentParser(description="Generate receding-horizon MILP expert trajectories")
    parser.add_argument("--split", choices=["train", "val", "both"], default="train",
                        help="Which temporal split to generate")
    parser.add_argument("--pf", action="store_true",
                        help="Also generate clairvoyant (full PF-MIP) trajectory for comparison")
    parser.add_argument("--horizon", type=int, default=24,
                        help="Look-ahead horizon in hours (default: 24)")
    parser.add_argument("--commit", type=int, default=1,
                        help="Commit window in hours (default: 1)")
    parser.add_argument("--solver", type=str, default=None,
                        help="CVXPY solver (default: auto-select)")
    parser.add_argument("--time-limit", type=float, default=5.0,
                        help="Per-window solver time limit in seconds (default: 5.0)")
    parser.add_argument("--out", type=str, default="data/expert_trajectories",
                        help="Output directory")
    args = parser.parse_args()

    splits_to_run = ["train", "val"] if args.split == "both" else [args.split]

    for split in splits_to_run:
        start_date, end_date = SPLITS[split]
        logger.info(f"\n{'='*60}")
        logger.info(f"Generating {split} trajectory ({start_date} → {end_date})")
        logger.info(f"Horizon: {args.horizon}h  Commit: {args.commit}h")
        logger.info(f"{'='*60}")

        prices = load_prices(start_date, end_date)

        # Receding-horizon trajectory
        traj = generate_trajectory(
            prices=prices,
            params=BATTERY_PARAMS,
            horizon_hours=args.horizon,
            commit_hours=args.commit,
            solver=args.solver,
            split_name=split,
            time_limit=args.time_limit,
        )

        out_path = Path(args.out) / f"receding_horizon_{split}.npz"
        save_trajectory(traj, out_path)
        sanity_check(traj, prices)

        # Optional clairvoyant (upper bound)
        if args.pf:
            traj_pf = generate_clairvoyant_trajectory(
                prices=prices,
                params=BATTERY_PARAMS,
                solver=args.solver,
                split_name=split,
            )
            pf_path = Path(args.out) / f"clairvoyant_{split}.npz"
            save_trajectory(traj_pf, pf_path)
            sanity_check(traj_pf, prices)

            # Revenue comparison
            rh_daily = traj["meta"]["daily_avg_raw"]
            pf_daily = traj_pf["meta"]["daily_avg_raw"]
            ratio = rh_daily / pf_daily if pf_daily > 0 else float("nan")
            logger.info(f"\nRevenue comparison ({split}):")
            logger.info(f"  Receding-horizon: ${rh_daily:.0f}/day")
            logger.info(f"  Clairvoyant PF:   ${pf_daily:.0f}/day")
            logger.info(f"  Ratio (RH/PF):    {ratio:.3f}  (target: 0.85–0.95)")


if __name__ == "__main__":
    main()
