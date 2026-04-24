"""Data characterization for offline RL method selection.

Produces structured statistics for experiments/data_characterization.md.
One-shot script: loads everything, prints sections, done.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
TRAJ = ROOT / "data" / "expert_trajectories"
OUT_JSON = ROOT / "experiments" / "_characterize_output.json"

# 7-atom action grid in p.u. of P_max. P_max = 10 MW.
ATOM_LEVELS = np.array([-1.0, -2.0/3.0, -1.0/3.0, 0.0, 1.0/3.0, 2.0/3.0, 1.0])
P_MAX = 10.0
E_MAX = 20.0

RTCB_START = pd.Timestamp("2025-12-05", tz="UTC")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def distributional_stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    out = {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "skew": float(sps.skew(x)),
        "kurtosis": float(sps.kurtosis(x)),  # excess kurtosis
    }
    for p in [1, 5, 25, 50, 75, 95, 99]:
        out[f"p{p}"] = float(np.percentile(x, p))
    return out


def autocorr_lags(x: np.ndarray, lags) -> dict:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    denom = np.dot(x, x)
    out = {}
    for L in lags:
        if L >= len(x):
            out[L] = float("nan")
        else:
            out[L] = float(np.dot(x[:-L], x[L:]) / denom)
    return out


def run_lengths(arr: np.ndarray, value) -> np.ndarray:
    """Return lengths of consecutive runs where arr==value."""
    mask = (arr == value).astype(np.int8)
    if mask.sum() == 0:
        return np.array([], dtype=int)
    # pad with zeros and find transitions
    padded = np.concatenate([[0], mask, [0]])
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    return ends - starts


# ---------------------------------------------------------------------------
# Load market data
# ---------------------------------------------------------------------------
print("Loading market data...")

def load_table(name: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(PROC / name / "*.parquet")))
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs).sort_index()
    return df

energy = load_table("energy_prices")
as_p = load_table("as_prices")
system = load_table("system_conditions")
print(f"  energy:  {energy.shape}  {energy.index.min()} → {energy.index.max()}")
print(f"  as:      {as_p.shape}    {as_p.index.min()} → {as_p.index.max()}")
print(f"  system:  {system.shape}  {system.index.min()} → {system.index.max()}")

# Regime masks on raw market data
pre_mask = energy.index < RTCB_START
post_mask = energy.index >= RTCB_START
print(f"  pre rows: {pre_mask.sum()}, post rows: {post_mask.sum()}")

# ---------------------------------------------------------------------------
# Load MILP trajectories
# ---------------------------------------------------------------------------
print("\nLoading MILP trajectories...")
train = np.load(TRAJ / "receding_horizon_train_option_d.npz")
val = np.load(TRAJ / "receding_horizon_val_option_d.npz")

train_rewards = train["rewards"]
val_rewards = val["rewards"]
train_actions = train["actions"]
val_actions = val["actions"]
train_static = train["static_features"]
val_static = val["static_features"]
train_ph = train["price_history"]     # (N, 32, 12)
val_ph = val["price_history"]

print(f"  train: rewards {train_rewards.shape}, actions {train_actions.shape}, static {train_static.shape}")
print(f"  val:   rewards {val_rewards.shape}, actions {val_actions.shape}, static {val_static.shape}")

# Trajectory timestamps (5-min resolution, aligned to start of data).
# Training split: 2020-01-01 → 2023-12-31. Val: 2024-01-01 → 2025-09-30.
# Reconstruct by counting 5-min intervals from known starts.
train_start = pd.Timestamp("2020-01-01 00:00:00", tz="UTC")
val_start = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
# Trajectories skip first seq_len=32 steps (needed for lookback), and possibly some leading gaps.
# We'll check by matching energy price history to the trajectory's price_history at t=0.
# But for most analysis (rewards/actions on trajectory-internal time), the exact calendar
# timing matters only for daily-hour patterns — handled via hour-of-day inference below.

# The trajectories are contiguous 5-min steps; we can infer hour-of-day from time cyclical
# features in static_features dims 7..12.
# time = [sin_h, cos_h, sin_dow, cos_dow, sin_m, cos_m].
# hour = atan2(sin_h, cos_h) * 12 / pi (mod 24)
def recover_hour(static):
    sin_h = static[:, 7]
    cos_h = static[:, 8]
    theta = np.arctan2(sin_h, cos_h)
    hour = (theta * 12.0 / np.pi) % 24.0
    return hour

def recover_dow(static):
    sin_d = static[:, 9]
    cos_d = static[:, 10]
    theta = np.arctan2(sin_d, cos_d)
    dow = (theta * 7.0 / (2 * np.pi)) % 7.0
    return dow

train_hour = recover_hour(train_static)
val_hour = recover_hour(val_static)

# SoC (dim 13 is soc fraction → MWh = frac * E_max)
train_soc = train_static[:, 13] * E_MAX
val_soc = val_static[:, 13] * E_MAX

# ---------------------------------------------------------------------------
# SECTION 1: Reward distributions
# ---------------------------------------------------------------------------
print("\n=== Section 1: Reward distributions ===")
results = {}
results["section1"] = {}

def reward_concentration(r):
    pos = r[r > 0]
    if pos.size == 0:
        return {"top1": float("nan"), "top5": float("nan"), "top10": float("nan"),
                "sum_pos": 0.0, "n_pos": 0}
    sorted_desc = np.sort(pos)[::-1]
    total = sorted_desc.sum()
    n = sorted_desc.size
    top1 = sorted_desc[: max(1, int(round(n * 0.01)))].sum() / total
    top5 = sorted_desc[: max(1, int(round(n * 0.05)))].sum() / total
    top10 = sorted_desc[: max(1, int(round(n * 0.10)))].sum() / total
    return {"top1": float(top1), "top5": float(top5), "top10": float(top10),
            "sum_pos": float(total), "n_pos": int(n)}

for name, r in [("train", train_rewards), ("val", val_rewards)]:
    stats_ = distributional_stats(r)
    n = len(r)
    frac_zero = float(np.mean(r == 0.0))
    frac_near_zero = float(np.mean((r >= -1) & (r <= 1)))
    pos = r[r > 0]
    if pos.size:
        p99_pos = float(np.percentile(pos, 99))
        frac_tail_events = float(np.mean(r > p99_pos))
    else:
        p99_pos = float("nan")
        frac_tail_events = 0.0
    conc = reward_concentration(r)
    stats_.update({
        "frac_zero": frac_zero,
        "frac_near_zero_pm1": frac_near_zero,
        "p99_positive_rewards": p99_pos,
        "frac_above_p99_positive": frac_tail_events,
        "reward_concentration_top1pct_of_pos": conc["top1"],
        "reward_concentration_top5pct_of_pos": conc["top5"],
        "reward_concentration_top10pct_of_pos": conc["top10"],
        "sum_positive_rewards": conc["sum_pos"],
        "n_positive": conc["n_pos"],
    })
    results["section1"][f"milp_{name}"] = stats_

# RT LMP by regime
for name, mask in [("pre_rtcb", pre_mask), ("post_rtcb", post_mask)]:
    sub = energy.loc[mask, "rt_lmp"].dropna().values
    stats_ = distributional_stats(sub)
    # Scarcity
    scarcity_count = int((sub > 1000.0).sum())
    stats_["scarcity_count_gt_1000"] = scarcity_count
    stats_["scarcity_rate"] = float(scarcity_count / sub.size) if sub.size else 0.0
    results["section1"][f"rt_lmp_{name}"] = stats_

# Scarcity count by year (pre and post combined)
rt_lmp_all = energy["rt_lmp"].dropna()
scarcity_mask = rt_lmp_all > 1000.0
by_year = rt_lmp_all[scarcity_mask].groupby(rt_lmp_all[scarcity_mask].index.year).size()
results["section1"]["scarcity_by_year"] = {int(y): int(c) for y, c in by_year.items()}

# Longest consecutive run of scarcity intervals (in raw RT LMP series)
# NB: drop NaNs but keep time order; treat NaN as non-scarcity
rt_lmp_series = energy["rt_lmp"].copy()
scar = (rt_lmp_series.fillna(0.0) > 1000.0).values.astype(np.int8)
rl = run_lengths(scar, 1)
results["section1"]["longest_scarcity_run_intervals"] = int(rl.max()) if rl.size else 0
results["section1"]["longest_scarcity_run_minutes"] = int(rl.max() * 5) if rl.size else 0
results["section1"]["scarcity_run_count"] = int(rl.size)
if rl.size:
    results["section1"]["scarcity_run_mean_len"] = float(rl.mean())
    results["section1"]["scarcity_run_p95_len"] = float(np.percentile(rl, 95))

print("Section 1 done.")

# ---------------------------------------------------------------------------
# SECTION 2: Action distributions
# ---------------------------------------------------------------------------
print("\n=== Section 2: Action distributions ===")
results["section2"] = {}

def analyze_actions(a: np.ndarray, name: str) -> dict:
    counts = np.bincount(a, minlength=7)
    pct = counts / counts.sum()
    charge_idx = {0, 1, 2}
    discharge_idx = {4, 5, 6}
    idle_idx = {3}
    charge_mask = np.isin(a, list(charge_idx))
    discharge_mask = np.isin(a, list(discharge_idx))
    idle_mask = a == 3

    # Magnitudes (in p.u.)
    mag_when_charge = np.abs(ATOM_LEVELS[a[charge_mask]]).mean() if charge_mask.any() else 0.0
    mag_when_dch = np.abs(ATOM_LEVELS[a[discharge_mask]]).mean() if discharge_mask.any() else 0.0

    # Bang-bang: fraction of non-idle actions that are ±P_max (idx 0 or 6)
    non_idle = a[~idle_mask]
    bb_frac = float(np.mean(np.isin(non_idle, [0, 6]))) if non_idle.size else float("nan")

    # Autocorr (treat as signed p.u.)
    signed = ATOM_LEVELS[a]
    ac = autocorr_lags(signed, [1, 3, 6, 12])

    # Run lengths by category
    cat = np.where(charge_mask, -1, np.where(discharge_mask, 1, 0)).astype(np.int8)
    charge_runs = run_lengths(cat, -1)
    dch_runs = run_lengths(cat, 1)
    idle_runs = run_lengths(cat, 0)

    def summarize_runs(rl):
        if rl.size == 0:
            return {"n": 0, "mean": 0.0, "median": 0.0, "max": 0}
        return {"n": int(rl.size), "mean": float(rl.mean()),
                "median": float(np.median(rl)), "max": int(rl.max())}

    # Switching frequency: count direction changes (cat != previous cat among {-1,0,1})
    # Count only non-zero transitions (charge ↔ discharge or charge/dch ↔ idle)
    changes = int((cat[1:] != cat[:-1]).sum())
    days = len(a) / (288.0)  # 288 5-min intervals per day
    changes_per_day = changes / days if days else 0.0

    out = {
        "counts": {i: int(counts[i]) for i in range(7)},
        "pct": {i: float(pct[i]) for i in range(7)},
        "charge_frac": float(charge_mask.mean()),
        "discharge_frac": float(discharge_mask.mean()),
        "idle_frac": float(idle_mask.mean()),
        "mean_magnitude_when_charging_pu": float(mag_when_charge),
        "mean_magnitude_when_discharging_pu": float(mag_when_dch),
        "bang_bang_fraction": bb_frac,
        "autocorr": ac,
        "charge_runs": summarize_runs(charge_runs),
        "discharge_runs": summarize_runs(dch_runs),
        "idle_runs": summarize_runs(idle_runs),
        "switches_per_day": float(changes_per_day),
        "switches_total": changes,
        "days_approx": days,
    }
    return out

results["section2"]["train"] = analyze_actions(train_actions, "train")
results["section2"]["val"] = analyze_actions(val_actions, "val")

print("Section 2 done.")

# ---------------------------------------------------------------------------
# SECTION 3: State distributions + regime shift
# ---------------------------------------------------------------------------
print("\n=== Section 3: State distributions ===")
results["section3"] = {}

def soc_stats(soc: np.ndarray, hour_arr: np.ndarray) -> dict:
    s = {
        "mean_mwh": float(soc.mean()),
        "std_mwh": float(soc.std()),
        "min_mwh": float(soc.min()),
        "max_mwh": float(soc.max()),
        "pct_at_floor_le3": float((soc <= 3.0).mean()),
        "pct_at_ceiling_ge17": float((soc >= 17.0).mean()),
        "pct_midrange_4_16": float(((soc >= 4.0) & (soc <= 16.0)).mean()),
        "autocorr": autocorr_lags(soc, [1, 3, 6, 12]),
    }
    return s

results["section3"]["soc_train"] = soc_stats(train_soc, train_hour)
results["section3"]["soc_val"] = soc_stats(val_soc, val_hour)

# Correlate SoC with current-step RT LMP and DAM SPP (price_history[:, -1, 0] / :, -1, 6])
train_rt_now = train_ph[:, -1, 0]
train_da_now = train_ph[:, -1, 6]
val_rt_now = val_ph[:, -1, 0]
val_da_now = val_ph[:, -1, 6]

def pearson(x, y):
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return float("nan")
    return float(np.corrcoef(x[m], y[m])[0, 1])

results["section3"]["soc_rtlmp_corr_train"] = pearson(train_soc, train_rt_now)
results["section3"]["soc_damspp_corr_train"] = pearson(train_soc, train_da_now)
results["section3"]["soc_rtlmp_corr_val"] = pearson(val_soc, val_rt_now)
results["section3"]["soc_damspp_corr_val"] = pearson(val_soc, val_da_now)

# Observation dimension groupings — trajectory obs is structured (price_history 32x12, static_features 14)
# The "90-dim" mentioned in the task is the post-TTFE encoding. We report the pre-TTFE input layout.
results["section3"]["obs_layout"] = {
    "note": "MILP trajectories store structured obs: price_history (32 steps × 12 prices) and static_features (14). The 90-dim 'obs' referenced in CLAUDE.md is the TTFE-encoded observation built at training time from these inputs: 64-dim TTFE output + 12 current prices + 7 system + 6 cyclical time + 1 SoC = 90.",
    "price_history_shape": list(train_ph.shape[1:]),
    "price_cols": ["rt_lmp", "rt_mcpc_regup", "rt_mcpc_regdn", "rt_mcpc_rrs",
                    "rt_mcpc_ecrs", "rt_mcpc_nsrs", "dam_spp", "dam_as_regup",
                    "dam_as_regdn", "dam_as_rrs", "dam_as_ecrs", "dam_as_nsrs"],
    "static_features_layout": {
        "dims_0_6_system": ["total_load_mw", "load_forecast_mw", "wind_actual_mw",
                            "wind_forecast_mw", "solar_actual_mw", "solar_forecast_mw",
                            "net_load_mw"],
        "dims_7_12_time_cyclical": ["sin_hour", "cos_hour", "sin_dow", "cos_dow",
                                    "sin_month", "cos_month"],
        "dim_13_soc": "soc_fraction = soc / 20 MWh",
    },
    "post_ttfe_90dim_grouping": {
        "price_like_64_plus_12": 76,
        "system_like": 7,
        "time_like_cyclical": 6,
        "soc_like": 1,
    },
}

# --- Regime shift on raw market data ---
pre_rt = energy.loc[pre_mask, "rt_lmp"].dropna().values
post_rt = energy.loc[post_mask, "rt_lmp"].dropna().values
# KS test
ks = sps.ks_2samp(pre_rt, post_rt)
results["section3"]["ks_rtlmp_pre_vs_post"] = {
    "statistic": float(ks.statistic),
    "pvalue": float(ks.pvalue),
}
results["section3"]["rtlmp_moments_pre"] = {
    "mean": float(pre_rt.mean()), "std": float(pre_rt.std()),
    "p99": float(np.percentile(pre_rt, 99)),
}
results["section3"]["rtlmp_moments_post"] = {
    "mean": float(post_rt.mean()), "std": float(post_rt.std()),
    "p99": float(np.percentile(post_rt, 99)),
}

# AS clearing prices
for col in ["rt_mcpc_regup", "rt_mcpc_ecrs"]:
    s_all = as_p[col]
    pre = s_all.loc[energy.index[pre_mask].intersection(s_all.index)].dropna().values
    post = s_all.loc[energy.index[post_mask].intersection(s_all.index)].dropna().values
    if pre.size == 0:
        continue
    results["section3"][f"{col}_pre"] = {
        "n": int(pre.size), "mean": float(pre.mean()), "std": float(pre.std()),
        "p99": float(np.percentile(pre, 99)),
    }
    if post.size > 0:
        results["section3"][f"{col}_post"] = {
            "n": int(post.size), "mean": float(post.mean()), "std": float(post.std()),
            "p99": float(np.percentile(post, 99)),
        }
    else:
        results["section3"][f"{col}_post"] = {"n": 0}

# Correlation RT LMP vs RegUp RT MCPC
def corr_cols(df, c1, c2, mask):
    x = df.loc[mask, c1]
    y = df.loc[mask, c2]
    join = pd.concat([x, y], axis=1).dropna()
    if len(join) < 100:
        return float("nan")
    return float(join.corr().iloc[0, 1])

merged = energy[["rt_lmp"]].join(as_p[["rt_mcpc_regup", "rt_mcpc_ecrs"]])
pre_m = merged.index < RTCB_START
post_m = merged.index >= RTCB_START
results["section3"]["corr_rtlmp_regup_pre"] = corr_cols(merged, "rt_lmp", "rt_mcpc_regup", pre_m)
results["section3"]["corr_rtlmp_regup_post"] = corr_cols(merged, "rt_lmp", "rt_mcpc_regup", post_m)
results["section3"]["corr_rtlmp_ecrs_pre"] = corr_cols(merged, "rt_lmp", "rt_mcpc_ecrs", pre_m)
results["section3"]["corr_rtlmp_ecrs_post"] = corr_cols(merged, "rt_lmp", "rt_mcpc_ecrs", post_m)

# Scarcity in post period (rate comparison)
pre_scarc_rate = float((pre_rt > 1000.0).mean())
post_scarc_rate = float((post_rt > 1000.0).mean())
results["section3"]["scarcity_rate_pre"] = pre_scarc_rate
results["section3"]["scarcity_rate_post"] = post_scarc_rate
results["section3"]["post_sample_size_months"] = float(post_rt.size / (288 * 30))

print("Section 3 done.")

# ---------------------------------------------------------------------------
# SECTION 4: Expert policy structure
# ---------------------------------------------------------------------------
print("\n=== Section 4: Expert policy structure ===")
results["section4"] = {}

# Daily SoC pattern: avg SoC by hour of day (training split)
hours = np.floor(train_hour).astype(int)
soc_by_hour_train = np.array([train_soc[hours == h].mean() if (hours == h).any() else np.nan
                              for h in range(24)])
hours_v = np.floor(val_hour).astype(int)
soc_by_hour_val = np.array([val_soc[hours_v == h].mean() if (hours_v == h).any() else np.nan
                            for h in range(24)])

# Action direction signal by hour (mean of signed p.u. per hour)
signed_train = ATOM_LEVELS[train_actions]
signed_val = ATOM_LEVELS[val_actions]
act_by_hour_train = np.array([signed_train[hours == h].mean() if (hours == h).any() else np.nan
                              for h in range(24)])
act_by_hour_val = np.array([signed_val[hours_v == h].mean() if (hours_v == h).any() else np.nan
                            for h in range(24)])

results["section4"]["soc_by_hour_train"] = [float(v) for v in soc_by_hour_train]
results["section4"]["soc_by_hour_val"] = [float(v) for v in soc_by_hour_val]
results["section4"]["signed_action_by_hour_train"] = [float(v) for v in act_by_hour_train]
results["section4"]["signed_action_by_hour_val"] = [float(v) for v in act_by_hour_val]

# Correlation of action with same-period RT LMP (overall, and conditioned on non-idle)
results["section4"]["corr_action_rtlmp_train"] = pearson(signed_train, train_rt_now)
results["section4"]["corr_action_rtlmp_val"] = pearson(signed_val, val_rt_now)

charge_mask_t = train_actions < 3
dch_mask_t = train_actions > 3
results["section4"]["charge_rt_lmp_mean_train"] = float(train_rt_now[charge_mask_t].mean())
results["section4"]["discharge_rt_lmp_mean_train"] = float(train_rt_now[dch_mask_t].mean())
results["section4"]["idle_rt_lmp_mean_train"] = float(train_rt_now[train_actions == 3].mean())
results["section4"]["charge_rt_lmp_mean_val"] = float(val_rt_now[val_actions < 3].mean())
results["section4"]["discharge_rt_lmp_mean_val"] = float(val_rt_now[val_actions > 3].mean())

# Forward-looking: correlation of current action with RT LMP 12 intervals ahead.
# Use next_price_history's last slice if available. Safer: shift price_history within the array
# by 12 — but trajectories are contiguous, so price at step i+12 is train_ph[i+12, -1, 0].
L = 12
rt_future_train = train_rt_now[L:]
rt_future_val = val_rt_now[L:]
signed_t_trim = signed_train[:-L]
signed_v_trim = signed_val[:-L]
results["section4"]["corr_charge_action_vs_rtlmp_plus12_train"] = pearson(
    -signed_t_trim * (signed_t_trim < 0), rt_future_train  # charge magnitude (positive) vs future price
)
# More meaningful: among currently-charging steps, does future price trend up?
# Compute corr(signed action, future RT LMP): if expert charges (negative action) ahead of price spikes,
# the correlation should be negative.
results["section4"]["corr_signed_action_vs_rtlmp_plus12_train"] = pearson(signed_t_trim, rt_future_train)
results["section4"]["corr_signed_action_vs_rtlmp_plus12_val"] = pearson(signed_v_trim, rt_future_val)

# Daily revenue stats (on training split)
# Use 288 intervals per day; reshape if contiguous.
def daily_revenue(rewards, n_per_day=288):
    n = (len(rewards) // n_per_day) * n_per_day
    r = rewards[:n].reshape(-1, n_per_day)
    daily = r.sum(axis=1)
    return daily

daily_train = daily_revenue(train_rewards)
daily_val = daily_revenue(val_rewards)
results["section4"]["daily_revenue_train"] = {
    "mean": float(daily_train.mean()),
    "std": float(daily_train.std()),
    "median": float(np.median(daily_train)),
    "max": float(daily_train.max()),
    "min": float(daily_train.min()),
    "n_days": int(len(daily_train)),
}
results["section4"]["daily_revenue_val"] = {
    "mean": float(daily_val.mean()),
    "std": float(daily_val.std()),
    "median": float(np.median(daily_val)),
    "max": float(daily_val.max()),
    "min": float(daily_val.min()),
    "n_days": int(len(daily_val)),
}

# Scarcity response
for name, actions, rt_now, soc in [("train", train_actions, train_rt_now, train_soc),
                                    ("val", val_actions, val_rt_now, val_soc)]:
    scar_mask = rt_now > 500.0
    n_scar = int(scar_mask.sum())
    if n_scar > 0:
        scar_actions = actions[scar_mask]
        scar_soc = soc[scar_mask]
        cnt = np.bincount(scar_actions, minlength=7)
        results["section4"][f"scarcity_response_{name}"] = {
            "n_events_rt_gt_500": n_scar,
            "action_counts": {i: int(cnt[i]) for i in range(7)},
            "action_pct": {i: float(cnt[i] / n_scar) for i in range(7)},
            "discharge_frac": float((scar_actions > 3).mean()),
            "charge_frac": float((scar_actions < 3).mean()),
            "idle_frac": float((scar_actions == 3).mean()),
            "mean_soc_mwh": float(scar_soc.mean()),
            "p5_soc_mwh": float(np.percentile(scar_soc, 5)),
            "p95_soc_mwh": float(np.percentile(scar_soc, 95)),
        }
    # also a higher-bar scarcity (>$1000)
    scar_mask_1k = rt_now > 1000.0
    n_scar_1k = int(scar_mask_1k.sum())
    if n_scar_1k > 0:
        scar_actions = actions[scar_mask_1k]
        scar_soc = soc[scar_mask_1k]
        cnt = np.bincount(scar_actions, minlength=7)
        results["section4"][f"scarcity_response_{name}_gt1k"] = {
            "n_events_rt_gt_1000": n_scar_1k,
            "action_counts": {i: int(cnt[i]) for i in range(7)},
            "discharge_frac": float((scar_actions > 3).mean()),
            "idle_frac": float((scar_actions == 3).mean()),
            "mean_soc_mwh": float(scar_soc.mean()),
        }

print("Section 4 done.")

# ---------------------------------------------------------------------------
# Dump JSON
# ---------------------------------------------------------------------------
with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nWrote {OUT_JSON}")
