"""
Option D preprocessor: MILP raw traces → IQL transition dataset,
WITHOUT replaying through env.step.

Path (b): standalone numpy encoder. We directly read the parquet data tables
(as env does) and reproduce env._get_observation exactly for each MILP step,
using MILP's recorded SoC. MILP's rewards_env are used unchanged (zero
reward-mismatch by construction).

Output: data/expert_trajectories/receding_horizon_{split}_option_d.npz
  price_history        (N, 32, 12) float32
  static_features      (N, 14)     float32     [system(7) + time(6) + soc(1)]
  next_price_history   (N, 32, 12) float32
  next_static_features (N, 14)     float32
  actions              (N,)        int64       discrete atom idx 0..6
  rewards              (N,)        float32     = MILP's rewards_env
  dones                (N,)        bool        all False (MILP is SoC-feasible)
  truncateds           (N,)        bool        True when next transition crosses a UTC-date boundary
  signed_continuous    (N,)        float32     MILP's original signed magnitude in [-1, 1] (diagnostic)
  quantization_error   (N,)        float32     |signed − snapped| (diagnostic)

Run:
  python -u -m scripts.preprocess_milp_option_d --split both
"""
import argparse
import glob
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.env.ercot_env import PRICE_COLS, SYSTEM_COLS, SEQ_LEN, MODE_CHARGE, MODE_DISCHARGE
from src.models.networks import TIER2A_ACTION_LEVELS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants (must match src/env/ercot_env.py) ──
E_MAX = 20.0
SYSTEM_SCALES = np.array([50000, 50000, 15000, 15000, 10000, 10000, 40000], dtype=np.float32)
N_PRICES = len(PRICE_COLS)   # 12
N_SYSTEM = len(SYSTEM_COLS)  # 7
N_STATIC = N_SYSTEM + 6 + 1  # 14
ATOM_LEVELS = np.array(TIER2A_ACTION_LEVELS, dtype=np.float32)  # (7,) in [-1, 1]

SPLITS = {
    "train": ("2020-01-01", "2023-12-31"),
    "val":   ("2024-01-01", "2025-09-30"),
}


# ──────────────────────────────────────────────────────────────────────
# Data loading — replicates src/env/ercot_env.py _load_data exactly
# ──────────────────────────────────────────────────────────────────────
def _read_all_parquets(directory: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(directory, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files in {directory}")
    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs).sort_index()


def load_merged_tables(data_dir: str, date_range):
    t0 = time.time()
    ep = _read_all_parquets(os.path.join(data_dir, "energy_prices"))
    ap = _read_all_parquets(os.path.join(data_dir, "as_prices"))
    sc = _read_all_parquets(os.path.join(data_dir, "system_conditions"))
    for df in (ep, ap, sc):
        if "is_post_rtcb" in df.columns:
            df.drop(columns=["is_post_rtcb"], inplace=True)
    merged = ep.join(ap, how="outer").join(sc, how="outer")
    if date_range:
        merged = merged.loc[date_range[0]:date_range[1]]
    merged[PRICE_COLS]  = merged[PRICE_COLS].fillna(0.0)
    merged[SYSTEM_COLS] = merged[SYSTEM_COLS].ffill().fillna(0.0)
    logger.info(f"Loaded + merged parquets in {time.time() - t0:.1f}s: {merged.shape}")
    return merged


# ──────────────────────────────────────────────────────────────────────
# Feature construction
# ──────────────────────────────────────────────────────────────────────
def build_time_features(timestamps_utc: pd.DatetimeIndex) -> np.ndarray:
    """(T, 6) cyclical [sin_h, cos_h, sin_dow, cos_dow, sin_m, cos_m]; Central Time."""
    ts_local = timestamps_utc.tz_convert("US/Central")
    hour  = ts_local.hour + ts_local.minute / 60.0
    dow   = ts_local.dayofweek.values
    month = ts_local.month.values
    feats = np.stack([
        np.sin(2 * np.pi * hour / 24.0),
        np.cos(2 * np.pi * hour / 24.0),
        np.sin(2 * np.pi * dow / 7.0),
        np.cos(2 * np.pi * dow / 7.0),
        np.sin(2 * np.pi * month / 12.0),
        np.cos(2 * np.pi * month / 12.0),
    ], axis=1).astype(np.float32)
    return feats


def milp_to_signed(mode: np.ndarray, magnitude: np.ndarray) -> np.ndarray:
    """(mode ∈ {0,1,2}, mag ∈ [0,1]) → signed ∈ [-1, +1]."""
    signed = np.zeros(len(mode), dtype=np.float32)
    signed[mode == MODE_CHARGE]    = -magnitude[mode == MODE_CHARGE]
    signed[mode == MODE_DISCHARGE] = +magnitude[mode == MODE_DISCHARGE]
    return signed


def snap_signed_to_atom_idx(signed: np.ndarray) -> np.ndarray:
    """Vectorised nearest-atom lookup → (N,) int64."""
    # diffs[i, k] = |signed[i] − ATOM_LEVELS[k]|
    diffs = np.abs(signed[:, None] - ATOM_LEVELS[None, :])
    return diffs.argmin(axis=1).astype(np.int64)


# ──────────────────────────────────────────────────────────────────────
# Main preprocessing per split
# ──────────────────────────────────────────────────────────────────────
def preprocess_split(split: str, milp_npz: Path, out_npz: Path, data_dir: str):
    date_range = SPLITS[split]
    logger.info(f"── Preprocessing {split} split ({date_range[0]} → {date_range[1]}) ──")

    # Load MILP raw trace
    raw = np.load(milp_npz, allow_pickle=False)
    milp_ts    = pd.to_datetime(raw["timestamps"], utc=True)
    milp_mode  = raw["modes"]
    milp_mag   = raw["magnitudes"]
    milp_socs  = raw["socs"]
    milp_r_env = raw["rewards_env"].astype(np.float32)
    N_milp = len(milp_ts)
    logger.info(f"MILP trace: {N_milp:,} intervals")

    # Load merged parquet tables for this date range
    merged = load_merged_tables(data_dir, date_range)
    parquet_ts = merged.index
    price_arr  = merged[PRICE_COLS].values.astype(np.float32)     # (T, 12)
    system_arr = merged[SYSTEM_COLS].values.astype(np.float32)    # (T, 7)
    time_arr   = build_time_features(parquet_ts)                  # (T, 6)

    # MILP timestamp → parquet index (O(N) hash map)
    ts_to_idx = pd.Series(np.arange(len(parquet_ts)), index=parquet_ts)
    parquet_idx_raw = ts_to_idx.reindex(milp_ts).to_numpy()
    missing = pd.isna(parquet_idx_raw)
    if missing.any():
        logger.warning(f"{int(missing.sum())} MILP timestamps missing in parquets → dropped")
    valid = ~missing & (parquet_idx_raw.astype("float64") >= (SEQ_LEN - 1))  # need 32-step history
    # Require next step to exist and be in-range (for (s, s') pairs)
    # Next step exists iff i+1 < N_milp AND parquet_idx[i+1] is valid AND in-range
    idx = parquet_idx_raw.astype("float64")
    has_next = np.zeros(N_milp, dtype=bool)
    has_next[:-1] = (~missing[:-1]) & (~missing[1:]) & (idx[1:] >= (SEQ_LEN - 1))
    mask = valid & has_next
    usable = np.where(mask)[0]
    N = len(usable)
    logger.info(f"Usable transitions: {N:,} / {N_milp:,} "
                f"(dropped {N_milp - N} for missing timestamps or insufficient lookback)")

    pi  = parquet_idx_raw.astype(np.int64)   # (N_milp,)
    pi_next = np.empty(N_milp, dtype=np.int64)
    pi_next[:-1] = pi[1:]
    pi_next[-1]  = pi[-1]   # unused; mask drops it

    # ── Gather s and s' for every usable transition ──
    t1 = time.time()
    u_pi      = pi[usable]           # (N,)
    u_pi_next = pi_next[usable]      # (N,)

    # price_history (N, 32, 12) = price_arr[u_pi[:, None] + offsets, :]
    offsets = np.arange(-(SEQ_LEN - 1), 1, dtype=np.int64)   # (32,) = -31..0
    ph      = price_arr[u_pi[:, None] + offsets[None, :], :].astype(np.float32)
    nph     = price_arr[u_pi_next[:, None] + offsets[None, :], :].astype(np.float32)

    # static_features (N, 14) = [system/scale (7), time (6), soc_frac (1)]
    sys_scaled      = (system_arr[u_pi]      / SYSTEM_SCALES[None, :]).astype(np.float32)
    sys_scaled_next = (system_arr[u_pi_next] / SYSTEM_SCALES[None, :]).astype(np.float32)
    time_u      = time_arr[u_pi]
    time_u_next = time_arr[u_pi_next]

    soc_u      = (milp_socs[usable]     / E_MAX).astype(np.float32)
    soc_u_next = (milp_socs[usable + 1] / E_MAX).astype(np.float32)

    sf  = np.concatenate([sys_scaled,      time_u,      soc_u[:, None]],      axis=1)
    nsf = np.concatenate([sys_scaled_next, time_u_next, soc_u_next[:, None]], axis=1)
    logger.info(f"Built obs tensors in {time.time() - t1:.1f}s "
                f"(ph={ph.shape}, sf={sf.shape}, nph={nph.shape}, nsf={nsf.shape})")

    # ── Actions, rewards, dones, truncateds ──
    signed    = milp_to_signed(milp_mode, milp_mag)          # (N_milp,) in [-1, 1]
    atom_idx  = snap_signed_to_atom_idx(signed)              # (N_milp,) int64
    u_signed  = signed[usable]
    u_atoms   = atom_idx[usable]
    u_rewards = milp_r_env[usable]
    u_qerr    = np.abs(u_signed - ATOM_LEVELS[u_atoms]).astype(np.float32)

    # Per-day boundary: truncated=True when next-step UTC date differs from current
    utc_date      = milp_ts.date
    utc_date_next = np.empty(N_milp, dtype=object)
    utc_date_next[:-1] = utc_date[1:]
    utc_date_next[-1]  = utc_date[-1]
    crosses_day = np.array([utc_date[i] != utc_date_next[i] for i in range(N_milp)], dtype=bool)
    u_truncateds = crosses_day[usable].astype(bool)
    u_dones      = np.zeros(N, dtype=bool)

    # ── Save ──
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        price_history=ph,
        static_features=sf,
        next_price_history=nph,
        next_static_features=nsf,
        actions=u_atoms,
        rewards=u_rewards,
        dones=u_dones,
        truncateds=u_truncateds,
        signed_continuous=u_signed,
        quantization_error=u_qerr,
    )
    size_mb = out_npz.stat().st_size / 1e6
    logger.info(f"Saved: {out_npz}  ({size_mb:.1f} MB)")

    # ── Stats ──
    action_dist = np.bincount(u_atoms, minlength=7)
    q_mean_mw   = float(u_qerr.mean() * 10.0)
    q_max_mw    = float(u_qerr.max()  * 10.0)

    logger.info("=" * 68)
    logger.info(f"Split: {split}")
    logger.info(f"  Transitions N        = {N:,}")
    logger.info(f"  Action dist (0..6)   = {action_dist.tolist()}")
    logger.info(f"  Action pct (0..6)    = {(100 * action_dist / N).round(1).tolist()}")
    logger.info(f"  Quant error          = mean {q_mean_mw:.3f} MW  max {q_max_mw:.3f} MW")
    logger.info(f"  Truncateds (day end) = {int(u_truncateds.sum()):,}")
    logger.info(f"  Dones                = {int(u_dones.sum()):,}  (MILP is SoC-feasible)")
    logger.info(f"  Reward: mean {u_rewards.mean():+.3f}  min {u_rewards.min():+.1f}  max {u_rewards.max():+.1f}")

    # ── Sanity: rewards match MILP exactly by construction ──
    r_saved = np.load(out_npz)["rewards"]
    r_milp  = milp_r_env[usable]
    diff = np.abs(r_saved.astype(np.float64) - r_milp.astype(np.float64))
    if diff.max() == 0.0:
        logger.info(f"  ✅ rewards ≡ MILP.rewards_env (max |Δ| = 0.00)")
    else:
        logger.error(f"  ❌ rewards MISMATCH: max |Δ| = {diff.max()}")
    logger.info("=" * 68)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "both"], default="both")
    parser.add_argument("--in-dir",  type=Path, default=Path("data/expert_trajectories"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/expert_trajectories"))
    parser.add_argument("--data-dir", default="data/processed")
    args = parser.parse_args()

    splits = ["train", "val"] if args.split == "both" else [args.split]
    for split in splits:
        inp = args.in_dir / f"receding_horizon_{split}.npz"
        out = args.out_dir / f"receding_horizon_{split}_option_d.npz"
        if not inp.exists():
            raise FileNotFoundError(inp)
        preprocess_split(split, inp, out, data_dir=args.data_dir)


if __name__ == "__main__":
    main()
