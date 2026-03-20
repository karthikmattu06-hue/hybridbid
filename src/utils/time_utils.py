"""
Time utilities for ERCOT data.

ERCOT uses Central Prevailing Time (CPT) with Hour Ending (HE) convention:
  - HE 1:00 covers 00:00–01:00
  - During fall DST transition, HE 2:00 appears twice (flagged with DSTFlag)

This module provides conversions between ERCOT's native timestamps and
our canonical UTC-indexed 5-minute intervals.
"""

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import pytz

# ERCOT operates in US/Central timezone
ERCOT_TZ = pytz.timezone("US/Central")
UTC = pytz.UTC

# RTC+B go-live date
RTCB_GO_LIVE = pd.Timestamp("2025-12-05", tz=UTC)

# Canonical interval duration
INTERVAL_5MIN = timedelta(minutes=5)
INTERVALS_PER_HOUR = 12
INTERVALS_PER_DAY = 288


def cpt_to_utc(ts: pd.Timestamp, dst_flag: Optional[bool] = None) -> pd.Timestamp:
    """
    Convert ERCOT Central Prevailing Time to UTC.

    Parameters
    ----------
    ts : pd.Timestamp
        Naive or CPT-localized timestamp.
    dst_flag : bool, optional
        If True, the timestamp is in DST (CDT, UTC-5).
        If False, standard time (CST, UTC-6).
        Used to disambiguate fall-back overlap.

    Returns
    -------
    pd.Timestamp
        UTC-localized timestamp.
    """
    if ts.tzinfo is None:
        if dst_flag is not None:
            is_dst = dst_flag
        else:
            is_dst = False  # Default to standard time if ambiguous
        ts = ERCOT_TZ.localize(ts, is_dst=is_dst)
    return ts.astimezone(UTC)


def utc_to_cpt(ts: pd.Timestamp) -> pd.Timestamp:
    """Convert UTC timestamp to ERCOT Central Prevailing Time."""
    if ts.tzinfo is None:
        ts = UTC.localize(ts)
    return ts.astimezone(ERCOT_TZ)


def hour_ending_to_interval_start(he_hour: int, date: pd.Timestamp) -> pd.Timestamp:
    """
    Convert ERCOT Hour Ending notation to interval start time.

    HE 1 → 00:00, HE 2 → 01:00, ..., HE 24 → 23:00
    """
    return date.normalize() + timedelta(hours=he_hour - 1)


def make_5min_index(
    start: str,
    end: str,
    tz: str = "UTC",
) -> pd.DatetimeIndex:
    """
    Create a canonical 5-minute DatetimeIndex.

    Parameters
    ----------
    start, end : str
        Date strings (e.g., '2024-01-01').
    tz : str
        Timezone for the index. Default 'UTC'.

    Returns
    -------
    pd.DatetimeIndex
        5-minute frequency index.
    """
    return pd.date_range(start=start, end=end, freq="5min", tz=tz, inclusive="left")


def is_post_rtcb(ts: pd.Timestamp) -> bool:
    """Check if a timestamp falls after the RTC+B go-live date."""
    if ts.tzinfo is None:
        ts = UTC.localize(ts)
    return ts >= RTCB_GO_LIVE


def add_post_rtcb_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add an 'is_post_rtcb' boolean column to a DataFrame with a datetime index.
    """
    df = df.copy()
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize(UTC)
    df["is_post_rtcb"] = idx >= RTCB_GO_LIVE
    return df


def resample_to_5min(
    df: pd.DataFrame,
    method: str = "ffill",
) -> pd.DataFrame:
    """
    Resample a DataFrame to 5-minute intervals.

    Parameters
    ----------
    df : pd.DataFrame
        Input with DatetimeIndex at any frequency.
    method : str
        'ffill' (forward fill) or 'interpolate'.

    Returns
    -------
    pd.DataFrame
        Resampled to 5-minute frequency.
    """
    df_5min = df.resample("5min").first()

    if method == "ffill":
        df_5min = df_5min.ffill()
    elif method == "interpolate":
        df_5min = df_5min.interpolate(method="time")
    else:
        raise ValueError(f"Unknown resample method: {method}")

    return df_5min


def get_ercot_operating_day(ts: pd.Timestamp) -> pd.Timestamp:
    """
    Get the ERCOT operating day for a given timestamp.
    ERCOT operating day runs from HE 1 (00:00 CPT) to HE 24 (23:55 CPT).
    """
    cpt_ts = utc_to_cpt(ts)
    return cpt_ts.normalize()
