"""
Canonical schema definitions for HybridBid data tables.

All data is stored in Parquet format with these schemas.
The canonical timestamp is UTC at 5-minute intervals.
Pre-RTC+B periods have NaN for RT AS price columns.
"""

import pyarrow as pa

# ──────────────────────────────────────────────
# Energy Prices Table
# ──────────────────────────────────────────────
ENERGY_PRICES_SCHEMA = pa.schema([
    ("timestamp", pa.timestamp("us", tz="UTC")),
    # Real-time settlement point prices (15-min, resampled to 5-min via ffill)
    ("rt_spp_hub", pa.float64()),           # Hub average SPP [$/MWh]
    ("rt_spp_north", pa.float64()),         # North zone
    ("rt_spp_south", pa.float64()),         # South zone
    ("rt_spp_west", pa.float64()),          # West zone
    ("rt_spp_houston", pa.float64()),       # Houston zone
    # Day-ahead prices (hourly, resampled to 5-min via ffill)
    ("dam_spp_hub", pa.float64()),          # DAM hub average [$/MWh]
    # Metadata
    ("is_post_rtcb", pa.bool_()),           # True if after Dec 5, 2025
])

# ──────────────────────────────────────────────
# Ancillary Service Prices Table
# ──────────────────────────────────────────────
AS_PRICES_SCHEMA = pa.schema([
    ("timestamp", pa.timestamp("us", tz="UTC")),
    # Real-time MCPCs (5-min, only exist post-RTC+B — NaN before Dec 5, 2025)
    ("rt_mcpc_regup", pa.float64()),        # [$/MW]
    ("rt_mcpc_regdown", pa.float64()),
    ("rt_mcpc_rrs", pa.float64()),
    ("rt_mcpc_ecrs", pa.float64()),
    ("rt_mcpc_nsrs", pa.float64()),
    # Day-ahead AS clearing prices (hourly, available full history)
    ("dam_as_regup", pa.float64()),         # [$/MW]
    ("dam_as_regdown", pa.float64()),
    ("dam_as_rrs", pa.float64()),
    ("dam_as_ecrs", pa.float64()),
    ("dam_as_nsrs", pa.float64()),
    # Metadata
    ("is_post_rtcb", pa.bool_()),
])

# ──────────────────────────────────────────────
# System Conditions Table
# ──────────────────────────────────────────────
SYSTEM_CONDITIONS_SCHEMA = pa.schema([
    ("timestamp", pa.timestamp("us", tz="UTC")),
    # Load
    ("total_load_mw", pa.float64()),        # System-wide load [MW]
    ("load_forecast_mw", pa.float64()),     # 7-day ahead forecast [MW]
    # Renewables — actuals
    ("wind_actual_mw", pa.float64()),       # Wind generation [MW]
    ("solar_actual_mw", pa.float64()),      # Solar generation [MW]
    # Renewables — forecasts
    ("wind_forecast_mw", pa.float64()),     # Wind forecast [MW]
    ("solar_forecast_mw", pa.float64()),    # Solar forecast [MW]
    # Derived
    ("net_load_mw", pa.float64()),          # total_load - wind - solar [MW]
    # Metadata
    ("is_post_rtcb", pa.bool_()),
])

# ──────────────────────────────────────────────
# Column name mappings from gridstatus to canonical
# ──────────────────────────────────────────────
# These may need adjustment based on actual gridstatus output columns.
# Update during Day 1 data exploration.

GRIDSTATUS_COLUMN_MAP = {
    # Energy prices — adjust after inspecting actual gridstatus output
    "SPP": "rt_spp_hub",
    "LMP": "rt_lmp_hub",
    # Add mappings as discovered during exploration
}

# ──────────────────────────────────────────────
# Parquet write configuration
# ──────────────────────────────────────────────
PARQUET_CONFIG = {
    "compression": "snappy",
    "row_group_size": 100_000,    # ~1 week of 5-min data
    "use_dictionary": True,
}

# Table names and their output paths (relative to data/processed/)
TABLES = {
    "energy_prices": {
        "schema": ENERGY_PRICES_SCHEMA,
        "partition_by": "month",  # YYYY-MM partitioning
        "path": "energy_prices",
    },
    "as_prices": {
        "schema": AS_PRICES_SCHEMA,
        "partition_by": "month",
        "path": "as_prices",
    },
    "system_conditions": {
        "schema": SYSTEM_CONDITIONS_SCHEMA,
        "partition_by": "month",
        "path": "system_conditions",
    },
}
