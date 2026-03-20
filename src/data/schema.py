"""
Canonical schema definitions for HybridBid data tables.

All data is stored in Parquet format with UTC timestamps at 5-minute intervals.
Column mappings are based on Phase 1/1b exploration of actual gridstatus output.

Three canonical tables:
  - energy_prices:      RT LMP + DAM SPP
  - as_prices:          RT MCPCs (post-RTC+B) + DAM AS
  - system_conditions:  load, wind, solar
"""

import pyarrow as pa

# RTC+B go-live boundary (UTC)
RTCB_BOUNDARY_UTC = "2025-12-05T06:00:00"  # midnight CPT = 06:00 UTC

# ──────────────────────────────────────────────
# Energy Prices Table
# ──────────────────────────────────────────────
ENERGY_PRICES_SCHEMA = pa.schema([
    ("timestamp_utc", pa.timestamp("us", tz="UTC")),
    ("rt_lmp", pa.float64()),         # RT LMP at HB_HUBAVG [$/MWh], 5-min native
    ("dam_spp", pa.float64()),        # DAM SPP at HB_HUBAVG [$/MWh], hourly ffill
    ("is_post_rtcb", pa.bool_()),
])

# ──────────────────────────────────────────────
# Ancillary Service Prices Table
# ──────────────────────────────────────────────
AS_PRICES_SCHEMA = pa.schema([
    ("timestamp_utc", pa.timestamp("us", tz="UTC")),
    # RT MCPCs — 5-min SCED, NaN pre-RTC+B
    ("rt_mcpc_regup", pa.float64()),   # [$/MW]
    ("rt_mcpc_regdn", pa.float64()),
    ("rt_mcpc_rrs", pa.float64()),
    ("rt_mcpc_ecrs", pa.float64()),
    ("rt_mcpc_nsrs", pa.float64()),
    # DAM AS — hourly, forward-filled to 5-min
    ("dam_as_regup", pa.float64()),    # [$/MW]
    ("dam_as_regdn", pa.float64()),
    ("dam_as_rrs", pa.float64()),
    ("dam_as_ecrs", pa.float64()),     # NaN → 0 pre-June 2023
    ("dam_as_nsrs", pa.float64()),
    ("is_post_rtcb", pa.bool_()),
])

# ──────────────────────────────────────────────
# System Conditions Table
# ──────────────────────────────────────────────
SYSTEM_CONDITIONS_SCHEMA = pa.schema([
    ("timestamp_utc", pa.timestamp("us", tz="UTC")),
    ("total_load_mw", pa.float64()),       # ERCOT system total [MW]
    ("load_forecast_mw", pa.float64()),    # Operational forecast [MW]
    ("wind_actual_mw", pa.float64()),      # System-wide wind gen [MW]
    ("wind_forecast_mw", pa.float64()),    # STWPF system-wide [MW]
    ("solar_actual_mw", pa.float64()),     # System-wide solar gen [MW]
    ("solar_forecast_mw", pa.float64()),   # STPPF system-wide [MW]
    ("net_load_mw", pa.float64()),         # total_load - wind - solar [MW]
    ("is_post_rtcb", pa.bool_()),
])

# ──────────────────────────────────────────────
# Column mappings: gridstatus → canonical
# Based on Phase 1/1b exploration findings
# ──────────────────────────────────────────────

# DAM AS: gridstatus wide-format column names → canonical
DAM_AS_COLUMN_MAP = {
    "Regulation Up": "dam_as_regup",
    "Regulation Down": "dam_as_regdn",
    "Responsive Reserves": "dam_as_rrs",
    "ERCOT Contingency Reserve Service": "dam_as_ecrs",
    "Non-Spinning Reserves": "dam_as_nsrs",
}

# RT SCED MCPC: ASType values → canonical column names
RT_MCPC_ASTYPE_MAP = {
    "REGUP": "rt_mcpc_regup",
    "REGDN": "rt_mcpc_regdn",
    "RRS": "rt_mcpc_rrs",
    "ECRS": "rt_mcpc_ecrs",
    "NSPIN": "rt_mcpc_nsrs",
}

# ──────────────────────────────────────────────
# Parquet write configuration
# ──────────────────────────────────────────────
PARQUET_CONFIG = {
    "compression": "snappy",
    "row_group_size": 100_000,
    "use_dictionary": True,
}

TABLES = {
    "energy_prices": {
        "schema": ENERGY_PRICES_SCHEMA,
        "partition_by": "month",
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
