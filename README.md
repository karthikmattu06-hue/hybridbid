# HybridBid — Battery Storage Bidding Optimization for ERCOT Post-RTC+B

An AI system that helps small battery storage operators (5-20 MW) optimize their bidding in ERCOT's post-RTC+B electricity market.

**Target:** 75-82% of perfect foresight revenue (vs. 40-56% for current small operator strategies).

## Quick Start

```bash
# 1. Create environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Set up Gurobi academic license
# Visit: https://www.gurobi.com/academia/academic-program-and-licenses/
# HiGHS is used as the default solver until Gurobi is configured.

# 4. Run data exploration notebook
jupyter notebook notebooks/01_data_exploration.ipynb

# 5. Run the data pipeline
python -m src.data.pipeline --start 2024-01-01 --end 2026-02-01

# 6. Run baselines
python -m src.baselines.run_baselines --test-start 2025-10-01 --test-end 2026-02-01
```

## Project Structure

```
hybridbid/
├── configs/
│   ├── battery.yaml          # Battery parameters (10MW/20MWh reference)
│   └── data_products.yaml    # ERCOT data product IDs and access config
├── data/
│   ├── raw/                  # Downloaded ERCOT files
│   ├── processed/            # Clean Parquet files (canonical schema)
│   └── mappings/             # ESR combo-to-single model mapping
├── notebooks/
│   └── 01_data_exploration.ipynb
├── src/
│   ├── data/
│   │   ├── pipeline.py       # Main ingestion orchestrator
│   │   ├── ercot_fetcher.py  # gridstatus wrapper (API + scraping)
│   │   ├── schema.py         # Canonical Parquet schema definitions
│   │   └── preprocessing.py  # Cleaning, alignment, resampling
│   ├── baselines/
│   │   ├── tbx.py            # Time-based arbitrage baseline
│   │   ├── perfect_foresight.py  # Energy-only MIP (upper bound)
│   │   └── run_baselines.py  # CLI runner
│   ├── evaluation/
│   │   ├── metrics.py        # Revenue, TB2 capture, constraint compliance
│   │   └── visualization.py  # Plots: revenue curves, SoC trajectories
│   └── utils/
│       ├── time_utils.py     # CPT/UTC conversion, ERCOT hour-ending
│       └── battery_sim.py    # Battery state simulator
├── tests/
│   └── test_battery_sim.py
├── requirements.txt
└── README.md
```

## Week 1 Implementation Plan

| Day | Focus | Deliverable |
|-----|-------|-------------|
| 1 | Environment + data exploration | Exploration notebook with sample data from all products |
| 2 | Data audit + schema design | Schema spec, gap analysis |
| 3 | Build ingestion pipeline | Working pipeline producing Parquet files |
| 4 | TBx + energy-only perfect foresight | Baseline revenue numbers |
| 5 | Evaluation framework + Go/No-Go | Validated baselines, metrics, visualizations |

## Key Design Decisions

- **Schema:** 5-minute canonical timestamp index (UTC). Pre-RTC+B AS columns are NaN.
- **Solver:** HiGHS (default) → Gurobi (once academic license is set up).
- **Perfect foresight:** Energy-only first (Week 1), add AS co-optimization in Week 2.
- **Data access:** `gridstatus` API for 2023+, web scraping for 2020-2023.
