"""
Locked evaluation harness for Stage 1 3-day sprint.
DO NOT MODIFY. This file is the fixed yardstick.

Wraps src.evaluation.evaluate_stage1.evaluate() with a reproducible
multi-seed interface and machine-parseable RESULT line output.

Seeding note: the env is data-driven (no stochasticity) and evaluate()
uses deterministic=True, so all 5 seeds produce identical numbers.
The multi-seed IQM infrastructure is kept for interface compatibility
and future use with stochastic evaluation modes.

Fixed constants:
  - Test range: 2025-10-01 → 2025-12-04 (in evaluate_stage1.py)
  - TBx baseline: $870/day (pre-RTC+B, CLAUDE.md)
  - Eval seeds: [10, 11, 12, 13, 14]
  - IQM: mean of middle 3 of 5 sorted seed results
"""

import argparse
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.evaluate_stage1 import evaluate
from src.training.config import Stage1Config, Stage1V60Config

# ── Fixed constants — never change these ──
EVAL_SEEDS = [10, 11, 12, 13, 14]
TBX_BASELINE_DAILY = 870.0   # $/day, pre-RTC+B (CLAUDE.md)


def evaluate_multi_seed(
    checkpoint_path: str,
    config: Stage1Config,
    verbose_first: bool = True,
) -> dict:
    """
    Run evaluation over all EVAL_SEEDS and return IQM summary.

    Because the env is deterministic and select_action uses deterministic=True,
    all seeds return the same numbers. The loop is kept so that if stochastic
    modes are introduced later, the harness requires no changes.

    IQM with n=5: drop sorted[0] (min) and sorted[4] (max), mean of middle 3.
    """
    revenues = []
    soc_violations_total = 0
    num_eval_days = 0

    for i, seed in enumerate(EVAL_SEEDS):
        np.random.seed(seed)
        result = evaluate(checkpoint_path, config=config, verbose=(i == 0 and verbose_first))
        revenues.append(result["avg_daily_revenue"])
        # Deterministic eval: violations/n_days are identical across seeds; take first seed.
        if i == 0:
            soc_violations_total = int(result["soc_violations"])
            num_eval_days = int(result["n_days"])

    revenues_sorted = sorted(revenues)
    iqm = float(np.mean(revenues_sorted[1:-1]))   # drop min and max
    # Primary metric: net_return = gross − 50 × total_violations (absolute).
    # Penalises dump-and-terminate policies that inflate gross via early termination.
    net_return = iqm - 50.0 * soc_violations_total

    return {
        "iqm_daily_revenue": iqm,
        "net_daily_revenue": net_return,
        "min_daily_revenue": float(min(revenues)),
        "max_daily_revenue": float(max(revenues)),
        "capture_rate": iqm / TBX_BASELINE_DAILY,
        "soc_violations": soc_violations_total,
        "n_days": num_eval_days,
        "per_seed_revenues": revenues,
    }


def report_result(checkpoint_path: str, experiment_name: str, config: Stage1Config) -> dict:
    """
    Evaluate checkpoint and print the machine-parseable RESULT line.

    Format (always one line, always last line of stdout):
      RESULT experiment=<name> iqm_return=<X.XX> net_return=<X.XX> capture=<Y.YY> violations=<N> min=<A.AA> max=<B.BB>

    net_return = iqm_return − 50 × (violations / n_days). This is the primary
    sprint metric; iqm_return is kept for backward compatibility with baselines
    measured before the net_return patch.
    """
    summary = evaluate_multi_seed(checkpoint_path, config)

    print(
        f"RESULT experiment={experiment_name} "
        f"iqm_return={summary['iqm_daily_revenue']:.2f} "
        f"net_return={summary['net_daily_revenue']:.2f} "
        f"capture={summary['capture_rate']:.4f} "
        f"violations={summary['soc_violations']} "
        f"min={summary['min_daily_revenue']:.2f} "
        f"max={summary['max_daily_revenue']:.2f}"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Locked eval harness — DO NOT MODIFY"
    )
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--experiment", required=True, help="Experiment name for RESULT line")
    parser.add_argument(
        "--v60", action="store_true",
        help="Use Stage1V60Config (enriched 36-dim TTFE obs, obs_dim=108)",
    )
    parser.add_argument("--device", default=None, help="Override compute device")
    args = parser.parse_args()

    cfg = Stage1V60Config() if args.v60 else Stage1Config()
    if args.device:
        cfg.device = args.device

    report_result(args.checkpoint, args.experiment, cfg)
