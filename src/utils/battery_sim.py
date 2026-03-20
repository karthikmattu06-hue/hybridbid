"""
Battery state simulator for backtesting bidding strategies.

Simulates physical battery operation at 5-minute intervals:
  - Tracks State of Charge (SoC) with efficiency losses
  - Enforces power limits, SoC bounds, ramp rates, mutual exclusivity
  - Computes revenue from energy arbitrage and AS capacity payments
  - Logs constraint violations for compliance checking

Used by both baselines and the full HybridBid system for evaluation.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import yaml


@dataclass
class BatteryParams:
    """Battery physical and operational parameters."""

    power_max_mw: float = 10.0
    energy_max_mwh: float = 20.0
    soc_min_frac: float = 0.10
    soc_max_frac: float = 0.90
    soc_initial_frac: float = 0.50
    eta_charge: float = 0.92
    eta_discharge: float = 0.92
    degradation_cost_per_mwh: float = 2.0
    ramp_rate_mw_per_min: float = 10.0

    @property
    def soc_min_mwh(self) -> float:
        return self.soc_min_frac * self.energy_max_mwh

    @property
    def soc_max_mwh(self) -> float:
        return self.soc_max_frac * self.energy_max_mwh

    @property
    def soc_initial_mwh(self) -> float:
        return self.soc_initial_frac * self.energy_max_mwh

    @property
    def usable_energy_mwh(self) -> float:
        return self.soc_max_mwh - self.soc_min_mwh

    @classmethod
    def from_yaml(cls, path: str, scenario: str = "reference_battery") -> "BatteryParams":
        """Load battery parameters from YAML config file."""
        with open(path) as f:
            config = yaml.safe_load(f)
        params = config[scenario]
        return cls(**{k: v for k, v in params.items() if k in cls.__dataclass_fields__})


@dataclass
class BatteryAction:
    """Action to take at a single 5-minute interval."""

    p_charge_mw: float = 0.0       # Charging power [MW] (>= 0)
    p_discharge_mw: float = 0.0    # Discharging power [MW] (>= 0)
    # AS capacity offers [MW] (all >= 0, for future use)
    as_regup_mw: float = 0.0
    as_regdown_mw: float = 0.0
    as_rrs_mw: float = 0.0
    as_ecrs_mw: float = 0.0
    as_nsrs_mw: float = 0.0


@dataclass
class StepResult:
    """Result of simulating one 5-minute interval."""

    timestamp: pd.Timestamp
    soc_before_mwh: float
    soc_after_mwh: float
    p_charge_mw: float
    p_discharge_mw: float
    p_net_mw: float               # discharge - charge (positive = injecting)
    energy_revenue_usd: float
    as_revenue_usd: float
    degradation_cost_usd: float
    net_revenue_usd: float
    constraint_violations: list = field(default_factory=list)


class BatterySimulator:
    """
    Simulates battery operation over a sequence of 5-minute intervals.

    Usage:
        sim = BatterySimulator(params)
        for t, prices in enumerate(price_series):
            action = my_strategy.decide(sim.state, prices)
            result = sim.step(action, prices)
    """

    DELTA_T_HOURS = 5.0 / 60.0  # 5 minutes in hours

    def __init__(self, params: BatteryParams):
        self.params = params
        self.soc_mwh = params.soc_initial_mwh
        self.prev_p_net_mw = 0.0
        self.history: list[StepResult] = []

    def reset(self, soc_mwh: Optional[float] = None):
        """Reset simulator to initial state."""
        self.soc_mwh = soc_mwh if soc_mwh is not None else self.params.soc_initial_mwh
        self.prev_p_net_mw = 0.0
        self.history = []

    @property
    def state(self) -> dict:
        """Current battery state for strategy decision-making."""
        return {
            "soc_mwh": self.soc_mwh,
            "soc_frac": self.soc_mwh / self.params.energy_max_mwh,
            "prev_p_net_mw": self.prev_p_net_mw,
            "power_max_mw": self.params.power_max_mw,
            "soc_min_mwh": self.params.soc_min_mwh,
            "soc_max_mwh": self.params.soc_max_mwh,
        }

    def step(
        self,
        action: BatteryAction,
        energy_price: float,
        as_prices: Optional[dict[str, float]] = None,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> StepResult:
        """
        Simulate one 5-minute interval.

        Parameters
        ----------
        action : BatteryAction
            Charge/discharge power and AS offers.
        energy_price : float
            Energy price [$/MWh] for this interval.
        as_prices : dict, optional
            AS clearing prices {'regup': $/MW, 'regdown': $/MW, ...}.
        timestamp : pd.Timestamp, optional
            Timestamp for logging.

        Returns
        -------
        StepResult
            Detailed result of the interval.
        """
        violations = []
        p = self.params
        dt = self.DELTA_T_HOURS

        # ── Clip and validate action ──
        p_ch = max(0.0, min(action.p_charge_mw, p.power_max_mw))
        p_dch = max(0.0, min(action.p_discharge_mw, p.power_max_mw))

        # Mutual exclusivity: can't charge and discharge simultaneously
        if p_ch > 0 and p_dch > 0:
            violations.append("mutual_exclusivity")
            # Resolve: keep the larger one
            if p_ch >= p_dch:
                p_dch = 0.0
            else:
                p_ch = 0.0

        # Ramp rate check
        p_net = p_dch - p_ch
        ramp = abs(p_net - self.prev_p_net_mw)
        max_ramp = p.ramp_rate_mw_per_min * 5.0  # 5-minute interval
        if ramp > max_ramp * 1.01:  # 1% tolerance
            violations.append(f"ramp_rate:{ramp:.2f}>{max_ramp:.2f}")

        # ── SoC dynamics ──
        soc_before = self.soc_mwh
        energy_in = p.eta_charge * p_ch * dt
        energy_out = (p_dch / p.eta_discharge) * dt
        soc_after = soc_before + energy_in - energy_out

        # SoC limit enforcement (clip and log violation)
        if soc_after < p.soc_min_mwh - 0.001:
            violations.append(f"soc_min:{soc_after:.3f}<{p.soc_min_mwh:.3f}")
            # Reduce action to stay within bounds
            soc_after = p.soc_min_mwh
        if soc_after > p.soc_max_mwh + 0.001:
            violations.append(f"soc_max:{soc_after:.3f}>{p.soc_max_mwh:.3f}")
            soc_after = p.soc_max_mwh

        # ── Revenue calculation ──
        # Energy revenue: net injection * price * interval duration
        energy_rev = p_net * energy_price * dt

        # AS revenue: capacity offered * AS price * interval duration
        as_rev = 0.0
        if as_prices:
            as_rev += action.as_regup_mw * as_prices.get("regup", 0.0) * dt
            as_rev += action.as_regdown_mw * as_prices.get("regdown", 0.0) * dt
            as_rev += action.as_rrs_mw * as_prices.get("rrs", 0.0) * dt
            as_rev += action.as_ecrs_mw * as_prices.get("ecrs", 0.0) * dt
            as_rev += action.as_nsrs_mw * as_prices.get("nsrs", 0.0) * dt

        # Degradation cost: throughput-based
        throughput = (p_ch + p_dch) * dt
        deg_cost = p.degradation_cost_per_mwh * throughput

        net_rev = energy_rev + as_rev - deg_cost

        # ── Update state ──
        self.soc_mwh = soc_after
        self.prev_p_net_mw = p_net

        result = StepResult(
            timestamp=timestamp,
            soc_before_mwh=soc_before,
            soc_after_mwh=soc_after,
            p_charge_mw=p_ch,
            p_discharge_mw=p_dch,
            p_net_mw=p_net,
            energy_revenue_usd=energy_rev,
            as_revenue_usd=as_rev,
            degradation_cost_usd=deg_cost,
            net_revenue_usd=net_rev,
            constraint_violations=violations,
        )
        self.history.append(result)
        return result

    def get_history_df(self) -> pd.DataFrame:
        """Convert simulation history to a DataFrame."""
        if not self.history:
            return pd.DataFrame()

        records = []
        for r in self.history:
            records.append({
                "timestamp": r.timestamp,
                "soc_before_mwh": r.soc_before_mwh,
                "soc_after_mwh": r.soc_after_mwh,
                "p_charge_mw": r.p_charge_mw,
                "p_discharge_mw": r.p_discharge_mw,
                "p_net_mw": r.p_net_mw,
                "energy_revenue_usd": r.energy_revenue_usd,
                "as_revenue_usd": r.as_revenue_usd,
                "degradation_cost_usd": r.degradation_cost_usd,
                "net_revenue_usd": r.net_revenue_usd,
                "n_violations": len(r.constraint_violations),
                "violations": "|".join(r.constraint_violations) if r.constraint_violations else "",
            })

        df = pd.DataFrame(records)
        if "timestamp" in df.columns and df["timestamp"].notna().any():
            df = df.set_index("timestamp")
        return df
