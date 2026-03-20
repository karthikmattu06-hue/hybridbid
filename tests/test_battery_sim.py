"""
Tests for battery simulator.

Run: pytest tests/test_battery_sim.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.utils.battery_sim import BatteryAction, BatteryParams, BatterySimulator


@pytest.fixture
def params():
    return BatteryParams(
        power_max_mw=10.0,
        energy_max_mwh=20.0,
        soc_min_frac=0.10,
        soc_max_frac=0.90,
        soc_initial_frac=0.50,
        eta_charge=0.92,
        eta_discharge=0.92,
        degradation_cost_per_mwh=2.0,
    )


@pytest.fixture
def sim(params):
    return BatterySimulator(params)


class TestBatteryParams:
    def test_derived_properties(self, params):
        assert params.soc_min_mwh == pytest.approx(2.0)
        assert params.soc_max_mwh == pytest.approx(18.0)
        assert params.soc_initial_mwh == pytest.approx(10.0)
        assert params.usable_energy_mwh == pytest.approx(16.0)


class TestBatterySimulator:
    def test_initial_state(self, sim):
        assert sim.soc_mwh == pytest.approx(10.0)
        assert sim.state["soc_frac"] == pytest.approx(0.5)

    def test_idle_step(self, sim):
        result = sim.step(BatteryAction(), energy_price=50.0)
        assert result.soc_after_mwh == pytest.approx(10.0)
        assert result.energy_revenue_usd == pytest.approx(0.0)
        assert result.net_revenue_usd == pytest.approx(0.0)
        assert len(result.constraint_violations) == 0

    def test_charge_step(self, sim, params):
        action = BatteryAction(p_charge_mw=10.0)
        result = sim.step(action, energy_price=30.0)

        # SoC should increase: 10 + 0.92 * 10 * (5/60) = 10.767 MWh
        dt = 5.0 / 60.0
        expected_soc = 10.0 + params.eta_charge * 10.0 * dt
        assert result.soc_after_mwh == pytest.approx(expected_soc, abs=0.01)

        # Revenue should be negative (buying energy)
        # p_net = 0 - 10 = -10 MW, revenue = -10 * 30 * dt
        expected_energy_rev = -10.0 * 30.0 * dt
        assert result.energy_revenue_usd == pytest.approx(expected_energy_rev, abs=0.01)

    def test_discharge_step(self, sim, params):
        action = BatteryAction(p_discharge_mw=10.0)
        result = sim.step(action, energy_price=100.0)

        # SoC should decrease
        dt = 5.0 / 60.0
        expected_soc = 10.0 - (10.0 / params.eta_discharge) * dt
        assert result.soc_after_mwh == pytest.approx(expected_soc, abs=0.01)

        # Revenue should be positive (selling energy)
        expected_energy_rev = 10.0 * 100.0 * dt
        assert result.energy_revenue_usd == pytest.approx(expected_energy_rev, abs=0.01)

    def test_mutual_exclusivity_violation(self, sim):
        action = BatteryAction(p_charge_mw=5.0, p_discharge_mw=5.0)
        result = sim.step(action, energy_price=50.0)
        assert "mutual_exclusivity" in result.constraint_violations

    def test_soc_min_enforcement(self, params):
        # Start near min SoC
        sim = BatterySimulator(params)
        sim.soc_mwh = 2.5  # Just above min (2.0)

        # Try to discharge more than available
        action = BatteryAction(p_discharge_mw=10.0)
        result = sim.step(action, energy_price=100.0)

        # SoC should be clipped to min
        assert result.soc_after_mwh >= params.soc_min_mwh

    def test_soc_max_enforcement(self, params):
        sim = BatterySimulator(params)
        sim.soc_mwh = 17.5  # Just below max (18.0)

        action = BatteryAction(p_charge_mw=10.0)
        result = sim.step(action, energy_price=20.0)

        assert result.soc_after_mwh <= params.soc_max_mwh

    def test_degradation_cost(self, sim, params):
        dt = 5.0 / 60.0
        action = BatteryAction(p_discharge_mw=10.0)
        result = sim.step(action, energy_price=100.0)

        expected_deg = params.degradation_cost_per_mwh * 10.0 * dt
        assert result.degradation_cost_usd == pytest.approx(expected_deg, abs=0.01)

    def test_history_tracking(self, sim):
        for i in range(10):
            sim.step(BatteryAction(p_discharge_mw=5.0), energy_price=50.0)

        df = sim.get_history_df()
        assert len(df) == 10
        assert "net_revenue_usd" in df.columns
        assert "soc_after_mwh" in df.columns

    def test_reset(self, sim):
        sim.step(BatteryAction(p_discharge_mw=10.0), energy_price=100.0)
        assert sim.soc_mwh != sim.params.soc_initial_mwh

        sim.reset()
        assert sim.soc_mwh == pytest.approx(sim.params.soc_initial_mwh)
        assert len(sim.history) == 0

    def test_full_cycle_revenue(self, params):
        """A full charge-then-discharge cycle should be profitable if spread > costs."""
        sim = BatterySimulator(params)
        dt = 5.0 / 60.0
        n_intervals = 24  # 2 hours

        # Charge at $20/MWh for 2 hours
        for _ in range(n_intervals):
            sim.step(BatteryAction(p_charge_mw=10.0), energy_price=20.0)

        # Discharge at $80/MWh for 2 hours
        for _ in range(n_intervals):
            sim.step(BatteryAction(p_discharge_mw=10.0), energy_price=80.0)

        df = sim.get_history_df()
        total_rev = df["net_revenue_usd"].sum()

        # With $60/MWh spread, ~84.6% round-trip efficiency, minus degradation:
        # should be meaningfully positive
        assert total_rev > 0, f"Full cycle should be profitable, got ${total_rev:.2f}"


class TestBatteryParamsFromYaml:
    def test_load_from_yaml(self, tmp_path):
        config = tmp_path / "battery.yaml"
        config.write_text("""
reference_battery:
  power_max_mw: 10.0
  energy_max_mwh: 20.0
  soc_min_frac: 0.10
  soc_max_frac: 0.90
  soc_initial_frac: 0.50
  eta_charge: 0.92
  eta_discharge: 0.92
  degradation_cost_per_mwh: 2.0
""")
        params = BatteryParams.from_yaml(str(config))
        assert params.power_max_mw == 10.0
        assert params.energy_max_mwh == 20.0
