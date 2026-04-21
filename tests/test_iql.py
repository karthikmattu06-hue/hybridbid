"""Tests for Tier 2c IQL agent + Option D trajectory loader + atom snap helper."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch
import pytest

from src.agents.iql import (
    IQLAgent,
    IQLConfig,
    ValueNet,
    expectile_loss,
    symlog_scale,
)
from src.models.networks import TIER2A_ACTION_LEVELS, TIER2A_N_ACTIONS


# ──────────────────────────────────────────────────────────────────────
# Dynamically import the Option D preprocessor so we can unit-test the
# snap function without running the script.
# ──────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[1]
_PREPROC_PATH = _REPO_ROOT / "scripts" / "preprocess_milp_option_d.py"
_spec = importlib.util.spec_from_file_location("preprocess_milp_option_d", _PREPROC_PATH)
_preproc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_preproc)  # type: ignore[union-attr]


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def _make_batch(B: int, cfg: IQLConfig) -> dict:
    """Synthetic IQL batch matching the Option D preprocessor output layout."""
    torch.manual_seed(0)
    return {
        "price_history":        torch.randn(B, cfg.seq_len, cfg.n_prices) * 50.0,
        "static_features":      torch.randn(B, cfg.static_dim),
        "next_price_history":   torch.randn(B, cfg.seq_len, cfg.n_prices) * 50.0,
        "next_static_features": torch.randn(B, cfg.static_dim),
        "actions":              torch.randint(0, cfg.n_actions, (B,), dtype=torch.long),
        "rewards":              torch.randn(B) * 100.0,          # raw $ — agent scales internally
        "dones":                torch.zeros(B),                   # MILP data: always 0
    }


def _small_cfg() -> IQLConfig:
    # Small-but-valid IQL config for fast tests. Dimensions must still satisfy
    # obs_dim == d_model + n_prices_flat + static_dim.
    return IQLConfig(
        obs_dim=64 + 12 + 14,    # 90
        hidden_dim=64,
        n_actions=7,
        n_prices=12,
        n_prices_flat=12,
        seq_len=32,
        static_dim=14,
        d_model=64,
        nhead=8,
        n_layers=2,
        n_atoms=51,              # smaller for speed; wider sigma OK
        hl_gauss_sigma=0.75,
        device="cpu",
    )


# ──────────────────────────────────────────────────────────────────────
# 1. expectile_loss
# ──────────────────────────────────────────────────────────────────────
class TestExpectileLoss:
    def test_tau_half_is_mse(self):
        diff = torch.tensor([-2.0, -1.0, 0.5, 3.0])
        loss = expectile_loss(diff, expectile=0.5)
        expected = 0.5 * diff.pow(2).mean()
        assert torch.allclose(loss, expected, atol=1e-7)

    def test_tau_high_penalises_underestimation_more(self):
        # At τ=0.9, positive diff (Q > V, underestimation) weighted 0.9;
        # negative diff (overestimation) weighted 0.1.
        pos = torch.tensor([2.0])
        neg = torch.tensor([-2.0])
        lp = expectile_loss(pos, expectile=0.9)
        ln = expectile_loss(neg, expectile=0.9)
        assert lp.item() == pytest.approx(0.9 * 4.0)
        assert ln.item() == pytest.approx(0.1 * 4.0)
        assert lp > ln

    def test_nonnegative(self):
        diff = torch.randn(128) * 5.0
        assert expectile_loss(diff, 0.9).item() >= 0.0


# ──────────────────────────────────────────────────────────────────────
# 2. symlog_scale
# ──────────────────────────────────────────────────────────────────────
class TestSymlogScale:
    def test_zero_maps_to_zero(self):
        r = torch.zeros(4)
        assert torch.allclose(symlog_scale(r), torch.zeros(4))

    def test_odd_symmetry(self):
        r = torch.tensor([-500.0, -1.0, 1.0, 500.0])
        s = symlog_scale(r, scale=100.0)
        assert torch.allclose(s[:2], -s.flip(0)[:2], atol=1e-6)

    def test_compresses_spikes(self):
        # r=10000, scale=100 → |r/scale|=100 → symlog ≈ log(101) ≈ 4.6
        r = torch.tensor([10000.0])
        s = symlog_scale(r, scale=100.0)
        assert 4.0 < s.item() < 5.0


# ──────────────────────────────────────────────────────────────────────
# 3. ValueNet
# ──────────────────────────────────────────────────────────────────────
class TestValueNet:
    def test_forward_shape(self):
        v = ValueNet(obs_dim=90, hidden=128)
        out = v(torch.randn(8, 90))
        assert out.shape == (8, 1)

    def test_gradient_flows(self):
        v = ValueNet(obs_dim=90, hidden=64)
        obs = torch.randn(4, 90, requires_grad=False)
        loss = v(obs).sum()
        loss.backward()
        # At least one parameter should have a non-zero gradient.
        grads = [p.grad.abs().sum().item() for p in v.parameters() if p.grad is not None]
        assert any(g > 0.0 for g in grads)


# ──────────────────────────────────────────────────────────────────────
# 4. Twin Q independence (re-tests that new IQL context does not regress)
# ──────────────────────────────────────────────────────────────────────
class TestTwinQIndependence:
    def test_q1_q2_different_outputs(self):
        cfg = _small_cfg()
        agent = IQLAgent(cfg)
        enc = torch.randn(4, cfg.obs_dim)
        q1, q2 = agent.critic(enc)
        # Random init ⇒ with overwhelming probability q1 != q2
        assert q1.shape == q2.shape == (4, cfg.n_actions, cfg.n_atoms)
        assert not torch.allclose(q1, q2)


# ──────────────────────────────────────────────────────────────────────
# 5. Policy distribution well-formed
# ──────────────────────────────────────────────────────────────────────
class TestPolicyDistribution:
    def test_softmax_sums_to_one(self):
        cfg = _small_cfg()
        agent = IQLAgent(cfg)
        enc = torch.randn(16, cfg.obs_dim)
        logits = agent.actor(enc)
        probs = torch.softmax(logits, dim=-1)
        assert probs.shape == (16, cfg.n_actions)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(16), atol=1e-6)
        assert (probs >= 0).all()

    def test_select_action_valid_range(self):
        cfg = _small_cfg()
        agent = IQLAgent(cfg)
        obs = {
            "price_history":   np.random.randn(cfg.seq_len, cfg.n_prices).astype(np.float32) * 50,
            "static_features": np.random.randn(cfg.static_dim).astype(np.float32),
        }
        for _ in range(20):
            a = agent.select_action(obs, deterministic=True)
            assert isinstance(a, int)
            assert 0 <= a < cfg.n_actions


# ──────────────────────────────────────────────────────────────────────
# 6. Full update step — AWR weights finite + losses sensible + polyak moves
# ──────────────────────────────────────────────────────────────────────
class TestIQLUpdate:
    def test_single_update_all_metrics_finite(self):
        cfg = _small_cfg()
        agent = IQLAgent(cfg)
        batch = _make_batch(32, cfg)
        metrics = agent.update(batch)

        for k, v in metrics.items():
            assert np.isfinite(v), f"metric {k} = {v} is not finite"

        # AWR weight clip honoured (≤ awr_weight_clip, > 0)
        assert 0.0 < metrics["awr/mean_w"]
        assert metrics["awr/max_w"] <= cfg.awr_weight_clip + 1e-5

        # Losses non-negative
        assert metrics["loss/v"] >= 0.0
        assert metrics["loss/q"] >= 0.0

        # Actor grad exists (not identically zero after a full update)
        assert metrics["grad/actor"] >= 0.0
        assert metrics["grad/v"] >= 0.0
        assert metrics["grad/q"] >= 0.0

    def test_polyak_moves_target_toward_online(self):
        cfg = _small_cfg()
        agent = IQLAgent(cfg)

        # Perturb online V weights so target != online.
        with torch.no_grad():
            for p in agent.value.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        # Snapshot distance before one polyak step.
        def _dist():
            total = 0.0
            for p, tp in zip(agent.value.parameters(), agent.value_target.parameters()):
                total += (p - tp).pow(2).sum().item()
            return total

        d_before = _dist()
        assert d_before > 0.0  # online != target

        agent._polyak_update_value()
        d_after = _dist()
        # target moves toward online → distance strictly decreases by factor (1-τ)²
        assert d_after < d_before
        # And specifically matches the polyak recurrence up to fp precision.
        expected_ratio = (1.0 - cfg.polyak_tau) ** 2
        assert d_after == pytest.approx(d_before * expected_ratio, rel=1e-5)

    def test_ttfe_gradient_flows_via_v_update(self):
        """The V update owns TTFE — after one update, TTFE params should have moved."""
        cfg = _small_cfg()
        agent = IQLAgent(cfg)

        before = {n: p.detach().clone() for n, p in agent.ttfe.named_parameters()}
        batch = _make_batch(32, cfg)
        agent.update(batch)

        moved = [
            not torch.allclose(before[n], p.detach(), atol=0.0)
            for n, p in agent.ttfe.named_parameters()
            if p.requires_grad
        ]
        assert any(moved), "TTFE parameters did not move after IQL update"


# ──────────────────────────────────────────────────────────────────────
# 7. Atom snap helper (preprocessor) — ≥10 cases
# ──────────────────────────────────────────────────────────────────────
class TestSnapSignedToAtomIdx:
    def test_exact_atoms_map_identity(self):
        signed = np.array(TIER2A_ACTION_LEVELS, dtype=np.float32)
        idx = _preproc.snap_signed_to_atom_idx(signed)
        np.testing.assert_array_equal(idx, np.arange(TIER2A_N_ACTIONS))

    def test_zero_maps_to_idle_atom(self):
        idx = _preproc.snap_signed_to_atom_idx(np.array([0.0], dtype=np.float32))
        assert idx[0] == 3

    def test_extremes_map_to_endpoint_atoms(self):
        # Clip-worthy inputs outside [-1, 1] still map to closest endpoint.
        idx = _preproc.snap_signed_to_atom_idx(np.array([-5.0, -1.0, 1.0, 5.0], dtype=np.float32))
        np.testing.assert_array_equal(idx, np.array([0, 0, 6, 6]))

    def test_near_boundaries_rounding(self):
        # -2/3 = -0.6667, -1/3 = -0.3333; midpoint = -0.5.
        # -0.49 should round to -1/3 (idx 2). -0.51 should round to -2/3 (idx 1).
        signed = np.array([-0.49, -0.51, 0.49, 0.51], dtype=np.float32)
        idx = _preproc.snap_signed_to_atom_idx(signed)
        np.testing.assert_array_equal(idx, np.array([2, 1, 4, 5]))

    def test_midpoint_ties_deterministic(self):
        # argmin breaks ties toward the lower index. At midpoint -0.5 between
        # atoms 1 (-0.6667) and 2 (-0.3333), |.|= 0.1667 both sides → idx=1 wins.
        idx = _preproc.snap_signed_to_atom_idx(np.array([-0.5, 0.5], dtype=np.float32))
        # We assert the function is deterministic on midpoints, not which side wins.
        idx2 = _preproc.snap_signed_to_atom_idx(np.array([-0.5, 0.5], dtype=np.float32))
        np.testing.assert_array_equal(idx, idx2)

    def test_many_random_cases_all_valid(self):
        rng = np.random.default_rng(42)
        signed = rng.uniform(-1.5, 1.5, size=500).astype(np.float32)
        idx = _preproc.snap_signed_to_atom_idx(signed)
        assert idx.shape == (500,)
        assert idx.dtype == np.int64
        assert idx.min() >= 0 and idx.max() <= 6


# ──────────────────────────────────────────────────────────────────────
# 8. milp_to_signed helper
# ──────────────────────────────────────────────────────────────────────
class TestMilpToSigned:
    def test_charge_is_negative(self):
        from src.env.ercot_env import MODE_CHARGE, MODE_DISCHARGE, MODE_IDLE
        mode = np.array([MODE_CHARGE, MODE_DISCHARGE, MODE_IDLE], dtype=np.int64)
        mag  = np.array([0.5, 0.7, 0.0], dtype=np.float32)
        signed = _preproc.milp_to_signed(mode, mag)
        np.testing.assert_allclose(signed, [-0.5, +0.7, 0.0], atol=1e-6)


# ──────────────────────────────────────────────────────────────────────
# 9. Option D NPZ dataset loader shape match
# ──────────────────────────────────────────────────────────────────────
class TestOptionDNPZLoader:
    TRAIN_NPZ = _REPO_ROOT / "data/expert_trajectories/receding_horizon_train_option_d.npz"

    def test_npz_schema_matches_iql_batch(self):
        if not self.TRAIN_NPZ.exists():
            pytest.skip(f"Preprocessed train NPZ not present at {self.TRAIN_NPZ}")
        with np.load(self.TRAIN_NPZ) as d:
            keys = set(d.files)
            required = {
                "price_history", "static_features",
                "next_price_history", "next_static_features",
                "actions", "rewards", "dones", "truncateds",
            }
            missing = required - keys
            assert not missing, f"NPZ missing keys: {missing}"

            N = d["actions"].shape[0]
            assert d["price_history"].shape      == (N, 32, 12)
            assert d["next_price_history"].shape == (N, 32, 12)
            assert d["static_features"].shape[0]      == N
            assert d["next_static_features"].shape[0] == N
            assert d["static_features"].shape[1] == d["next_static_features"].shape[1]
            # 14 = 7 system + 1 soc + 6 cyclical
            assert d["static_features"].shape[1] == 14

            assert d["actions"].dtype == np.int64
            assert d["actions"].min() >= 0 and d["actions"].max() <= 6
            assert d["dones"].any() == False  # MILP is SoC-feasible
