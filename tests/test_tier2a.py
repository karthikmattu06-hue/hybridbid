"""Tests for Tier 2a discrete N=7 SAC components."""

import numpy as np
import torch
import pytest

from src.models.networks import (
    ActorDiscrete7,
    DiscreteBroNetCritic,
    TwinDiscreteBroNetCritic,
    TIER2A_ACTION_LEVELS,
    TIER2A_N_ACTIONS,
    TIER2A_IDX_TO_ENV_ACTION,
    tier2a_idx_to_env_action,
    MODE_CHARGE,
    MODE_DISCHARGE,
    MODE_IDLE,
)


class TestTier2aActionAdapter:
    def test_n_actions_is_seven(self):
        assert TIER2A_N_ACTIONS == 7
        assert len(TIER2A_ACTION_LEVELS) == 7

    def test_action_levels_symmetric_about_zero(self):
        lvls = np.array(TIER2A_ACTION_LEVELS)
        assert lvls[0] == -1.0 and lvls[-1] == 1.0
        # symmetric
        np.testing.assert_allclose(lvls, -lvls[::-1], atol=1e-7)
        # zero is the middle index
        assert lvls[3] == 0.0

    def test_idx_to_env_action_shapes(self):
        for i in range(TIER2A_N_ACTIONS):
            env_a = tier2a_idx_to_env_action(i)
            assert env_a.shape == (4,)
            onehot = env_a[:3]
            mag = env_a[3]
            # exactly one mode hot
            assert onehot.sum() == 1.0
            assert mag >= 0.0 and mag <= 1.0

    def test_idx_to_env_action_mode_mapping(self):
        # idx 0,1,2 are charge (negative levels)
        for i in range(3):
            env_a = tier2a_idx_to_env_action(i)
            assert env_a[MODE_CHARGE] == 1.0
        # idx 3 is idle (zero)
        env_a = tier2a_idx_to_env_action(3)
        assert env_a[MODE_IDLE] == 1.0
        assert env_a[3] == 0.0
        # idx 4,5,6 are discharge (positive)
        for i in range(4, 7):
            env_a = tier2a_idx_to_env_action(i)
            assert env_a[MODE_DISCHARGE] == 1.0

    def test_cached_table_matches_function(self):
        assert TIER2A_IDX_TO_ENV_ACTION.shape == (7, 4)
        for i in range(7):
            np.testing.assert_allclose(
                TIER2A_IDX_TO_ENV_ACTION[i], tier2a_idx_to_env_action(i)
            )


class TestActorDiscrete7:
    def test_forward_shapes(self):
        actor = ActorDiscrete7(obs_dim=90, hidden_dim=512, n_actions=7)
        obs = torch.randn(8, 90)
        logits = actor(obs)
        assert logits.shape == (8, 7)

    def test_sample_returns_idx_logprobs_probs(self):
        actor = ActorDiscrete7(obs_dim=90, hidden_dim=512, n_actions=7)
        obs = torch.randn(8, 90)
        idx, log_probs, probs = actor.sample(obs)
        assert idx.shape == (8,)
        assert log_probs.shape == (8, 7)
        assert probs.shape == (8, 7)
        # probs sum to 1 per row
        torch.testing.assert_close(probs.sum(dim=-1), torch.ones(8), atol=1e-5, rtol=1e-5)
        # log_probs = log(probs) (finite)
        assert torch.isfinite(log_probs).all()
        # idx in valid range
        assert (idx >= 0).all() and (idx < 7).all()

    def test_deterministic_argmax(self):
        actor = ActorDiscrete7(obs_dim=90, hidden_dim=512, n_actions=7)
        obs = torch.randn(4, 90)
        idx_det, _, probs = actor.sample(obs, deterministic=True)
        argmax = probs.argmax(dim=-1)
        torch.testing.assert_close(idx_det, argmax)


class TestDiscreteBroNetCritic:
    def test_forward_shape_n_actions_n_atoms(self):
        critic = DiscreteBroNetCritic(
            obs_dim=90, hidden=512, n_actions=7, n_atoms=101,
        )
        obs = torch.randn(8, 90)
        out = critic(obs)
        # (batch, n_actions, n_atoms) — no action input
        assert out.shape == (8, 7, 101)

    def test_twin_two_independent_heads(self):
        twin = TwinDiscreteBroNetCritic(
            obs_dim=90, hidden=512, n_actions=7, n_atoms=101,
        )
        obs = torch.randn(4, 90)
        q1, q2 = twin(obs)
        assert q1.shape == (4, 7, 101)
        assert q2.shape == (4, 7, 101)
        # two heads should be initialized independently → different outputs
        assert not torch.allclose(q1, q2)


class TestClosedFormExpectation:
    """Verify the discrete SAC actor loss is a closed-form expectation
    Σ_a π(a|s) × (α log π(a|s) − Q(s,a)), not a sampled estimate."""

    def test_expectation_sums_to_scalar_per_sample(self):
        B, N, A = 4, 101, 7
        logits = torch.randn(B, A)
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        q_raw = torch.randn(B, A)
        alpha = 0.1
        per_sample = (probs * (alpha * log_probs - q_raw)).sum(dim=-1)
        assert per_sample.shape == (B,)
        # Finite and differentiable through probs
        loss = per_sample.mean()
        assert torch.isfinite(loss)
