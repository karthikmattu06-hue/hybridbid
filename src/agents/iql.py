"""
Implicit Q-Learning (IQL) agent for Tier 2c (offline learning on MILP trajectories).

Reference: Kostrikov, Nair, Levine. "Offline Reinforcement Learning with Implicit
Q-Learning" (ICLR 2022, arXiv:2110.06169).

Design choices (paper-standard, no tuning):
  - Value V(s): scalar head, BroNet trunk, expectile regression with τ = 0.9
  - Twin Q(s): TwinDiscreteBroNetCritic from Tier 2a — (B, N, n_atoms) HL-Gauss
  - Policy π(a|s): ActorDiscrete7 from Tier 2a — categorical over 7 atoms
  - Advantage-weighted BC: loss_π = -E[exp(β·A(s,a)).clamp(max=100) · log π(a|s)]
  - β_advantage = 5.0, γ = 0.97, lr = 3e-4 (AdamW, weight_decay = 0.01), batch 256
  - Polyak τ = 0.005 on **target V only** (Q has no target)

Reward pre-scaling (reused from Tier 1 / Tier 2a stack):
    r_scaled = sign(r/100) * log1p(|r/100|)      # symlog(r / 100)
Everything else (Q, V, bootstrap targets) lives in symlog-space.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.networks import (
    ActorDiscrete7,
    TwinDiscreteBroNetCritic,
    HLGaussLoss,
    TIER2A_N_ACTIONS,
)
from src.models.ttfe import TTFE


# ─────────────────────────────────────────────────────────────────────────────
# Value network — scalar head, BroNet trunk (no HL-Gauss, V is symlog-scalar)
# ─────────────────────────────────────────────────────────────────────────────
class ValueNet(nn.Module):
    """Scalar V(s) in symlog space. BroNet-style trunk."""

    def __init__(self, obs_dim: int = 90, hidden: int = 512):
        super().__init__()
        self.entry = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.block1 = self._make_block(hidden)
        self.block2 = self._make_block(hidden)
        self.head = nn.Linear(hidden, 1)

    @staticmethod
    def _make_block(hidden: int) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns scalar V(s) of shape (batch, 1) in symlog space."""
        h = self.entry(obs)
        h = h + self.block1(h)
        h = h + self.block2(h)
        return self.head(h)


# ─────────────────────────────────────────────────────────────────────────────
# Loss helpers
# ─────────────────────────────────────────────────────────────────────────────
def expectile_loss(diff: torch.Tensor, expectile: float = 0.9) -> torch.Tensor:
    """L_τ(u) = |τ - 1{u<0}| · u². diff = Q(s,a) - V(s) in symlog-space."""
    weight = torch.where(diff > 0, expectile, 1.0 - expectile)
    return (weight * diff.pow(2)).mean()


def symlog_scale(r: torch.Tensor, scale: float = 100.0) -> torch.Tensor:
    """symlog(r / scale) — same as Tier 1 / Tier 2a reward pre-scale."""
    r_scaled = r / scale
    return torch.sign(r_scaled) * torch.log1p(torch.abs(r_scaled))


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class IQLConfig:
    obs_dim: int = 90
    hidden_dim: int = 512
    n_actions: int = TIER2A_N_ACTIONS      # 7
    n_prices: int = 12
    n_prices_flat: int = 12
    seq_len: int = 32
    static_dim: int = 14
    d_model: int = 64
    nhead: int = 8
    n_layers: int = 2

    # IQL hyperparameters (paper-standard)
    gamma: float = 0.97
    expectile_tau: float = 0.9
    beta_advantage: float = 5.0
    awr_weight_clip: float = 100.0
    polyak_tau: float = 0.005

    # Optimizer
    lr: float = 3e-4
    weight_decay: float = 0.01

    # HL-Gauss for Q
    n_atoms: int = 101
    hl_gauss_min: float = -20.0
    hl_gauss_max: float = 20.0
    hl_gauss_sigma: float = 0.75

    # Reward pre-scale
    reward_scale: float = 100.0

    # Gradient clipping (safety belt)
    max_grad_norm: float = 1.0

    device: str = "cpu"


class IQLAgent:
    """IQL agent with TTFE encoder, Twin Q, V + target V, and a 7-way policy."""

    PRICE_NORM = 1000.0  # same as SACAgent — scales raw ERCOT prices into attention-safe range

    def __init__(self, cfg: IQLConfig):
        self.cfg = cfg
        self.device = cfg.device

        # TTFE encoder — same architecture as Tier 1/2a
        self.ttfe = TTFE(
            n_prices=cfg.n_prices,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            n_layers=cfg.n_layers,
            seq_len=cfg.seq_len,
        ).to(self.device)

        # Policy (categorical over 7 atoms)
        self.actor = ActorDiscrete7(
            obs_dim=cfg.obs_dim, hidden_dim=cfg.hidden_dim, n_actions=cfg.n_actions,
        ).to(self.device)

        # Twin Q (s) → (B, N, n_atoms), HL-Gauss
        self.critic = TwinDiscreteBroNetCritic(
            obs_dim=cfg.obs_dim, n_actions=cfg.n_actions,
            hidden=cfg.hidden_dim, n_atoms=cfg.n_atoms,
        ).to(self.device)

        # Value + polyak target
        self.value = ValueNet(obs_dim=cfg.obs_dim, hidden=cfg.hidden_dim).to(self.device)
        self.value_target = copy.deepcopy(self.value).to(self.device)
        for p in self.value_target.parameters():
            p.requires_grad = False

        # HL-Gauss loss helper (symlog-space support)
        self.hl_gauss = HLGaussLoss(
            min_value=cfg.hl_gauss_min,
            max_value=cfg.hl_gauss_max,
            n_atoms=cfg.n_atoms,
            sigma=cfg.hl_gauss_sigma,
        ).to(self.device)

        # Optimisers — AdamW paper-standard, wd=0.01
        self.actor_optim  = torch.optim.AdamW(self.actor.parameters(),  lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.critic_optim = torch.optim.AdamW(self.critic.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.value_optim  = torch.optim.AdamW(self.value.parameters(),  lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.ttfe_optim   = torch.optim.AdamW(self.ttfe.parameters(),   lr=cfg.lr, weight_decay=cfg.weight_decay)

    # ── Encoding ──
    def _encode_obs(self, price_history: torch.Tensor, static_features: torch.Tensor) -> torch.Tensor:
        """Match SACAgent._encode_obs. Returns (B, obs_dim) = (B, d_model + n_prices_flat + static_dim)."""
        ph_norm = price_history / self.PRICE_NORM
        temporal = self.ttfe(ph_norm)                                      # (B, d_model)
        current_prices = ph_norm[:, -1, :self.cfg.n_prices_flat]           # (B, n_prices_flat)
        return torch.cat([temporal, current_prices, static_features], dim=-1)

    # ── Polyak ──
    def _polyak_update_value(self):
        with torch.no_grad():
            for p, tp in zip(self.value.parameters(), self.value_target.parameters()):
                tp.data.mul_(1.0 - self.cfg.polyak_tau).add_(self.cfg.polyak_tau * p.data)

    # ── Select action (eval) ──
    @torch.no_grad()
    def select_action(self, obs: dict, deterministic: bool = True) -> int:
        """Returns a single int atom idx (0..6) for a dict obs."""
        self.ttfe.eval(); self.actor.eval()
        ph = torch.as_tensor(obs["price_history"], dtype=torch.float32, device=self.device).unsqueeze(0)
        sf = torch.as_tensor(obs["static_features"], dtype=torch.float32, device=self.device).unsqueeze(0)
        encoded = self._encode_obs(ph, sf)
        idx, _, _ = self.actor.sample(encoded, deterministic=deterministic)
        self.ttfe.train(); self.actor.train()
        return int(idx.item())

    # ── Single update ──
    def update(self, batch: dict) -> dict:
        """
        batch keys (all tensors on self.device):
            price_history         (B, seq_len, n_prices)
            static_features       (B, static_dim)
            next_price_history    (B, seq_len, n_prices)
            next_static_features  (B, static_dim)
            actions               (B,) long
            rewards               (B,) float   — raw $ (pre-scale happens here)
            dones                 (B,) float   — 0/1 (MILP data: always 0)

        Returns a dict of scalar metrics.
        """
        ph      = batch["price_history"]
        sf      = batch["static_features"]
        next_ph = batch["next_price_history"]
        next_sf = batch["next_static_features"]
        actions = batch["actions"].long()
        rewards = batch["rewards"].float()
        dones   = batch["dones"].float()

        B = ph.shape[0]
        batch_idx = torch.arange(B, device=self.device)

        # Encode once; detach for V and Q heads so TTFE is owned by actor/V update only.
        # Paper-standard IQL trains all three heads on the same frozen representation
        # in some implementations; here we let V's gradient flow into TTFE (so TTFE is
        # shaped by the bootstrap target too), matching Tier 1/2a's "value-head owns TTFE"
        # choice. Actor / critic use a detached copy.
        encoded_full      = self._encode_obs(ph, sf)
        with torch.no_grad():
            encoded_next  = self._encode_obs(next_ph, next_sf)
        encoded_detached  = encoded_full.detach()

        # Reward pre-scale → symlog space
        r_sl = symlog_scale(rewards, scale=self.cfg.reward_scale)

        # ── 1. V update (expectile regression against min-twin Q on behavior action) ──
        with torch.no_grad():
            q1_logits_all, q2_logits_all = self.critic(encoded_detached)   # (B, N, n_atoms) each
            q1_a = q1_logits_all[batch_idx, actions, :]                    # (B, n_atoms)
            q2_a = q2_logits_all[batch_idx, actions, :]
            # Use symlog-expectation (scalar in symlog space) for both heads
            probs1 = F.softmax(q1_a, dim=-1)
            probs2 = F.softmax(q2_a, dim=-1)
            q1_sl = self.hl_gauss.transform_from_probs(probs1).squeeze(-1) # (B,)
            q2_sl = self.hl_gauss.transform_from_probs(probs2).squeeze(-1)
            q_min_sl = torch.minimum(q1_sl, q2_sl)                         # (B,)

        v_sl = self.value(encoded_full).squeeze(-1)                        # (B,)
        diff = q_min_sl - v_sl
        v_loss = expectile_loss(diff, self.cfg.expectile_tau)

        self.value_optim.zero_grad()
        self.ttfe_optim.zero_grad()
        v_loss.backward()
        v_grad_norm = nn.utils.clip_grad_norm_(self.value.parameters(),  self.cfg.max_grad_norm)
        ttfe_grad_norm = nn.utils.clip_grad_norm_(self.ttfe.parameters(), self.cfg.max_grad_norm)
        self.value_optim.step()
        self.ttfe_optim.step()

        # ── 2. Q update (HL-Gauss fit of r + γ V_target(s') in symlog space) ──
        with torch.no_grad():
            v_next_sl = self.value_target(encoded_next).squeeze(-1)        # (B,)
            td_target_sl = r_sl + self.cfg.gamma * (1.0 - dones) * v_next_sl

        q1_logits_all, q2_logits_all = self.critic(encoded_detached)
        q1_a = q1_logits_all[batch_idx, actions, :]                        # (B, n_atoms)
        q2_a = q2_logits_all[batch_idx, actions, :]
        q_loss = (
            self.hl_gauss.loss(q1_a, td_target_sl).mean()
            + self.hl_gauss.loss(q2_a, td_target_sl).mean()
        )
        self.critic_optim.zero_grad()
        q_loss.backward()
        q_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
        self.critic_optim.step()

        # ── 3. Actor update (advantage-weighted BC) ──
        with torch.no_grad():
            q1_logits_all, q2_logits_all = self.critic(encoded_detached)
            q1_a = q1_logits_all[batch_idx, actions, :]
            q2_a = q2_logits_all[batch_idx, actions, :]
            probs1 = F.softmax(q1_a, dim=-1)
            probs2 = F.softmax(q2_a, dim=-1)
            q1_sl = self.hl_gauss.transform_from_probs(probs1).squeeze(-1)
            q2_sl = self.hl_gauss.transform_from_probs(probs2).squeeze(-1)
            q_min_sl = torch.minimum(q1_sl, q2_sl)
            v_sl_no_grad = self.value(encoded_detached).squeeze(-1)
            adv_sl = q_min_sl - v_sl_no_grad                               # (B,)
            weights = torch.exp(self.cfg.beta_advantage * adv_sl).clamp(max=self.cfg.awr_weight_clip)

        logits = self.actor(encoded_detached)                              # (B, N)
        log_probs_all = F.log_softmax(logits, dim=-1)
        logp_a = log_probs_all[batch_idx, actions]                         # (B,)
        actor_loss = -(weights * logp_a).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
        self.actor_optim.step()

        # ── 4. Polyak update on V target ──
        self._polyak_update_value()

        # Metrics
        with torch.no_grad():
            policy_entropy = -(log_probs_all.exp() * log_probs_all).sum(dim=-1).mean()
            mean_weight = weights.mean()
            max_weight  = weights.max()

        return {
            "loss/v":         float(v_loss.item()),
            "loss/q":         float(q_loss.item()),
            "loss/actor":     float(actor_loss.item()),
            "grad/v":         float(v_grad_norm),
            "grad/q":         float(q_grad_norm),
            "grad/actor":     float(actor_grad_norm),
            "grad/ttfe":      float(ttfe_grad_norm),
            "v/mean_symlog":  float(v_sl.mean().item()),
            "q/min_mean_sl":  float(q_min_sl.mean().item()),
            "adv/mean":       float(adv_sl.mean().item()) if "adv_sl" in dir() else 0.0,
            "adv/std":        float(adv_sl.std().item())  if "adv_sl" in dir() else 0.0,
            "awr/mean_w":     float(mean_weight.item()),
            "awr/max_w":      float(max_weight.item()),
            "pi/entropy":     float(policy_entropy.item()),
        }

    # ── Checkpoint ──
    def state_dict(self) -> dict:
        return {
            "ttfe":          self.ttfe.state_dict(),
            "actor":         self.actor.state_dict(),
            "critic":        self.critic.state_dict(),
            "value":         self.value.state_dict(),
            "value_target":  self.value_target.state_dict(),
            "actor_optim":   self.actor_optim.state_dict(),
            "critic_optim":  self.critic_optim.state_dict(),
            "value_optim":   self.value_optim.state_dict(),
            "ttfe_optim":    self.ttfe_optim.state_dict(),
            "cfg":           self.cfg.__dict__,
        }

    def load_state_dict(self, sd: dict):
        self.ttfe.load_state_dict(sd["ttfe"])
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        self.value.load_state_dict(sd["value"])
        self.value_target.load_state_dict(sd["value_target"])
        self.actor_optim.load_state_dict(sd["actor_optim"])
        self.critic_optim.load_state_dict(sd["critic_optim"])
        self.value_optim.load_state_dict(sd["value_optim"])
        self.ttfe_optim.load_state_dict(sd["ttfe_optim"])

    def load_checkpoint(self, path: str) -> None:
        """
        Load a checkpoint saved by train_iql.py:
            {"agent": agent.state_dict(), "step": int, "cfg": dict}

        Matches the interface of SACAgent.load_checkpoint for parity with the
        locked evaluation harness (src/evaluation/evaluate_stage1.py).
        """
        sd = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state_dict(sd["agent"])
