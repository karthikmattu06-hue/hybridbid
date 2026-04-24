"""
Actor and Critic networks for SAC.

Actor: Hybrid discrete-continuous policy.
  - Mode selection (charge/discharge/idle) via Gumbel-Softmax
  - Continuous magnitude via squashed Gaussian
  Matches Li et al. (2024) Eq. 1, 23.

Critic (standard): Twin Q-networks for clipped double-Q learning.

Critic (Tier 1): BroNetCritic — LayerNorm + residual blocks + HL-Gauss categorical head.
  Reference: Farebrother et al., "Stop Regressing: Training Value Functions via
  Classification for Scalable Deep RL", ICML 2024 (arXiv:2403.03950).
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN = -20
LOG_STD_MAX = 2

# Mode indices
MODE_CHARGE = 0
MODE_DISCHARGE = 1
MODE_IDLE = 2


class Actor(nn.Module):
    """
    Hybrid discrete-continuous actor for SAC.

    Discrete part: 3-class mode (charge / discharge / idle) via Gumbel-Softmax.
    Continuous part: energy magnitude [squashed Gaussian in (-1,1)].
    Stage 2 extension: additional AS magnitudes (n_as_dims=5).

    Total action dim = 3 + 1 + n_as_dims
      Stage 1: 4  (3 mode + 1 energy mag)
      Stage 2: 9  (3 mode + 1 energy mag + 5 AS mags)

    Input:  90-dim (TTFE 64 + current prices 12 + static 14)
    """

    def __init__(self, obs_dim: int = 90, n_as_dims: int = 0, hidden_dim: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_as_dims = n_as_dims
        self.action_dim = 3 + 1 + n_as_dims  # total flattened action

        # Shared trunk
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Mode head: 3 logits (charge / discharge / idle)
        self.mode_head = nn.Linear(hidden_dim, 3)

        # Energy magnitude head
        self.energy_mag_mean_head = nn.Linear(hidden_dim, 1)
        self.energy_mag_log_std_head = nn.Linear(hidden_dim, 1)

        # AS magnitude heads (Stage 2 only)
        if n_as_dims > 0:
            self.as_mag_mean_head = nn.Linear(hidden_dim, n_as_dims)
            self.as_mag_log_std_head = nn.Linear(hidden_dim, n_as_dims)

    def forward(self, obs: torch.Tensor):
        """
        Returns
        -------
        mode_logits      : (batch, 3)
        energy_mag_mean  : (batch, 1)
        energy_mag_log_std : (batch, 1)
        [as_mag_mean     : (batch, n_as_dims)]  — only when n_as_dims > 0
        [as_mag_log_std  : (batch, n_as_dims)]
        """
        h = F.relu(self.fc1(obs))
        h = F.relu(self.fc2(h))

        mode_logits = self.mode_head(h)
        energy_mag_mean = self.energy_mag_mean_head(h)
        energy_mag_log_std = torch.clamp(
            self.energy_mag_log_std_head(h), LOG_STD_MIN, LOG_STD_MAX
        )

        if self.n_as_dims > 0:
            as_mag_mean = self.as_mag_mean_head(h)
            as_mag_log_std = torch.clamp(
                self.as_mag_log_std_head(h), LOG_STD_MIN, LOG_STD_MAX
            )
            return mode_logits, energy_mag_mean, energy_mag_log_std, as_mag_mean, as_mag_log_std

        return mode_logits, energy_mag_mean, energy_mag_log_std

    def sample(self, obs: torch.Tensor, tau: float = 1.0, hard: bool = False,
               idle_logit_bonus: float = 0.0):
        """
        Sample action via Gumbel-Softmax (mode) + reparameterization (magnitudes).

        Parameters
        ----------
        tau : float
            Gumbel-Softmax temperature. Start at 1.0, anneal toward 0.1.
        hard : bool
            If True, use straight-through hard one-hot (for deterministic eval).
        idle_logit_bonus : float
            Additive offset on the idle logit (index 2) before Gumbel-Softmax.
            Prevents zero-idle collapse without materially changing a healthy policy.
            0.0 = off (v5.9.1 default). 0.1 = v5.9.2 default.

        Returns
        -------
        action    : (batch, action_dim) — cat([mode_soft/hard, energy_mag, as_mags?])
        log_prob  : (batch, 1) — sum of mode + magnitude log-probs
        det_action : (batch, action_dim) — deterministic: argmax mode + tanh(mean)
        """
        outputs = self.forward(obs)
        mode_logits = outputs[0]
        energy_mag_mean = outputs[1]
        energy_mag_log_std = outputs[2]

        # Apply idle logit bonus (v5.9.2+): small additive offset on idle class
        # to prevent zero-idle degenerate collapse. Affects both the sampled mode
        # and the log_prob (via mode_probs below), so entropy accounting is correct.
        if idle_logit_bonus != 0.0:
            bonus = torch.zeros_like(mode_logits)
            bonus[:, MODE_IDLE] = idle_logit_bonus
            mode_logits = mode_logits + bonus

        # --- Mode: Gumbel-Softmax ---
        mode_sample = F.gumbel_softmax(mode_logits, tau=tau, hard=hard)  # (batch, 3)

        # Log-prob of mode: E_{q}[log p(mode)] under categorical
        mode_probs = F.softmax(mode_logits, dim=-1)  # (batch, 3)
        log_prob_mode = (mode_sample * torch.log(mode_probs + 1e-8)).sum(dim=-1, keepdim=True)

        # --- Energy magnitude: squashed Gaussian ---
        energy_std = energy_mag_log_std.exp()
        energy_normal = Normal(energy_mag_mean, energy_std)
        x_energy = energy_normal.rsample()
        energy_mag = torch.tanh(x_energy)  # ∈ (-1, 1)

        log_prob_energy = energy_normal.log_prob(x_energy)
        log_prob_energy -= torch.log(1 - energy_mag.pow(2) + 1e-6)
        log_prob_energy = log_prob_energy.sum(dim=-1, keepdim=True)

        log_prob = log_prob_mode + log_prob_energy
        action_parts = [mode_sample, energy_mag]

        # Deterministic counterparts
        hard_mode = F.one_hot(mode_logits.argmax(dim=-1), 3).float()
        det_energy_mag = torch.tanh(energy_mag_mean)
        det_parts = [hard_mode, det_energy_mag]

        # --- AS magnitudes (Stage 2) ---
        if self.n_as_dims > 0:
            as_mag_mean = outputs[3]
            as_mag_log_std = outputs[4]
            as_std = as_mag_log_std.exp()
            as_normal = Normal(as_mag_mean, as_std)
            x_as = as_normal.rsample()
            as_mag = torch.tanh(x_as)  # ∈ (-1, 1), env takes abs

            log_prob_as = as_normal.log_prob(x_as)
            log_prob_as -= torch.log(1 - as_mag.pow(2) + 1e-6)
            log_prob_as = log_prob_as.sum(dim=-1, keepdim=True)

            log_prob = log_prob + log_prob_as
            action_parts.append(as_mag)
            det_parts.append(torch.tanh(as_mag_mean))

        action = torch.cat(action_parts, dim=-1)
        det_action = torch.cat(det_parts, dim=-1)

        return action, log_prob, det_action

    @classmethod
    def init_stage2_from_stage1(cls, stage1_actor: "Actor", n_as_dims: int = 5) -> "Actor":
        """
        Create a Stage 2 actor initialized from a trained Stage 1 actor.

        - All Stage 1 components (trunk, mode head, energy mag heads) copied exactly.
        - AS magnitude heads initialized near-zero (small Gaussian weights, zero bias).
        """
        actor2 = cls(
            obs_dim=stage1_actor.obs_dim,
            n_as_dims=n_as_dims,
            hidden_dim=stage1_actor.fc1.out_features,
        )

        # Copy all Stage 1 components
        actor2.fc1.load_state_dict(stage1_actor.fc1.state_dict())
        actor2.fc2.load_state_dict(stage1_actor.fc2.state_dict())
        actor2.mode_head.load_state_dict(stage1_actor.mode_head.state_dict())
        actor2.energy_mag_mean_head.load_state_dict(stage1_actor.energy_mag_mean_head.state_dict())
        actor2.energy_mag_log_std_head.load_state_dict(stage1_actor.energy_mag_log_std_head.state_dict())

        # AS heads: near-zero initialization
        with torch.no_grad():
            actor2.as_mag_mean_head.weight.normal_(0, 0.01)
            actor2.as_mag_mean_head.bias.zero_()
            actor2.as_mag_log_std_head.weight.normal_(0, 0.01)
            actor2.as_mag_log_std_head.bias.zero_()

        return actor2

    @classmethod
    def init_stage2_from_stage1_new_obs(
        cls, stage1_actor: "Actor", n_as_dims: int, new_obs_dim: int
    ) -> "Actor":
        """
        Create a Stage 2 actor when obs_dim has changed (e.g. v3a: 90→108).

        fc1 input dim differs from Stage 1 → cannot copy fc1 weights.
        fc2, mode_head, energy heads: copied from Stage 1.
        AS heads: near-zero init (same as init_stage2_from_stage1).
        """
        hidden_dim = stage1_actor.fc1.out_features
        actor2 = cls(obs_dim=new_obs_dim, n_as_dims=n_as_dims, hidden_dim=hidden_dim)

        # fc1: intentionally NOT copied — new input dim, fresh PyTorch default init
        actor2.fc2.load_state_dict(stage1_actor.fc2.state_dict())
        actor2.mode_head.load_state_dict(stage1_actor.mode_head.state_dict())
        actor2.energy_mag_mean_head.load_state_dict(stage1_actor.energy_mag_mean_head.state_dict())
        actor2.energy_mag_log_std_head.load_state_dict(stage1_actor.energy_mag_log_std_head.state_dict())

        # AS heads: near-zero init
        with torch.no_grad():
            actor2.as_mag_mean_head.weight.normal_(0, 0.01)
            actor2.as_mag_mean_head.bias.zero_()
            actor2.as_mag_log_std_head.weight.normal_(0, 0.01)
            actor2.as_mag_log_std_head.bias.zero_()

        return actor2


class Critic(nn.Module):
    """Single Q-network: Q(obs, action) -> scalar."""

    def __init__(self, obs_dim: int = 90, action_dim: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)


class TwinCritic(nn.Module):
    """Twin Q-networks for clipped double-Q learning."""

    def __init__(self, obs_dim: int = 90, action_dim: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.q1 = Critic(obs_dim, action_dim, hidden_dim)
        self.q2 = Critic(obs_dim, action_dim, hidden_dim)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        """Returns (Q1, Q2) both as (batch, 1)."""
        return self.q1(obs, action), self.q2(obs, action)


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1 critic: BroNet + HL-Gauss categorical head
# ─────────────────────────────────────────────────────────────────────────────


class HLGaussLoss:
    """
    HL-Gauss: convert scalar regression targets to soft categorical distributions
    over a fixed support, then use cross-entropy as the critic loss.

    Reference: Farebrother et al., "Stop Regressing: Training Value Functions via
    Classification for Scalable Deep RL", ICML 2024 (arXiv:2403.03950).

    The support spans [-20, 20] in symlog space.  Since rewards are symlog-
    transformed before entering the buffer, symlog(return) is bounded: a reward
    of ±50 symlogs to ±3.9, and even the τ=0.97 infinite-horizon sum is ≤ symlog
    of the raw sum, which stays well inside ±20 for ERCOT arbitrage.

    Parameters
    ----------
    min_value, max_value : float
        Left and right edges of the support (symlog scale).  Default: [-20, 20].
    n_atoms : int
        Number of support atoms.  Default: 101.
    sigma : float
        Gaussian smoothing bandwidth (bins).  Default: 0.75.
    """

    def __init__(
        self,
        min_value: float = -20.0,
        max_value: float = 20.0,
        n_atoms: int = 101,
        sigma: float = 0.75,
    ):
        self.min_value = min_value
        self.max_value = max_value
        self.n_atoms = n_atoms
        self.sigma = sigma
        self.support = torch.linspace(min_value, max_value, n_atoms)
        # Midpoints between consecutive atoms, used as bin boundaries.
        self._midpoints = 0.5 * (self.support[:-1] + self.support[1:])  # (n_atoms-1,)

    def to(self, device: str) -> "HLGaussLoss":
        self.support = self.support.to(device)
        self._midpoints = self._midpoints.to(device)
        return self

    def transform_to_probs(self, target: torch.Tensor) -> torch.Tensor:
        """
        Convert scalar targets to HL-Gauss soft categorical distributions.

        Uses midpoints between adjacent support atoms as bin boundaries.
        The leftmost bin absorbs mass from -inf; the rightmost from +inf.

        Parameters
        ----------
        target : (batch,) float tensor — values in symlog space.

        Returns
        -------
        probs : (batch, n_atoms) — non-negative, sums to 1 along dim=-1.
        """
        # CDF of N(target, sigma) evaluated at each midpoint: (batch, n_atoms-1)
        cdf_at_mid = 0.5 * (
            1.0 + torch.erf(
                (self._midpoints.unsqueeze(0) - target.unsqueeze(-1))
                / (math.sqrt(2.0) * self.sigma)
            )
        )
        # Prepend 0 and append 1 to capture boundary mass.
        zeros = torch.zeros(target.shape[0], 1, device=target.device)
        ones = torch.ones(target.shape[0], 1, device=target.device)
        cdf_full = torch.cat([zeros, cdf_at_mid, ones], dim=-1)  # (batch, n_atoms+1)
        probs = cdf_full[:, 1:] - cdf_full[:, :-1]              # (batch, n_atoms)
        return probs

    def transform_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        """Scalar expectation of the categorical distribution (symlog scale)."""
        return (probs * self.support.to(probs.device)).sum(dim=-1, keepdim=True)  # (batch, 1)

    def compute_q(self, logits: torch.Tensor):
        """
        Convert critic logits to (q_raw, q_symlog).

        q_symlog : expectation over support (symlog scale).
        q_raw    : symexp(q_symlog) — natural scale, used in actor loss.
        """
        probs = F.softmax(logits, dim=-1)
        q_symlog = self.transform_from_probs(probs)                           # (batch, 1)
        q_raw = torch.sign(q_symlog) * (torch.exp(torch.abs(q_symlog)) - 1)  # symexp
        return q_raw, q_symlog

    def loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Cross-entropy loss between predicted logits and HL-Gauss target distribution.

        Parameters
        ----------
        logits : (batch, n_atoms)
        target : (batch,) — scalar values in symlog space.

        Returns
        -------
        loss : (batch,) — per-sample cross-entropy.
        """
        target_probs = self.transform_to_probs(target)        # (batch, n_atoms)
        log_pred = F.log_softmax(logits, dim=-1)
        return -(target_probs * log_pred).sum(dim=-1)         # (batch,)


class BroNetCritic(nn.Module):
    """
    BroNet-style critic: Entry block + 2 pre-LN residual blocks + HL-Gauss head.

    Returns logits (n_atoms) over the HL-Gauss support — NOT a scalar.
    Use HLGaussLoss.compute_q() to convert logits to scalar Q estimates.

    Reference architecture from Björck et al. "Is Deep Learning Good for
    Reinforcement Learning?" (BroNet / LayerNorm critic family).
    """

    def __init__(
        self,
        obs_dim: int = 90,
        act_dim: int = 4,
        hidden: int = 512,
        n_atoms: int = 101,
    ):
        super().__init__()
        self.entry = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.block1 = self._make_block(hidden)
        self.block2 = self._make_block(hidden)
        self.head = nn.Linear(hidden, n_atoms)
        self.n_atoms = n_atoms

    @staticmethod
    def _make_block(hidden: int) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Returns logits of shape (batch, n_atoms)."""
        x = torch.cat([obs, act], dim=-1)
        h = self.entry(x)
        h = h + self.block1(h)
        h = h + self.block2(h)
        return self.head(h)


class TwinBroNetCritic(nn.Module):
    """Twin BroNet critics for clipped double-Q learning (Tier 1)."""

    def __init__(
        self,
        obs_dim: int = 90,
        action_dim: int = 4,
        hidden: int = 512,
        n_atoms: int = 101,
    ):
        super().__init__()
        self.q1 = BroNetCritic(obs_dim, action_dim, hidden, n_atoms)
        self.q2 = BroNetCritic(obs_dim, action_dim, hidden, n_atoms)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        """Returns (logits_q1, logits_q2), each (batch, n_atoms)."""
        return self.q1(obs, action), self.q2(obs, action)


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2a: Discrete N=7 actor + BroNet/HL-Gauss Q(s)→[N, n_atoms] critic
# ─────────────────────────────────────────────────────────────────────────────

# Discrete action grid (Tier 2a): 7 levels in p.u. of P_max.
# Negative = charge, positive = discharge, 0 = idle.
TIER2A_ACTION_LEVELS = [-1.0, -2.0 / 3.0, -1.0 / 3.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]
TIER2A_N_ACTIONS = 7


def tier2a_idx_to_env_action(idx: int) -> np.ndarray:
    """
    Map a discrete action index (0..6) to the 4D env action format
    `[mode_onehot(3), energy_mag(1)]` expected by ercot_env.py.

    Idx → level mapping:
      0: -P_max   → charge,    mag=1.0
      1: -2P/3    → charge,    mag=2/3
      2: -P/3     → charge,    mag=1/3
      3:  0       → idle,      mag=0.0
      4: +P/3     → discharge, mag=1/3
      5: +2P/3    → discharge, mag=2/3
      6: +P_max   → discharge, mag=1.0
    """
    level = TIER2A_ACTION_LEVELS[idx]
    mode_onehot = np.zeros(3, dtype=np.float32)
    if level < 0:
        mode_onehot[MODE_CHARGE] = 1.0
    elif level > 0:
        mode_onehot[MODE_DISCHARGE] = 1.0
    else:
        mode_onehot[MODE_IDLE] = 1.0
    mag = abs(level)
    return np.concatenate([mode_onehot, [mag]]).astype(np.float32)


# Pre-materialize the (7, 4) translation table for fast batched sim lookups.
TIER2A_IDX_TO_ENV_ACTION = np.stack(
    [tier2a_idx_to_env_action(i) for i in range(TIER2A_N_ACTIONS)]
).astype(np.float32)


class ActorDiscrete7(nn.Module):
    """
    Discrete SAC actor: Categorical over 7 action levels.

    No Gumbel-Softmax, no continuous magnitude, no reparameterization. Outputs
    raw logits; the SAC update consumes the full softmax distribution to form
    closed-form expectations over the action space.

    Input : obs (batch, obs_dim)
    Output: logits (batch, 7)
    """

    def __init__(self, obs_dim: int = 90, hidden_dim: int = 512,
                 n_actions: int = TIER2A_N_ACTIONS):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.logit_head = nn.Linear(hidden_dim, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(obs))
        h = F.relu(self.fc2(h))
        return self.logit_head(h)

    def sample(self, obs: torch.Tensor, deterministic: bool = False):
        """
        Returns
        -------
        action_idx    : (batch,) long — sampled (or argmax) action index.
        log_probs_all : (batch, n_actions) — full distribution log-probs.
        probs_all     : (batch, n_actions) — full softmax.
        """
        logits = self.forward(obs)
        log_probs_all = F.log_softmax(logits, dim=-1)
        probs_all = log_probs_all.exp()
        if deterministic:
            action_idx = logits.argmax(dim=-1)
        else:
            action_idx = torch.distributions.Categorical(probs=probs_all).sample()
        return action_idx, log_probs_all, probs_all


class DiscreteBroNetCritic(nn.Module):
    """
    BroNet-style critic for Tier 2a: Q(s) → logits of shape (batch, N, n_atoms).

    Entry block + 2 pre-LN residual blocks + head that emits N * n_atoms logits,
    reshaped to (batch, N, n_atoms). Each slice along dim=1 is an independent
    HL-Gauss distribution over the same symlog support.

    Input: obs only — no action. This removes the `∂Q/∂a` symexp-amplification
    path that was the surviving spike driver in Tier 1 v2.1.
    """

    def __init__(
        self,
        obs_dim: int = 90,
        n_actions: int = TIER2A_N_ACTIONS,
        hidden: int = 512,
        n_atoms: int = 101,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.entry = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.block1 = self._make_block(hidden)
        self.block2 = self._make_block(hidden)
        self.head = nn.Linear(hidden, n_actions * n_atoms)

    @staticmethod
    def _make_block(hidden: int) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns logits of shape (batch, n_actions, n_atoms)."""
        h = self.entry(obs)
        h = h + self.block1(h)
        h = h + self.block2(h)
        flat = self.head(h)                                        # (batch, N*A)
        return flat.view(-1, self.n_actions, self.n_atoms)


class TwinDiscreteBroNetCritic(nn.Module):
    """Twin discrete BroNet critics for clipped double-Q (Tier 2a)."""

    def __init__(
        self,
        obs_dim: int = 90,
        n_actions: int = TIER2A_N_ACTIONS,
        hidden: int = 512,
        n_atoms: int = 101,
    ):
        super().__init__()
        self.q1 = DiscreteBroNetCritic(obs_dim, n_actions, hidden, n_atoms)
        self.q2 = DiscreteBroNetCritic(obs_dim, n_actions, hidden, n_atoms)

    def forward(self, obs: torch.Tensor):
        """Returns (logits_q1, logits_q2), each (batch, n_actions, n_atoms)."""
        return self.q1(obs), self.q2(obs)
