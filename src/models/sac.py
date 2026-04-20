"""
Soft Actor-Critic agent with TTFE encoder.

Two-stage architecture:
  Stage 1: energy-only (4D action: 3 mode + 1 energy_mag), pretrain on pre-RTC+B
  Stage 2: co-optimize (9D action: 3 mode + 1 energy_mag + 5 AS_mags), finetune on post-RTC+B

Action format matches Li et al. (2024) with Gumbel-Softmax discrete mode selection.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.ttfe import TTFE
from src.models.networks import Actor, TwinCritic, TwinBroNetCritic, HLGaussLoss
from src.models.replay_buffer import ReplayBuffer


def has_nan_params(model):
    """Check if any parameter in model contains NaN or Inf."""
    for name, param in model.named_parameters():
        if param.requires_grad and (torch.isnan(param).any() or torch.isinf(param).any()):
            return True, name
    return False, None


def _grad_norm(params):
    """Compute L2 gradient norm for a list of parameters (pre-clip)."""
    grads = [p.grad.detach().flatten() for p in params if p.grad is not None]
    if not grads:
        return 0.0
    return torch.cat(grads).norm().item()


class SACAgent:
    """
    SAC agent encapsulating TTFE + Actor + TwinCritic + target networks.

    Parameters
    ----------
    stage : int
        1 for energy-only, 2 for co-optimize.
    device : str
        'cpu', 'cuda', or 'mps'.
    tau_gumbel : float
        Initial Gumbel-Softmax temperature (anneal from 1.0 → 0.1 during training).
    """

    def __init__(
        self,
        stage: int = 1,
        device: str = "cpu",
        n_prices: int = 12,
        n_prices_flat: int = None,
        d_model: int = 64,
        nhead: int = 8,
        n_layers: int = 2,
        seq_len: int = 32,
        static_dim: int = 14,
        hidden_dim: int = 256,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_ttfe: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_capacity: int = None,
        batch_size: int = None,
        max_grad_norm: float = 1.0,
        max_grad_norm_critic: float = None,
        alpha_max: float = float("inf"),
        idle_logit_bonus: float = 0.0,
        tau_gumbel: float = 1.0,
    ):
        self.stage = stage
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.max_grad_norm = max_grad_norm
        # Critic uses a separate (tighter) clip in v5.9.2+. Falls back to
        # max_grad_norm when not specified (preserves v5.9.1 behavior).
        self.max_grad_norm_critic = max_grad_norm_critic if max_grad_norm_critic is not None else max_grad_norm
        self.alpha_max = alpha_max
        self.idle_logit_bonus = idle_logit_bonus
        self.tau_gumbel = tau_gumbel

        # Action dimensions
        # Stage 1: 3 mode + 1 energy_mag = 4
        # Stage 2: 3 mode + 1 energy_mag + 5 AS_mags = 9
        self.n_as_dims = 0 if stage == 1 else 5
        self.action_dim = 3 + 1 + self.n_as_dims  # 4 or 9
        self.n_continuous = 1 + self.n_as_dims     # 1 or 6 (continuous dims only)

        self.n_prices = n_prices
        # n_prices_flat: how many dims from price_history[-1] appear in the flat obs.
        # For enriched obs (v6.0): n_prices=36 (TTFE input) but only 12 go into flat obs.
        # Defaults to n_prices for backward compatibility with all prior stages.
        self.n_prices_flat = n_prices_flat if n_prices_flat is not None else n_prices
        self.obs_dim = d_model + self.n_prices_flat + static_dim  # e.g. 64+12+14=90 or 64+12+32=108

        # Default buffer/batch sizes per stage
        if buffer_capacity is None:
            buffer_capacity = 1_000_000 if stage == 1 else 50_000
        if batch_size is None:
            batch_size = 256 if stage == 1 else 128
        self.batch_size = batch_size

        # Networks
        self.ttfe = TTFE(n_prices=n_prices, d_model=d_model, nhead=nhead,
                         n_layers=n_layers, seq_len=seq_len).to(device)
        self.actor = Actor(obs_dim=self.obs_dim, n_as_dims=self.n_as_dims,
                           hidden_dim=hidden_dim).to(device)
        self.critic = TwinCritic(obs_dim=self.obs_dim, action_dim=self.action_dim,
                                 hidden_dim=hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)

        for p in self.critic_target.parameters():
            p.requires_grad = False

        # Target entropy calibrated per stage:
        # Stage 1 (4D): log(3) ≈ 1.099 — accounts for 3-mode discrete entropy.
        # Stage 2 (9D): 5.0 — corrected for 5 additional continuous AS heads.
        #   Each Gaussian head contributes ~1.42 nats at unit variance; theoretical
        #   max is log(3) + 5 * 0.5 * log(2πe) ≈ 8.2, but AS heads start near-zero
        #   with bounded outputs, so 5.0 is a conservative floor that prevents collapse
        #   without forcing excess exploration.
        self.target_entropy = float(np.log(3)) if stage == 1 else 5.0
        self.log_alpha = torch.zeros(1, device=device, requires_grad=True)

        # Optimizers
        self.ttfe_optimizer = torch.optim.Adam(self.ttfe.parameters(), lr=lr_ttfe)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_actor)

        # Replay buffer
        self.buffer = ReplayBuffer(
            capacity=buffer_capacity,
            seq_len=seq_len,
            n_prices=n_prices,
            static_dim=static_dim,
            action_dim=self.action_dim,
        )

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    # Scale factor for raw ERCOT prices ($/MWh). Dividing keeps attention Q/K
    # dot products from overflowing float32 during storm events ($9000+/MWh).
    # Normal trading is $20-200 → 0.02-0.2 after scaling; storms → 9.0 max.
    PRICE_NORM = 1000.0

    def _encode_obs(self, price_history: torch.Tensor, static_features: torch.Tensor) -> torch.Tensor:
        """Run TTFE on price history and concatenate with current prices + static features.

        price_history shape: (batch, seq_len, n_prices) — n_prices=12 standard, 36 enriched.
        Only the first n_prices_flat dims of the last timestep enter the flat observation,
        so the 24 appended DA LMP dims don't double-count into the flat part.
        """
        ph_norm = price_history / self.PRICE_NORM             # scale $/MWh → ~[0, 9]
        temporal = self.ttfe(ph_norm)                         # (batch, d_model)
        current_prices = ph_norm[:, -1, :self.n_prices_flat]  # (batch, n_prices_flat)
        return torch.cat([temporal, current_prices, static_features], dim=-1)  # (batch, obs_dim)

    @torch.no_grad()
    def select_action(self, obs: dict, deterministic: bool = False) -> np.ndarray:
        """
        Select action given observation dict.

        Parameters
        ----------
        obs : dict with 'price_history' (seq_len, n_prices) and 'static_features' (static_dim,)
        deterministic : bool
            If True, use argmax mode + tanh(mean) magnitude (no sampling noise).

        Returns
        -------
        np.ndarray of shape (action_dim,)
        """
        self.ttfe.eval()
        self.actor.eval()

        ph = torch.tensor(obs["price_history"], dtype=torch.float32, device=self.device).unsqueeze(0)
        sf = torch.tensor(obs["static_features"], dtype=torch.float32, device=self.device).unsqueeze(0)

        encoded = self._encode_obs(ph, sf)

        if deterministic:
            _, _, action = self.actor.sample(encoded, tau=self.tau_gumbel, hard=True,
                                             idle_logit_bonus=self.idle_logit_bonus)
        else:
            action, _, _ = self.actor.sample(encoded, tau=self.tau_gumbel, hard=False,
                                             idle_logit_bonus=self.idle_logit_bonus)

        self.ttfe.train()
        self.actor.train()

        return action.squeeze(0).cpu().numpy()

    def update(
        self,
        batch: dict = None,
        tau_gumbel: float = None,
        phase: str = "A",
    ) -> dict:
        """
        Perform one SAC update step.

        Parameters
        ----------
        tau_gumbel : float, optional
            Override Gumbel temperature for this update. Uses self.tau_gumbel if None.
        phase : str
            Training phase ('A', 'B'). Phase B applies TTFE gradient scaling (×0.1)
            to stabilize the freshly-unfrozen top TTFE layer. Phase C is removed in
            Stage 2 v2 — full TTFE unfreeze eroded pretrained representations.

        Returns dict of losses/metrics.
        """
        if tau_gumbel is None:
            tau_gumbel = self.tau_gumbel

        if batch is None:
            if len(self.buffer) < self.batch_size:
                return {}
            batch = self.buffer.sample(self.batch_size, device=self.device)

        ph = batch["price_history"]
        sf = batch["static_features"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_ph = batch["next_price_history"]
        next_sf = batch["next_static_features"]
        dones = batch["dones"]

        # Rewards are symlog-transformed before entering the replay buffer
        # (see train_stage1.py). No clipping needed here.

        # Encode observations
        obs_encoded = self._encode_obs(ph, sf)
        with torch.no_grad():
            next_obs_encoded = self._encode_obs(next_ph, next_sf)

        # --- Critic update ---
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(
                next_obs_encoded, tau=tau_gumbel, hard=False,
                idle_logit_bonus=self.idle_logit_bonus,
            )
            q1_target, q2_target = self.critic_target(next_obs_encoded, next_actions)
            q_target = torch.min(q1_target, q2_target) - self.alpha * next_log_probs
            td_target = rewards + (1.0 - dones) * self.gamma * q_target

        q1, q2 = self.critic(obs_encoded.detach(), actions)
        # Huber loss (smooth L1) instead of MSE: quadratic for small TD errors,
        # linear for large ones — directly reduces gradient magnitude from outlier batches.
        critic_loss = F.huber_loss(q1, td_target) + F.huber_loss(q2, td_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Per-component grad norms (pre-clip)
        grad_q1 = _grad_norm(self.critic.q1.parameters())
        grad_q2 = _grad_norm(self.critic.q2.parameters())
        # v5.9.2+: separate (tighter) critic clip; falls back to max_grad_norm in v5.9.1.
        critic_grad_norm = nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.max_grad_norm_critic
        )
        grad_c_pre_clip = critic_grad_norm.item()   # clip_grad_norm_ returns pre-clip norm
        grad_c_post_clip = min(grad_c_pre_clip, self.max_grad_norm_critic)
        self.critic_optimizer.step()

        # NaN check: critic
        nan_found, nan_name = has_nan_params(self.critic)
        if nan_found:
            return {"nan_detected": True, "nan_source": f"critic.{nan_name}"}

        # --- Actor + TTFE update ---
        # obs_encoded retains the TTFE computation graph (not detached above).
        # TTFE is updated here via actor loss — NOT via critic loss. This removes
        # the amplification path: TTFE → critic → Q-values → (critic weights × Q)
        # → TTFE gradient, which grew to 314T in v5.4 as critic weights accumulated.
        # Actor gradient to TTFE is small (~0.5-1.4 norm, observed) and does not
        # scale with Q-value magnitude.
        new_actions, log_probs, _ = self.actor.sample(
            obs_encoded, tau=tau_gumbel, hard=False,
            idle_logit_bonus=self.idle_logit_bonus,
        )
        # Mode distribution across the batch (Gumbel-soft samples ≈ policy probs).
        mode_probs_mean = new_actions[:, :3].mean(dim=0).detach().cpu().tolist()

        # Detach obs before critic so critic state weights don't amplify TTFE grad.
        q1_new, q2_new = self.critic(obs_encoded.detach(), new_actions)
        q_new = torch.min(q1_new, q2_new)
        q_value_mean = q_new.mean().item()
        q_value_max_abs = q_new.abs().max().item()
        actor_loss = (self.alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        self.ttfe_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.max_grad_norm
        )
        grad_a_pre_clip = actor_grad_norm.item()
        grad_a_post_clip = min(grad_a_pre_clip, self.max_grad_norm)
        grad_ttfe_proj = _grad_norm(
            [self.ttfe.input_proj.weight, self.ttfe.input_proj.bias, self.ttfe.pos_embedding]
        )
        grad_ttfe_attn = _grad_norm(self.ttfe.transformer.parameters())
        # Phase B: 10× TTFE gradient damping when top layer is unfrozen
        # (belt-and-suspenders with 3e-5 lr; prevents TTFE disruption from a fresh critic)
        if phase == "B":
            for p in self.ttfe.parameters():
                if p.grad is not None:
                    p.grad *= 0.1
        ttfe_grad_norm = nn.utils.clip_grad_norm_(
            self.ttfe.parameters(), self.max_grad_norm
        )
        self.actor_optimizer.step()
        self.ttfe_optimizer.step()

        # NaN check: actor + TTFE
        nan_found, nan_name = has_nan_params(self.actor)
        if nan_found:
            return {"nan_detected": True, "nan_source": f"actor.{nan_name}"}
        nan_found, nan_name = has_nan_params(self.ttfe)
        if nan_found:
            return {"nan_detected": True, "nan_source": f"ttfe.{nan_name}"}

        # --- Alpha update ---
        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        # Clamp log_alpha to [log(0.05), log(alpha_max)].
        # Floor (0.05): prevents alpha collapse in Stage 2's 9D action space.
        # Ceiling (alpha_max): v5.9.2+ prevents spike-and-crash cycles where
        # alpha hit 0.42 in v5.9.1 then overcorrected to 0.14 causing mode collapse.
        # alpha_max=inf (default) preserves original v5.9.1 behavior.
        with torch.no_grad():
            log_alpha_max = float(np.log(self.alpha_max)) if self.alpha_max != float("inf") else float("inf")
            self.log_alpha.clamp_(min=float(np.log(0.05)), max=log_alpha_max)

        # --- Soft update target networks ---
        self._soft_update()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
            "q_mean": q_value_mean,
            "q_max_abs": q_value_max_abs,
            # Critic gradient norms (pre/post clip, v5.9.2+)
            "critic_grad_norm": grad_c_pre_clip,  # kept for log compat (pre-clip)
            "grad_c_pre_clip": grad_c_pre_clip,
            "grad_c_post_clip": grad_c_post_clip,
            # Actor/TTFE gradient norms
            "actor_grad_norm": grad_a_pre_clip,   # kept for log compat (pre-clip)
            "grad_a_pre_clip": grad_a_pre_clip,
            "grad_a_post_clip": grad_a_post_clip,
            "ttfe_grad_norm": ttfe_grad_norm.item(),
            "grad_q1": grad_q1,
            "grad_q2": grad_q2,
            "grad_ttfe_proj": grad_ttfe_proj,
            "grad_ttfe_attn": grad_ttfe_attn,
            # Mode distribution across training batch (charge/discharge/idle)
            "mode_probs_ch": mode_probs_mean[0],
            "mode_probs_dc": mode_probs_mean[1],
            "mode_probs_id": mode_probs_mean[2],
        }

    def _soft_update(self):
        """Polyak averaging for target networks."""
        for p, p_target in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_target.data.mul_(1.0 - self.tau)
            p_target.data.add_(self.tau * p.data)

    def snapshot_state(self):
        """Return cloned state dicts for emergency recovery. ~1.6MB, <1ms on GPU."""
        return {
            "ttfe": {k: v.clone() for k, v in self.ttfe.state_dict().items()},
            "actor": {k: v.clone() for k, v in self.actor.state_dict().items()},
            "critic": {k: v.clone() for k, v in self.critic.state_dict().items()},
            "critic_target": {k: v.clone() for k, v in self.critic_target.state_dict().items()},
            "log_alpha": self.log_alpha.data.clone(),
        }

    def save_emergency_checkpoint(self, path: str, snapshot: dict):
        """Save an emergency checkpoint from a previous good state snapshot."""
        torch.save({
            "stage": self.stage,
            "tau_gumbel": self.tau_gumbel,
            **snapshot,
        }, path)

    def save_checkpoint(self, path: str):
        """Save all model weights and optimizer states."""
        torch.save({
            "stage": self.stage,
            "tau_gumbel": self.tau_gumbel,
            "ttfe": self.ttfe.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha.data,
            "ttfe_optimizer": self.ttfe_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, path: str, weights_only_mode: bool = False):
        """Load model weights and (optionally) optimizer states.

        Parameters
        ----------
        weights_only_mode : bool
            If True, load only model weights (TTFE, actor, critic). Skip optimizer
            states. Use for evaluation, or when optimizer param groups may not match
            the current agent config (e.g. Phase B checkpoints with partial TTFE).
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.ttfe.load_state_dict(ckpt["ttfe"])
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.log_alpha.data.copy_(ckpt["log_alpha"])
        if "tau_gumbel" in ckpt:
            self.tau_gumbel = ckpt["tau_gumbel"]
        if not weights_only_mode:
            self.ttfe_optimizer.load_state_dict(ckpt["ttfe_optimizer"])
            self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
            self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
            self.alpha_optimizer.load_state_dict(ckpt["alpha_optimizer"])

    def init_from_stage1(self, stage1_checkpoint_path: str):
        """
        Initialize Stage 2 agent from Stage 1 checkpoint.

        - TTFE weights copied from Stage 1 (always compatible — 12-dim input unchanged).
        - Actor:
            * If obs_dim unchanged (v2): all layers copied, AS heads near-zero.
            * If obs_dim changed (v3a: 90→108): fc2+heads copied, fc1 fresh (dim mismatch).
        - Critics: fresh random initialization.
        - Buffer: empty.
        """
        assert self.stage == 2, "init_from_stage1 only for Stage 2"

        ckpt = torch.load(stage1_checkpoint_path, map_location=self.device, weights_only=True)

        # TTFE: always compatible (12-dim input, never changed)
        self.ttfe.load_state_dict(ckpt["ttfe"])

        # Infer Stage 1 obs_dim from saved fc1 weight shape
        stage1_obs_dim = ckpt["actor"]["fc1.weight"].shape[1]
        hidden_dim = self.actor.fc1.out_features

        # Build Stage 1 actor with its original obs_dim
        stage1_actor = Actor(obs_dim=stage1_obs_dim, n_as_dims=0, hidden_dim=hidden_dim)
        stage1_actor.load_state_dict(ckpt["actor"])

        if stage1_obs_dim == self.actor.obs_dim:
            # Dims match (v2 / standard path): copy fc1 too
            new_actor = Actor.init_stage2_from_stage1(stage1_actor, n_as_dims=self.n_as_dims)
        else:
            # Dims differ (v3a: 90→108): fc1 stays at fresh init, copy fc2+heads
            print(
                f"  init_from_stage1: obs_dim mismatch "
                f"({stage1_obs_dim} → {self.actor.obs_dim}). "
                f"fc1 initialized fresh; fc2+heads copied from Stage 1."
            )
            new_actor = Actor.init_stage2_from_stage1_new_obs(
                stage1_actor, n_as_dims=self.n_as_dims, new_obs_dim=self.actor.obs_dim
            )

        self.actor.load_state_dict(new_actor.state_dict())

    def freeze_ttfe(self):
        """Freeze all TTFE parameters (for Stage 2 Phase 1)."""
        for p in self.ttfe.parameters():
            p.requires_grad = False

    def unfreeze_ttfe_top_layers(self, n_layers: int = 1, lr: float = 3e-5):
        """Unfreeze top N transformer layers (for Stage 2 Phase 2)."""
        total_layers = len(self.ttfe.transformer.layers)
        for i in range(total_layers - n_layers, total_layers):
            for p in self.ttfe.transformer.layers[i].parameters():
                p.requires_grad = True

        unfrozen = [p for p in self.ttfe.parameters() if p.requires_grad]
        if unfrozen:
            self.ttfe_optimizer = torch.optim.Adam(unfrozen, lr=lr)

    def unfreeze_ttfe_all(self, lr: float = 3e-5):
        """Unfreeze all TTFE parameters (for Stage 2 Phase C)."""
        for p in self.ttfe.parameters():
            p.requires_grad = True
        self.ttfe_optimizer = torch.optim.Adam(self.ttfe.parameters(), lr=lr)


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1 agent: BroNetCritic + HL-Gauss loss + fixed alpha + γ=0.97 + τ=0.001
# ─────────────────────────────────────────────────────────────────────────────


class SACAgentTier1:
    """
    Stage 1 Tier 1 agent — architectural changes to fix Q-value explosion:

    1. BroNetCritic (LayerNorm + 2 residual blocks + HL-Gauss categorical head)
    2. HL-Gauss cross-entropy critic loss (Farebrother et al. ICML 2024)
    3. AdamW with weight_decay=0.1 for critic optimizer
    4. γ = 0.97 (effective horizon ~33 steps @ 5-min)
    5. Fixed alpha = 0.1 (no auto-tuning — breaks Q-swing → α-swing feedback loop)
    6. Polyak τ = 0.001 (slower target updates contain spike propagation)
    7. Gradient clip max_norm=1.0 on combined critic params
    8. No idle_logit_bonus, no alpha clamp (band-aids removed)

    Actor and TTFE are unchanged from v5.9.x.
    """

    PRICE_NORM = 1000.0  # same as SACAgent

    def __init__(
        self,
        stage: int = 1,
        device: str = "cpu",
        n_prices: int = 12,
        d_model: int = 64,
        nhead: int = 8,
        n_layers: int = 2,
        seq_len: int = 32,
        static_dim: int = 14,
        hidden_dim: int = 512,        # BroNet uses 512
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-4,
        lr_ttfe: float = 3e-4,
        gamma: float = 0.97,
        tau: float = 0.001,
        alpha: float = 0.1,           # Fixed — no auto-tuning
        weight_decay: float = 0.1,    # AdamW weight decay for critic
        buffer_capacity: int = None,
        batch_size: int = None,
        max_grad_norm: float = 1.0,
        max_grad_norm_ttfe: float = None,  # separate tighter TTFE clip; None → use max_grad_norm
        n_atoms: int = 101,
        hl_gauss_min: float = -20.0,
        hl_gauss_max: float = 20.0,
        hl_gauss_sigma: float = 0.75,
        tau_gumbel: float = 1.0,
    ):
        self.stage = stage
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha            # Fixed scalar — no log_alpha, no optimizer
        self.max_grad_norm = max_grad_norm
        self.max_grad_norm_ttfe = max_grad_norm_ttfe if max_grad_norm_ttfe is not None else max_grad_norm
        self.tau_gumbel = tau_gumbel

        self.n_as_dims = 0 if stage == 1 else 5
        self.action_dim = 3 + 1 + self.n_as_dims
        self.n_prices = n_prices
        self.obs_dim = d_model + n_prices + static_dim  # e.g. 64+12+14=90

        if buffer_capacity is None:
            buffer_capacity = 1_000_000 if stage == 1 else 50_000
        if batch_size is None:
            batch_size = 256 if stage == 1 else 128
        self.batch_size = batch_size

        # HL-Gauss loss (device-aware support tensor)
        self.hl_gauss = HLGaussLoss(
            min_value=hl_gauss_min,
            max_value=hl_gauss_max,
            n_atoms=n_atoms,
            sigma=hl_gauss_sigma,
        ).to(device)

        # Networks
        self.ttfe = TTFE(
            n_prices=n_prices, d_model=d_model, nhead=nhead,
            n_layers=n_layers, seq_len=seq_len,
        ).to(device)
        self.actor = Actor(
            obs_dim=self.obs_dim, n_as_dims=self.n_as_dims, hidden_dim=hidden_dim,
        ).to(device)
        self.critic = TwinBroNetCritic(
            obs_dim=self.obs_dim, action_dim=self.action_dim,
            hidden=hidden_dim, n_atoms=n_atoms,
        ).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # Optimizers
        self.ttfe_optimizer = torch.optim.Adam(self.ttfe.parameters(), lr=lr_ttfe)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        # AdamW with weight decay for critic (Change 3)
        self.critic_optimizer = torch.optim.AdamW(
            list(self.critic.q1.parameters()) + list(self.critic.q2.parameters()),
            lr=lr_critic,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

        # Replay buffer
        self.buffer = ReplayBuffer(
            capacity=buffer_capacity,
            seq_len=seq_len,
            n_prices=n_prices,
            static_dim=static_dim,
            action_dim=self.action_dim,
        )

    def _encode_obs(
        self, price_history: torch.Tensor, static_features: torch.Tensor
    ) -> torch.Tensor:
        ph_norm = price_history / self.PRICE_NORM
        temporal = self.ttfe(ph_norm)
        current_prices = ph_norm[:, -1, :]
        return torch.cat([temporal, current_prices, static_features], dim=-1)

    @torch.no_grad()
    def select_action(self, obs: dict, deterministic: bool = False) -> np.ndarray:
        self.ttfe.eval()
        self.actor.eval()

        ph = torch.tensor(obs["price_history"], dtype=torch.float32, device=self.device).unsqueeze(0)
        sf = torch.tensor(obs["static_features"], dtype=torch.float32, device=self.device).unsqueeze(0)
        encoded = self._encode_obs(ph, sf)

        if deterministic:
            _, _, action = self.actor.sample(encoded, tau=self.tau_gumbel, hard=True)
        else:
            action, _, _ = self.actor.sample(encoded, tau=self.tau_gumbel, hard=False)

        self.ttfe.train()
        self.actor.train()
        return action.squeeze(0).cpu().numpy()

    def update(self, batch: dict = None, tau_gumbel: float = None) -> dict:
        """
        One Tier 1 SAC update step.

        Critic: HL-Gauss cross-entropy loss on symlog-space TD targets.
        Actor:  Standard policy gradient with fixed alpha, Q in raw scale (symexp).
        TTFE:   Updated only via actor gradient (no critic path).
        Alpha:  Fixed — no update.
        """
        if tau_gumbel is None:
            tau_gumbel = self.tau_gumbel

        if batch is None:
            if len(self.buffer) < self.batch_size:
                return {}
            batch = self.buffer.sample(self.batch_size, device=self.device)

        ph = batch["price_history"]
        sf = batch["static_features"]
        actions = batch["actions"]
        rewards = batch["rewards"]          # (batch, 1), symlog-transformed
        next_ph = batch["next_price_history"]
        next_sf = batch["next_static_features"]
        dones = batch["dones"]              # (batch, 1)

        obs_encoded = self._encode_obs(ph, sf)
        with torch.no_grad():
            next_obs_encoded = self._encode_obs(next_ph, next_sf)

        # ── Critic update ────────────────────────────────────────────────────
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(
                next_obs_encoded, tau=tau_gumbel, hard=False,
            )
            # Get next Q in symlog space from target networks
            next_q1_logits, next_q2_logits = self.critic_target(next_obs_encoded, next_actions)
            _, next_q1_symlog = self.hl_gauss.compute_q(next_q1_logits)   # (batch, 1)
            _, next_q2_symlog = self.hl_gauss.compute_q(next_q2_logits)
            next_q_symlog = torch.min(next_q1_symlog, next_q2_symlog)

            # TD target in symlog space.
            # rewards are already symlog-transformed; log_probs in nats.
            # alpha * log_prob is small (≈0.1 * [-2,2] = [-0.2,0.2]) vs Q (±5 symlog).
            target_symlog = rewards + (1.0 - dones) * self.gamma * (
                next_q_symlog - self.alpha * next_log_probs
            )  # (batch, 1)

        q1_logits, q2_logits = self.critic(obs_encoded.detach(), actions)
        target_1d = target_symlog.squeeze(-1)  # (batch,) for HL-Gauss
        critic1_loss = self.hl_gauss.loss(q1_logits, target_1d).mean()
        critic2_loss = self.hl_gauss.loss(q2_logits, target_1d).mean()
        critic_loss = critic1_loss + critic2_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Combined clip across both sub-critics
        all_critic_params = list(self.critic.q1.parameters()) + list(self.critic.q2.parameters())
        critic_grad_norm = nn.utils.clip_grad_norm_(all_critic_params, max_norm=self.max_grad_norm)
        grad_c_pre_clip = critic_grad_norm.item()
        grad_c_post_clip = min(grad_c_pre_clip, self.max_grad_norm)
        self.critic_optimizer.step()

        # NaN check
        nan_found, nan_name = has_nan_params(self.critic)
        if nan_found:
            return {"nan_detected": True, "nan_source": f"critic.{nan_name}"}

        # ── Actor + TTFE update ──────────────────────────────────────────────
        # TTFE updated only via actor gradient (not critic path — avoids Q-amplification).
        new_actions, log_probs, _ = self.actor.sample(obs_encoded, tau=tau_gumbel, hard=False)
        mode_probs_mean = new_actions[:, :3].mean(dim=0).detach().cpu().tolist()

        q1_new_logits, q2_new_logits = self.critic(obs_encoded.detach(), new_actions)
        q1_new_raw, q1_new_symlog = self.hl_gauss.compute_q(q1_new_logits)
        q2_new_raw, q2_new_symlog = self.hl_gauss.compute_q(q2_new_logits)
        q_new = torch.min(q1_new_raw, q2_new_raw)  # raw scale for policy gradient
        q_new_symlog = torch.min(q1_new_symlog, q2_new_symlog)
        q_value_mean = q_new.mean().item()
        q_value_max_abs = q_new.abs().max().item()
        # symlog-scale Q — directly comparable to HL-Gauss support bounds [±20]
        q_symlog_mean_actor = q_new_symlog.mean().item()
        q_symlog_maxabs_actor = q_new_symlog.abs().max().item()

        actor_loss = (self.alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        self.ttfe_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        grad_a_pre_clip = actor_grad_norm.item()
        grad_a_post_clip = min(grad_a_pre_clip, self.max_grad_norm)
        grad_ttfe_proj = _grad_norm(
            [self.ttfe.input_proj.weight, self.ttfe.input_proj.bias, self.ttfe.pos_embedding]
        )
        grad_ttfe_attn = _grad_norm(self.ttfe.transformer.parameters())
        ttfe_grad_norm = nn.utils.clip_grad_norm_(self.ttfe.parameters(), self.max_grad_norm_ttfe)
        self.actor_optimizer.step()
        self.ttfe_optimizer.step()

        nan_found, nan_name = has_nan_params(self.actor)
        if nan_found:
            return {"nan_detected": True, "nan_source": f"actor.{nan_name}"}
        nan_found, nan_name = has_nan_params(self.ttfe)
        if nan_found:
            return {"nan_detected": True, "nan_source": f"ttfe.{nan_name}"}

        # ── Soft update target ───────────────────────────────────────────────
        self._soft_update()

        # ── HL-Gauss diagnostic metrics (from Q1 on current obs) ─────────────
        with torch.no_grad():
            probs1 = F.softmax(q1_logits.detach(), dim=-1)  # (batch, n_atoms)
            bin_entropy = -(probs1 * torch.log(probs1 + 1e-8)).sum(dim=-1).mean().item()
            argmax_idx = probs1.argmax(dim=-1).float()       # (batch,)
            support = self.hl_gauss.support                  # (n_atoms,)
            argmax_support_val = support[probs1.argmax(dim=-1)].mean().item()
            # Q in symlog space for logging (bounded — useful to track saturation)
            _, q1_symlog = self.hl_gauss.compute_q(q1_logits.detach())
            _, q2_symlog = self.hl_gauss.compute_q(q2_logits.detach())
            q_symlog_mean = torch.min(q1_symlog, q2_symlog).mean().item()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            # Q metrics (raw scale — actor loss signal)
            "q_mean": q_value_mean,
            "q_max_abs": q_value_max_abs,
            # HL-Gauss specific (Tier 1 enhanced logging)
            # q_expected_* are in SYMLOG space — directly comparable to support bounds ±20
            "q_expected_mean": q_symlog_mean_actor,
            "q_expected_max_abs": q_symlog_maxabs_actor,
            "q_symlog_mean": q_symlog_mean,
            "critic_bin_entropy": bin_entropy,
            "critic_bin_argmax_support_value": argmax_support_val,
            # Gradient norms
            "grad_c_pre_clip": grad_c_pre_clip,
            "grad_c_post_clip": grad_c_post_clip,
            "critic_grad_norm": grad_c_pre_clip,
            "grad_a_pre_clip": grad_a_pre_clip,
            "grad_a_post_clip": grad_a_post_clip,
            "actor_grad_norm": grad_a_pre_clip,
            "ttfe_grad_norm": ttfe_grad_norm.item(),
            "grad_ttfe_proj": grad_ttfe_proj,
            "grad_ttfe_attn": grad_ttfe_attn,
            # Mode distribution
            "mode_probs_ch": mode_probs_mean[0],
            "mode_probs_dc": mode_probs_mean[1],
            "mode_probs_id": mode_probs_mean[2],
        }

    def _soft_update(self):
        for p, p_target in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_target.data.mul_(1.0 - self.tau)
            p_target.data.add_(self.tau * p.data)

    def snapshot_state(self) -> dict:
        return {
            "ttfe": {k: v.clone() for k, v in self.ttfe.state_dict().items()},
            "actor": {k: v.clone() for k, v in self.actor.state_dict().items()},
            "critic": {k: v.clone() for k, v in self.critic.state_dict().items()},
            "critic_target": {k: v.clone() for k, v in self.critic_target.state_dict().items()},
        }

    def save_emergency_checkpoint(self, path: str, snapshot: dict):
        torch.save({"stage": self.stage, "tau_gumbel": self.tau_gumbel, **snapshot}, path)

    def save_checkpoint(self, path: str):
        torch.save({
            "stage": self.stage,
            "tau_gumbel": self.tau_gumbel,
            "ttfe": self.ttfe.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "ttfe_optimizer": self.ttfe_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, path: str, weights_only_mode: bool = False):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.ttfe.load_state_dict(ckpt["ttfe"])
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        if "tau_gumbel" in ckpt:
            self.tau_gumbel = ckpt["tau_gumbel"]
        if not weights_only_mode:
            self.ttfe_optimizer.load_state_dict(ckpt["ttfe_optimizer"])
            self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
            self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
