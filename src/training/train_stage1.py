"""
Stage 1: Energy-only pretraining on pre-RTC+B data.

Full training loop with logging, checkpointing, and numerical stability.
"""

import argparse
import os
import sys
import time

import math

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def symlog(x: float) -> float:
    """DreamerV3 symmetric logarithmic transform (Hafner et al., 2023, arXiv:2301.04104).

    Compresses large reward magnitudes while preserving sign and ordering.
    symlog(0)=0, symlog(1)≈0.69, symlog(100)≈4.62, symlog(9000)≈9.10.
    Applied to the economic reward component only (NOT to the SoC penalty).
    """
    return math.copysign(math.log1p(abs(x)), x)

from src.env.ercot_env import ERCOTBatteryEnv
from src.models.sac import SACAgent, SACAgentTier1, SACAgentTier2a
from src.training.config import (
    Stage1Config,
    Stage1V60Config,
    Stage1V592Config,
    Stage1Tier1Config,
    Stage1Tier2aConfig,
)


def train_stage1(config: Stage1Config = None, enriched_obs: bool = False,
                 resume_from: str = None, resume_step: int = 0,
                 reset_optimizers: bool = False):
    if config is None:
        config = Stage1Config()

    # Tier 2a inherits from Tier 1, so check it FIRST for branching; only treat
    # as generic "tier1" when the config is Stage1Tier1Config but not Tier 2a.
    is_tier2a = isinstance(config, Stage1Tier2aConfig)
    is_tier1 = isinstance(config, Stage1Tier1Config) and not is_tier2a

    if is_tier2a:
        version = "tier2a_discrete7"
    elif is_tier1:
        version = "tier1_v1"
    elif isinstance(config, Stage1V592Config):
        version = "v5.9.2"
    elif enriched_obs:
        version = "v6.0"
    else:
        version = "v5.9"
    print(f"=== Stage 1: Energy-Only Training ({version}) ===")
    print(f"Data: {config.train_start} to {config.train_end}")
    print(f"Device: {config.device}")
    print(f"Total steps: {config.total_steps}")

    if is_tier2a:
        print("=== Stage 1 Tier 2a Configuration (discrete N=7) ===")
        print(f"Action space: Categorical({config.n_actions}) over "
              "{-P, -2P/3, -P/3, 0, +P/3, +2P/3, +P}")
        print("Critic: DiscreteBroNet Q(s) -> (batch, N, n_atoms), no action input")
        print(f"Critic optimizer: AdamW lr={config.lr_critic} weight_decay={config.weight_decay_critic}")
        print(f"Actor optimizer: Adam lr={config.lr_actor}")
        print(f"Gamma: {config.gamma} | Alpha: {config.alpha_fixed} (FIXED)")
        print(f"Polyak tau: {config.tau}")
        print(f"HL-Gauss: support=[{config.hl_gauss_min}, {config.hl_gauss_max}], "
              f"n_atoms={config.n_atoms}, sigma={config.hl_gauss_sigma}")
        print(f"Reward pre-scale: ÷{config.reward_scale} before symlog")
        print(f"Gradient clip: actor={config.max_grad_norm} critic={config.max_grad_norm} "
              f"ttfe={config.max_grad_norm_ttfe}")
        print(f"TTFE optimizer: Adam lr={config.lr_ttfe}")
    elif is_tier1:
        print("=== Stage 1 Tier 1 v1 Configuration ===")
        print("Critic: BroNet (LayerNorm + 2 residual blocks + HL-Gauss 101 bins)")
        print(f"Critic optimizer: AdamW lr={config.lr_critic} weight_decay={config.weight_decay_critic}")
        print(f"Actor optimizer: Adam lr={config.lr_actor} (unchanged)")
        print(f"Gamma: {config.gamma}")
        print(f"Alpha: {config.alpha_fixed} (FIXED, no auto-tuning)")
        print(f"Polyak tau: {config.tau}")
        print(f"HL-Gauss: support=[{config.hl_gauss_min}, {config.hl_gauss_max}], "
              f"n_atoms={config.n_atoms}, sigma={config.hl_gauss_sigma}")
        print(f"Reward pre-scale: ÷{config.reward_scale} before symlog")
        print(f"Gradient clip: actor={config.max_grad_norm} critic={config.max_grad_norm} "
              f"ttfe={config.max_grad_norm_ttfe}")
        print(f"TTFE optimizer: Adam lr={config.lr_ttfe}")
    else:
        print(f"Max grad norm: actor/ttfe={config.max_grad_norm} "
              f"critic={getattr(config, 'max_grad_norm_critic', None) or config.max_grad_norm}")
        print(f"LR: actor={config.lr_actor} critic={config.lr_critic} ttfe={config.lr_ttfe}")
        print(f"τ_gumbel: {config.tau_gumbel_init} → {config.tau_gumbel_final}")
        print(f"Alpha bounds: [0.05, {getattr(config, 'alpha_max', 'inf')}]  "
              f"idle_logit_bonus={getattr(config, 'idle_logit_bonus', 0.0)}")
    if enriched_obs:
        n_prices_flat = getattr(config, "n_prices_flat", config.n_prices)
        obs_dim = config.d_model + n_prices_flat + config.static_dim
        print(f"Enriched obs: TTFE={config.n_prices}-dim input, obs_dim={obs_dim}, static_dim={config.static_dim}")

    # Create environment
    battery_config = dict(
        p_max=config.p_max, e_max=config.e_max,
        soc_min_frac=config.soc_min_frac, soc_max_frac=config.soc_max_frac,
        soc_initial_frac=config.soc_initial_frac,
        eta_ch=config.eta_ch, eta_dch=config.eta_dch,
        degradation_cost=config.degradation_cost,
    )
    env = ERCOTBatteryEnv(
        data_dir=config.data_dir,
        mode="energy_only",
        battery_config=battery_config,
        seq_len=config.seq_len,
        date_range=(config.train_start, config.train_end),
        enriched_obs=enriched_obs,
    )

    # Create SAC agent
    if is_tier2a:
        agent = SACAgentTier2a(
            device=config.device,
            n_prices=config.n_prices,
            d_model=config.d_model,
            nhead=config.nhead,
            n_layers=config.n_layers,
            seq_len=config.seq_len,
            static_dim=config.static_dim,
            hidden_dim=config.hidden_dim,
            lr_actor=config.lr_actor,
            lr_critic=config.lr_critic,
            lr_ttfe=config.lr_ttfe,
            gamma=config.gamma,
            tau=config.tau,
            alpha=config.alpha_fixed,
            weight_decay=config.weight_decay_critic,
            buffer_capacity=config.buffer_capacity,
            batch_size=config.batch_size,
            max_grad_norm=config.max_grad_norm,
            max_grad_norm_ttfe=getattr(config, "max_grad_norm_ttfe", None),
            n_actions=config.n_actions,
            n_atoms=config.n_atoms,
            hl_gauss_min=config.hl_gauss_min,
            hl_gauss_max=config.hl_gauss_max,
            hl_gauss_sigma=config.hl_gauss_sigma,
        )
    elif is_tier1:
        agent = SACAgentTier1(
            stage=1,
            device=config.device,
            n_prices=config.n_prices,
            d_model=config.d_model,
            nhead=config.nhead,
            n_layers=config.n_layers,
            seq_len=config.seq_len,
            static_dim=config.static_dim,
            hidden_dim=config.hidden_dim,
            lr_actor=config.lr_actor,
            lr_critic=config.lr_critic,
            lr_ttfe=config.lr_ttfe,
            gamma=config.gamma,
            tau=config.tau,
            alpha=config.alpha_fixed,
            weight_decay=config.weight_decay_critic,
            buffer_capacity=config.buffer_capacity,
            batch_size=config.batch_size,
            max_grad_norm=config.max_grad_norm,
            max_grad_norm_ttfe=getattr(config, "max_grad_norm_ttfe", None),
            n_atoms=config.n_atoms,
            hl_gauss_min=config.hl_gauss_min,
            hl_gauss_max=config.hl_gauss_max,
            hl_gauss_sigma=config.hl_gauss_sigma,
            tau_gumbel=config.tau_gumbel_init,
        )
    else:
        agent = SACAgent(
            stage=1,
            device=config.device,
            n_prices=config.n_prices,
            n_prices_flat=getattr(config, "n_prices_flat", None),
            d_model=config.d_model,
            nhead=config.nhead,
            n_layers=config.n_layers,
            seq_len=config.seq_len,
            static_dim=config.static_dim,
            hidden_dim=config.hidden_dim,
            lr_actor=config.lr_actor,
            lr_critic=config.lr_critic,
            lr_ttfe=config.lr_ttfe,
            gamma=config.gamma,
            tau=config.tau,
            buffer_capacity=config.buffer_capacity,
            batch_size=config.batch_size,
            max_grad_norm=config.max_grad_norm,
            max_grad_norm_critic=getattr(config, "max_grad_norm_critic", None),
            alpha_max=getattr(config, "alpha_max", float("inf")),
            idle_logit_bonus=getattr(config, "idle_logit_bonus", 0.0),
            tau_gumbel=config.tau_gumbel_init,
        )

    # Gumbel temperature annealing schedule
    tau_gumbel_range = config.tau_gumbel_init - config.tau_gumbel_final

    # Resume from checkpoint (network + optimizer state; replay buffer starts empty)
    if resume_from is not None:
        print(f"Resuming from checkpoint: {resume_from} at step {resume_step}")
        if reset_optimizers:
            print("  reset_optimizers=True — loading weights only; Adam moments reset to zero")
        agent.load_checkpoint(resume_from, reset_optimizers=reset_optimizers)
        if not is_tier2a:
            # tau_gumbel is restored from checkpoint; override with scheduled value
            # in case checkpoint predates the current schedule.
            frac_resume = min(1.0, resume_step / max(config.total_steps, 1))
            agent.tau_gumbel = config.tau_gumbel_init - frac_resume * tau_gumbel_range
            print(f"  tau_gumbel set to {agent.tau_gumbel:.4f} (step {resume_step})")

        # Confirm Adam state is fresh if reset was requested
        if reset_optimizers:
            def _n_state_entries(optim):
                return sum(len(v) for v in optim.state_dict()["state"].values())
            ttfe_n = _n_state_entries(agent.ttfe_optimizer)
            actor_n = _n_state_entries(agent.actor_optimizer)
            critic_n = _n_state_entries(agent.critic_optimizer)
            print(f"  Adam state entries per param group (should be 0 pre-first-step): "
                  f"ttfe={ttfe_n} actor={actor_n} critic={critic_n}")

    # Training loop
    obs, _ = env.reset()
    episode_reward = 0.0      # symlog-transformed (what the agent trains on)
    episode_raw_reward = 0.0  # pre-symlog (for comparison with v5.7/v5.8)
    episode_count = 0
    step = resume_step
    log_interval = config.log_interval
    save_interval = config.save_every
    t_start = time.time()

    # Rolling metrics for logging
    recent_rewards = []      # symlog-transformed episode totals
    recent_raw_rewards = []  # pre-symlog episode totals
    recent_socs = []
    mode_counts = {0: 0, 1: 0, 2: 0}  # charge=0, discharge=1, idle=2

    # Rolling enriched feature values for sanity logging (v6.0 only)
    recent_pct_rank_24h = []
    recent_z_24h = []
    recent_da_rt_basis = []

    # Last-good-state snapshot for NaN recovery
    prev_snapshot = None

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    print(f"Warming up for {config.warmup_steps} steps...")

    while step < config.total_steps:
        # Select action
        if is_tier2a:
            env_action, action_idx = agent.select_action(obs)
            action = env_action                      # 4D, for env.step
            buffer_action = np.array([action_idx], dtype=np.float32)  # for buffer
        else:
            action = agent.select_action(obs)
            buffer_action = action

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Apply symlog to economic reward component.
        raw_econ = info["energy_revenue"] + info["timing_bonus"]
        if is_tier1 or is_tier2a:
            # Pre-scale before symlog: ERCOT rewards of $0–1500/step compress to
            # $0–15, so symlog ≈ [0, 2.8], discounted return stays in HL-Gauss
            # support. Without scaling: q_symlog≈10 → symexp grad ≈ 22k → actor
            # grad explosion → TTFE corruption (observed: grad_a=1274 at step 6k).
            # SoC penalty scaled proportionally: -50/100 = -0.5 → symlog(-0.5) ≈ -0.41
            reward_scale = config.reward_scale
            scaled_econ = raw_econ / reward_scale
            soc_penalty_raw = -50.0 / reward_scale if info["soc_violated"] else 0.0
            transformed_reward = symlog(scaled_econ) + symlog(soc_penalty_raw)
        else:
            soc_penalty = -50.0 if info["soc_violated"] else 0.0
            transformed_reward = symlog(raw_econ) + soc_penalty

        episode_reward += transformed_reward
        episode_raw_reward += reward  # original env reward (pre-symlog) for logging
        recent_socs.append(info["soc"])
        mode_counts[info["mode"]] += 1

        # Track enriched feature values for sanity logging (indices in static_features)
        # static_features layout (enriched): [system(7), time(6), soc(1), price_feats(18)]
        # price_feats: [pct_rank_4h, pct_rank_12h, pct_rank_24h, z_4h, z_12h, z_24h, ...]
        if enriched_obs and "static_features" in obs:
            sf = obs["static_features"]
            if len(sf) >= 32:
                recent_pct_rank_24h.append(float(sf[16]))   # pct_rank_24h
                recent_z_24h.append(float(sf[19]))           # z_24h
                recent_da_rt_basis.append(float(sf[29]))     # da_rt_basis

        # Store symlog-transformed reward in replay buffer
        agent.buffer.add(obs, buffer_action, transformed_reward, next_obs, terminated)

        # Anneal Gumbel temperature (not used for Tier 2a — no Gumbel path)
        if not is_tier2a:
            frac = min(1.0, step / max(config.total_steps, 1))
            agent.tau_gumbel = config.tau_gumbel_init - frac * tau_gumbel_range

        # Update agent
        metrics = {}
        if step >= config.warmup_steps:
            if is_tier2a:
                metrics = agent.update()
            else:
                # Snapshot state every 100 steps for NaN recovery (non-Tier2a only;
                # Tier 2a relies on in-update NaN detection alone).
                if step % 100 == 0:
                    prev_snapshot = agent.snapshot_state()
                metrics = agent.update(tau_gumbel=agent.tau_gumbel)

            # NaN guard: check if update() detected NaN in parameters
            if metrics.get("nan_detected"):
                nan_source = metrics.get("nan_source", "unknown")
                print(
                    f"\nFATAL: NaN detected in {nan_source} at step {step}.",
                    flush=True,
                )
                if prev_snapshot is not None:
                    emergency_path = os.path.join(
                        config.checkpoint_dir,
                        f"emergency_pre_nan_step{step}.pt",
                    )
                    agent.save_emergency_checkpoint(emergency_path, prev_snapshot)
                    print(f"  Emergency checkpoint (last good state) saved: {emergency_path}")
                else:
                    ckpt_path = os.path.join(config.checkpoint_dir, f"nan_at_step{step}.pt")
                    agent.save_checkpoint(ckpt_path)
                    print(f"  NaN checkpoint (post-corruption) saved: {ckpt_path}")
                return agent, recent_rewards

        obs = next_obs
        step += 1

        if terminated or truncated:
            episode_count += 1
            recent_rewards.append(episode_reward)
            recent_raw_rewards.append(episode_raw_reward)
            episode_reward = 0.0
            episode_raw_reward = 0.0
            obs, _ = env.reset()

        # Logging
        if step % log_interval == 0 and metrics:
            elapsed = time.time() - t_start
            steps_per_sec = step / elapsed if elapsed > 0 else 0
            avg_reward = np.mean(recent_rewards[-10:]) if recent_rewards else 0
            avg_raw_reward = np.mean(recent_raw_rewards[-10:]) if recent_raw_rewards else 0
            avg_soc = np.mean(recent_socs[-288:]) if recent_socs else 0

            # Mode distribution over the logging window
            total_modes = sum(mode_counts.values())
            if total_modes > 0:
                mode_pct_charge = 100.0 * mode_counts[0] / total_modes
                mode_pct_discharge = 100.0 * mode_counts[1] / total_modes
                mode_pct_idle = 100.0 * mode_counts[2] / total_modes
            else:
                mode_pct_charge = mode_pct_discharge = mode_pct_idle = 0.0
            mode_counts = {0: 0, 1: 0, 2: 0}  # reset window

            gumbel_temperature = agent.tau_gumbel

            # Check for NaN in metrics values (belt-and-suspenders with param check)
            has_nan = any(
                np.isnan(v) for v in metrics.values() if isinstance(v, float)
            )
            nan_flag = " *** NaN DETECTED ***" if has_nan else ""

            # Enriched feature summary (v6.0 only)
            feat_str = ""
            if enriched_obs and recent_pct_rank_24h:
                avg_pct = np.mean(recent_pct_rank_24h[-200:])
                avg_z   = np.mean(recent_z_24h[-200:])
                avg_basis = np.mean(recent_da_rt_basis[-200:])
                feat_str = (f" | pct24h={avg_pct:.2f} z24h={avg_z:.2f}"
                            f" da_rt={avg_basis:.4f}")

            # Batch-level mode distribution (from policy, not env execution)
            b_ch = metrics.get('mode_probs_ch', 0) * 100
            b_dc = metrics.get('mode_probs_dc', 0) * 100
            b_id = metrics.get('mode_probs_id', 0) * 100

            if is_tier2a:
                # Tier 2a: HL-Gauss + policy entropy + per-action distribution
                action_dist = " ".join(
                    f"p{i}={metrics.get(f'action_p{i}', 0):.2f}" for i in range(config.n_actions)
                )
                t2a_str = (
                    f" | bin_ent={metrics.get('critic_bin_entropy', 0):.3f}"
                    f" bin_argmax={metrics.get('critic_bin_argmax_support_value', 0):.2f}"
                    f" q_exp_mean={metrics.get('q_expected_mean', 0):.2f}"
                    f" q_exp_maxabs={metrics.get('q_expected_max_abs', 0):.1f}"
                    f" | pi_H={metrics.get('policy_entropy', 0):.3f}"
                    f" | {action_dist}"
                )
                if metrics.get("grad_a_pre_clip", 0) > 50.0:
                    t2a_str += (
                        f" | SPIKE batch_price_max=${metrics.get('batch_price_max', 0):.0f}"
                        f" n>2k={metrics.get('batch_price_n_gt_2k', 0)}"
                        f" n>5k={metrics.get('batch_price_n_gt_5k', 0)}"
                    )
                print(
                    f"Step {step:>7d}/{config.total_steps} | "
                    f"ep={episode_count} | "
                    f"critic={metrics.get('critic_loss', 0):.4f} | "
                    f"actor={metrics.get('actor_loss', 0):.4f} | "
                    f"avg_reward={avg_reward:.1f} | "
                    f"avg_raw_reward={avg_raw_reward:.1f} | "
                    f"avg_soc={avg_soc:.2f} | "
                    f"grad_c={metrics.get('grad_c_pre_clip', 0):.3f}"
                    f"→{metrics.get('grad_c_post_clip', 0):.3f} | "
                    f"grad_a={metrics.get('grad_a_pre_clip', 0):.3f}"
                    f"→{metrics.get('grad_a_post_clip', 0):.3f} | "
                    f"grad_t={metrics.get('ttfe_grad_norm', 0):.3f} "
                    f"[proj={metrics.get('grad_ttfe_proj', 0):.1f} attn={metrics.get('grad_ttfe_attn', 0):.1f}] | "
                    f"mode_env=[ch={mode_pct_charge:.0f}% dc={mode_pct_discharge:.0f}% id={mode_pct_idle:.0f}%] "
                    f"mode_batch=[ch={b_ch:.0f}% dc={b_dc:.0f}% id={b_id:.0f}%] | "
                    f"{steps_per_sec:.1f} steps/s{t2a_str}{nan_flag}",
                    flush=True,
                )
            elif is_tier1:
                # Tier 1: no alpha logging; add HL-Gauss diagnostics
                tier1_str = (
                    f" | bin_ent={metrics.get('critic_bin_entropy', 0):.3f}"
                    f" bin_argmax={metrics.get('critic_bin_argmax_support_value', 0):.2f}"
                    f" q_exp_mean={metrics.get('q_expected_mean', 0):.2f}"
                    f" q_exp_maxabs={metrics.get('q_expected_max_abs', 0):.1f}"
                )
                # Spike-gated ERCOT price diagnostic (v2.1): only log batch price
                # distribution when grad_a_pre > 50 — characterizes secondary
                # amplification path without polluting normal logs.
                if metrics.get("grad_a_pre_clip", 0) > 50.0:
                    tier1_str += (
                        f" | SPIKE batch_price_max=${metrics.get('batch_price_max', 0):.0f}"
                        f" n>2k={metrics.get('batch_price_n_gt_2k', 0)}"
                        f" n>5k={metrics.get('batch_price_n_gt_5k', 0)}"
                    )
                print(
                    f"Step {step:>7d}/{config.total_steps} | "
                    f"ep={episode_count} | "
                    f"critic={metrics.get('critic_loss', 0):.4f} | "
                    f"actor={metrics.get('actor_loss', 0):.4f} | "
                    f"avg_reward={avg_reward:.1f} | "
                    f"avg_raw_reward={avg_raw_reward:.1f} | "
                    f"avg_soc={avg_soc:.2f} | "
                    f"grad_c={metrics.get('grad_c_pre_clip', 0):.3f}"
                    f"→{metrics.get('grad_c_post_clip', 0):.3f} | "
                    f"grad_a={metrics.get('grad_a_pre_clip', 0):.3f}"
                    f"→{metrics.get('grad_a_post_clip', 0):.3f} | "
                    f"grad_t={metrics.get('ttfe_grad_norm', 0):.3f} "
                    f"[proj={metrics.get('grad_ttfe_proj', 0):.1f} attn={metrics.get('grad_ttfe_attn', 0):.1f}] | "
                    f"mode_env=[ch={mode_pct_charge:.0f}% dc={mode_pct_discharge:.0f}% id={mode_pct_idle:.0f}%] "
                    f"mode_batch=[ch={b_ch:.0f}% dc={b_dc:.0f}% id={b_id:.0f}%] | "
                    f"tau_g={gumbel_temperature:.3f} | "
                    f"{steps_per_sec:.1f} steps/s{tier1_str}{nan_flag}",
                    flush=True,
                )
            else:
                print(
                    f"Step {step:>7d}/{config.total_steps} | "
                    f"ep={episode_count} | "
                    f"critic={metrics.get('critic_loss', 0):.4f} | "
                    f"actor={metrics.get('actor_loss', 0):.4f} | "
                    f"alpha={metrics.get('alpha', 0):.4f} | "
                    f"avg_reward={avg_reward:.1f} | "
                    f"avg_raw_reward={avg_raw_reward:.1f} | "
                    f"avg_soc={avg_soc:.2f} | "
                    f"grad_c={metrics.get('grad_c_pre_clip', metrics.get('critic_grad_norm', 0)):.3f}"
                    f"→{metrics.get('grad_c_post_clip', metrics.get('critic_grad_norm', 0)):.3f} "
                    f"[q1={metrics.get('grad_q1', 0):.1f} q2={metrics.get('grad_q2', 0):.1f}] | "
                    f"grad_a={metrics.get('grad_a_pre_clip', metrics.get('actor_grad_norm', 0)):.3f}"
                    f"→{metrics.get('grad_a_post_clip', metrics.get('actor_grad_norm', 0)):.3f} | "
                    f"grad_t={metrics.get('ttfe_grad_norm', 0):.3f} "
                    f"[proj={metrics.get('grad_ttfe_proj', 0):.1f} attn={metrics.get('grad_ttfe_attn', 0):.1f}] | "
                    f"q_mean={metrics.get('q_mean', 0):.2f} q_maxabs={metrics.get('q_max_abs', 0):.1f} | "
                    f"mode_env=[ch={mode_pct_charge:.0f}% dc={mode_pct_discharge:.0f}% id={mode_pct_idle:.0f}%] "
                    f"mode_batch=[ch={b_ch:.0f}% dc={b_dc:.0f}% id={b_id:.0f}%] | "
                    f"tau_g={gumbel_temperature:.3f} | "
                    f"{steps_per_sec:.1f} steps/s{feat_str}{nan_flag}",
                    flush=True,
                )

            if has_nan:
                print("FATAL: NaN detected in metrics. Saving emergency checkpoint and stopping.")
                if prev_snapshot is not None:
                    emergency_path = os.path.join(
                        config.checkpoint_dir, f"emergency_step{step}.pt"
                    )
                    agent.save_emergency_checkpoint(emergency_path, prev_snapshot)
                    print(f"  Emergency checkpoint saved: {emergency_path}")
                else:
                    ckpt_path = os.path.join(
                        config.checkpoint_dir, f"nan_metrics_step{step}.pt"
                    )
                    agent.save_checkpoint(ckpt_path)
                    print(f"  NaN checkpoint saved: {ckpt_path}")
                return agent, []

            # Clear old history to avoid memory growth
            if len(recent_socs) > 1000:
                recent_socs = recent_socs[-500:]
            if len(recent_pct_rank_24h) > 2000:
                recent_pct_rank_24h = recent_pct_rank_24h[-1000:]
                recent_z_24h = recent_z_24h[-1000:]
                recent_da_rt_basis = recent_da_rt_basis[-1000:]

        # Checkpointing
        if step % save_interval == 0:
            ckpt_path = os.path.join(config.checkpoint_dir, f"checkpoint_step{step}.pt")
            agent.save_checkpoint(ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}", flush=True)

    # Final checkpoint
    final_path = os.path.join(config.checkpoint_dir, "checkpoint_final.pt")
    agent.save_checkpoint(final_path)

    elapsed = time.time() - t_start
    print(f"\n=== Training Complete ===")
    print(f"Total steps: {step}")
    print(f"Episodes: {episode_count}")
    print(f"Time: {elapsed/3600:.2f} hours")
    print(f"Final checkpoint: {final_path}")

    if recent_rewards:
        print(f"Last 10 episode avg reward: {np.mean(recent_rewards[-10:]):.2f}")

    return agent, recent_rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1 Training")
    parser.add_argument("--steps", type=int, default=None, help="Override total_steps")
    parser.add_argument("--start", type=str, default=None, help="Override train_start date")
    parser.add_argument("--end", type=str, default=None, help="Override train_end date")
    parser.add_argument("--device", type=str, default=None, help="Override device")
    parser.add_argument("--log-interval", type=int, default=None, help="Override log interval")
    parser.add_argument(
        "--v60", action="store_true",
        help="Stage 1 v6.0: enriched obs (36-dim TTFE + 18 engineered features, obs_dim=108)"
    )
    parser.add_argument(
        "--v592", action="store_true",
        help="Stage 1 v5.9.2: stability fixes (lr_critic=1e-4, critic_clip=0.5, "
             "alpha_max=0.5, idle_logit_bonus=0.1). 500k validation run."
    )
    parser.add_argument(
        "--tier1", action="store_true",
        help="Stage 1 Tier 1 v1: BroNet critic + HL-Gauss loss + fixed alpha=0.1 "
             "+ gamma=0.97 + tau=0.001 + AdamW. 500k validation run."
    )
    parser.add_argument(
        "--tier2a", action="store_true",
        help="Stage 1 Tier 2a: discrete N=7 categorical action space atop Tier 1 "
             "stack. Replaces Gumbel+continuous with Categorical(7) over "
             "{-P, -2P/3, -P/3, 0, +P/3, +2P/3, +P}. 500k from-scratch run."
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Override checkpoint directory")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path (e.g., checkpoints/x/checkpoint_step50000.pt)")
    parser.add_argument("--resume-step", type=int, default=0,
                        help="Step counter to resume at; drives tau_gumbel anneal and save intervals")
    parser.add_argument("--reset-optimizers", action="store_true",
                        help="Resume model weights only; reset Adam optimizer moments to zero. "
                             "Use after a gradient-topology change to purge stale momentum.")
    args = parser.parse_args()

    # Set seed before anything else
    if args.seed is not None:
        import torch
        import random
        import numpy as np
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    if args.tier2a:
        config = Stage1Tier2aConfig()
    elif args.tier1:
        config = Stage1Tier1Config()
    elif args.v592:
        config = Stage1V592Config()
    elif args.v60:
        config = Stage1V60Config()
    else:
        config = Stage1Config()
    if args.steps is not None:
        config.total_steps = args.steps
    if args.start is not None:
        config.train_start = args.start
    if args.end is not None:
        config.train_end = args.end
    if args.device is not None:
        config.device = args.device
    if args.log_interval is not None:
        config.log_interval = args.log_interval
    if args.checkpoint_dir is not None:
        config.checkpoint_dir = args.checkpoint_dir

    train_stage1(config, enriched_obs=args.v60,
                 resume_from=args.resume, resume_step=args.resume_step,
                 reset_optimizers=args.reset_optimizers)
