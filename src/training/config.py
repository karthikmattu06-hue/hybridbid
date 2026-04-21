"""
Hyperparameter configuration for TempDRL training.

Values aligned with Li et al. (2024) Table I where specified.
"""

from dataclasses import dataclass, field

import torch


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class TrainConfig:
    """Base training hyperparameters (Li et al. Table I defaults)."""

    # Data
    data_dir: str = "data/processed"
    seq_len: int = 32
    n_prices: int = 12
    static_dim: int = 14

    # TTFE — Li et al.: N_MHA=2, h=8
    d_model: int = 64
    nhead: int = 8       # paper: 8 heads
    n_layers: int = 2

    # Networks
    hidden_dim: int = 256

    # SAC — Li et al. Table I
    gamma: float = 0.99
    tau: float = 0.005   # τ_ψ target network smoothing (standard SAC default; 0.01 tracks volatile critic too closely under heavy-tailed rewards)

    # Gradient clipping (not in paper, kept as numerical safety)
    max_grad_norm: float = 1.0
    # Separate critic clip (v5.9.2+). None = use max_grad_norm for critic too.
    max_grad_norm_critic: float = None  # type: ignore[assignment]

    # Alpha entropy temperature bounds (v5.9.2+)
    # alpha_max=inf means no upper clamp (original behavior).
    alpha_max: float = float("inf")

    # Idle logit bonus (v5.9.2+): additive offset on idle mode logit before
    # Gumbel-Softmax to prevent zero-idle collapse. 0.0 = off.
    idle_logit_bonus: float = 0.0

    # Gumbel-Softmax temperature annealing
    tau_gumbel_init: float = 1.0   # start temperature
    tau_gumbel_final: float = 0.1  # end temperature

    # Battery — Li et al. uses η=0.95 for both
    p_max: float = 10.0
    e_max: float = 20.0
    soc_min_frac: float = 0.10
    soc_max_frac: float = 0.90
    soc_initial_frac: float = 0.50
    eta_ch: float = 0.95   # paper: 0.95
    eta_dch: float = 0.95  # paper: 0.95
    degradation_cost: float = 2.0  # kept for info tracking, not used in step reward

    # Device — auto-detect CUDA > MPS > CPU
    device: str = field(default_factory=_detect_device)


@dataclass
class Stage1Config(TrainConfig):
    """Stage 1: Energy-only pretraining."""

    # Training
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4   # v5.3: reverted from 1e-4 — lower LR worsened mode collapse in v5.2
    lr_ttfe: float = 3e-4
    buffer_capacity: int = 1_000_000
    batch_size: int = 256
    total_steps: int = 1_000_000  # v5.2: extended from 500k
    warmup_steps: int = 1000
    updates_per_step: int = 1

    # Data range
    train_start: str = "2020-01-01"
    train_end: str = "2023-12-31"

    # Logging / Checkpoint
    log_interval: int = 1_000
    checkpoint_dir: str = "checkpoints/stage1"
    save_every: int = 25_000   # v5.2: finer granularity from 50k


@dataclass
class Stage1V60Config(Stage1Config):
    """Stage 1 v6.0: Enriched observations with DAM LMPs + engineered price features.

    Key changes from v5.9:
      - n_prices=36: TTFE input expanded to 12 orig + 24 hourly DA LMP values
      - n_prices_flat=12: only original 12 dims used in flat obs (no double-count)
      - static_dim=32: 14 original + 18 engineered price features
      - obs_dim: 64 + 12 + 32 = 108 (was 64 + 12 + 14 = 90)
      - checkpoint_dir: checkpoints/stage1_v60
      All training dynamics identical to v5.9 (symlog, SAC v2, τ=0.005, etc.)
    """
    n_prices: int = 36        # TTFE input: 12 orig + 24 DA LMP
    n_prices_flat: int = 12   # flat obs: first 12 dims only (orig prices)
    static_dim: int = 32      # 14 orig + 18 engineered price features
    checkpoint_dir: str = "checkpoints/stage1_v60"
    save_every: int = 50_000  # checkpoints every 50k (same granularity as v5.9)


@dataclass
class Stage1V592Config(Stage1Config):
    """Stage 1 v5.9.2: four targeted stability fixes, 500k validation run.

    Diagnosed root cause from v5.9.1: critic gradients chronically at 50–300
    (clipped every step) caused unstable Q-values, which drove alpha to oscillate
    wildly (0.12 → 0.42 → 0.14 over 50k steps), causing mode flips and zero-idle
    collapse.

    Fixes applied:
      1. lr_critic 3e-4 → 1e-4: tame Q-gradient overload at the source.
      2. max_grad_norm_critic 1.0 → 0.5: tighter per-component critic clip.
      3. alpha_max 0.5: prevent spike-and-crash cycles (v5.9.1 hit 0.42 and collapsed).
      4. idle_logit_bonus 0.1: additive idle logit bonus to prevent zero-idle collapse.

    Budget: 500k steps (validation). Extend to 1M if trajectory is clean.
    """
    lr_critic: float = 1e-4
    max_grad_norm_critic: float = 0.5
    alpha_max: float = 0.5
    idle_logit_bonus: float = 0.1
    total_steps: int = 500_000
    checkpoint_dir: str = "checkpoints/stage1_v592"


@dataclass
class Stage1Tier1Config(Stage1Config):
    """Stage 1 Tier 1: BroNet critic + HL-Gauss loss + fixed alpha + γ=0.97.

    Root cause addressed: scalar-regression critic + Cauchy-tailed ERCOT rewards
    under γ=0.99 → Q-value explosion → alpha swing → mode collapse.

    Changes from Stage1Config:
      1. Critic: BroNetCritic (LayerNorm + 2 residual blocks + HL-Gauss head, hidden=512)
      2. Critic loss: HL-Gauss cross-entropy over 101-bin symlog support [-20, 20]
      3. Critic optimizer: AdamW lr=1e-4, weight_decay=0.1
      4. gamma: 0.99 → 0.97 (effective horizon ~33 steps @ 5-min resolution)
      5. alpha: 0.1 fixed (no auto-tuning; removes Q-swing → alpha-swing feedback)
      6. tau: 0.005 → 0.001 (slower target update contains spike propagation)
      7. Critic grad clip: max_norm=1.0 on combined critic params
      8. idle_logit_bonus=0.0, alpha_max=inf (v5.9.2 band-aids removed)
    """
    # BroNet hidden dim (larger than standard 256)
    hidden_dim: int = 512

    # Critic optimizer
    lr_critic: float = 1e-4
    weight_decay_critic: float = 0.1  # AdamW weight decay

    # HL-Gauss categorical head
    n_atoms: int = 101
    hl_gauss_min: float = -20.0
    hl_gauss_max: float = 20.0
    hl_gauss_sigma: float = 0.75

    # SAC hyperparams (Changes 4–6)
    gamma: float = 0.97
    alpha_fixed: float = 0.1          # fixed entropy temperature
    tau: float = 0.001

    # Remove v5.9.2 band-aids (Change 8)
    alpha_max: float = float("inf")
    idle_logit_bonus: float = 0.0
    max_grad_norm_critic: float = None  # type: ignore[assignment]  # unified clip

    # Reward pre-scaling: divide raw $ reward by REWARD_SCALE before symlog.
    # ERCOT rewards span $0–1500 normally (spikes to $10k+). At REWARD_SCALE=100,
    # symlog maps to [0, 2.8], discounted return ≈ 6.2 in symlog → symexp gradient
    # amplification ≈ 500×. Without scaling, raw rewards of $600+ compound to
    # q_symlog ≈ 10 → symexp gradient 22,000× → actor grad explosion → TTFE corruption.
    reward_scale: float = 100.0

    # TTFE clip (kept as secondary safety belt alongside reward scaling)
    lr_ttfe: float = 3e-4              # restored — reward scaling fixes root cause
    max_grad_norm_ttfe: float = 0.5    # moderate TTFE clip (not as aggressive as 0.1)

    # Run config
    total_steps: int = 500_000
    checkpoint_dir: str = "checkpoints/stage1_tier1_v1"
    log_interval: int = 1_000


@dataclass
class Stage1Tier2aConfig(Stage1Tier1Config):
    """Stage 1 Tier 2a: discrete N=7 categorical action space atop Tier 1 stack.

    Replaces the Gumbel-Softmax mode + squashed-Gaussian magnitude head with a
    single Categorical over 7 action levels in p.u. of P_max:
        {-P, -2P/3, -P/3, 0, +P/3, +2P/3, +P}

    Critic is rebuilt as Q(s) → (batch, N, n_atoms) with no action input, which
    removes the `∂Q/∂a` symexp-amplification path identified as the residual
    spike driver in Tier 1 v2.1.

    All other Tier 1 choices unchanged (BroNet, HL-Gauss, reward prescale ÷100,
    stop-gradient on actor→TTFE, fixed α=0.1, γ=0.97, τ=0.001, AdamW wd=0.1).
    """
    # Discrete action config
    n_actions: int = 7

    # Run config
    checkpoint_dir: str = "checkpoints/tier2a_seed42"
    save_every: int = 25_000     # mandatory checkpoint every 25k (eval cadence)
    log_interval: int = 1_000


@dataclass
class Stage1Tier2cConfig(Stage1Tier2aConfig):
    """Stage 1 Tier 2c: offline IQL on MILP expert trajectories (Option D path b).

    Paper-standard IQL (Kostrikov et al., ICLR 2022, arXiv:2110.06169), no tuning:
      - Expectile τ = 0.9, β_advantage = 5.0
      - γ = 0.97, AdamW lr = 3e-4, weight_decay = 0.01
      - Batch size 256, 200k total gradient steps
      - Eval every 25k; mandatory process-level pause at 50k for user review
      - Polyak τ = 0.005 on **target V only** (Q has no target)
      - Advantage-weight clip = 100.0
      - Reward pre-scale: symlog(r / 100)

    Trajectories: preprocessed from MILP expert via `scripts/preprocess_milp_option_d.py`:
      - train: 420,423 transitions (2020-01-01 → 2023-12-31)
      - val:   183,871 transitions (2024-01-01 → 2025-09-30)
      - Actions: 7-atom discrete lattice (98% mass on atoms 0, 3, 6)
      - Dones: all False; truncateds at UTC-date boundaries (1,460 in train)
      - Rewards ≡ MILP.rewards_env (zero mismatch by construction)

    TTFE ownership: V-head owns TTFE gradient (actor + Q use detached encoding),
    mirroring the Tier 1 / Tier 2a ownership pattern.
    """
    # IQL-specific hyperparameters
    expectile_tau: float = 0.9
    beta_advantage: float = 5.0
    awr_weight_clip: float = 100.0
    polyak_tau: float = 0.005

    # Optimizer — unified IQL lr/wd across all four heads
    lr: float = 3e-4
    weight_decay: float = 0.01

    # Offline run config
    batch_size: int = 256
    total_steps: int = 200_000
    pause_step: int = 50_000        # non-negotiable process-level halt
    eval_interval: int = 25_000
    log_interval: int = 1_000
    save_every: int = 25_000

    # Trajectory paths (produced by preprocess_milp_option_d.py)
    train_npz: str = "data/expert_trajectories/receding_horizon_train_option_d.npz"
    val_npz:   str = "data/expert_trajectories/receding_horizon_val_option_d.npz"

    checkpoint_dir: str = "checkpoints/tier2c_seed42"


@dataclass
class Stage2Config(TrainConfig):
    """Stage 2: Post-RTC+B co-optimization fine-tuning (stage2_v2).

    Key changes from v1:
      - target_entropy corrected to 5.0 in SACAgent (9D action space)
      - Alpha floor at 0.05 (log_alpha.clamp_(min=log(0.05)))
      - Phase C removed — two-phase only (A: 0–40%, B: 40–100%)
      - Gumbel τ: 0.8→0.5 in Phase A, hold 0.5 in Phase B
      - Total steps reduced from 150k to 120k (Phase C consumed wasted steps)
    """

    # Training
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_ttfe: float = 3e-5   # 10× lower for TTFE unfreezing phases
    buffer_capacity: int = 60_000
    batch_size: int = 256
    total_steps: int = 120_000
    warmup_steps: int = 5_000
    updates_per_step: int = 1

    # Gumbel temperature — higher start encourages mode exploration while AS heads learn
    tau_gumbel_init: float = 0.8   # Phase A start; anneals to 0.5 at Phase A end
    tau_gumbel_final: float = 0.5  # Phase B holds at 0.5 (no further annealing)

    # Two-phase TTFE unfreezing (Phase C removed — destabilized pretrained representations)
    phase_b_start_frac: float = 0.40   # Phase A: 0–40% frozen TTFE (steps 0–47999)
                                        # Phase B: 40–100% unfreeze top layer (steps 48000–119999)

    # Data range — post-RTC+B train; test = March 2026 (held out)
    train_start: str = "2025-12-05"
    train_end: str = "2026-02-28"

    # Stage 1 best checkpoint (v5.9 300k, $267/day)
    stage1_checkpoint: str = "checkpoints/stage1/checkpoint_step300000.pt"

    # Logging / Checkpoint
    log_interval: int = 1_000
    checkpoint_dir: str = "checkpoints/stage2_v2"
    save_every: int = 5_000


@dataclass
class Stage2V3aConfig(Stage2Config):
    """Stage 2 v3a: Enriched flat observation (18 price features), TTFE unchanged.

    Key changes from v2:
      - static_dim: 14 → 32 (adds 18 engineered price features to flat obs only)
      - TTFE input stays 12-dim — loads perfectly from v5.9 300k checkpoint
      - obs_dim: 90 → 108 (64 TTFE + 12 raw prices + 32 static)
      - Actor fc1 (90→256) cannot load from v5.9 (dim mismatch) — fresh init
      - Actor fc2 + mode/energy heads: copied from v5.9
      - total_steps: 200k (extended for broader search given larger obs space)
      - phase_b_start_frac: 0.40 → Phase B at step 80k
      - checkpoint_dir: checkpoints/stage2_v3a
    """
    static_dim: int = 32         # 14 orig + 18 engineered price features
    total_steps: int = 200_000   # extended from 120k
    checkpoint_dir: str = "checkpoints/stage2_v3a"
