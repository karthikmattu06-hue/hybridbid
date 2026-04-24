"""
Stage 1 Tier 2c: offline IQL training on MILP expert trajectories (Option D path b).

Training protocol (paper-standard; do NOT tune):
    - 200,000 gradient steps, batch size 256, uniform sampling with replacement
    - Eval every 25k steps on the val NPZ (loss + policy action distribution)
    - Checkpoints every 25k steps
    - Mandatory process-level `sys.exit()` at step 50_000 (non-negotiable)
      unless --resume-from-pause is passed.

Invocation:
    # First launch (will halt at 50k)
    python -m src.training.train_iql --seed 42 --device cuda

    # Smoke test (5k steps, report every 1k)
    python -m src.training.train_iql --smoke --steps 5000 --log-interval 1000 --device cpu

    # Resume post-review
    python -m src.training.train_iql --seed 42 --device cuda \
        --resume-from checkpoints/tier2c_seed42/checkpoint_step50000.pt \
        --resume-step 50000 --resume-from-pause
"""
from __future__ import annotations

import argparse
import logging
import os
import resource
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

# Make package imports work when launched as `python src/training/train_iql.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.agents.iql import IQLAgent, IQLConfig
from src.training.config import Stage1Tier2cConfig


# ───────────────────────────────────────────────────────────────────────
# Logging
# ───────────────────────────────────────────────────────────────────────
def _make_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("train_iql")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, mode="a"); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger


# ───────────────────────────────────────────────────────────────────────
# Dataset — memory-mapped NPZ with uniform sampling
# ───────────────────────────────────────────────────────────────────────
class OfflineNPZDataset:
    """
    Wraps an Option D preprocessor NPZ file. All arrays are loaded into RAM once;
    `sample(batch_size)` returns a dict of torch tensors on `device`.
    """

    REQUIRED_KEYS = (
        "price_history", "static_features",
        "next_price_history", "next_static_features",
        "actions", "rewards", "dones",
    )

    def __init__(self, npz_path: str | Path, device: str = "cpu"):
        self.path = Path(npz_path)
        self.device = device
        with np.load(self.path) as d:
            for k in self.REQUIRED_KEYS:
                if k not in d.files:
                    raise KeyError(f"NPZ {self.path} missing required key: {k}")
            self.price_history        = d["price_history"].astype(np.float32)
            self.static_features      = d["static_features"].astype(np.float32)
            self.next_price_history   = d["next_price_history"].astype(np.float32)
            self.next_static_features = d["next_static_features"].astype(np.float32)
            self.actions              = d["actions"].astype(np.int64)
            self.rewards              = d["rewards"].astype(np.float32)
            self.dones                = d["dones"].astype(np.float32)

        self.N = self.actions.shape[0]

    def __len__(self) -> int:
        return self.N

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict:
        idx = rng.integers(0, self.N, size=batch_size)
        def _t(x):
            return torch.from_numpy(x).to(self.device, non_blocking=True)
        return {
            "price_history":        _t(self.price_history[idx]),
            "static_features":      _t(self.static_features[idx]),
            "next_price_history":   _t(self.next_price_history[idx]),
            "next_static_features": _t(self.next_static_features[idx]),
            "actions":              _t(self.actions[idx]),
            "rewards":              _t(self.rewards[idx]),
            "dones":                _t(self.dones[idx]),
        }

    def iter_batches(self, batch_size: int, shuffle: bool = False, rng: np.random.Generator | None = None):
        """Full-pass iterator (used by eval). Drops final incomplete batch."""
        order = np.arange(self.N)
        if shuffle:
            assert rng is not None
            rng.shuffle(order)
        for start in range(0, self.N - batch_size + 1, batch_size):
            idx = order[start : start + batch_size]
            def _t(x):
                return torch.from_numpy(x).to(self.device, non_blocking=True)
            yield {
                "price_history":        _t(self.price_history[idx]),
                "static_features":      _t(self.static_features[idx]),
                "next_price_history":   _t(self.next_price_history[idx]),
                "next_static_features": _t(self.next_static_features[idx]),
                "actions":              _t(self.actions[idx]),
                "rewards":              _t(self.rewards[idx]),
                "dones":                _t(self.dones[idx]),
            }


# ───────────────────────────────────────────────────────────────────────
# Eval
# ───────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(agent: IQLAgent, val_ds: OfflineNPZDataset, batch_size: int = 512,
             max_batches: int = 100, logger: logging.Logger | None = None) -> dict:
    """Val-side losses + policy action distribution. Does NOT touch env."""
    agent.ttfe.eval(); agent.actor.eval(); agent.critic.eval(); agent.value.eval()

    import torch.nn.functional as F
    from src.agents.iql import symlog_scale, expectile_loss

    v_losses, q_losses = [], []
    v_means = []
    action_counts = np.zeros(agent.cfg.n_actions, dtype=np.int64)
    seen = 0

    rng = np.random.default_rng(0)
    for b_i, batch in enumerate(val_ds.iter_batches(batch_size, shuffle=True, rng=rng)):
        if b_i >= max_batches:
            break

        ph, sf = batch["price_history"], batch["static_features"]
        nph, nsf = batch["next_price_history"], batch["next_static_features"]
        a = batch["actions"].long()
        r = batch["rewards"].float()
        dn = batch["dones"].float()
        B = a.shape[0]
        batch_idx = torch.arange(B, device=agent.device)

        enc = agent._encode_obs(ph, sf)
        enc_next = agent._encode_obs(nph, nsf)

        # V
        q1_l, q2_l = agent.critic(enc)
        q1_a = q1_l[batch_idx, a, :]
        q2_a = q2_l[batch_idx, a, :]
        q1_sl = agent.hl_gauss.transform_from_probs(F.softmax(q1_a, dim=-1)).squeeze(-1)
        q2_sl = agent.hl_gauss.transform_from_probs(F.softmax(q2_a, dim=-1)).squeeze(-1)
        q_min_sl = torch.minimum(q1_sl, q2_sl)
        v_sl = agent.value(enc).squeeze(-1)
        v_losses.append(expectile_loss(q_min_sl - v_sl, agent.cfg.expectile_tau).item())
        v_means.append(v_sl.mean().item())

        # Q
        v_next_sl = agent.value_target(enc_next).squeeze(-1)
        r_sl = symlog_scale(r, scale=agent.cfg.reward_scale)
        td_target_sl = r_sl + agent.cfg.gamma * (1.0 - dn) * v_next_sl
        q_loss = (agent.hl_gauss.loss(q1_a, td_target_sl).mean()
                  + agent.hl_gauss.loss(q2_a, td_target_sl).mean()).item()
        q_losses.append(q_loss)

        # Policy action distribution
        logits = agent.actor(enc)
        a_pred = logits.argmax(dim=-1).detach().cpu().numpy()
        for k in range(agent.cfg.n_actions):
            action_counts[k] += int((a_pred == k).sum())
        seen += B

    agent.ttfe.train(); agent.actor.train(); agent.critic.train(); agent.value.train()

    dist = action_counts / max(seen, 1)
    metrics = {
        "val/v_loss": float(np.mean(v_losses)),
        "val/q_loss": float(np.mean(q_losses)),
        "val/v_mean_symlog": float(np.mean(v_means)),
        "val/N": seen,
        "val/action_dist": dist.tolist(),
    }
    if logger is not None:
        dist_str = " ".join(f"{d*100:5.1f}%" for d in dist)
        logger.info(
            f"  [eval N={seen}] v_loss={metrics['val/v_loss']:.4f}  "
            f"q_loss={metrics['val/q_loss']:.4f}  "
            f"v_mean_sl={metrics['val/v_mean_symlog']:+.3f}  "
            f"π_dist=[{dist_str}]"
        )
    return metrics


# ───────────────────────────────────────────────────────────────────────
# Memory helper
# ───────────────────────────────────────────────────────────────────────
def _rss_gb() -> float:
    # macOS reports ru_maxrss in bytes; Linux in KB. Normalize.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024**3)
    return rss / (1024**2)  # KB → GB


# ───────────────────────────────────────────────────────────────────────
# Train
# ───────────────────────────────────────────────────────────────────────
def train_iql(cfg: Stage1Tier2cConfig,
              resume_from: str | None = None,
              resume_step: int = 0,
              resume_from_pause: bool = False,
              smoke: bool = False) -> None:

    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path("logs") / f"{ckpt_dir.name}_train.log"
    logger = _make_logger(log_path)

    logger.info("=" * 70)
    logger.info(f"Stage 1 Tier 2c — Offline IQL (paper-standard, no tuning)")
    logger.info(f"  device={cfg.device}  total_steps={cfg.total_steps}  batch={cfg.batch_size}")
    logger.info(f"  expectile_tau={cfg.expectile_tau}  beta_adv={cfg.beta_advantage}  "
                f"gamma={cfg.gamma}  lr={cfg.lr}  wd={cfg.weight_decay}")
    logger.info(f"  checkpoint_dir={ckpt_dir}")
    logger.info("=" * 70)

    # Datasets
    logger.info(f"Loading train NPZ: {cfg.train_npz}")
    train_ds = OfflineNPZDataset(cfg.train_npz, device=cfg.device)
    logger.info(f"  train N = {len(train_ds):,}")
    logger.info(f"Loading val   NPZ: {cfg.val_npz}")
    val_ds = OfflineNPZDataset(cfg.val_npz, device=cfg.device)
    logger.info(f"  val   N = {len(val_ds):,}")

    # Agent
    n_prices_flat = getattr(cfg, "n_prices_flat", cfg.n_prices)
    obs_dim = cfg.d_model + n_prices_flat + cfg.static_dim   # TTFE(64) + raw(12) + static(14) = 90
    iql_cfg = IQLConfig(
        obs_dim=obs_dim,
        hidden_dim=cfg.hidden_dim,
        n_actions=cfg.n_actions,
        n_prices=cfg.n_prices,
        n_prices_flat=n_prices_flat,
        seq_len=cfg.seq_len,
        static_dim=cfg.static_dim,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        n_layers=cfg.n_layers,
        gamma=cfg.gamma,
        expectile_tau=cfg.expectile_tau,
        beta_advantage=cfg.beta_advantage,
        awr_weight_clip=cfg.awr_weight_clip,
        polyak_tau=cfg.polyak_tau,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        n_atoms=cfg.n_atoms,
        hl_gauss_min=cfg.hl_gauss_min,
        hl_gauss_max=cfg.hl_gauss_max,
        hl_gauss_sigma=cfg.hl_gauss_sigma,
        reward_scale=cfg.reward_scale,
        max_grad_norm=cfg.max_grad_norm,
        device=cfg.device,
    )
    agent = IQLAgent(iql_cfg)
    logger.info(f"Agent built. obs_dim={iql_cfg.obs_dim}  n_actions={iql_cfg.n_actions}  "
                f"n_atoms={iql_cfg.n_atoms}")

    step = 0
    if resume_from is not None:
        logger.info(f"Resuming from checkpoint: {resume_from}  (step {resume_step})")
        sd = torch.load(resume_from, map_location=cfg.device, weights_only=False)
        agent.load_state_dict(sd["agent"])
        step = resume_step

    # Non-negotiable pause gate
    pause_step = cfg.pause_step
    if step >= pause_step and not resume_from_pause:
        logger.error(f"Cannot start past pause step {pause_step} without --resume-from-pause. "
                     f"User review required before continuing.")
        sys.exit(2)

    # Training loop
    rng = np.random.default_rng(int(os.environ.get("IQL_SAMPLER_SEED", "42")))
    total_steps = cfg.total_steps if not smoke else min(cfg.total_steps, 5_000)
    log_interval = cfg.log_interval
    eval_interval = cfg.eval_interval
    save_every = cfg.save_every

    logger.info(f"Starting IQL training at step {step} → {total_steps}")
    t0 = time.time()

    # Rolling metrics
    acc = {k: 0.0 for k in (
        "loss/v", "loss/q", "loss/actor",
        "grad/v", "grad/q", "grad/actor", "grad/ttfe",
        "v/mean_symlog", "q/min_mean_sl",
        "awr/mean_w", "awr/max_w", "pi/entropy",
    )}
    acc_n = 0

    while step < total_steps:
        batch = train_ds.sample(cfg.batch_size, rng)
        m = agent.update(batch)
        step += 1
        acc_n += 1
        for k in acc:
            acc[k] += m.get(k, 0.0)

        if step % log_interval == 0:
            avg = {k: v / acc_n for k, v in acc.items()}
            dt = time.time() - t0
            sps = step / dt if dt > 0 else float("nan")
            logger.info(
                f"step {step:6d}/{total_steps}  "
                f"v={avg['loss/v']:.4f} q={avg['loss/q']:.4f} π={avg['loss/actor']:+.4f}  "
                f"v_sl={avg['v/mean_symlog']:+.3f} q_sl={avg['q/min_mean_sl']:+.3f}  "
                f"awr⟨w⟩={avg['awr/mean_w']:.2f} max={avg['awr/max_w']:.1f}  "
                f"H={avg['pi/entropy']:.3f}  "
                f"grads v/q/π/ttfe={avg['grad/v']:.2f}/{avg['grad/q']:.2f}/"
                f"{avg['grad/actor']:.2f}/{avg['grad/ttfe']:.2f}  "
                f"[{sps:.1f} steps/s, RSS={_rss_gb():.2f} GB]"
            )
            # Sanity guards
            if not np.isfinite(avg["loss/v"]) or not np.isfinite(avg["loss/q"]):
                logger.error("Non-finite loss detected — aborting.")
                sys.exit(3)
            if abs(avg["v/mean_symlog"]) > 18.0:
                logger.warning(
                    f"V(s) symlog mean |{avg['v/mean_symlog']:+.2f}| > 18 — approaching HL-Gauss "
                    f"support edge ±20. Investigate before next eval."
                )
            acc = {k: 0.0 for k in acc}
            acc_n = 0

        # Eval + checkpoint
        if step % eval_interval == 0:
            logger.info(f"── Eval at step {step} ──")
            evaluate(agent, val_ds, batch_size=512, max_batches=100, logger=logger)

        if step % save_every == 0:
            ckpt_path = ckpt_dir / f"checkpoint_step{step}.pt"
            torch.save({"agent": agent.state_dict(), "step": step, "cfg": asdict(cfg)}, ckpt_path)
            logger.info(f"  saved checkpoint → {ckpt_path}")

        # ── Mandatory process-level pause at 50k ──
        if step == pause_step and not resume_from_pause:
            logger.info("=" * 70)
            logger.info(f"MANDATORY PAUSE: reached step {pause_step}.")
            logger.info(f"Last checkpoint: {ckpt_dir/f'checkpoint_step{step}.pt'}")
            logger.info(f"Review required before continuing. Re-launch with:")
            logger.info(f"  --resume-from {ckpt_dir/f'checkpoint_step{step}.pt'} "
                        f"--resume-step {step} --resume-from-pause")
            logger.info("=" * 70)
            sys.exit(0)

    # Final save
    final_path = ckpt_dir / "checkpoint_final.pt"
    torch.save({"agent": agent.state_dict(), "step": step, "cfg": asdict(cfg)}, final_path)
    logger.info(f"Done. Final checkpoint → {final_path}")
    logger.info(f"Total wallclock: {(time.time()-t0)/60:.1f} min")


# ───────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Tier 2c IQL offline training")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--steps", type=int, default=None, help="Override total_steps")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--log-interval", type=int, default=None)
    p.add_argument("--eval-interval", type=int, default=None)
    p.add_argument("--save-every", type=int, default=None)
    p.add_argument("--checkpoint-dir", type=str, default=None)
    p.add_argument("--train-npz", type=str, default=None)
    p.add_argument("--val-npz", type=str, default=None)
    p.add_argument("--resume-from", type=str, default=None)
    p.add_argument("--resume-step", type=int, default=0)
    p.add_argument("--resume-from-pause", action="store_true",
                   help="Acknowledge user review complete — permit training past step 50,000.")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke test: cap total steps at 5,000 (no pause gate engagement).")
    args = p.parse_args()

    # Seed everything
    import random
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = Stage1Tier2cConfig()
    if args.device is not None:         cfg.device = args.device
    if args.steps is not None:          cfg.total_steps = args.steps
    if args.batch_size is not None:     cfg.batch_size = args.batch_size
    if args.log_interval is not None:   cfg.log_interval = args.log_interval
    if args.eval_interval is not None:  cfg.eval_interval = args.eval_interval
    if args.save_every is not None:     cfg.save_every = args.save_every
    if args.checkpoint_dir is not None: cfg.checkpoint_dir = args.checkpoint_dir
    if args.train_npz is not None:      cfg.train_npz = args.train_npz
    if args.val_npz is not None:        cfg.val_npz = args.val_npz

    train_iql(cfg,
              resume_from=args.resume_from,
              resume_step=args.resume_step,
              resume_from_pause=args.resume_from_pause,
              smoke=args.smoke)
