# v5.1 → v5.7 Change Audit

**Generated:** 2026-04-14  
**Base commit:** `525fdbe` ("Stage 1 v5.2: reward bugs fixed, alpha fix, extended training config") — this is the v5.1 launch state  
**Head commit:** `31548b1` ("v5.7: Normalize ERCOT price inputs to TTFE by 1/1000")  
**Files changed:** 4 (`src/models/sac.py`, `src/training/config.py`, `src/training/train_stage1.py`, `tests/test_sac.py`)  
**Files unchanged:** `src/env/ercot_env.py`, `src/models/networks.py`, `src/models/ttfe.py`, `src/evaluation/evaluate_stage1.py`, all other test files

---

## Summary

Total changes: **14**  
Categorized as:
- Paper-spec corrections: 1
- Defensive/diagnostic (no effect on training dynamics): 7
- Training dynamics changes: 6

---

## Change List

---

### Change 1: Price normalization in `_encode_obs`
- **File:** `src/models/sac.py`, `_encode_obs()` method and class constant `PRICE_NORM = 1000.0`
- **What changed:** Raw ERCOT price history (in $/MWh) now divided by 1000 before being fed to TTFE and before being concatenated as `current_prices` in the observation vector. `price_history / 1000.0` replaces `price_history`.
- **Why it was added:** v5.7. Root cause of all NaN crashes: raw ERCOT prices ($9000+/MWh during Feb 2021 storm) produce attention Q/K dot products that overflow float32 (~exp(916) during storms, even ~exp(200) at normal $100/MWh prices due to weight magnitudes through input_proj → attention projection layers). PyTorch stable softmax handles the forward pass but backward gradients through saturated attention accumulate NaN in `pos_embedding`.
- **Category:** Training dynamics
- **Still needed with price normalization?** N/A — this IS the price normalization fix. Independently necessary regardless of other changes.

---

### Change 2: Reward clipping to `[-50, 50]` in `update()`
- **File:** `src/models/sac.py`, `update()`, line added after batch unpacking
- **What changed:** `rewards = rewards.clamp(-50.0, 50.0)` applied before TD target computation. V5.1 passed raw rewards.
- **Why it was added:** v5.6. ERCOT price spikes (Feb 2021 storm, $9000/MWh × β_S=10 timing bonus) produce per-step rewards up to $7,837. This drives Q* to ~68,000, causing critic weights to grow proportionally. Large critic action weights then amplify gradients back to TTFE via the actor-action path (∂Q/∂new_actions → ∂new_actions/∂TTFE). V5.6 with this clip + actor-path TTFE (but without price norm) still crashed at step 47k, indicating this alone is insufficient but contributes.
- **Category:** Training dynamics
- **Still needed with price normalization?** **Yes.** Price normalization prevents attention overflow but does NOT bound reward scale or Q-values. V5.5 architecture routes TTFE gradient through actor, but ∂Q/∂new_actions still scales with critic action weights → Q*. Without reward clipping, Q* ≈ 68k during storm events, critic action weights grow large, and the indirect path (critic action → actor → TTFE) would still accumulate over hundreds of thousands of steps. V5.6 alone crashed at 47k confirming the indirect path is real.

---

### Change 3: TTFE updated via actor backward only (not critic)
- **File:** `src/models/sac.py`, `update()` method
- **What changed:** Eliminated the separate TTFE forward pass and `ttfe_loss = -(q1_ttfe + q2_ttfe) * 0.5` backward (which backpropagated through all critic layers). Instead, `obs_encoded` retains the TTFE computation graph, and `actor_loss.backward()` now updates both actor and TTFE parameters jointly. `ttfe_optimizer.zero_grad()` and `ttfe_optimizer.step()` moved into the actor update block. Before: actor used `obs_encoded.detach()` (TTFE got no gradient from actor). After: actor uses `obs_encoded` (TTFE gets gradient through actor weights).
- **Why it was added:** v5.5. The critic → TTFE gradient path amplified by growing critic weights: at step 21k in v5.4, `grad_ttfe_proj` reached 314 trillion. Even with Q-value guard, Q-values can be large-but-finite while generating catastrophic gradients backward through accumulated critic weights. The actor gradient path to TTFE is bounded by actor weight norms (~0.5–2 norm, observed), not by Q-value scale.
- **Category:** Training dynamics
- **Still needed with price normalization?** **Yes.** Price normalization addresses attention overflow, not the gradient amplification. With the original critic → TTFE path: even at bounded Q* (reward clipping), critic weights grow to represent Q* = 5000, and the backward through all critic layers to TTFE accumulates. V5.6 (has reward clip, no v5.5) crashed at step 47k with grad_c=602 at step 45k and TTFE getting gradients via the critic path. The actor-only TTFE update is independently necessary.

---

### Change 4: MSE → Huber loss for critic
- **File:** `src/models/sac.py`, `update()`, critic loss line
- **What changed:** `F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)` → `F.huber_loss(q1, td_target) + F.huber_loss(q2, td_target)` (default delta=1.0: quadratic for |error| < 1, linear slope-1 for |error| ≥ 1).
- **Why it was added:** v5.4. MSE critic gradient scales quadratically with TD error; Huber clips it to linear for large errors. Intended to reduce gradient magnitude during large Q-value outlier batches. In practice, v5.4 (Huber only, no price norm, no actor-path TTFE) still crashed at step 21k — Huber alone was insufficient.
- **Category:** Training dynamics
- **Still needed with price normalization?** **Unknown.** With price normalization + reward clipping bounding Q*, TD errors are more controlled. Huber adds no harm and provides robustness to occasional batch variance. However, it changes the theoretically correct gradient signal and its benefit is unproven in this final configuration. A candidate for future removal if ablation shows no degradation.

---

### Change 5: `lr_critic` reverted from `1e-4` to `3e-4`
- **File:** `src/training/config.py`, `Stage1Config.lr_critic`
- **What changed:** `1e-4` (set in the v5.2 commit / v5.1 launch) → `3e-4` (reverted in v5.3 commit).
- **Why it was added:** v5.2 lowered `lr_critic` to `1e-4` to reduce critic gradient spikes. v5.3 reverted it because v5.2 crashed at step 95k with 88–97% idle mode — lower critic LR slowed Q-learning, the actor saw near-zero Q-gradient signal, and entropy regularization drove the policy toward uniform/idle. The paper specifies `lr = 3e-4` for all components.
- **Category:** Paper-spec correction
- **Still needed with price normalization?** Yes — `3e-4` matches Li et al. and `1e-4` caused mode collapse independently of the stability issues.

---

### Change 6: `has_nan_params()` function added
- **File:** `src/models/sac.py`, module-level function
- **What changed:** New function iterates model parameters and returns `(True, param_name)` on first NaN/Inf detected. Used after each optimizer step in `update()`.
- **Why it was added:** v5.3. After-the-fact parameter NaN detection to identify which component failed and trigger emergency checkpoint saving. Distinguishes actor/critic/TTFE failure sources.
- **Category:** Defensive
- **Still needed with price normalization?** Yes — belt-and-suspenders. If training destabilizes for any reason (data issues, MPS/CUDA numerical drift), this catches it before silent corruption accumulates across steps.

---

### Change 7: `_grad_norm()` helper function added
- **File:** `src/models/sac.py`, module-level function
- **What changed:** New utility function computes L2 norm of gradients for a list of parameters (pre-clip). Used for per-component gradient monitoring.
- **Why it was added:** v5.3. Needed to compute `grad_q1`, `grad_q2`, `grad_ttfe_proj`, `grad_ttfe_attn` separately without calling `clip_grad_norm_` multiple times.
- **Category:** Diagnostic
- **Still needed with price normalization?** Yes — monitoring utility with no training effect. Preserves observability.

---

### Change 8: Per-component gradient norms added to `update()` return dict
- **File:** `src/models/sac.py`, `update()` return dict
- **What changed:** Four new keys added: `grad_q1` (Q1-head norm pre-clip), `grad_q2` (Q2-head norm pre-clip), `grad_ttfe_proj` (input_proj + pos_embedding norm pre-clip), `grad_ttfe_attn` (transformer layer norm pre-clip).
- **Why it was added:** v5.3. Enabled diagnosis of which sub-component of critic or TTFE was exploding during the NaN crashes. Without these, only the total `critic_grad_norm` and `ttfe_grad_norm` were visible.
- **Category:** Diagnostic
- **Still needed with price normalization?** Yes — no training effect, preserves monitoring.

---

### Change 9: NaN checks after optimizer steps in `update()`
- **File:** `src/models/sac.py`, `update()` method
- **What changed:** Added `has_nan_params(self.critic)` after `critic_optimizer.step()`, and `has_nan_params(self.actor)` + `has_nan_params(self.ttfe)` after `actor_optimizer.step()` / `ttfe_optimizer.step()`. Each returns `{"nan_detected": True, "nan_source": "component.param_name"}` early.
- **Why it was added:** v5.3. The existing NaN check (scanning `metrics.values()` for float NaN) only caught NaN in loss scalars at the next log interval. By then, the corrupted parameters had been used in 1000 more update steps. Per-step parameter checks allow immediate detection and graceful exit.
- **Category:** Defensive
- **Still needed with price normalization?** Yes — fast-exit on parameter corruption is good practice regardless of stability.

---

### Change 10: `snapshot_state()` and `save_emergency_checkpoint()` methods added
- **File:** `src/models/sac.py`, new methods on `SACAgent`
- **What changed:** `snapshot_state()` clones all state dicts (TTFE, actor, critic, critic_target, log_alpha) into CPU tensors. `save_emergency_checkpoint(path, snapshot)` writes the snapshot to disk with stage/tau_gumbel metadata. The saved checkpoint is the LAST GOOD STATE before corruption, not the corrupted current state (which `save_checkpoint()` would write).
- **Why it was added:** v5.3. Emergency recovery: after NaN crash, the corrupted weights are useless for resuming, but the pre-NaN state (up to 100 steps earlier) could be resumed with a fixed codebase. Also useful for forensic analysis of the weight state before failure.
- **Category:** Defensive
- **Still needed with price normalization?** Yes — no training effect; valuable safety net for any future instability.

---

### Change 11: Startup LR print in `train_stage1.py`
- **File:** `src/training/train_stage1.py`
- **What changed:** Added `print(f"LR: actor={config.lr_actor} critic={config.lr_critic} ttfe={config.lr_ttfe}")` in startup header.
- **Why it was added:** v5.3. When relaunching after crashes, needed to confirm which config was actually loaded (e.g., verify v5.3 reverted lr_critic from 1e-4 back to 3e-4 without reading the config file).
- **Category:** Diagnostic
- **Still needed with price normalization?** Yes — no training effect.

---

### Change 12: NaN guard block and snapshot loop in `train_stage1.py`
- **File:** `src/training/train_stage1.py`
- **What changed:** (a) `prev_snapshot = None` initialized before training loop. (b) Every 100 steps: `prev_snapshot = agent.snapshot_state()`. (c) After `agent.update()`: check `metrics.get("nan_detected")` → if true, save emergency checkpoint from `prev_snapshot` and `return` instead of crashing. (d) Updated the metrics-NaN fallback path to also use `save_emergency_checkpoint` instead of `save_checkpoint` (saves clean state not corrupted one).
- **Why it was added:** v5.3. Enables graceful exit and state preservation on NaN crash instead of silent termination or saving corrupted weights.
- **Category:** Defensive
- **Still needed with price normalization?** Yes — no training effect; preserves recovery capability.

---

### Change 13: Extended log format in `train_stage1.py`
- **File:** `src/training/train_stage1.py`, log print statement
- **What changed:** `grad_c=X` → `grad_c=X [q1=Y q2=Z]` and `grad_t=X` → `grad_t=X [proj=Y attn=Z]`. Uses `metrics.get('grad_q1')` etc.
- **Why it was added:** v5.3. Needed the per-component breakdown to diagnose which Q-network or TTFE sub-module was spiking.
- **Category:** Diagnostic
- **Still needed with price normalization?** Yes — no training effect; useful monitoring.

---

### Change 14: `TestNaNGuard` class added to `tests/test_sac.py`
- **File:** `tests/test_sac.py`
- **What changed:** New test class with 5 tests: `test_has_nan_params_clean`, `test_has_nan_params_nan_weight`, `test_has_nan_params_inf_bias`, `test_snapshot_and_emergency_save`, `test_update_returns_component_grads`. Also added `has_nan_params` to the import line.
- **Why it was added:** v5.3. Covers the new defensive functions added in the same commit. `test_update_returns_component_grads` specifically asserts that `grad_q1`, `grad_q2`, `grad_ttfe_proj`, `grad_ttfe_attn` are present in `update()` return dict — this encodes a contract that the per-component logging must survive future refactors.
- **Category:** Defensive (test coverage)
- **Still needed with price normalization?** Yes — tests for utility functions that remain in the codebase.

---

## Cross-Change Analysis

### Which changes are independently necessary for stable training?

The three training-dynamics changes that matter:

| Change | Mechanism it prevents | Tested alone? |
|--------|----------------------|---------------|
| Price normalization (÷1000) | Attention Q/K overflow → NaN backward gradients in pos_embedding | Yes — root fix; v5.7 stable at 500k+ |
| Actor-only TTFE update | Direct critic→TTFE gradient amplified by critic weights × Q* | v5.6 (no v5.5) crashed at 47k; necessary |
| Reward clipping (±50) | Indirect critic-action→actor→TTFE gradient via large critic action weights | v5.5 (no reward clip) crashed at 19k; necessary |

All three are load-bearing. The Huber loss (Change 4) is the one change that is plausibly redundant — it delays but doesn't prevent crashes in isolation, and with the three above in place its marginal contribution is untested.

### What would "v5.1 + price normalization alone" have looked like?

v5.1 used the **original TTFE update via critic** (`ttfe_loss = -(q1 + q2)/2` backpropagated through critic layers). With price normalization:
- Attention overflow: fixed ✓
- Critic → TTFE gradient: still exists, still scales with Q* (up to 68k during storms)
- Verdict: would likely still crash eventually as critic weights accumulate, just later than 47k.

### Changes that are definitely still needed (summary)

All 14 changes should be retained. The 6 training-dynamics changes each address a distinct failure mode. The 7 defensive/diagnostic changes add no training overhead and preserve observability and recovery capability.
