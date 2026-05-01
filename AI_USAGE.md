# AI Usage Documentation

This document records the use of AI tools for code generation and code architecture decisions across the HybridBid project. Each entry follows the required format: tool used, request, what was generated, modifications made, and what was learned. A reflection section appears at the end.

## Tools Used

- **Claude Code (CLI agent)** code generation, refactoring, debugging assistance, repo restructuring.
- **Claude.ai (web interface, Opus 4.x)** code architecture decisions, root-cause diagnosis of training failures, methodology decisions that shaped code structure.

## Code Generation

### 1. Stage 1 Training Loop (SAC + TTFE)

- **Tool:** Claude Code
- **Request:** Implement the Stage 1 training loop adapted from Li et al. (2024, TempDRL), using SAC v2 with automatic entropy tuning and a Transformer Temporal Feature Extractor.
- **Generated:** Initial training script with SAC actor/critic networks, TTFE encoder, replay buffer, and training loop.
- **Modifications:** Multiple iterations across v1–v5.9.2. Significant rewrite at v5 (paper-spec reset) as deviations from Li et al. were diagnosed as the root cause of SoC drift. Kept gradient clipping (max_norm=1.0) as a non-paper addition. Spotted and flagged various issues with early implementations of the code that were leading to SoC drift and required significant debugging to identify and resolve.
- **Learned:** Every deviation from the paper specification required a major fix. Cross check specs at each stage and smoke test periodically

### 2. MILP Baselines (TBx, Perfect Foresight, Receding-Horizon Demonstrations)

- **Tool:** Claude Code
- **Request:** Implement three MILP baselines: (1) TBx energy-only with perfect price foresight, (2) Perfect Foresight oracle with full-horizon look-ahead, (3) receding-horizon MILP demonstration generator for offline RL training data.
- **Generated:** 3 baseline scripts.
- **Modifications:** Added the ERCOT AS sustain duration constraints by hand after reading ERCOT's documentation these were missing from the AI-generated formulation. Tuned the receding-horizon length and settled on 24 hours. Used HiGHS at defaults; ECOS was slower on the joint formulation.
- **Learned:** The reward computation that goes with the demonstrations needs to be checked against an independent calculation before training a downstream model on them. The Δt scaling and MW vs. p.u. bugs only surfaced when the recomputed rewards were compared against a separate physical-dollar calculation.

### 3. Offline RL Method Implementations (Cal-QL, Diffusion-QL, QDT)

- **Tool:** Claude Code
- **Request:** Implement Cal-QL, Diffusion-QL, and QDT for offline training on the MILP demonstration dataset, with shared evaluation harness integration.
- **Generated:** Three method implementations with twin-Q critics, calibration anchor (Cal-QL), diffusion policy parameterization (Diffusion-QL), and three-stage CQL→RTG-relabel→DT pipeline (QDT).
- **Modifications:** Added per-method diagnostics (calibration activation, Q-magnitude trajectory, RTG distribution) and unified all three evaluate() outputs to the eval harness's physical-dollar format.
- **Learned:** You can't tell why a method failed until you log the right things without tracking calibration activation and infeasibility rates, Cal-QL just looked like a bad model, not one whose actions were being silently corrected at every step.

### 4. Evaluation Harness

- **Tool:** Claude Code
- **Request:** Build a frozen evaluation harness over a 54-day post-RTC+B test window enforcing continuous battery state-of-charge, applying silent feasibility projection to proposed actions, and computing physical-dollar revenue separated by all-days, ex-Fern, and Fern-only.
- **Generated:**  Harness including a MILP-replay validation canary that returns \$58.40/kW-yr in every consistent eval.
- **Modifications:** Added per-step logging of the silent feasibility projection average AS scaling factor and count of steps requiring projection which the AI-generated harness did not include. Added the ex-Fern revenue split.
- **Learned:** Logging the projection's behavior was what made the Cal-QL infeasibility numbers visible at all. Without that logging, Cal-QL looked like a generic underperformer rather than a method whose deployed actions were being silently corrected.

### 5. ERCOT Data Pipeline

- **Tool:** Claude Code (initial scrapers); manual debugging
- **Request:** Build data ingestion for ERCOT public API across RT LMP, RT MCPC (5 products), DAM SPP, DAM AS clearing prices, and system variables (load, wind, solar forecasts).
- **Generated:** Initial scraper using the gridstatus library.
- **Modifications:** Substantial. The gridstatus scraper was largely broken following ERCOT's CSV→XML migration. Replaced with direct ErcotAPI calls. RT LMP required `get_lmp_by_settlement_point` (NP6-788-CD) for true 5-minute resolution rather than 15-minute bulk files. RT SCED MCPC required `NP6-332-CD` via the data API endpoint, not the archive. ECRS data has NaN before June 2023 and required handling. Wind/solar forecasts required deduplication by latest publish time.
- **Learned:** AI-generated scrapers built on top of outdated library docs can produce silently broken pipelines. A 429 rate-limit error was initially read as "no data exists" would have dropped about 170 days of data if not spot-checked against ERCOT's web UI.



## Code Architecture & Design Decisions

These entries cover Claude.ai conversations that shaped what code got written.

### 7. Two-Stage Architecture Design

- **Tool:** Claude.ai
- **Request:** Adapt TempDRL (SAC + TTFE) to ERCOT's post-RTC+B market break, designing a pretrain→finetune system that retains pre-RTC+B knowledge while adapting to the new joint-clearing structure.
- **Generated:** Claude.ai returned analysis of how a two-stage approach could work, describing the general structure and tradeoffs.
- **Design decisions (ours):** Two-stage design Stage 1 energy-only pretrain (1D action space, replay buffer 1M, batch size 256); Stage 2 fine-tune with 6D action space (replay buffer 30–50k, batch size 128); progressive TTFE unfreezing at 10× lower LR per ULMFiT; fresh critic re-initialization; partial actor initialization (energy from Stage 1, AS dimensions near-zero). Each decision about hyperparameters, initialization strategy, and code structure (separate config files per stage, weight-loading utilities, frozen-layer parameter groups) was made and validated by me.
- **Modifications:** I refined the design as Stage 1 instability emerged. When I pivoted the project to offline RL on post-RTC+B data, the structure I had built (separate Stage 2 entry point, MILP demonstration data loader) made the pivot a matter of swapping training scripts rather than rewriting.
- **Learned:** Designing for the possibility of a pivot before knowing whether one would happen kept the codebase flexible at low cost.

### 8. Paper-Spec Reset (v5)

- **Tool:** Claude.ai
- **Request:** Diagnose why version 1 to version 4 implementations were producing SoC-pinning and mode collapse despite multiple compensatory fixes in code.
- **Generated:** Root-cause analysis identifying that the cascade of code-level deviations from Li et al. (continuous-only action space, reward scaling, price normalization, alpha floor) were each masking symptoms of an upstream issue. Recommendation: code reset to paper specification (Gumbel-Softmax 3-class mode + continuous magnitude in the actor head, EMA arbitrage bonus τ=0.9 β=10 in the reward function, episode termination penalty), keeping only gradient clipping as a non-paper addition. This translated into a substantial code rewrite, not just a config change.
- **Modifications:** Implemented the reset (v5) and verified 89 tests passing. Subsequently identified mode collapse in v5.1 traceable to SAC v2's learned alpha being pulled down by the continuous magnitude component, a separate structural issue requiring a different code change (fixed alpha) rather than tuning.
- **Learned:** When patches accumulate, reverting to the reference and re-deriving each addition can be faster than continuing to patch.
### 9. Stage 2 AS Revenue Decoupling

- **Tool:** Claude.ai
- **Request:** Reconcile Li et al.'s binary mode formulation (which ties AS revenue to the active mode) with ERCOT's ADER/ESR framework (which allows AS availability payments while idle).
- **Generated:** Analysis showing that Li et al.'s binary mode could not be applied directly to our reward function; AS revenue needs to be decoupled from the action mode in Stage 2.
- **Modifications:** Decoupling was deferred to future work. The Stage 2 reward function uses Li et al.'s baseline formulation. Since the project pivoted to offline RL on demonstrations before Stage 2 finetune was reached, this did not become blocking.
- **Learned:** Adapting a published method to a different market structure means reading the paper's assumptions carefully, not just transcribing the algorithm.


## Reflection

### Where AI tools were most helpful for code

- **Drafting Long length of code with ease** The overall architechture of TTFE, SAC and the offline RL techniques span across 1500 lines of code. Drafting, implementing and changing this was a humungus task for us. Claude was a godsend as it not only made the code files for us, but also perform training, evaluation on the deep RL policy.
- **Simpler Diagnosis** When training failed (alpha collapse, critic-instability cascade, Cal-QL calibration deactivation, Diffusion-QL Q-divergence), having a conversation partner to walk through training logs, propose hypotheses, and rule out compensatory fixes was substantially faster than working in a silo. Our team strategy to work on different approaches was amazing as it helped us cover a lot of ground faster.
- **Refactoring under structure** Repository restructuring, separating Stage 1 and Stage 2 entry points, extracting hyperparameter blocks into configs.
- **Conversation with a virtual AI Engineer** Several times the AI suggestion was to add a compensatory mechanism (reward scaling, alpha floor, action penalty) when the right move was to revert and find the root cause. Pushing back on these suggestions — and having the AI then reason about why the deeper issue was real — was useful, but required me to recognize the pattern.

### Where AI tools were not helpful for code

- **Hallucination!!** Shubh faced a very big problem, he was implementing the project using his methodology, the claude gave him positive reinforcement and continued implementing the project. 3 days passed by only to know that the implementation was wrong and Claude hallucinated heavily. After a lot of working on claude, we observed a severe problem of hallucinating with the coding agent, which gave the code effectively but worked in the wrong direction.
- **Subtle bugs in generated code** The Δt scaling factor and MW vs. p.u. confusion both came from AI-generated code and survived multiple AI-assisted reviews. AI tools were poor at flagging when their own generated code had subtle bugs. External validation via the MILP replay canary, by spot-checking against known references, by running smoke tests caught what the AI didn't.
- **Definitive method recommendations** When asked "should we use Cal-QL or Diffusion-QL," AI tools could enumerate trade-offs but couldn't replace the empirical work of running both. The value was in framing the comparison and structuring the experiments, not in answering the comparison.
- **Domain-specific market knowledge** ERCOT's ADER/ESR framework, AS sustain duration requirements, and the structural change introduced by RTC+B required reading source documents directly. AI summaries often glossed over the implementation details that turned out to matter for the reward function and constraint code.

### How AI-generated code was verified

- **Smoke tests** Every significant code change was run through smoke tests before a full training run. Smoke tests caught most reward function and dimension mismatches early.
- **Eval-harness canary** The MILP-replay baseline returned \$58.40/kW-yr in every consistent eval. A different number meant the harness was broken before any policy result could be trusted. This invariant caught several silent regressions in the eval code.
- **Independent reward computation** Recomputed rewards verified against an independently computed physical dollar baseline within 1% tolerance.
- **Cross-codebase agreement** Stage 1 failure analysis used two independent implementations (Implementation A and B by different team members). Mechanism-level agreement across both was treated as stronger evidence than agreement within either alone.
- **Paper-spec reference** When in doubt, the paper spec (Li et al. 2024, arXiv:2402.19110) was the source of truth, not the AI-generated implementation. Several bugs were caught by re-reading the paper rather than re-reading the code.
- **Manual log inspection** Training logs (alpha trajectories, critic loss curves, action distributions, infeasibility ratios) were read by hand. AI tools were good at proposing what to look for but did not reliably catch anomalies on their own.

### What I'd do differently

- Work with better quality of RTC+B Data.
- Look at more methods, RL is not the sole solution provider just because other researchers have started implementing RL did not necessarily mean for us to do the same.
- Come out of the hallucinating LLM trap much faster to change the path and work in a much more efficient manner.
