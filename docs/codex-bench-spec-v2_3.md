# Codex-Bench: Harness-Aware Adaptation for Agentic Coding

## Project Spec — v2.3

---

## Changelog

- v0.1–v0.4: Design iteration (see prior versions)
- v0.5: Serving optimization, autoresearch, correct Codex invocation, honest timeline
- v1.0: Consolidated spec
- v1.1: Fixed personality, reasoning_effort, wire_api, 2×2 eval scope, Gate 1 soak
- v1.2: Codex-only training, SWE-smith as reference, honest 5–6 month timeline
- v1.3: Removed 9B entirely. 27B is sole RL target (no fallback — if Gate 3
  fails, Contribution B is dropped). Gate 4 tightened: pilot runs 30 tasks ×
  2 Codex seeds × 1 SWE-Agent seed, gates on projected unique matched task IDs
  (≥30) not raw traces. Split Codex-SFT-all (headline) from Codex-SFT-matched
  (2×2 comparison). 35B-trace fallback pre-registered as separate cross-model
  distillation branch. Non-Qwen defaults to GLM-4.7 with Preserved Thinking
  caveat (published numbers used SGLang-only mode). Sprint 3 arithmetic fixed:
  1000 core runs = 800 Codex + 200 SWE-Agent = ~69 days.
  Honest total: ~5.5–7 months, ~157–176 Spark days.
- v2.3: **Smaller-v1 escape hatch fully pre-registered; two cleanups.**
  B2-only proceed rule added for 35-family path: Contribution B survives only
  if projected Codex traces ≥ 50 AND projected Train-Long collection wall-clock
  ≤ 25 Spark days; otherwise project reduces to Contribution A only. "Assess
  whether B2 is worth the investment" language removed — decision is now
  pre-registered, not discretionary. Cleanup 1: Dev-Bench run count (300)
  marked as baseline contingent on full lineup passing Gates 1b and 1c.
  Cleanup 2: fairness asymmetry reworded — the real limitation is that
  Responses-capable providers can still differ in reasoning-effort behavior
  and semantics, not just that non-Responses models are excluded.
- v2.2: **Final blocker patched — green-target.** Added Base model through
  SWE-Agent on SWE Final-Test (100 × 1 seed) to B2 roster. Provides the
  missing denominator: ΔCodex = Codex-SFT-all/Codex − Base/Codex; ΔSWE =
  Codex-SFT-all/SWE-Agent − Base/SWE-Agent. Both deltas now computable.
  B2 core updated to 1100 runs (~73 days); B1+B2 combined ~90–91 days core.
  Sections 1.9, 3, 4, Sprint 3, and Section 7 updated consistently. Cleanup 1:
  Gate 4 matched-families formula example parametrized as
  M / pilot_family_count × N_train_long_families instead of hardcoded ×30.
  Cleanup 2: Proto-fixture wording tightened to "lightweight committed fixture
  workspace + prompts."
- v2.1: **Three final P0 patches + two cleanups — targeting sign-off.**
  P0-1: Gate 1b explicitly scoped to locally-served vLLM models only. Frontier
  API ceiling model gets a separate named provider-side compatibility check
  (5 sequential + 3 parallel tool-call tasks against provider endpoint; different
  failure surface, separate procedure). P0-2: B2 arithmetic fully reconciled:
  core B2 (800 Codex + 200 SWE-Agent) = ~69 days; SWE-smith reference arm
  adds ~8 days separately (~77 days total with reference); fixed bottlenecks
  bullet corrected from stale ~66 to ~69 days; Sections 1.9 and 7 now
  consistent. P0-3: Hard Test-Long family floor added — if Test-Long family
  count < 8 at freeze, B1 is dropped regardless of Gate 4 outcome. Prevents
  35-family path (Test-Long ~6 families) from producing underpowered B1 claim
  even if Train-Long matched-ID thresholds pass. Cleanup 1: Comparison slot
  reference scores replaced with TBD/candidate-dependent. Cleanup 2: Mandatory
  ablation run (Codex-SFT-all through SWE-Agent) explicitly framed as required
  diagnostic attached to B2, not a standalone claim.
- v2.0: **Final three P0 patches + two cleanups — targeting sign-off.**
  P0-1: Gate 1b fixture sequencing fixed. Proto-fixtures (lightweight,
  Sprint-0-authorable, independent of Sprint 0b scenario design) distinguish
  from the full Sprint 0b fixture set. Order of operations is now consistent.
  P0-2: Missing mandatory ablation added to B2 roster: Codex-SFT-all through
  SWE-Agent on SWE Final-Test (100 × 1 seed). Section 3 claim now matches
  the run list. P0-3: Projected matched families formula pre-registered:
  matched_families_projected = (matched_families_in_pilot / pilot_family_count)
  × total_Train-Long_families. Extrapolation block in Gate 4 updated.
  Cleanup 1: "Non-Qwen open" slot renamed to "comparison slot" with
  "(fallback: second Qwen)" annotation in all tables. Cleanup 2: Reduced-split
  geometry pre-declared for smaller-v1 (35-family path): Train-Long ~20,
  Val-Long ~7, Test-Long ~6, Public-Dev ~2.
- v1.9: **Three P0 patches, two P1 cleanups toward sign-off.** P0-1: Gate 1b
  now has an explicit kill rule — if Qwen3.5-27B fails Gate 1b, Contribution B
  is killed and project reduces to Contribution A only (27B exclusion is not
  a lineup change, it is the end of adaptation). P0-2: Smaller-v1 escape hatch
  parameterized: Option B pre-registered — smaller-v1 ships Contribution A +
  B2 only by default; B1 is only included if the full Gate 4 thresholds (matched
  IDs ≥ 50, matched families ≥ 8, ≥ 4 types) are met on the actual shipped
  family count. P0-3: Non-Qwen slot reframed — second Qwen variant is the
  practical default fallback; GLM-4.7 and Step 3.5 Flash are aspirational
  add-ons pending Gate 1b + Gate 6, not a presumptive slot. P1-1: Removed
  remaining "Chat API models silently ignore" sentence from Section 5. P1-2:
  Gate 1b now requires fixed benchmark repo fixtures/prompts for sequential
  and parallel tool-call elicitation, not ad hoc prompting.
- v1.8: **Final consistency sweep and protocol-risk tightening.** (1) Sections 6
  and 7 consistency pass: budget total split into fixed-without-optional
  (~105–110) and fixed-with-Sprint-0.5 (~107–112); stale "Bench-Train" language
  in Sprint 0.5 removed; B1 mitigation corrected from "run at 1 seed" (already
  default) to "skip optional second seed"; Gate 4 borderline-pass range updated
  to match raised ≥50 threshold. (2) Gate 1b strengthened: added parallel
  multi-tool-call case (not just sequential ≥3) and official OpenAI Python SDK
  stream-replay requirement alongside Codex CLI verification. (3) Purged all
  remaining chat/legacy-shim language from Section 3; wire_api = responses is
  the only supported path, no exceptions. (4) Gate 4 matched-family coverage
  criterion added: projected matched families ≥ 8 AND ≥ 4 scenario types
  contribute matched families — ID count alone is no longer sufficient to
  PROCEED. (5) Sprint 0b escape hatch added: explicit smaller v1 release
  option if authoring quality slips.
- v1.7: **Arithmetic reconciliation and remaining blockers.** (1) B1 eval table
  corrected: added missing Base-through-SWE-Agent row; B1 = 300 runs (150
  Codex + 150 SWE-Agent), ~18 days. B2 arithmetic fixed: 800 Codex + 100
  SWE-Agent = 900 runs, ~66 days. Section 1.7 trace projection updated to
  120–160. Section 3 Gate 4→Gate 5 logprob reference fixed. (2) Added Gate 1b:
  explicit pass/fail on Codex-consumable streamed tool-call event shape from
  local vLLM — streamed function-call semantics, OpenAI SDK compatibility,
  `/v1/responses` protocol correctness. (3) Sprint 0b audit minimum raised:
  one fully shortcut-audited variant per scenario type (5 minimum) plus 20%
  random adversarial audit across all families before split is frozen.
  (4) B1 reframed throughout as directional pilot evidence, not precision
  result; Gate 4 matched-ID minimum raised from ≥30 to ≥50. Section 3
  "RL'd model" corrected to "SFT model." Codex-SFT-all clarified as B2-only
  headline; B1 headline is the 2×2 matched comparison. SWE-Bench-Control-SFT
  removed from Sprint 3 main checklist, noted as appendix-only run.
- v1.6: **Six P0 patches.** (1) Reconciled Sections 1.9/4/7: B1 eval on Test-Long,
  B2 eval on SWE Final-Test, arithmetic shown explicitly. (2) Rewrote B1
  causal claim — "family-disjoint" means within-benchmark generalization across
  scenario templates, not same-distribution held-out; tightened language
  throughout. (3) Added explicit Codex-Long anti-cheat / hidden-test /
  oracle-isolation protocol (verifier access, test injection timing, oracle
  solution handling). (4) De-risked Sprint 1: benchmark authoring promoted to
  a dedicated Sprint 0b (4–6 weeks) before orchestrator Sprint 1; removed
  the false claim that 55+ families can be authored in 2 weeks alongside
  infrastructure work. (5) Gate 4 expanded to 5 pilot families, one per named
  scenario type. (6) B1 statistics switched to family-clustered bootstrap;
  seed count on Test-Long gated on Gate 4 wall-clock. P1 patches: Bench-Control
  demoted to diagnostic appendix; Gate 6 provider language tightened to
  Responses-API-only reality; Phase 2b reframed as pre-killed until Gate 5
  passes; Contribution A novelty claim narrowed.
- v1.5: **Hybrid structure.** Restored SWE-bench as both anchor eval and small
  in-domain control arm (Bench-Control ~50 tasks, explicit design control only —
  not primary adaptation engine). Codex-Long expanded to 50+ scenario families
  with four-way **family-disjoint** split: Train-Long / Val-Long / Test-Long /
  Public-Dev. Contribution B formally split into two distinct claims: B1
  (harness specificity, in-domain, 2×2 on Test-Long) and B2 (transfer to
  SWE-bench Final-Test). Gate 4 redesigned as family-aware Codex-Long pilot
  measuring unique solved families, wall-clock by family type, and turn/tool-call
  counts — not just raw solve rate. Sprint timeline savings explicitly not banked
  until Gate 4 pilot completes.
- v1.4: **Separated SWE-bench Verified into an evaluation-only role.**
  Introduced the Codex-Long track (~30 scenario families × 5–10 variants each)
  as the dedicated training data source for Contribution B. SWE-bench Verified
  is now used exclusively for Dev-Bench (Contribution A) and Final-Test
  evaluation (Contribution B) — never as training data. The 350-task Bench-Train
  SWE-bench pool is removed. Sprint 2 Bench-Train is replaced by Codex-Long
  trajectory collection. Sprint 1 gains a scenario design deliverable.
  Timeline compresses: Codex-Long collection ~30 days vs prior ~54 days.
  Revised total: ~125–137 Spark days (~5–6 months).

---

## 1. Feasibility Math

### 1.1 DGX Spark Inference Throughput

Single-stream decode (batch 1, interactive). Context-degraded averages
account for growing context across ~30-turn agentic sessions.

| Model                          | Baseline (FP8) | Avg (context-degraded) | Optimized (NVFP4+MTP)* |
|--------------------------------|---------------|------------------------|------------------------|
| Qwen3.5-27B                   | ~12–14        | ~10                    | ~18–22 (est.)          |
| Qwen3.5-35B-A3B (MoE)         | ~42–50        | ~35                    | ~55–60 (est.)          |
| Qwen3-Coder-Next 80B-A3B (MoE)| ~42           | ~33                    | ~67 (Avarok measured)  |
| Qwen3.5-122B-A10B (MoE)       | ~38–51        | ~38                    | ~50–65 (est.)          |

*Optimized plan contingent on NVFP4 ARM64 stability. There is an open vLLM
bug for Qwen3.5 NVFP4 crashing on ARM64 GB10 with CUDA illegal instructions,
and a separate issue reporting poor accuracy on a Qwen3.5 NVFP4 checkpoint.
Optimized results reported separately from baseline, never as primary condition.

**All timeline estimates use baseline FP8 context-degraded numbers.**

Planning estimates only. Sprint 0 measures our own baselines on our exact
stack. Sprint 0.5 (optional autoresearch) may improve these further.

**Agentic latency profile** — report all four per model × task:

| Metric             | What it measures                                 |
|--------------------|--------------------------------------------------|
| TTFT (ms)          | Time to first token — dominated by prefill speed |
| Prefill throughput | tok/s for processing input context                |
| Decode throughput  | tok/s for generating output (steady state)        |
| Cache hit rate     | % of prefill tokens served from prefix cache      |

Prefix caching ON by default. Cache-OFF as ablation.

### 1.2 Training Throughput

| Model            | Method  | Peak tok/s | Source       |
|------------------|---------|------------|--------------|
| Llama 3.1 8B     | LoRA    | 53,657     | NVIDIA blog  |
| Llama 3.3 70B    | QLoRA   | 5,079      | NVIDIA blog  |
| Qwen3.5-27B      | QLoRA   | ~8,000–12K | Extrapolated |

### 1.3 Model Selection

#### Benchmark Lineup (Contribution A — inference only)

| Model                          | Total / Active | SWE-bench V. | T-Bench 2 | Baseline Decode | License    |
|--------------------------------|---------------|-------------|-----------|-----------------|------------|
| Qwen3.5-35B-A3B (MoE)         | 35B / 3B      | 69.2%       | 40.5      | ~35 avg tok/s   | Apache 2.0 |
| Qwen3-Coder-Next 80B-A3B (MoE)| 80B / 3B      | 70.6%       | 36.2*     | ~33 avg tok/s   | Apache 2.0 |
| Qwen3.5-122B-A10B (MoE)       | 122B / 10B    | 72.0%       | 49.4      | ~38 avg tok/s   | Apache 2.0 |
| Comparison slot (fallback: 2nd Qwen)† | TBD           | TBD (candidate-dependent) | TBD | TBD             | Open       |
| 1× Frontier API (ceiling)     | —             | 78–81%      | 58–67     | N/A             | Proprietary|

*T-Bench 2 varies by scaffold.
†**Practical default fallback: second Qwen variant** (e.g., an additional
Qwen3.5 size or Qwen3-Coder variant confirmed compatible in Sprint 0).
GLM-4.7 and Step 3.5 Flash are **aspirational add-ons** pending Gate 1b
and Gate 6 — not a presumptive slot. Recent vLLM issue traffic shows GLM
streaming tool-call failures are an active risk. Do not plan around GLM
until it passes Gate 1b. If neither GLM nor Step passes, the non-Qwen slot
becomes a second Qwen variant and the benchmark remains internally valid.

**Important**: All SWE-bench and T-Bench scores in this table are
**vendor-published capability references**, not expected reproduction
under our frozen Codex + vLLM baseline stack. Each vendor's published
numbers were obtained with their own scaffold, prompts, thinking modes,
and serving configs (e.g., GLM-4.7's scores require Preserved Thinking
mode via SGLang; Qwen scores may assume specific prompt templates).
Codex-Bench measures what these models actually achieve through our
pinned Codex CLI configuration — which may be materially different.
Do not interpret this table as a performance prediction.

#### RL Training Target (Contribution B — requires gradients)

| Model              | Total / Active | Baseline Decode | Training?       | Rollout/Task | Notes              |
|--------------------|---------------|-----------------|-----------------|--------------|---------------------|
| Qwen3.5-27B (dense)| 27B / 27B     | ~10 avg t/s     | ✅ QLoRA fits   | ~110 min     | **Primary target**  |
| Qwen3.5-35B-A3B    | 35B / 3B      | ~35 avg t/s     | ⚠️ MoE unproven | ~40 min      | HIGH RISK stretch   |

Qwen3.5-27B is the RL target. It has published SWE-bench Verified (72.4%)
and Terminal-Bench 2 (41.6%) scores, making baseline comparisons meaningful.
Sprint 0 validates feasibility (Gate 3).

#### Model × Role Matrix

| Model                    | Codex-Bench? | RL Target?    | Role                         |
|--------------------------|-------------|---------------|------------------------------|
| Qwen3.5-35B-A3B          | ✅ Primary   | ⚠️ Stretch     | Primary open-weight baseline |
| Qwen3-Coder-Next 80B-A3B | ✅           | ❌             | Coding-specialized comparison|
| Qwen3.5-122B-A10B        | ✅           | ❌             | Open-weight ceiling          |
| Comparison slot (fallback: 2nd Qwen) | ✅           | ❌             | Cross-vendor or 2nd Qwen     |
| Qwen3.5-27B (dense)      | ✅           | ✅ **Primary** | RL target                    |
| 1× Frontier API          | ✅ Ceiling   | ❌             | Upper-bound reference        |

### 1.4 Task Scoping

Two separate tracks with strictly separated roles.

#### Track 1 — SWE-bench Verified

SWE-bench Verified tasks serve two roles: **public anchor evaluation** (Dev-Bench,
Final-Test) and a small **in-domain control arm** (Bench-Control). The control
arm is not expected to provide training scale — it exists to preserve an
interpretable in-domain baseline against which Codex-Long training can be compared.

| Pool                | Tasks | Models       | Seeds | Role                                         |
|---------------------|-------|--------------|-------|----------------------------------------------|
| **Dev-Bench**       | 50    | All 6 models | 1     | Contribution A leaderboard (published)       |
| **Bench-Control**   | ~50   | 27B only     | 1–2   | In-domain SFT control arm only — not primary adaptation |
| **Final-Test**      | 100   | 27B only     | 3     | Contribution B headline eval (sealed)        |

All three pools are disjoint subsets of SWE-bench Verified. Final-Test is sealed
until Sprint 3. **Bench-Control exists for interpretability, not scale**: at ~15%
solve rate and ~50 tasks it yields only ~7–8 successful traces per seed (~15–20
at 2 seeds). This is a design control, not a primary training source.

#### Track 2 — Codex-Long (Primary Training + Realism Track)

A purpose-built long-horizon benchmark that serves as the primary source of
training trajectories AND as a secondary reported benchmark. Both roles require
rigorous internal splitting.

**Split is family-disjoint, not just environment-disjoint.** A family is a
scenario design template (repo pattern + grading invariant + milestone structure).
Variants within a family share structural DNA. Val-Long and Test-Long must
contain families not seen in Train-Long — not merely unseen variants of
Train-Long families.

| Pool               | Families | Envs/Family | Total Envs | Role                                        |
|--------------------|----------|-------------|------------|---------------------------------------------|
| **Train-Long**     | ~30      | ~5–8        | ~150–240   | Primary SFT/RL trajectory collection        |
| **Val-Long**       | ~10      | ~3–5        | ~30–50     | RL early stopping and hyperparameter selection only — no gradient updates |
| **Test-Long**      | ~10      | ~4–6        | ~40–60     | Reported secondary benchmark (sealed until Sprint 3) |
| **Public-Dev**     | ~5       | ~3–4        | ~15–20     | Published dev set for reproducibility       |
| **Total**          | **~55**  |             | **~235–370** |                                           |

**Scenario types targeted (across all splits):**
- **Feature evolution**: implement a spec delta across code, tests, and docs
- **Migration / refactor**: update an API or dependency across multiple call sites
- **Build / CI breakage**: dependency drift, toolchain mismatch, config skew
- **Investigate-then-fix**: start from logs or a failing integration test
- **Cross-layer changes**: backend + CLI + config + tests in one session

**Grading**: State-based verifier over final container state (not patch match).
Each scenario includes milestone checks for partial-credit RL reward signals.

**Difficulty calibration**: Train-Long scenarios filtered to 20–80% solve rate
for DAPO group coverage (G=4–8). Tasks outside this range excluded from RL
training but may be retained for SFT if solve rate > 0.

**Family assignment protocol**: Families are assigned to Train/Val/Test/Public-Dev
before any collection run. Assignment is frozen and published. No post-hoc
reassignment based on solve rates.

**Public-Dev release**: Published alongside results. Covers all five scenario
types. Enough to understand benchmark structure and reproduce the grading
pipeline.

#### Codex-Long Integrity Protocol

Grading is state-based over final container state. The following rules prevent
shortcut exploitation and must be enforced at the harness level, not relied on
via agent cooperation.

**Verifier and oracle isolation:**
- Verifier scripts and oracle solutions are stored outside the agent container
  and are never mounted or accessible during a run. Grading is run in a
  separate post-run container with the verifier injected after the agent
  session terminates — matching Terminal-Bench's integrity model.
- Milestone check scripts follow the same rule: injected post-run, not
  available to the agent at any point during execution.
- Oracle solutions are used only to prove task solvability during scenario
  authoring. They are never included in the agent's context, AGENTS.md,
  or any file visible during a run.

**Test resource injection timing:**
- Any test files referenced by the verifier are copied into the evaluation
  container **after** the agent session ends. The agent sees only the
  repository state as checked out at run start.
- This prevents agents from reading test expectations and reverse-engineering
  solutions to match them.

**Shortcut prevention:**
- Scenarios must not be solvable by deleting failing tests, disabling CI
  checks, or mocking external state. Verifier scripts must explicitly check
  that the full test suite is present and unmodified.
- Each scenario's verifier is reviewed for shortcut resistance during
  benchmark authoring (Sprint 0b).

**Reproducibility:**
- Each scenario variant is pinned to a specific Docker base image digest.
- The full verifier suite is open-sourced alongside the Public-Dev split.
- Reproduction instructions allow independent grading of any submitted
  trajectory without access to oracle solutions.

### 1.5 Benchmark Time Estimation

**Per-task time (baseline FP8, context-degraded averages):**

Note: These estimates are dominated by decode time. For long-horizon Codex
sessions that repeatedly resend growing shared prefixes, TTFT and prefill
latency are also significant runtime drivers (especially at 100K+ context
where prefill latency grows sharply). Sprint 0 measures actual per-task
wall time on 5 real SWE-bench tasks to replace these decode-only estimates.

| Model Class              | Avg tok/s | Output tok/task | Time/task |
|--------------------------|----------|----------------|-----------|
| 27B dense                | ~10      | ~60,000        | ~110 min  |
| 35B-A3B MoE              | ~35      | ~60,000        | ~40 min   |
| 80B-A3B MoE              | ~33      | ~60,000        | ~42 min   |
| 122B-A10B MoE            | ~38      | ~60,000        | ~35 min   |

**Total timeline by phase (baseline estimates — not banked until Gate 4 pilot):**

| Phase                                           | Runs   | Est. Hours | Est. Days  |
|-------------------------------------------------|--------|-----------|------------|
| Dev-Bench (6 models × 50, baseline†)           | 300†   | ~170 hrs  | ~7 days    |
| Bench-Control (27B × ~50 × 1–2 seeds)          | ~100   | ~180 hrs  | ~8 days    |
| Codex-Long Train-Long (27B × ~200 envs × 2 seeds) | ~400 | TBD by Gate 4 pilot | TBD |
| Codex-Long Val-Long (27B × ~40 envs × 1 seed)  | ~40    | TBD       | TBD        |
| SWE-Agent arm (27B × ~200 Train-Long envs × 1 seed) | ~200 | TBD    | TBD        |

**Codex-Long wall-clock estimates are deliberately left as TBD.** Long-horizon
Codex cases likely run longer per attempt than SWE-bench bugfixes (more turns,
larger context, more tool calls). Gate 4 pilot measures actual wall-clock by
family type before any collection run is committed.

†Dev-Bench run count assumes the full six-model lineup passes Gates 1b and 1c.
Models excluded by Gate 1b (local vLLM protocol failure) or the frontier slot
dropped by Gate 1c reduce this count proportionally. The ~7-day estimate and
~300-run figure are baseline planning numbers; actual counts are determined
after all per-model gate checks complete. Do not use v1.4 estimates
as planning inputs.

Codex-Long Train-Long collection is the dominant Sprint 2 bottleneck. Wall-clock
per scenario is measured in Gate 4 pilot before total is estimated.

### 1.6 Trace Budget

**Two training sources. Different roles. Do not conflate them.**

#### SWE Bench-Control (design control arm only)

At ~15% solve rate on ~50 tasks, Bench-Control yields approximately:

| Seeds | Bench-Control Tasks | Solve Rate | Expected Codex Traces |
|-------|---------------------|------------|-----------------------|
| 1     | ~50                 | ~15%       | ~7–8                  |
| 2     | ~50                 | ~15%       | ~15–20                |

These numbers are too small to serve as a primary training source. Bench-Control
exists to produce a matched in-domain SFT variant (SWE-Bench-Control-SFT) that
makes Contribution B claims interpretable. It is explicitly not expected to
provide sufficient scale for primary adaptation. This is stated directly so no
reviewer can misread it as a primary data source.

#### Codex-Long Train-Long (primary adaptation source)

Solve rate depends on scenario difficulty calibration. Target: 30–40% on 27B.

| Seeds | Train-Long Envs | Target Solve Rate | Expected Codex Traces |
|-------|-----------------|-------------------|-----------------------|
| 1     | ~200            | ~30–40%           | ~60–80                |
| 2     | ~200            | ~30–40%           | ~120–160              |

At 2 seeds: ~120–160 traces — the primary SFT/RL dataset.

**Warning**: Higher solve rate increases useful trace yield but does not reduce
wall-clock runtime. If Codex-Long cases average longer than SWE-bench tasks
(likely), total collection time increases even at higher solve rate. Gate 4
pilot measures this directly.

**Warning**: Trace yield is necessary but not sufficient for RL quality.
If scenarios lack good verifier structure (milestone checks, partial-progress
signals, robust final-state grading, family diversity), long noisy traces
may substitute for the prior problem of too few successful ones. Both verifier
design and family diversity are required.

**Fallbacks if trace count is critically low (< 30 after Train-Long):**
1. Add seed 3 (+time TBD from Gate 4 pilot)
2. Add variants to existing Train-Long families (low-cost)
3. Frame as limitation result: "harness-native signal exists but requires more data"
4. **(Pre-registered branch only)**: Cross-model distillation from 35B-A3B traces.
   Reported as a separate explicitly labeled experiment, never mixed into main results.

### 1.7 Training Time

**Phase 2a — Filtered SFT on Codex traces (guaranteed path, no rollouts):**
- 120–160 successful Codex-native traces from Train-Long (no SWE-smith warm-start)
- Training: minutes per epoch at ~10K tok/s (27B QLoRA)
- Total: 1–2 days including hyperparameter sweeps

**Phase 2b — DAPO with on-policy rollouts (stretch goal):**

| RL Target | Time/rollout | G=4 group | 100 prompts total |
|-----------|-------------|-----------|-------------------|
| 27B       | ~110 min    | 7.3 hrs   | 30 days           |
| 35B-A3B   | ~40 min     | 2.7 hrs   | 11 days           |

Phase 2a is guaranteed. Phase 2b only if Phase 2a shows signal AND
rollout time is acceptable. Title and claims do not promise RL.

### 1.8 DGX Spark Serving Optimization

The Spark's memory bandwidth (~273 GB/s) is the primary decode bottleneck.

**Optimization stack (all contingent on ARM64 stability):**

| Stack                   | Qwen3-Next-80B tok/s | vs Baseline |
|-------------------------|----------------------|-------------|
| Stock vLLM + FP8        | ~42                  | baseline    |
| Avarok vLLM + NVFP4     | ~42 (+Marlin fix)    | +20%*       |
| Avarok + NVFP4 + MTP    | **~67**              | **+60%**    |

*NVFP4 advantage requires community-patched Marlin MoE backend.
Known ARM64 crash bug and accuracy issues. Not assumed in timelines.

Additional optimizations: FP8 KV cache, chunked prefill, CUDA graph
capture, prefix caching, environment variable tuning.

**Action**: Sprint 0 tests stock FP8. Sprint 0.5 (optional) searches for
optimized config. Results reported separately, never as primary benchmark.

### 1.9 Contribution B Eval Budget

Contribution B runs two separate evaluation tracks. They are not interchangeable.

#### B1 — Harness Specificity (Test-Long, ~50 envs)

| Evaluation                                         | Envs  | Seeds | Runs  | Notes                     |
|----------------------------------------------------|-------|-------|-------|---------------------------|
| Base model through Codex                           | ~50   | 1     | ~50   | B1 Codex baseline         |
| Base model through SWE-Agent                       | ~50   | 1     | ~50   | B1 SWE-Agent baseline     |
| **Codex-SFT-matched** through Codex                | ~50   | 1     | ~50   | 2×2 cell A                |
| **Codex-SFT-matched** through SWE-Agent            | ~50   | 1     | ~50   | 2×2 cell B                |
| **SWE-Agent-SFT-matched** through Codex            | ~50   | 1     | ~50   | 2×2 cell C                |
| **SWE-Agent-SFT-matched** through SWE-Agent        | ~50   | 1     | ~50   | 2×2 cell D                |
| **B1 core total**                                  |       |       | **~300** |                        |

Both baselines are required. Without Base-through-SWE-Agent, cells B and D
have no denominator for harness-matched improvement.

**Time estimate (B1, 27B baseline):**
- ~150 Codex runs × ~110 min = ~275 hrs
- ~150 SWE-Agent runs × ~60 min = ~150 hrs
- B1 core: **~17–18 Spark days** (Test-Long wall-clock TBD from Gate 4;
  adjust if actual env runtime differs from 110 min assumed for Codex)

Statistics: family-clustered bootstrap. Seeds on Test-Long: 1 minimum, 2 if
Gate 4 wall-clock allows. Do not run B1 at 2 seeds if it pushes total eval
beyond available Spark budget.

#### B2 — Transfer to Public Anchor (SWE Final-Test, 100 tasks)

| Evaluation                                         | Tasks | Seeds | Runs  | Notes                  |
|----------------------------------------------------|-------|-------|-------|------------------------|
| Base model through Codex                           | 100   | 3     | 300   | B2 baseline            |
| Base model through SWE-Agent                       | 100   | 1     | 100   | B2 SWE-Agent baseline — denominator for ΔSWE |
| **Codex-SFT-all** through Codex                    | 100   | 3     | 300   | Headline B2 result     |
| **Codex-SFT-all** through SWE-Agent                | 100   | 1     | 100   | Required diagnostic — numerator for ΔSWE |
| **Codex-SFT-matched** through Codex                | 100   | 1     | 100   | B2 comparison arm      |
| **SWE-Agent-SFT-matched** through Codex            | 100   | 1     | 100   | B2 comparison arm      |
| **SWE-Agent-SFT-matched** through SWE-Agent        | 100   | 1     | 100   | B2 comparison arm      |
| **B2 core total**                                  |       |       | **1100**|                     |
| SWE-smith-SFT through Codex (reference)            | 100   | 1     | 100   | Optional reference     |
| DAPO through Codex (stretch, Phase 2b only)        | 100   | 3     | 300   | Pre-killed until Gate 5|

**Note**: SWE-Bench-Control-SFT (diagnostic control arm) reported in appendix only —
not part of core B2 eval budget. Run at 100 tasks × 1 seed = 100 runs if time allows.

**Time estimate (B2 core, 27B baseline):**
- ~800 Codex runs × ~110 min = ~1,467 hrs = **~61 days**
- ~300 SWE-Agent runs × ~60 min = ~300 hrs = **~12 days**
- B2 core (1100 runs, no reference arm): **~73 days**
- SWE-smith reference arm (optional, +100 Codex × 110 min = ~183 hrs): **+~8 days → ~81 days**

The reference arm is a separate optional run, not included in the core total.

**The harness-specificity diagnostic now has both deltas:**
- ΔCodex = Codex-SFT-all/Codex − Base/Codex (300 − 300 runs)
- ΔSWE = Codex-SFT-all/SWE-Agent − Base/SWE-Agent (100 − 100 runs)

If ΔCodex > ΔSWE: improvement is harness-specific. If ΔCodex ≈ ΔSWE: improvement
is general. Both outcomes are reportable; neither requires suppression.

**Combined B1 + B2 core: ~90–91 Spark days** (B2 core ~73 + B1 ~17–18).
With SWE-smith reference: ~98–99 days. With B1 second seed (if Gate 4 allows): add ~17–18 more.

---

## 2. RL Algorithm — DAPO (Stretch Goal)

### 2.1 Why DAPO Over GRPO

GRPO has known problems: entropy collapse, sample-level loss bias, zero-advantage
groups. DAPO (ByteDance, 2025) fixes these with:
1. **Clip-Higher**: Asymmetric clipping to preserve exploration
2. **Dynamic Sampling**: Use **DAPO with DS disabled** (25%+ overhead, no benefit)
3. **Token-level loss**: Critical for long multi-turn trajectories
4. **No KL penalty**: Removes reference model, saves memory

**Decision**: DAPO (without DS) via VeRL or OpenRLHF. Fallback: REINFORCE++-baseline.
Note: DAPO is Phase 2b (stretch). Phase 2a (filtered SFT) does not need DAPO.

### 2.2 Framework Selection

| Framework            | Multi-turn? | Spark tested? | DAPO?  | Notes                      |
|----------------------|------------|---------------|--------|----------------------------|
| nanochat-dgxspark-rl | ✅          | ✅             | GRPO   | Minimal, may need mods     |
| Unsloth + TRL        | ⚠️          | ✅             | GRPO   | Easiest Sprint 0 setup     |
| OpenRLHF             | ✅          | ❌             | ✅ DAPO | Most complete, needs Ray   |
| VeRL                 | ✅          | ❌             | ✅ DAPO | DAPO was built on VeRL     |

Start with Unsloth + TRL for Sprint 0. Upgrade to VeRL + DAPO if Phase 2b.

---

## 3. Hard Review Questions

### Design Validity

**Train/test boundary**:
- Dev-Bench (50, SWE-bench Verified, Contribution A eval — published)
- Bench-Control (~50, SWE-bench Verified, in-domain SFT control arm — not eval)
- Final-Test (100, SWE-bench Verified, sealed until Sprint 3)
- Train-Long (~200 envs, Codex-Long, ~30 families — primary SFT/RL data)
- Val-Long (~40 envs, Codex-Long, ~10 families — RL early stopping only)
- Test-Long (~50 envs, Codex-Long, ~10 families — sealed secondary benchmark)

All Codex-Long splits are **family-disjoint**. Family assignment is frozen
before any collection run and published with results.

**RL or imitation?** Phase 2a IS filtered SFT on Codex-native traces only.
Two SFT variants: Codex-SFT-all (all traces, headline result) and
Codex-SFT-matched (matched task IDs only, for 2×2 comparison).
DAPO is stretch. SWE-smith-SFT is separate reference.

**What's "harness-aware"?** Model learns Codex's tool-call API contract (schemas,
observations, errors). NOT internal routing. Bash-trained models have never seen
these patterns.

**Tool agency?** Model chooses tools. Codex executes. Model has full agency.

**Fairness contract:**

Pinned across all models:
- Codex CLI version (git hash)
- `--yolo --json` flags
- `-c 'web_search="disabled"'`
- `-c 'model_reasoning_effort="high"'` (Responses API models only — see below)
- `-c 'personality="pragmatic"'`
- AGENTS.md content (identical)
- Max turns (50), max output tokens/turn (8,192)

Known asymmetries (documented, not fixable):
- `model_reasoning_effort` applies only to Responses API providers, and Gate 6
  restricts the lineup to Responses-capable models only — so the issue is no
  longer that some models are excluded. The real remaining limitation is that
  Responses-capable providers can still differ in how fully they implement
  reasoning-effort semantics: some may honor the `high` setting by allocating
  more compute to internal reasoning steps; others may treat it as a soft hint
  or no-op. This means reasoning budget is not perfectly uniform even within
  the compliant lineup. Documented as a known limitation; measured and reported
  per-model via the tool-call reliability and behavior taxonomy in Contribution A.

May vary per provider (documented, published alongside results):
- `model_verbosity` (Responses API only)
- Reasoning-effort interpretation depth (model/provider-specific)
- Tool-call schema compliance (model capability)
- Token counting methodology

**Adapter confound**: Codex-Bench measures **model × adapter × harness**, not
the model in isolation. Intentional. We report **tool-call reliability rate**
per model as a separate metric.

**Seeds**: 3 for RL target on SWE Final-Test. 1 for comparison models. 1 for
Test-Long (minimum); 2 seeds recommended if Gate 4 wall-clock allows.

**Statistics — B2 (SWE Final-Test)**: Environment-level bootstrap CIs are
appropriate. Tasks are drawn independently from SWE-bench Verified.

**Statistics — B1 (Test-Long 2×2)**: **Family-clustered bootstrap CIs are
required.** Variants within a family share repo structure, grading invariants,
and failure modes — they are not independent draws. Environment-level bootstrap
overstates confidence for B1. Report both family-level and environment-level
statistics; the family-level numbers are the primary claim. With ~10 Test-Long
families, power is modest — frame B1 as a directional harness-specificity result,
not a high-precision estimate.

**Harness-awareness proof**: Required diagnostic attached to B2. Two runs on
SWE Final-Test: Codex-SFT-all through SWE-Agent (100 × 1 seed) and Base model
through SWE-Agent (100 × 1 seed). Together with the Codex-path runs, these
provide both deltas needed for a valid comparison:

- ΔCodex = Codex-SFT-all/Codex − Base/Codex
- ΔSWE = Codex-SFT-all/SWE-Agent − Base/SWE-Agent

If ΔCodex > ΔSWE: improvement is harness-specific. If ΔCodex ≈ ΔSWE:
improvement is general. This diagnostic is reported alongside B2 main results,
not in an appendix. Both outcomes are honest results; neither requires
suppression.

### Engineering

**Batch execution**: `codex exec --yolo`. True unattended. Docker-isolated.

**Failure contract**: resolved / failed / no_patch / timeout / crash. All recorded.

**Trajectory + logprobs capture**:

```
┌──────────┐         ┌─────────────┐         ┌──────────┐
│ Codex    │ ──────▶ │ Logprob     │ ──────▶ │ vLLM     │
│ exec     │ ◀────── │ Capture     │ ◀────── │ Server   │
│ --json   │         │ Proxy       │         │ logprobs │
│ stdout   │         │ (FastAPI)   │         │ in resp. │
└──────────┘         └─────────────┘         └──────────┘
     │                     │
     ▼                     ▼
  Trajectory          Token logprobs
  (Codex events)      (Phase 2b only)
```

Codex `--json` is primary trajectory source (sufficient for Phase 2a SFT).
Logprob proxy only needed for Phase 2b DAPO.
Persistent session files kept as backup (no `--ephemeral` during collection).

**DAPO logprob fragility warning**: There is a documented vLLM feature request
noting that Responses API logprobs were empty, closed as "not planned." The
Chat API logprob path is better supported but may not be available if
`wire_api = "responses"` is required. Phase 2b remains a strict stretch
result until Sprint 0 **Gate 5** proves the logprob path works end-to-end.

**Cross-run contamination**: Prefix cache flushed between tasks, ON within tasks.

### RL-Specific

**Action masking**: Gradient only on `role: assistant` tokens.

**DAPO groups**: G=4–8 rollouts. Mixed success/failure → gradient. Need
20–80% solve rate tasks.

**2×2 matching (B1)**: Training traces matched on solved Codex-Long Train-Long
scenario IDs (scenarios both harnesses solved). Evaluation for B1 runs on the
full **Test-Long** (family-disjoint from Train-Long and Val-Long — sealed until
Sprint 3). Evaluation for B2 runs on **SWE Final-Test** (100 tasks).
These are reported as separate claims. Do NOT condition either test set on
post-hoc solvability.

---

## 4. Two Independent Contributions

### Contribution A: Codex-Bench

A systematic benchmark of open-weight models served locally via pinned vLLM
provider config through Codex CLI on SWE-bench Verified, measuring agentic
latency anatomy (TTFT, prefill, decode, cache-hit), tool-call reliability,
and failure-mode taxonomy.

**Novelty is specific, not broad.** Terminal-Bench already evaluates Codex CLI
as a harness through Harbor, which also supports SFT/RL data interfaces.
Contribution A does not claim novelty over Terminal-Bench's harness infrastructure.
The specific combination that is novel here is:
- **SWE-bench code-repair setting** (not terminal tasks)
- **Locally-served models** through a pinned Codex provider config (not hosted APIs)
- **Four-metric agentic latency anatomy** per model × task
- **Harness-native adaptation analysis** in Contribution B as the downstream use

Framed as a **pilot benchmark on a frozen baseline stack**. Optimized serving
results (if obtained via Sprint 0.5) reported as a separate appendix.

Standalone contribution regardless of Contribution B results.

### Contribution B: Harness-Native Adaptation

Training a model on Codex-native traces from the Codex-Long track improves its
performance when evaluated through Codex CLI. SWE-smith is tested separately as
a reference comparison. Bench-Control provides a small in-domain SFT variant
as a design control.

**Contribution B contains two distinct scientific claims. These must not be blurred.**

**B1 — Harness specificity (within-benchmark generalization).**
Train on Codex-Long Train-Long traces. Evaluate on **Codex-Long Test-Long**
using the 2×2 harness matrix. Train-Long and Test-Long are **family-disjoint**:
the model sees new scenario templates at evaluation time, not held-out variants
of training families. This means B1 tests within-benchmark generalization across
scenario templates, not held-out performance on a literally identical task
distribution. The causal question is: given generalization to new Codex-Long
scenarios, do Codex-native traces produce harness-specific advantage through
Codex, and do SWE-Agent-native traces produce the corresponding advantage
through SWE-Agent? If A > C and D > B in the 2×2, the answer is yes —
controlled for task family novelty.

**B2 — Transfer to public anchor.**
Take the same Codex-Long-trained model and evaluate on **SWE-bench Final-Test**
through Codex. This is a transfer result: Codex-Long training → SWE-bench
evaluation. It is not a harness-specificity proof. It answers: "does
harness-native long-horizon training generalize to a standard benchmark?"
Report it as transfer, not as the primary harness-specificity result.

**These are not the same claim. B1 is the core scientific contribution.
B2 is the public anchor that establishes real-world relevance.**

**Two distinct Codex-SFT models** (different training sets, same method):

- **Codex-SFT-all**: Trained on ALL available Codex-native Train-Long successes
  (~120–160 traces at 2 seeds). **B2 headline only.** Not used in B1.
- **Codex-SFT-matched**: Trained on Train-Long Codex traces whose scenario IDs
  were ALSO solved by SWE-Agent. **B1 primary arm** in the 2×2 comparison.

Similarly, **SWE-Agent-SFT-matched** is trained on SWE-Agent traces from the
same matched Train-Long scenario IDs. B1 reference arm.

**SWE-Bench-Control-SFT** (diagnostic appendix only): Trained on Bench-Control
SWE-bench traces (~15–20 traces at 2 seeds). Not a peer mainline arm — included
solely as an interpretable in-domain reference. Reported in a clearly labeled
appendix subsection, visually separated from all B1 and B2 results.

**B1 primary result — 2×2 on Test-Long (directional pilot evidence):**

|                              | Eval through Codex | Eval through SWE-Agent |
|------------------------------|--------------------|------------------------|
| **Codex-SFT-matched**        | **(A)**            | (B)                    |
| **SWE-Agent-SFT-matched**    | (C)                | **(D)**                |

If A > C and D > B: harness-specific training shows directional advantage,
controlled for task family. **B1 is directional pilot evidence** — ~10
Test-Long families and 1–2 seeds support a directional claim, not a
precision harness-specificity estimate. Report with family-clustered
bootstrap CIs and state this limitation explicitly alongside results.

**B2 transfer result — headline on SWE Final-Test:**
1. Base model through Codex (300 runs, 3 seeds)
2. **Codex-SFT-all** through Codex (300 runs, 3 seeds) ← B2 headline
3. Base model through SWE-Agent (100 runs, 1 seed) ← SWE-Agent denominator
4. **Codex-SFT-all** through SWE-Agent (100 runs, 1 seed) ← required diagnostic

If (2) > (1): Codex-Long harness-native training transfers to SWE-bench.
This is the primary publishable number and the result the leaderboard anchors on.

Runs (3) and (4) together provide the harness-specificity diagnostic:
- ΔCodex = (2) − (1)
- ΔSWE = (4) − (3)

If ΔCodex > ΔSWE: improvement is harness-specific. If ΔCodex ≈ ΔSWE: improvement
is general. Runs (3) and (4) are required diagnostics attached to B2, reported
in the same section. Without (3), the comparison of (2) and (4) is confounded
by scaffold difficulty and cannot support a harness-specificity interpretation.

**Stretch:** DAPO on Codex traces. Reported only if Phase 2b succeeds.

---

## 5. Codex CLI Invocation

**Pinned for all benchmark and collection runs (Sprint 2+3):**

```bash
codex exec \
  --yolo \
  --json \
  -m <model_name> \
  -c 'web_search="disabled"' \
  -c 'model_reasoning_effort="high"' \
  -c 'personality="pragmatic"' \
  -C /path/to/repo \
  "<prompt>" \
  > trajectory_${model}_${task_id}_${seed}.jsonl
```

| Flag / Option                    | Purpose                                     |
|----------------------------------|---------------------------------------------|
| `exec`                           | Non-interactive subcommand (batch execution) |
| `--yolo`                         | Bypass all approvals and sandbox             |
| `--json`                         | JSONL event stream to stdout                 |
| `-c 'web_search="disabled"'`     | No web access — reproducibility + no leakage |
| `-c 'model_reasoning_effort="high"'` | High reasoning budget. Applies to
|                                  | Responses API models only. Models without
|                                  | Responses API support are excluded by Gate 6
|                                  | and Gate 1b — no chat-fallback path exists. |
| `-c 'personality="pragmatic"'`   | Fixed personality (valid: none/friendly/pragmatic) |
| `-C /path`                       | Working directory                            |

No `--ephemeral` during collection — persistent sessions are the safety net.
Use `--ephemeral` only for Sprint 0 smoke tests.

Each run inside Docker: no host filesystem, no network except model server,
resource limits enforced, automatic cleanup.

---

## 6. Implementation Roadmap

### Sprint 0 — Kill Gates (1.5 weeks)

**Gate 1: Behavioral Reliability**
- [ ] Serve Qwen3.5-35B-A3B via stock vLLM FP8, configure Codex CLI
- [ ] **Phase A**: 50 consecutive trivial tasks (file creation, simple edits)
- [ ] Measure: tool-call schema correctness, JSON errors, drift
- **KILL**: Schema correctness < **98%** over 50 runs (≤1 failure)
- [ ] **Phase B (if Phase A passes)**: 24-hour soak with 100 mixed tasks
  (file edits, multi-file changes, shell commands, tool chaining patterns)
- [ ] Measure: schema correctness, crash rate, memory leaks, vLLM stability
- **KILL**: Any systematic failure pattern (>2% error rate, memory growth,
  model server crashes). Intermittent single failures are acceptable.

**Gate 1b: Streamed Tool-Call Event Semantics**

The entire benchmark depends on Codex consuming structured tool-call events
from local vLLM via `/v1/responses`. This is not covered by Gate 1's schema
correctness check. Gate 1b is a hard protocol-compatibility gate — run
immediately after Gate 1 Phase A passes, before any further infrastructure
investment.

Known failure modes from recent vLLM issues (target these explicitly):
malformed aggregation of multi-tool calls in one delta, tool calls disappearing
inside reasoning regions, GLM `/v1/responses` streaming failures, and Qwen
parallel tool-call events crashing the SDK client.

**Fixture tiers — two distinct sets, two distinct timelines:**

- **Proto-fixtures (Sprint 0, authored before Gate 1b runs):** A lightweight
  committed fixture workspace: a minimal repo state (e.g. a small stub project
  with a few files) plus task-description prompts crafted to force the model
  into the specific tool-call patterns Gate 1b needs to stress-test — sequential
  (≥3 tool calls across turns) and parallel (multiple tool calls in one model
  turn). The workspace is needed because verifying expected side effects through
  Codex CLI requires something for the model to operate on. These fixtures
  require no full Docker environments, no verifiers, and no Sprint 0b work.
  Author before Gate 1b, commit to repo. Estimated effort: 2–4 hours. Ad hoc
  prompting during the gate is not acceptable; the gate must be reproducible
  from the committed workspace.

- **Full fixture set (Sprint 0b, one per scenario type):** Extends proto-fixtures
  with scenario-typed elicitation prompts after Codex-Long families are frozen.
  Used for per-model Gate 1b re-runs before each model's Dev-Bench collection.
  Does not block Sprint 0 Gate 1b execution.

- [ ] Author and commit **proto-fixture set** (Sprint 0, before Gate 1b runs,
  independent of Sprint 0b)

- [ ] Capture raw `/v1/responses` event stream from vLLM under Codex invocation
  for the primary model (Qwen3.5-35B-A3B)
- [ ] Verify streamed function-call events arrive in the expected OpenAI SDK
  shape: correct `type`, `function.name`, `function.arguments` fields,
  proper `delta` streaming vs. complete object structure
- [ ] **Sequential tool-call test**: 10 tasks each requiring ≥3 sequential
  tool calls. Verify all tool calls are consumed and produce expected side
  effects through Codex CLI.
- [ ] **Parallel multi-tool-call test**: 5 tasks that elicit multiple tool calls
  emitted in a single model turn (parallel tool use). This is the exact failure
  mode recent vLLM `/v1/responses` bugs are centered on. Verify the event
  stream is correctly aggregated and Codex does not drop or partially parse
  the parallel calls.
- [ ] **Official SDK replay test**: Replay the same raw `/v1/responses` event
  stream from the parallel test through the official OpenAI Python SDK client
  (not through Codex CLI) and verify the SDK parses it without error. A stream
  that Codex silently recovers from may still break the SDK and reveal a latent
  protocol drift that will surface under load.
- [ ] **Repeat all three sub-tests for each locally-served model** before that
  model's Dev-Bench run begins. Gate 1b applies only to locally-served vLLM
  models — it cannot and does not cover the Frontier API ceiling model, which
  uses a provider-hosted endpoint with a different failure surface.
- **KILL (lineup)**: Any locally-served non-27B model where any sub-test fails
  → excluded from lineup. Document exclusion. No workaround shims.
- **KILL (project scope)**: **If Qwen3.5-27B fails Gate 1b, Contribution B is
  killed entirely.** There is no fallback RL target.
- **Note**: `model_reasoning_effort` and Responses-API semantics are
  model-specific. Gate 1b verifies the local event-stream contract; Gate 6
  verifies per-model Responses-API compatibility more broadly.

**Gate 1c: Frontier API Provider Compatibility Check**

The Frontier API ceiling model is served by a remote provider, not local vLLM.
Gate 1b's vLLM-specific sub-tests have no valid execution path for it.
Gate 1c is a separate, lighter compatibility check run before the Frontier
model's Dev-Bench collection begins.

- [ ] Run 5 sequential tool-call tasks (≥3 tool calls per task) through
  Codex CLI against the provider's endpoint. Verify Codex correctly consumes
  the tool-call event stream and tasks complete without protocol errors.
- [ ] Run 3 parallel tool-call tasks (multiple tool calls in a single model
  turn) through Codex CLI against the provider's endpoint. Verify all parallel
  calls are consumed without drop or misparse.
- **KILL (frontier slot)**: Provider fails either check → the frontier ceiling
  slot is dropped from the benchmark and noted in published results. This does
  not affect Contribution B or any locally-served model slots.
- **Note**: The provider endpoint may update independently of the pinned Codex
  version. Re-run Gate 1c if the provider API changes before Dev-Bench completes.

**Gate 2: Base Solve Rate**
- [ ] 20 SWE-bench tasks through Codex CLI + best local model
- **KILL**: Resolve rate < **5%** (< 1 task solved)
- If killed: investigate model vs adapter, try different model

**Gate 3: RL Target Feasibility (Qwen3.5-27B)**
- [ ] Serve Qwen3.5-27B via stock vLLM FP8
- [ ] Measure decode tok/s (batch 1, baseline)
- [ ] 1 full rollout (1 Codex-Long pilot scenario through Codex)
- [ ] 1 QLoRA training step
- **KILL**: 27B rollout > 150 min OR QLoRA training fails → project
  scope reduces to Contribution A only (benchmark without adaptation).
  27B is the only RL target; there is no 9B fallback.
- **DAPO**: If rollout_time × 4 × 100 < 14 days → Phase 2b attempted

**Gate 4: Codex-Long Pilot (family-aware, wall-clock-aware)**

Gate 4 is a Codex-Long pilot that gates the full Train-Long collection and
produces the wall-clock-per-family estimates that replace all TBD entries in
Sections 1.5 and 1.7.

- [ ] Select **5 pilot families, one per named scenario type**:
  feature evolution, migration/refactor, build/CI breakage,
  investigate-then-fix, cross-layer changes
- [ ] Run Qwen3.5-27B through Codex on all variants of those 5 families × 2 seeds
- [ ] Run Qwen3.5-27B through mini-SWE-Agent on the same variants × 1 seed
- [ ] Measure per family type:
  - Solve rate (fraction of variants solved per harness)
  - **Unique solved families** (not just unique solved variants)
  - **Median wall-clock per attempt** — computed separately by scenario type,
    never averaged across types (wall-clock varies materially by type)
  - Median turn count and tool-call count (trajectory richness proxy)
  - Matched scenario IDs (solved by both harnesses)
- [ ] Extrapolate projected yields across full Train-Long:
  - Projected Codex traces = (solve rate × total envs per type) × 2 seeds, summed across types
  - Projected matched scenario IDs = (matched rate × total envs), summed across types
  - **Projected matched families** = (matched families observed in pilot / pilot family count)
    × total Train-Long family count. Pre-registered formula: if M of the
    pilot families produce ≥1 matched scenario, project
    (M / pilot_family_count) × N_train_long_families matched families.
    Example for the full plan (5 pilot families, 30 Train-Long families):
    M/5 × 30. Example for the 35-family path (5 pilot families, 20
    Train-Long families): M/5 × 20. Use the actual planned counts; do not
    hardcode 30.
    This is a ratio extrapolation — crude but pre-registered and not gameable.
  - **Projected wall-clock = median wall-clock per type × env count per type, summed**
    (this is the only honest estimate; do not use a global average)
- **PROCEED**: ALL of the following must be true:
  - Projected matched scenario IDs ≥ 50
  - Projected Codex traces ≥ 80
  - ≥ 4 of 5 pilot families show 20–80% solve rate (diversity check)
  - **Projected matched families ≥ 8** (not just matched IDs — family coverage matters)
  - **≥ 4 distinct scenario types contribute at least one matched family** (structural diversity check)
  ID count alone is not sufficient; a benchmark with 50 matched IDs concentrated
  in 2–3 family types is structurally too narrow for a harness-specificity claim.
- **ADJUST**: Matched IDs 30–50 OR matched families 5–8 → add seed 3 OR expand
  variant count for low-yield families OR add more families to underrepresented types
- **KILL**: Matched IDs < 30 OR matched families < 5 OR ≤ 1 family type in 20–80%
  solve range AND difficulty calibration cannot fix it → redesign scenario families.
  B1 comparison will not be adequately powered. Contribution B reduces to B2 only.

Note: A single easy family type dominating trace yield is a failure mode even
if raw numbers look acceptable. The diversity checks catch this. Borderline
passes (projected matched IDs 45–55 or matched families 7–9) should add 5
more pilot variants across underrepresented types before committing full
Train-Long collection.

**Gate 5: Trajectory + Logprobs**
- [ ] Verify `codex exec --json` produces complete trajectories
- [ ] Set up logprob capture proxy, verify per-token logprobs end-to-end
- **Phase 2b is pre-killed.** It is not an active stretch branch. It becomes
  live only if Gate 5 confirms logprobs are reliably available through the
  pinned vLLM + Codex stack. Treat Phase 2b as dead weight in all planning
  until Gate 5 passes — do not allocate Spark budget, do not design RL
  infrastructure, do not count on it for timeline or claims.
- **KILL → SFT-only**: Logprobs unavailable or unreliable → Phase 2b stays dead.
  Project delivers Phase 2a SFT as the full Contribution B result.

**Gate 6: Non-Qwen Model Selection**

Current official Codex docs specify `wire_api = "responses"` only. Models
requiring `wire_api = "chat"` are not supported. The comparison set is
**"models with solid Responses-API compatibility through the pinned Codex
provider config"** — not arbitrary open models. State this explicitly in
published results.

**Practical default fallback: a second Qwen variant** (confirmed compatible
by Sprint 0 and Gate 1b). Planning and timeline do not depend on GLM or Step
passing. If neither passes, the non-Qwen slot becomes a second Qwen variant
and the benchmark remains internally valid — cross-vendor comparison is a
nice-to-have, not a requirement.

**Aspirational candidates (Gate 1b + Gate 6 required before inclusion):**
- **GLM-4.7** (73.8% SWE-bench V., `wire_api = "responses"` nominally compatible,
  but recent vLLM issue traffic shows active GLM streaming tool-call parser
  failures in exactly the patterns Gate 1b tests for. Do not assume compatibility
  before Gate 1b passes. Published scores used Preserved Thinking via SGLang —
  treat as capability ceiling only under vLLM.)
- **Step 3.5 Flash** (74.4% SWE-bench V., higher capability but ships with
  `wire_api = "chat"` — only include if confirmed `responses` compatibility
  in Sprint 0.)

Pre-registered inclusion criteria (must meet ALL before any model enters lineup):
  - Runs on stock vLLM FP8 on single Spark without crashes
  - Passes Gate 1b (all three sub-tests including parallel tool-call + SDK replay)
  - Confirmed `wire_api = "responses"` through Codex (no chat fallback)
  - Passes Gate 1 behavioral reliability (98% schema correctness)
  - Has published SWE-bench Verified score from vendor or third party
- [ ] Test GLM-4.7 on Spark through Codex CLI including Gate 1b fixtures
- [ ] If GLM-4.7 passes Gate 1b → it is the non-Qwen slot
- [ ] If GLM-4.7 fails → test Step 3.5 Flash with confirmed responses-API support
- [ ] If both fail → non-Qwen slot is filled by a second Qwen variant (not dropped)

### Sprint 0.5 — Autoresearch Optimization (OPTIONAL — 2 days)

**This step is recommended but not required.** All timelines use baseline
FP8 numbers. If Sprint 0.5 finds improvements, timelines compress.
If skipped, the project proceeds on baseline config.

Autoresearch results are reported as an appendix — NOT the primary
benchmark condition.

**Target A: Serving Speed**
- Objective: maximize decode tok/s for primary model on baseline stack
- Agent modifies vLLM launch config, quantization, MTP, env vars, CUDA
  graph settings, KV cache dtype, chunked prefill params
- Model weights and measurement methodology frozen
- 5-min experiment budget → ~50–100 experiments overnight

**Target B: Training Speed**
- Objective: maximize LoRA/QLoRA training throughput (tok/s)
- Agent modifies rank, alpha, optimizer, batch size, mixed precision,
  torch.compile settings
- 5-min training + 1-min eval per experiment → ~40–80 experiments overnight

**Implementation:**
```bash
git clone https://github.com/karpathy/autoresearch.git
# Replace train.py with serving benchmark or training script
# Replace program.md with optimization goals

codex exec --yolo --json \
  -c 'web_search="disabled"' \
  "Read program.md. Run experiments, keep improvements, discard failures."
```

**ROI**: 2 days. If serving speed improves 30–50%, Codex-Long collection and
Sprint 3 eval compress ~20–30%, potentially saving 15–25 Spark days combined.

### Sprint 0b — Benchmark Authoring (4–6 weeks, no Spark)

**This is the primary project risk and the primary Sprint 0b deliverable.**
Benchmark authoring is not a side task inside Sprint 1. It is a dedicated
phase that must complete before orchestrator development begins, because the
orchestrator is built against the finalized scenario specifications and
verifier contracts.

**Calibration reference**: Terminal-Bench 2.0 produced 89 final tasks from 229
contributions with ~3 reviewer-hours per accepted task, plus substantial
LM-assisted verification. We are building 55+ families with 3–8 variants each.
At 1–2 hours of authoring/verification per variant across ~275 environments,
this is 275–550 hours of non-parallelizable verification work. Four to six weeks
is the honest estimate assuming two contributors working in parallel.

**Deliverables (all must pass before Sprint 1 begins):**
- [ ] 55+ scenario family specs written: repo pattern, injected breakage class,
  grading invariant, milestone definitions, and shortcut-resistance notes
- [ ] Family-to-split assignment table finalized and frozen:
  Train-Long (~30 families), Val-Long (~10), Test-Long (~10), Public-Dev (~5)
- [ ] Docker environment factory implemented and validated
- [ ] State-based verifier implemented and shortcut-reviewed for each family
- [ ] Milestone check scripts implemented (post-run injection only)
- [ ] Oracle solution confirmed solvable for each family (not distributed)
- [ ] **Benchmark audit minimum** (two-tier, both required before freeze):
  - *Tier 1 — Type coverage*: At least one variant per scenario type (5 types
    minimum) fully audited end-to-end: env → oracle solve → verifier pass →
    attempted shortcut → verifier catch. This proves the integrity protocol
    works for each scenario category.
  - *Tier 2 — Random adversarial audit*: 20% of all families (≥11 families)
    selected at random and subjected to adversarial probing: attempt to pass
    the verifier by deleting tests, mocking external state, or producing a
    trivially wrong final state. Any family where the verifier passes on a
    clearly wrong output is disqualified and must be redesigned before freeze.
- [ ] Difficulty pre-screening: each family manually estimated in 20–80%
  target solve range before Gate 4 pilot

**LM-assisted authoring is encouraged.** Use LM to generate variant
permutations, write boilerplate verifier logic, and suggest milestone checks.
Human review for shortcut resistance and integrity is required for every
family before it enters the frozen split. The adversarial audit tier cannot
be delegated to LM alone — a human must attempt the exploit.

**Smaller v1 release escape hatch.** If Sprint 0b quality slips — verifiers
are failing adversarial audit, family designs are too similar across types,
or the 4–6 week budget is exhausted before all 55+ families are complete —
**freeze a smaller, higher-quality set rather than rushing to hit the count**.
A v1 release with 35 well-audited families across all 5 scenario types is
publishable and replicable. A v1 release with 55 families where 20 have weak
verifiers is not. The Gate 4 pilot will reveal whether the smaller set is
sufficient to meet matched-ID and matched-family thresholds. If it is not,
add families before Gate 4 commits — do not add families post-freeze.

**Pre-registered claim survival under smaller-v1 (Option B):**

Gate 4 thresholds are fixed regardless of final family count — matched IDs ≥ 50,
matched families ≥ 8, ≥ 4 scenario types contributing matched families. These
do not rescale. The survival rules are applied in order:

**Rule 1 — Hard Test-Long family floor (applied at freeze, before Gate 4):**
If the frozen Test-Long split has fewer than 8 families, B1 is dropped
automatically. This rule fires regardless of Gate 4 outcome. Gate 4 thresholds
are mostly about Train-Long matched yield — they do not protect against a
structurally underpowered Test-Long. With fewer than 8 Test-Long families,
family-clustered bootstrap produces intervals too wide to support a directional
harness-specificity claim. The 35-family path (Test-Long ~6 families) triggers
this rule: B1 is dropped on the 35-family path regardless of matched-ID count.

**Rule 2 — Gate 4 threshold check (applied after Gate 4 pilot):**
- If Rule 1 does not fire AND Gate 4 thresholds are met → full B1 + B2 claims survive.
- If Rule 1 does not fire AND Gate 4 thresholds are not met → B1 is dropped, B2 survives.

A directional B1 claim that survives both rules is still labeled "directional pilot
evidence" throughout — it is not promoted to a precision result at any scale.

**Pre-declared split geometry for the 35-family path:**

If Sprint 0b freezes at ~35 families (the smallest viable release), the
Train/Val/Test/Public-Dev split is pre-declared as follows. This is not
negotiated post-freeze.

| Split       | Families | Envs/Family | Total Envs | Notes                              |
|-------------|----------|-------------|------------|------------------------------------|
| Train-Long  | ~20      | ~5–8        | ~100–160   | Primary SFT/RL trajectories        |
| Val-Long    | ~7       | ~3–5        | ~21–35     | RL early stopping only             |
| Test-Long   | ~6       | ~4–6        | ~24–36     | Sealed secondary benchmark         |
| Public-Dev  | ~2       | ~3–4        | ~6–8       | Reproducibility release            |

At 35 families with Test-Long ~6 families, **Rule 1 fires: B1 is dropped on
the 35-family path**. The 35-family path ships Contribution A + B2 only,
subject to the following pre-registered B2-only proceed rule.

**Pre-registered B2-only proceed rule for the 35-family path:**

B2 depends on sufficient Codex-native training signal and acceptable collection
wall-clock — not on matched-family yield, which is a B1 concern. After Gate 4
pilot completes on the 35-family plan, apply this rule:

- **PROCEED (B2 survives)**: Projected Codex traces ≥ 50 AND projected
  Train-Long collection wall-clock ≤ 25 Spark days.
- **KILL (Contribution B dropped)**: Projected Codex traces < 50 OR projected
  wall-clock > 25 Spark days → project reduces to Contribution A only.

Rationale for thresholds: 50 traces is the stated minimum for a meaningful
Codex-SFT-all claim ("even ~50 harness-native traces produce measurable
improvement"). 25 Spark days is the scaled ceiling for ~100–160 envs at the
35-family Train-Long size — if collection exceeds this, the project economics
at reduced scale do not justify the Sprint 3 eval investment.

This rule fires before any Sprint 3 work begins. It is pre-registered here
and cannot be renegotiated post-Sprint-0b. Do not soften thresholds or add
families post-freeze to rescue either contribution.

This rule is pre-registered here so it cannot be rationalized away after Sprint 0b
produces a smaller family count than planned. The decision is made now, not then.

### Sprint 1 — Orchestrator (2 weeks, no Spark)
- [ ] Task orchestrator: Codex-Long env → Docker → `codex exec` → trajectory capture
- [ ] Task orchestrator: SWE-bench → Docker → `codex exec` → patch extraction
- [ ] Trajectory parser: Codex JSONL → SFT training format
- [ ] Patch → SWE-bench prediction JSONL converter (Dev-Bench and Bench-Control paths)
- [ ] 4-metric latency capture from Codex events
- [ ] Logprob capture proxy (FastAPI shim) for Phase 2b
- [ ] End-to-end validation: 10 Codex-Long scenarios + 10 SWE-bench tasks through
  the full orchestrator → verifier → trajectory parser pipeline

### Sprint 2 — Data Collection (timeline set by Gate 4 pilot)
- [ ] Dev-Bench: 50 tasks × 6 models × 1 seed (~7 days); SWE-bench eval on patches
- [ ] **Bench-Control (Codex)**: ~50 SWE-bench tasks × 27B × 1–2 seeds (~8 days)
- [ ] **Codex-Long Train-Long (Codex arm)**: ~200 envs × 27B × 1–2 seeds (TBD days)
- [ ] **Codex-Long Val-Long (Codex arm)**: ~40 envs × 27B × 1 seed (TBD days)
- [ ] **Codex-Long SWE-Agent arm**: ~200 Train-Long envs × 27B × 1 seed (TBD days)
- [ ] Generate Contribution A artifacts → **LinkedIn Post #1**

### Sprint 3 — Adaptation & Evaluation (8–10 weeks)
- [ ] Phase 2a: Train Codex-SFT-all on ALL Codex-native Train-Long traces (1–2 days)
- [ ] Phase 2a: Train Codex-SFT-matched on matched Train-Long scenario-ID Codex traces
- [ ] Phase 2a: Train SWE-Agent-SFT-matched on matched Train-Long scenario-ID SWE-Agent traces
- [ ] **B1 eval (directional pilot — harness specificity on Test-Long):**
  - Base model through Codex (Test-Long × 1 seed)
  - Base model through SWE-Agent (Test-Long × 1 seed)
  - Codex-SFT-matched through Codex (Test-Long × 1 seed) → cell A
  - Codex-SFT-matched through SWE-Agent (Test-Long × 1 seed) → cell B
  - SWE-Agent-SFT-matched through Codex (Test-Long × 1 seed) → cell C
  - SWE-Agent-SFT-matched through SWE-Agent (Test-Long × 1 seed) → cell D
- [ ] **B2 eval (transfer to public anchor — SWE Final-Test):**
  - Base model through Codex (100 tasks × 3 seeds = 300 runs)
  - **Base model through SWE-Agent** (100 tasks × 1 seed = 100 runs) ← SWE-Agent denominator
  - Codex-SFT-all through Codex (100 tasks × 3 seeds = 300 runs) ← B2 headline
  - **Codex-SFT-all through SWE-Agent** (100 tasks × 1 seed = 100 runs) ← required diagnostic
  - Codex-SFT-matched through Codex (100 tasks × 1 seed) ← B2 comparison arm
  - SWE-Agent-SFT-matched through Codex (100 tasks × 1 seed) ← B2 comparison arm
  - SWE-Agent-SFT-matched through SWE-Agent (100 tasks × 1 seed) ← B2 comparison arm
  - SWE-smith-SFT through Codex (100 tasks × 1 seed) ← reference comparison
- [ ] *Appendix only (if time allows)*: Train SWE-Bench-Control-SFT on Bench-Control
  traces; run through Codex on Final-Test (100 × 1 seed). Not in main results.
- [ ] Phase 2b (stretch, pre-killed until Gate 5): DAPO on Train-Long,
  Val-Long for early stopping → Final-Test + Test-Long eval
- [ ] Generate Contribution B artifacts → **LinkedIn Post #2**
### Sprint 4 — Publication (1 week)
- [ ] Open-source orchestrator, proxy, benchmark tooling
- [ ] Technical blog with all ablations
- [ ] Release data, leaderboard, reproduction instructions

---

## 7. Budget

| Phase                                   | Resource        | Spark Days  | Cost          |
|-----------------------------------------|-----------------|-------------|---------------|
| Sprint 0 (Gates 1, 1b, 2, 3)            | DGX Spark       | 5–7         | $0            |
| Sprint 0b (benchmark authoring)         | Laptop          | 0           | $0            |
| Sprint 1 (orchestrator)                 | Laptop          | 0           | $0            |
| Sprint 2 (Dev-Bench)                    | DGX Spark       | ~7          | $0            |
| Sprint 2 (Bench-Control)                | DGX Spark       | ~8          | $0            |
| Sprint 2 (Codex-Long Codex arm)         | DGX Spark       | **TBD**     | $0            |
| Sprint 2 (Codex-Long SWE-Agent arm)     | DGX Spark       | **TBD**     | $0            |
| Sprint 2 (frontier API)                 | Claude/GPT API  | —           | ~$400         |
| Sprint 3 (training)                     | DGX Spark       | 2–4         | $0            |
| Sprint 3 (B1 eval — Test-Long)          | DGX Spark       | ~17–18      | $0            |
| Sprint 3 (B2 eval — SWE Final-Test, core)| DGX Spark       | ~73         | $0            |
| Sprint 3 (SWE-smith reference arm, opt.) | DGX Spark       | ~8          | $0            |
| SWE-bench Docker eval                   | Any arm64 machine | Parallel    | $0            |
| **Subtotal (fixed, core — no optional items)** |            | **~112–117**| **~$400**     |
| Sprint 0.5 (OPTIONAL serving opt.)      | DGX Spark       | 2           | ~$10–30       |
| Sprint 3 (Phase 2b stretch, pre-killed) | DGX Spark       | 4–17        | $0            |
| Codex-Long collection                   | DGX Spark       | **TBD**     | $0            |
| **Grand total (core + Sprint 0.5)**     |                 | **~114–119 + TBD** | **~$410–430** |

Fixed Spark arithmetic (core, without optional items or reference arm):
Sprint 0 (5–7) + Dev-Bench (7) + Bench-Control (8) + training (2–4) + B1 (17–18) + B2 core (73) = **~112–117**
With SWE-smith reference arm (+8): **~120–125**

**Eval arithmetic (explicit):**
- B1: 150 Codex × 110 min + 150 SWE-Agent × 60 min = ~275 + ~150 hrs = **~17–18 days**
  (Test-Long wall-clock TBD from Gate 4; 110 min/env is the Codex-27B baseline assumption)
- B2: 800 Codex × 110 min + 300 SWE-Agent × 60 min = ~1,467 + ~300 hrs = **~73 days** (core, no reference)
  SWE-smith reference arm (optional): +100 Codex × 110 min = ~183 hrs = **+~8 days → ~81 days with reference**
- B1 + B2 combined: **~90–91 days core** (without reference arm); **~98–99 days with reference**

**Total calendar time: unknown until Gate 4 pilot completes.**

Fixed bottlenecks (known):
- Sprint 3 B2 eval core (SWE Final-Test, 1100 runs): ~73 days
- Sprint 3 B2 eval with reference arm: ~81 days
- Sprint 3 B1 eval (Test-Long): ~17–18 days (exact wall-clock confirmed by Gate 4)

Variable bottleneck (unknown until Gate 4):
- Sprint 2 Codex-Long collection: depends on wall-clock per Codex-Long family type

Gate 4 produces the wall-clock-per-family measurements that convert all TBD
entries into estimates. Do not commit Sprint 2 Codex-Long resources before
Gate 4 completes.

**Mitigation options if collection or eval timeline is too long (post Gate 4):**
1. Reduce Train-Long to 1 seed: halves collection time, halves expected traces
2. Reduce Train-Long env count: expand variants per existing family to maintain diversity
3. Skip optional B1 second seed: B1 defaults to 1 seed; do not add seed 2 if eval budget is tight
4. Drop SWE-smith reference: saves ~4 days
5. Run Codex-Long and SWE-Agent arms in parallel overnight: saves calendar time

---

*Document version: 2.3*
*Date: April 2026*
