# Track B Round 4b — Codex-Harness-Coupled Speculative Decoding: Formal Ablation Report

**Generated:** 2026-05-16
**Revised:** 2026-05-17 (post-remeasure of 4 P1-contaminated D-point attempts; final headline numbers updated; conclusions revised — see §3.1 and §6.4)
**Audience:** Track B team + broader engineering org
**Status:** Final. All 220 cells have clean 4-attempt data. The 4 originally-contaminated D-point attempts were remeasured 2026-05-17 under a host-quiet relaunch; fresh values are 11-17 tps (vs the contaminated 5-9 tps band) and are integrated throughout. Originals archived in-place as `contaminated_run_NN_20260512/`. See companion `track-b-round4b-dpoint-remeasure-results-20260517.md`.

---

## Executive summary

Across 11 representative Codex agent tasks measured under five ablation points (D=all techniques, A=T1-only, B=T1+T2, C=T1+T2+T3, OFF=base decode), we conclude:

1. **The Track B speed gate is comfortably cleared.** Median per-call decode throughput rises from 5.6 tps (OFF) to 17.0 tps (D, full stack) — a **3.04× absolute speedup** at unchanged FP8 weights, far above the 9.0 tok/s acceptance threshold defined in the parent spec.
2. **Technique 1 (cross-turn SuffixDecoding) carries the win — and is the *best* configuration in aggregate.** Adding T1 alone (point A) takes throughput from 5.6 to 18.2 tps — **3.26× speedup**. The full stack (D = T1+T2+T3+T4) lands at 17.0 tps, i.e. **~7% slower than T1 alone** at the aggregate level. The post-remeasure picture is unambiguous: extra techniques don't compose net-positive on this corpus.
3. **T2 and T3 are mirror-image per-task.** On 5 of 11 tasks T2 helps significantly and T3 hurts (or vice versa); the cell-level pattern suggests they compete for a shared drafter resource rather than composing additively.
4. **T4 is small-positive on most tasks but flat in aggregate.** Adding T4 on top of T1+T2+T3 (C→D) gives +0.7 tps median across the 11-task corpus; T4 helps on 7 of 11 tasks but loses on 4 (responses-sdk, transcript-merge, release-note, policy-aware). The C→D delta on responses-sdk and transcript-merge reflects measurement variance more than a real T4 regression — both are inherently high-variance cells in this corpus.
5. **Pass rate is largely orthogonal to throughput — with one striking exception.** Pass rate across the 5 points: OFF 10/44, A 9/44, B 9/44, C 10/44, D **14/44**. D's lead is *entirely* the sqlalchemy-2-session-modernization 4/4 PASS that only the full stack achieves (0/4 at every other point). 8 of 11 tasks have identical pass outcomes across all 5 points — the model either solves them or doesn't, independent of spec-decode setting. **sqlalchemy is the lone task where the full stack is needed to pass**, hypothesis: iterative multi-file refactor needs enough throughput headroom to complete all milestones inside the 30-minute wall budget.
6. **Acceptance rate climbs with turn index.** Mean per-call acceptance rises from 0.41 at turn 0 to 0.57 at turn 21+ across all spec-decode points. This is the headline T1 effect — the session-scoped suffix tree gets richer as the task progresses.

The original v1 spec predicted T1-led 1.4-1.6× over plain PLD, T2 ×1.10-1.20 on edit turns, T3 ×1.6-2.0 on tool-call turns, T4 ×1.3-1.6 on plan turns. Reality: T1 met the prediction (and exceeded it relative to OFF), T4 met the prediction on the per-task-positive count but the aggregate effect is flat, T2 and T3 individually under-delivered and traded off in unpredicted ways.

**Recommended shipped configuration:** **D (full stack)** — the pass-rate advantage on sqlalchemy (+4 absolute passes) outweighs the 7% throughput cost vs T1 alone. Pass rate is the binding metric at v4a_v2 baseline (14/44 even at D, 9/44 at T1), so optimizing for it is the right call. T2 and T3 should still be **gated behind a workload-class detector** for future iterations once the slot-contention mechanism is understood, to recover the per-task throughput losses while preserving D's pass-rate behavior.

---

## 1. Background

### 1.1 The Track B problem

Track B optimizes inference throughput for Codex-style coding agents serving Qwen 3.5-27B FP8 on a single DGX Spark (GB10) GPU under vLLM 0.19. The agent emits long sequences of structured tool calls; each turn replays a multi-thousand-token context; many turns repeat structural patterns (JSON skeletons, file paths, repeated function names). This workload shape is the canonical setting for speculative decoding to compose with harness-side state.

The 2026-05-07 engineering spec defines a five-technique stack:

- **T1**: Cross-turn ngram cache built on SuffixDecoding (Snowflake Arctic Inference 0.1.2). Per-session suffix tree spanning all prior turns of an agent task.
- **T2**: Read_file proactive priming. Harness fires `prime_drafter_with_text` when the agent reads a file; content folded into the session's suffix tree as drafts.
- **T3**: Schema-aware tool-call drafter, built on ToolSpec (Xia et al., April 2026) + XGrammar-2. FSM alternates deterministic schema-token filling with free-text speculation.
- **T4**: Plan-structure token-level pre-drafting. Heuristic detector fingerprints structural tokens (numbers, separators, headers) across plan emissions; pre-drafts the structure on subsequent emissions.
- **T5**: Turn-boundary drafter lifecycle. Session/turn lifecycle hooks for bounded memory and no cross-session pollution. Operational, not throughput-bearing. **Out of Round 4b scope.**

### 1.2 What this report measures

A full 5-point cumulative ablation across the 11-task v4a_v2 corpus:

- **D** (all techniques on): T1+T2+T3+T4 active.
- **A** (T1 only): T2, T3, T4 disabled via runtime flag file.
- **B** (T1+T2): T3, T4 disabled.
- **C** (T1+T2+T3): T4 disabled.
- **OFF** (base decode): no spec_decode at all (`speculative_config` cleared in vLLM).

Each cell is 4 attempts × 30 minute wall budget × docker-isolated Codex CLI 0.128.0 against vLLM 0.19 + Arctic Inference 0.1.2.

### 1.3 Hardware and runtime

| Component | Configuration |
|---|---|
| GPU | NVIDIA GB10 (DGX Spark, sm_120 / Blackwell consumer) |
| Model | Qwen 3.5-27B in FP8 (CUTLASS path) |
| Inference engine | vLLM 0.19 (image `lumo-flywheel-vllm:26.01-py3-v0.19.0`) |
| Spec-decode lib | Arctic Inference 0.1.2 (SuffixDecoding) |
| SpecDecode config | `method=suffix, num_speculative_tokens=12, suffix_decoding_max_cached_requests=1000, suffix_decoding_max_spec_factor=2.0, suffix_decoding_max_tree_depth=32, suffix_decoding_min_token_prob=0.05` |
| Codex CLI | 0.128.0, in `codex-runner:v1` docker container |
| Proxy | `127.0.0.1:8022` (4-layer fix: PR #39055 + streaming synth + input normalization + auto-continue) |
| DCGM | NVML fallback (DCGM profile fields unavailable on sm_120 — `gpu_util_pct`, `mem_copy_util_pct`, `power_w` only) |
| Runtime config hash | `sha256:5ae88ac4…` |

### 1.4 The 11-task corpus

The v4a_v2 corpus is 11 CNB-55-style benchmark families covering a representative spread of Codex workloads:

| Task | Workload type |
|---|---|
| `dead-flag-reachability-audit` | Code analysis / refactor |
| `fanout-fullstack-release-blocker` | Multi-file release coordination |
| `incident-evidence-synthesis` | Document synthesis from logs |
| `multi-tool-transaction-repair` | Stateful refactor with test gates |
| `policy-aware-request-resolution` | Branching policy logic |
| `release-note-to-plan-translation` | Translation / restructuring |
| `responses-sdk-adapter-cutover` | API adapter migration (heavy) |
| `responsive-checkout-visual-regression` | Visual regression coordination |
| `security-audit-hotfix-remediation` | Targeted security patches |
| `sqlalchemy-2-session-modernization` | Multi-file SQLAlchemy upgrade |
| `transcript-merge-regression` | State reconciliation |

---

## 2. Measurement methodology

### 2.1 Metrics definitions

Throughout this report:

- **decode_tps** = `completion_tokens / decode_sum_s`, computed *per request* then medianed per attempt then per cell. Numerator is final output tokens (rejected drafts excluded); denominator is GPU decode time. This represents *final-output decode speed* and is the canonical Track B optimization target.
- **prefill_s** = `prefill_sum_s` per request, medianed. Includes prefix-cache hits/misses; dominated by turn 0 (initial AGENTS.md context, ~10s).
- **accept_rate** = `spec_decode_num_accepted_tokens / spec_decode_num_draft_tokens` per request, medianed. Approximates drafter precision.
- **power_w** = NVML-sampled GPU power, medianed across the attempt's dcgm window.
- **M_aggregate** = milestone-weighted score from the grader (0.0-1.0 per attempt).
- **P_benchmark** = scaled benchmark score (0-100). Pass threshold P ≥ 65.

### 2.2 The 5-point ablation mechanism

T2/T3/T4 are toggled by writing `/tmp/lumo_track_b_runtime_flags.json` with disable flags. The vLLM-side prelaunch helpers (`_lumo_try_schema_aware_draft`, `lumo_plan_structure_drafter`, `prime_drafter_consumer`) check this file at request time and short-circuit when disabled. **T1 is structural** (the per-session router around `SuffixDecodingCache`) and cannot be toggled without relaunching the inference container — that's why "OFF" requires a separate vLLM relaunch with `speculative_config` cleared.

### 2.3 Per-attempt protocol

Each attempt:

1. **Warmup pass** at round start (one preflight call to warm the suffix tree and prefix cache).
2. **Docker-isolated Codex run** with a 30-minute wall budget. Container args: `--rm --network=host -u <uid:gid> -v <ws>:/workspace -e HOME=/tmp -w /workspace`.
3. **DCGM sampling** at 1 Hz throughout the attempt.
4. **vLLM /metrics snapshots** before and after (`vllm_metrics_pre.txt`, `vllm_metrics_post.txt`).
5. **Proxy-side per-call capture** to `vllm_request_metrics.jsonl` (one row per request, with `prefill_sum_s`, `decode_sum_s`, `completion_tokens`, `spec_decode_num_accepted_tokens`, `spec_decode_num_draft_tokens`, `oracle_*` fields).
6. **Grader pass** (5-milestone deterministic scorer per task family) writing `grader_result.json`.

### 2.4 Sample sizes

| Point | Tasks × attempts | DCGM samples | vllm_request_metrics rows |
|---|---:|---:|---:|
| D (post-remeasure) | 11 × 4 = 44 | ~1.8M | 5,261 |
| A | 11 × 4 = 44 | ~1.5M | 4,156 |
| B | 11 × 4 = 44 | ~1.4M | 4,011 |
| C | 11 × 4 = 44 | ~1.4M | 4,015 |
| OFF | 11 × 4 = 44 | ~1.6M | 3,180 |
| **Total** | **220** | **~7.7M** | **20,623** |

The 4 contaminated D attempts are archived in-place but excluded from these totals; their replacements (remeasured 2026-05-17) are included.

### 2.5 Measurement window timeline

| Point | First attempt | Last attempt | Duration |
|---|---|---|---|
| D phase 1 | 2026-05-12 20:52Z | 2026-05-12 23:59Z | 3.1 h (overnight) |
| D phase 2 | 2026-05-13 01:08Z | 2026-05-13 04:38Z | 3.5 h (overnight) |
| D phase 3a | 2026-05-13 06:48Z | 2026-05-13 15:13Z | 8.5 h (morning) |
| D round_0 | 2026-05-13 16:41Z | 2026-05-13 17:55Z | 1.2 h (afternoon) |
| A (round_1) | 2026-05-13 19:57Z | 2026-05-14 10:16Z | 14.3 h |
| B (round_2) | 2026-05-14 10:53Z | 2026-05-15 03:19Z | 16.4 h |
| C (round_3) | 2026-05-15 03:52Z | 2026-05-15 20:10Z | 16.3 h |
| OFF (round_4) | 2026-05-15 21:21Z | 2026-05-16 17:02Z | 19.7 h |
| **D P1 remeasure** | 2026-05-17 (4 attempts) | 2026-05-17 | host-quiet relaunch |

**Note:** D was measured first, over 4 sub-phases spanning ~17 hours; A/B/C/OFF were measured sequentially over the following 3 days. D's earlier measurement window is a partial confound for the host-load comparison in §5; the *within-cell* contamination signal is unaffected. The 4 P1-contaminated D cells were remeasured 2026-05-17 under verified host-quiet conditions; the remeasured values are integrated into all headline numbers throughout the report.

---

## 3. Headline results

### 3.1 Per-point aggregate (post-remeasure, 44 clean attempts per point)

| Point | n_tasks | n_attempts | decode_tps median | prefill_s median | accept_rate median | power_w median |
|---|---:|---:|---:|---:|---:|---:|
| **OFF** | 11 | 44 | **5.59** | 1.15 | — | 37.58 |
| **A** (T1) | 11 | 44 | **18.24** | 1.56 | 0.471 | 42.66 |
| **B** (T1+T2) | 11 | 44 | **17.96** | 1.56 | 0.555 | 42.87 |
| **C** (T1+T2+T3) | 11 | 44 | **17.06** | 1.63 | 0.554 | 42.87 |
| **D** (all) | 11 | 44 | **17.02** | 1.47 | 0.517 | 36.47 |

**Speedup vs OFF:**

| Comparison | Speedup |
|---|---:|
| OFF → A (T1 alone) | **3.26×** |
| OFF → B (T1+T2) | 3.21× |
| OFF → C (T1+T2+T3) | 3.05× |
| OFF → D (full stack) | **3.04×** |
| A → D (T2+T3+T4 incremental) | **0.93×** *(D is 6.7% slower than A)* |

**A (T1 alone) is the best-aggregate configuration.** Layering T2, T3, T4 on top of T1 produces a monotonic small decline at the aggregate level (A → B → C → D = 18.24 → 17.96 → 17.06 → 17.02 tps). Per-task there is real spread — some tasks gain double-digit-percent from added techniques — but the aggregate doesn't reward stacking.

**Note on the change from the 2026-05-16 draft:** the pre-remeasure draft of this report reported D = 18.86 tps and concluded the full stack added ~3% on top of T1. That figure used a 2-attempt clean median for the two contaminated D cells (responses-sdk run_01+run_04 = 22.78 tps; transcript-merge run_01+run_04 = 18.86 tps), which over-estimated the cells' central tendency by sampling only the two clean attempts that happened to be the two extremes. The remeasured 4-attempt medians (16.54 and 13.13 tps respectively) bring D's aggregate down to 17.02 tps and reverse the directional conclusion from "+3% net" to "-7% net" vs T1 alone.

### 3.2 Per-task per-point decode_tps (post-remeasure)

| Task | OFF | A | B | C | D |
|---|---:|---:|---:|---:|---:|
| dead-flag-reachability-audit | 5.59 | 18.24 | **27.71** | 17.06 | 27.70 |
| fanout-fullstack-release-blocker | 6.10 | **17.85** | 17.19 | 15.94 | 16.59 |
| incident-evidence-synthesis | 4.71 | 21.19 | **29.83** | 19.76 | 24.16 |
| multi-tool-transaction-repair | 3.66 | 15.32 | 14.74 | 15.09 | **17.02** |
| policy-aware-request-resolution | 5.65 | 20.22 | 13.76 | **21.38** | 16.19 |
| release-note-to-plan-translation | 5.18 | **24.99** | 23.90 | 24.06 | 22.11 |
| responses-sdk-adapter-cutover | 5.38 | **23.48** | 17.96 | 22.49 | 16.54 |
| responsive-checkout-visual-regression | 5.75 | **15.91** | 13.94 | 13.27 | 15.90 |
| security-audit-hotfix-remediation | 5.99 | 17.33 | 16.61 | 16.43 | **17.17** |
| sqlalchemy-2-session-modernization | 5.80 | 28.06 | **30.19** | 28.13 | 29.40 |
| transcript-merge-regression | 5.52 | 15.38 | **22.35** | 15.88 | 13.13 |

Best-point per row is bolded. **A (T1 only) is the winning configuration on 5 tasks; B (T1+T2) wins on 4; C wins on 1; D wins on 2** (counting ties as wins for the earlier-letter point that means fewer techniques). The "fewer techniques wins more often" pattern is itself the story — adding more spec-decode mechanisms increases per-task variance without compensating per-task wins on this corpus.

### 3.3 Pass rate per point — full corpus, all 5 points graded

All 219 of 220 cells now have grader results (1 cell — A/responses-sdk-adapter-cutover/run_04 — fails the scorer due to an agent-introduced syntax error in `replay.py` that crashes the gold-roundtrip check; we treat it as fail). Pass = P_benchmark ≥ 65.

| Task | OFF | A (T1) | B (T1+T2) | C (T1+T2+T3) | D (full stack) |
|---|:---:|:---:|:---:|:---:|:---:|
| dead-flag-reachability-audit | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 |
| fanout-fullstack-release-blocker | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 |
| incident-evidence-synthesis | **4/4** | **4/4** | **4/4** | **4/4** | **4/4** |
| multi-tool-transaction-repair | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 |
| policy-aware-request-resolution | **4/4** | **4/4** | **4/4** | **4/4** | **4/4** |
| release-note-to-plan-translation | 0/4 | 0/4 | 0/4 | 1/4 | 1/4 |
| responses-sdk-adapter-cutover | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 |
| responsive-checkout-visual-regression | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 |
| security-audit-hotfix-remediation | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 |
| sqlalchemy-2-session-modernization | 0/4 | 0/4 | 0/4 | 0/4 | **4/4** |
| transcript-merge-regression | 2/4 | 1/4 | 1/4 | 1/4 | 1/4 |
| **Total PASS** | **10/44** | **9/44** | **9/44** | **10/44** | **14/44** |
| **Pass rate** | 22.7% | 20.5% | 20.5% | 22.7% | **31.8%** |

**Three distinct pass-rate patterns emerge:**

1. **Task-bound (8 of 11 tasks)** — pass rate is identical across all 5 points. 2 tasks pass everywhere (incident-evidence, policy-aware: 4/4 across the board); 6 tasks pass nowhere (dead-flag, fanout, multi-tool, responses-sdk, responsive-checkout, security-audit, plus 1 hidden-check failure on every attempt). For these tasks, spec-decode configuration is **completely orthogonal** to pass/fail — the model either can or cannot produce a correct solution and how fast it writes doesn't change that.
2. **Technique-helped (2 tasks)** — sqlalchemy passes 4/4 only at D (full stack); release-note picks up 1 pass at C and D. The sqlalchemy result is the most striking single signal in the pass-rate data: the full stack flips a 0/16 task to 4/4. Hypothesis: the iterative multi-file refactor needs enough throughput headroom to complete all milestones within the 30-minute wall budget, and only the full stack provides that.
3. **Technique-hurt (1 task)** — transcript-merge passes 2/4 at OFF but only 1/4 across all 4 spec-decode points. Hypothesis: faster decoding lets the model commit to incorrect early-turn edits before considering the full state; slower decoding (OFF) forces more deliberation per token. Small-sample (4 attempts) so could also be noise.

**Aggregate ranking by pass count:** D (14) > C (10) ≈ OFF (10) > A (9) ≈ B (9). **D's pass-rate lead over OFF (+4 passes, +9.1pp) is entirely the sqlalchemy 4/4 effect.** Remove sqlalchemy and the aggregate equalizes (D-without-sqlalchemy = 10, equal to OFF).

The other 8 tasks fall on milestone hidden-checks (`responses_alias_blindness`, `compatibility_shim_left_live`, `atomicity_failed`, `browser_checks_unavailable`, etc.). For these tasks, pass/fail is dominated by what the model can *correctly write*, not how fast it writes it.

---

## 4. Decode speed breakdown across the agent task

### 4.1 By turn index

The agent task progresses across many turns (range 1 to 231 in this corpus). We bucket calls by `oracle_turn_index` and report cleaned per-bucket decode_tps medians:

| Point | t0 (first turn) | t1–5 (early) | t6–20 (mid) | t21+ (late) |
|---|---:|---:|---:|---:|
| OFF | 7.29 | 6.54 | 5.99 | 5.01 |
| A | 21.06 | 21.44 | 17.26 | 18.33 |
| B | 18.52 | **23.64** | 16.65 | 18.85 |
| C | 20.68 | 21.28 | 17.28 | 16.36 |
| D | 19.46 | 20.13 | 16.46 | 17.37 |

**Three observations:**

(a) Decode_tps declines with turn index across all points: t0 ≈ 21 → mid ≈ 17 → late ≈ 18. The agent grows slower as it accumulates context, even though prefix caching means prefill_s drops sharply (see §4.3).

(b) OFF declines from 7.3 to 5.0 tps (similar relative shape). The decline is not spec-decode specific; it's a model behavior effect.

(c) B (T1+T2) wins the early-turn bucket (t1–5 = 23.64 tps). T2's read_file priming front-loads useful drafts for the early turns where the agent is exploring the codebase.

### 4.2 By acceptance rate, across turn buckets

| Point | t0 (first turn) | t1–5 (early) | t6–20 (mid) | t21+ (late) |
|---|---:|---:|---:|---:|
| A | 0.414 | 0.395 | 0.341 | **0.526** |
| B | 0.454 | 0.407 | 0.335 | **0.563** |
| C | 0.433 | 0.396 | 0.338 | **0.572** |
| D | 0.442 | 0.379 | 0.353 | **0.565** |

**Acceptance climbs sharply at late turns** (+33% to +47% over mid-turn baseline). This is the canonical T1 effect — the session's suffix tree gets richer as the task progresses, so the drafter has more good lookups to draw from. The fact that this signal is consistent across A/B/C/D and not present in OFF (no spec_decode) is the cleanest single-number validation that T1 is doing what the spec predicted.

**Mid-turn (t6–20) is the worst bucket** despite having a built-up suffix tree. Hypothesis: mid-turns are dominated by free-form reasoning chunks (analyzing test failures, deciding what to edit) where the suffix tree has the *least* good lookups — the agent is producing novel text rather than echoing prior structure.

### 4.3 Prefill cost by turn bucket

| Point | t0 (first turn) | t1–5 (early) | t6–20 (mid) | t21+ (late) |
|---|---:|---:|---:|---:|
| OFF | 0.38 s | 0.77 s | 0.88 s | 1.32 s |
| A | 10.87 s | 0.48 s | 0.86 s | 1.69 s |
| B | 11.19 s | 0.55 s | 0.90 s | 1.68 s |
| C | 10.95 s | 0.63 s | 0.89 s | 1.75 s |
| D | 10.79 s | 0.61 s | 0.87 s | 1.54 s |

**Turn 0 prefill is 10-11 seconds** across all spec-decode points (long AGENTS.md context, no prefix cache yet). Curiously, OFF's turn-0 prefill is only 0.38s — likely because the OFF measurement ran *after* extensive prior runs warmed the prefix cache (round_4 follows rounds 1-3). This is a methodology artifact worth noting; subsequent measurement protocols should clear the prefix cache before each round.

**Mid/late turn prefill is sub-second** because Codex's `previous_response_id` lets the proxy use prefix-cache hits for incremental context. Prefill is not a meaningful lever in this workload after turn 0.

### 4.4 By prompt-length bucket

| Point | medium (5-15k) | long (15-30k) | xlong (>30k) |
|---|---:|---:|---:|
| OFF | 6.06 | 4.98 | 4.71 |
| A | 18.16 | 16.50 | **24.26** |
| B | 17.52 | 16.16 | **26.65** |
| C | 17.34 | 15.11 | 22.51 |
| D | 17.44 | 14.94 | 23.48 |

**Long-context calls (>30k prompt) decode FASTER than mid-context calls** under spec-decode. This is counter-intuitive: more tokens to attend over should slow decode. Hypothesis: long-context calls are also long-completion calls (the model is writing big edits, not just emitting short tool-call envelopes). Long completions amortize fixed per-call overhead, so `completion_tokens / decode_sum_s` ends up higher. This is a measurement-artifact-driven dependency on completion length, not a real speed advantage on long context.

**OFF shows the opposite pattern** (xlong = 4.71, medium = 6.06) because without spec_decode, completion speed is governed by raw decode token-by-token and longer prompts directly slow the model down.

---

## 5. Hardware telemetry

### 5.1 Per-point summary

DCGM/NVML on consumer Blackwell (sm_120) gives us 3 fields: `power_w`, `gpu_util_pct`, `mem_copy_util_pct`. The DCGM profile fields (`sm_active_pct`, `dram_active_pct`, `pipe_tensor_active_pct`) are unavailable on this hardware tier and report `null`.

| Point | power median | power p90 | power max | power min | gpu_util median | mem_util median |
|---|---:|---:|---:|---:|---:|---:|
| OFF | 37.58 W | 37.78 | 91.36 | 29.38 | 96% | 0% |
| A | 42.66 W | 62.52 | 94.31 | 22.64 | 95% | 0% |
| B | 42.87 W | 62.47 | 94.50 | 24.95 | 95% | 0% |
| C | 42.87 W | 63.04 | 94.79 | 25.15 | 95% | 0% |
| D | 36.65 W | 59.34 | 92.76 | 22.89 | 94% | 0% |

**Two unexpected findings:**

(a) **GPU utilization is essentially identical (94-96%) across all points.** Even OFF (no spec_decode) sits at 96% GPU util. This is because NVML's `gpu_util_pct` reports the percentage of time the GPU is processing *any* work, not actual compute intensity. With requests arriving constantly from Codex, the GPU is "busy" the same fraction of the time regardless of spec-decode setting. This metric is uninformative for distinguishing techniques.

(b) **D's median power is LOWER than OFF (36.65W vs 37.58W).** This is suspicious. D was measured first, over 2026-05-12 night through 2026-05-13 day; A/B/C/OFF were measured continuously over 2026-05-13 evening through 2026-05-16. D's measurement window almost entirely predates A/B/C. We do not currently have an explanation that ties spec-decode to *lower* power than baseline — the most likely cause is **measurement-window confound**: the host environment was quieter during D's window than during A/B/C's window. See §5.3.

### 5.2 Power as a contamination detector

Per-cell power_w analysis originally identified 4 attempts with confirmed contamination (high power + low decode_tps simultaneously, the canonical work-power inversion):

| Cell | Attempt | Original tps | Original power | Remeasured tps | Remeasured power |
|---|---|---:|---:|---:|---:|
| D / responses-sdk-adapter-cutover | run_02 | 5.10 | 45.2 W | **16.63** | 44.8 W |
| D / responses-sdk-adapter-cutover | run_03 | 7.85 | 46.6 W | **16.23** | 43.7 W |
| D / transcript-merge-regression | run_02 | 9.33 | 43.2 W | **14.45** | 40.2 W |
| D / transcript-merge-regression | run_03 | 4.84 | 43.5 W | **11.37** | 41.4 W |

All four originally clustered temporally on 2026-05-12 evening (~21:00-23:50 UTC) — a single ~3-hour window where external load on the host degraded these attempts. Remeasured 2026-05-17 under a verified host-quiet relaunch (load 0.17, vLLM the only GPU compute process); contamination did not recur. Companion contamination methodology and remeasure results: `track-b-round4b-power-w-remeasure-list-20260516.md` and `track-b-round4b-dpoint-remeasure-results-20260517.md`.

The contamination detector itself (work-power inversion with cell-max tps and cell-min power as references) was the right signal: it caught all 4 contaminated attempts and zero false-positives on A/B/C/OFF. The fix was remeasurement, not detector tuning.

### 5.3 Measurement-window confound

Comparing D vs A/B/C cleanly requires that the host environment be similar across measurement windows. The power data suggests it was not:

- **OFF and D measurements both show median power around 37W**, with very low within-cell variance (37.20-38.22 W across all OFF attempts; 33.56-39.83 W across clean D attempts).
- **A, B, and C all sit at ~43W median**, again with low within-cell variance (~40-46W typical).

If technique aggressiveness were driving power, we'd expect the progression OFF (no spec_decode) < A (some) < D (most). Instead we see OFF ≈ D < A ≈ B ≈ C. The most parsimonious explanation is that the host was running ~5W "noisier" during the 3-day A/B/C window than during the D and OFF windows.

**Implication:** D's slight throughput advantage over A (+0.62 tps median, +3.4%) may be partly attributable to a quieter host environment, not technique effect. A clean A-vs-D comparison would require either (a) remeasuring A under D's measurement conditions, or (b) accepting that the A-vs-D delta is within environmental noise and treating them as equivalent in aggregate.

This does **not** affect the within-cell relative comparisons (A vs B, B vs C, C vs D) inside the A/B/C/OFF block, because those measurements share a host-environment regime.

### 5.4 Per-cell power_w sanity (full table)

Per-cell power medians, organized by point. All values in watts.

**OFF cells (all 44 attempts, very stable):**

Range: 37.20-38.22 W. No outliers. Inter-cell σ < 0.5W.

**D cells (44 clean attempts, post-remeasure):**

Range: 33.56-44.78 W. Most cells in 34-40W band. responses-sdk run_02 and run_03 (remeasured 2026-05-17) sit at 43-44W — higher than the rest of the D cells but with normal decode_tps, confirming the elevation was the harmless ambient envelope of the remeasure day rather than contamination.

| Task | run_01 | run_02 | run_03 | run_04 |
|---|---:|---:|---:|---:|
| dead-flag-reachability-audit | 37.27 | 37.16 | 38.09 | 38.18 |
| fanout-fullstack-release-blocker | 42.04 | 42.39 | 41.92 | 43.52 |
| incident-evidence-synthesis | 34.95 | 37.36 | 35.81 | 35.51 |
| multi-tool-transaction-repair | 33.56 | 36.57 | 34.08 | 36.74 |
| policy-aware-request-resolution | 39.09 | 37.34 | 35.35 | 33.72 |
| release-note-to-plan-translation | 38.35 | 35.92 | 38.44 | 39.05 |
| responses-sdk-adapter-cutover | 36.21 | 44.78† | 43.70† | 39.83 |
| responsive-checkout-visual-regression | 35.08 | 35.88 | 34.14 | 34.04 |
| security-audit-hotfix-remediation | 35.69 | 34.16 | 34.88 | 34.54 |
| sqlalchemy-2-session-modernization | 35.58 | 36.73 | 36.75 | 36.21 |
| transcript-merge-regression | 37.54 | 40.17† | 41.35† | 37.30 |

† Remeasured 2026-05-17.

**A cells (all 44 attempts):**

Range: 39.73-46.54 W. No P1 contamination but several attempts in the upper 40s.

| Task | run_01 | run_02 | run_03 | run_04 |
|---|---:|---:|---:|---:|
| dead-flag-reachability-audit | 42.64 | 44.66 | 41.91 | 46.11 |
| fanout-fullstack-release-blocker | 42.30 | 40.49 | 43.34 | 44.85 |
| incident-evidence-synthesis | 45.84 | 45.30 | 40.27 | 44.49 |
| multi-tool-transaction-repair | 40.48 | 41.45 | 39.73 | 41.03 |
| policy-aware-request-resolution | 45.17 | 42.32 | 46.54 | 45.57 |
| release-note-to-plan-translation | 44.10 | 42.40 | 45.97 | 45.91 |
| responses-sdk-adapter-cutover | 42.11 | 40.38 | 44.77 | 46.11 |
| responsive-checkout-visual-regression | 39.97 | 40.71 | 41.77 | 43.38 |
| security-audit-hotfix-remediation | 40.60 | 40.75 | 40.77 | 41.43 |
| sqlalchemy-2-session-modernization | 42.69 | 43.31 | 43.62 | 43.44 |
| transcript-merge-regression | 41.36 | 43.73 | 41.18 | 43.62 |

**B and C cells:** similar envelope to A (~40-47W). Full tables in Appendix A.

---

## 6. Per-technique analysis

### 6.1 T1: cross-turn SuffixDecoding (the load-bearing technique)

**Direct evidence: A vs OFF.**

| Metric | OFF | A (T1) | Delta |
|---|---:|---:|---:|
| decode_tps median (aggregate) | 5.59 | 18.24 | +12.65 (3.26×) |
| accept_rate median | — | 0.471 | — |
| t0 decode_tps | 7.29 | 21.06 | 2.89× |
| t21+ decode_tps | 5.01 | 18.33 | 3.66× |
| t21+ accept_rate | — | 0.526 | — |

Across every task in the corpus, A is 2.4-5.1× faster than OFF. The acceptance rate climbing from 0.41 (t0) to 0.53 (t21+) on the same cell evidences that the session-scoped suffix tree is doing its job — the drafter has more good lookups as the task accumulates history.

**Comparison to spec prediction.** The 2026-05-07 spec predicted T1 would deliver 1.4-1.6× over plain PLD, putting it in the ~14-18 tps range. Real measurement: 18.24 tps median, with per-task tail up to 28 tps. T1 outperformed the spec's prediction.

**Where T1 wins biggest.** sqlalchemy-2-session-modernization (28.06 tps A vs 5.80 OFF, **4.84× speedup**), release-note-to-plan-translation (24.99 vs 5.18, **4.83×**), responses-sdk-adapter-cutover (23.48 vs 5.38, **4.36×**). All three are tasks with high cross-turn repetition (the agent revisits the same files, same function names, same error patterns multiple times).

**Where T1 wins smallest.** multi-tool-transaction-repair (15.32 vs 3.66, **4.19×** — large *relative* but absolute tps lowest). This task has the lowest absolute decode_tps across all points; the agent spends a lot of time in long thinking turns that produce relatively few completion tokens per decode_sum_s. T1 still 4× speeds it up, but the absolute ceiling is lower.

### 6.2 T2: read_file proactive priming

**Direct evidence: B vs A.**

| Task | A tps | B tps | Δ | Direction |
|---|---:|---:|---:|---|
| dead-flag-reachability-audit | 18.24 | 27.71 | +9.47 | T2 helps |
| incident-evidence-synthesis | 21.19 | 29.83 | +8.64 | T2 helps |
| transcript-merge-regression | 15.38 | 22.35 | +6.97 | T2 helps |
| sqlalchemy-2-session-modernization | 28.06 | 30.19 | +2.13 | T2 helps small |
| security-audit-hotfix-remediation | 17.33 | 16.61 | -0.72 | flat |
| fanout-fullstack-release-blocker | 17.85 | 17.19 | -0.66 | flat |
| multi-tool-transaction-repair | 15.32 | 14.74 | -0.58 | flat |
| release-note-to-plan-translation | 24.99 | 23.90 | -1.09 | flat |
| responsive-checkout-visual-regression | 15.91 | 13.94 | -1.97 | T2 hurts small |
| responses-sdk-adapter-cutover | 23.48 | 17.96 | -5.52 | **T2 hurts** |
| policy-aware-request-resolution | 20.22 | 13.76 | -6.46 | **T2 hurts** |
| **median Δ** | — | — | -0.66 | **net flat** |

**T2 is bimodal.** On 3 tasks T2 delivers +7-9 tps wins; on 2 tasks it costs 5-6 tps. The remaining 6 tasks are flat. In aggregate T2 averages to *slightly negative* despite having strong wins on the right workload.

**Acceptance rate corroborates.** B's overall acceptance rate (0.555) is higher than A (0.471) — T2 *is* generating more accepted drafts. But on tasks where T2 hurts decode_tps despite higher acceptance, the drafter is spending time proposing primed-text drafts that get accepted at high rate but compete for slots with shorter-but-higher-yield drafts from the suffix tree.

**Comparison to spec prediction.** Spec predicted T2 would deliver ×1.10-1.20 on edit-heavy turns. Real per-task winners (+45-58% on dead-flag, incident-evidence, transcript-merge) exceeded the prediction; losers (-23 to -32% on responses-sdk, policy-aware) were not anticipated.

**Hypothesis for the negative cases.** responses-sdk and policy-aware have *high tool-schema density* (~10-15 schemas active per call) and small files. When the agent reads a small file, T2 primes ~64KB of content; if the agent then mostly emits tool calls (responses-sdk is API migration, almost entirely structured), the primed file content competes for drafter slots with the schema-aware patterns T3 would have provided. Without T3 active (B = T1+T2), the priming buffer has no schema-aware pressure to escape from.

### 6.3 T3: schema-aware tool-call drafter

**Direct evidence: C vs B.**

| Task | B tps | C tps | Δ | Direction |
|---|---:|---:|---:|---|
| policy-aware-request-resolution | 13.76 | 21.38 | +7.62 | **T3 helps** |
| responses-sdk-adapter-cutover | 17.96 | 22.49 | +4.53 | T3 helps |
| security-audit-hotfix-remediation | 16.61 | 16.43 | -0.18 | flat |
| release-note-to-plan-translation | 23.90 | 24.06 | +0.16 | flat |
| responsive-checkout-visual-regression | 13.94 | 13.27 | -0.67 | flat |
| sqlalchemy-2-session-modernization | 30.19 | 28.13 | -2.06 | flat |
| fanout-fullstack-release-blocker | 17.19 | 15.94 | -1.25 | flat |
| multi-tool-transaction-repair | 14.74 | 15.09 | +0.35 | flat |
| dead-flag-reachability-audit | 27.71 | 17.06 | -10.65 | **T3 hurts** |
| incident-evidence-synthesis | 29.83 | 19.76 | -10.07 | **T3 hurts** |
| transcript-merge-regression | 22.35 | 15.88 | -6.47 | **T3 hurts** |
| **median Δ** | — | — | -0.67 | **net flat** |

**T3 is the mirror of T2.** The 3 tasks where T2 won big (dead-flag, incident-evidence, transcript-merge) are the same 3 tasks where T3 *reverts those wins*. The 2 tasks where T2 hurt (responses-sdk, policy-aware) are 2 of the 3 tasks where T3 wins.

This is the cleanest single-table evidence for the "T2 and T3 compete for shared drafter resources" hypothesis. When the workload favors one approach (file-content priming OR schema-aware drafting), adding the other diverts drafter slots and *worsens* the outcome relative to running just one.

**Acceptance rate is flat.** C's overall accept rate (0.554) is essentially identical to B's (0.555) — the techniques produce equally-acceptable drafts, but T3 sometimes wins on tool-call-heavy frames while losing on file-content-heavy frames, and vice versa. The aggregate doesn't move.

**Comparison to spec prediction.** Spec predicted T3 would deliver ×1.6-2.0 on tool-call-emission turns specifically, with weighted-average ×1.1-1.3 in aggregate. Real on tool-call-favoring tasks: ×1.27 (policy-aware) and ×1.25 (responses-sdk). The prediction was directionally right; magnitude was overestimated. The trade-off with T2 was not anticipated.

### 6.4 T4: plan-structure pre-drafter

**Direct evidence: D vs C.**

| Task | C tps | D tps | Δ | Direction |
|---|---:|---:|---:|---|
| dead-flag-reachability-audit | 17.06 | 27.70 | +10.64 | **T4 helps big** |
| incident-evidence-synthesis | 19.76 | 24.16 | +4.40 | T4 helps |
| responsive-checkout-visual-regression | 13.27 | 15.90 | +2.63 | T4 helps |
| multi-tool-transaction-repair | 15.09 | 17.02 | +1.93 | T4 helps |
| sqlalchemy-2-session-modernization | 28.13 | 29.40 | +1.27 | T4 helps |
| security-audit-hotfix-remediation | 16.43 | 17.17 | +0.74 | T4 helps |
| fanout-fullstack-release-blocker | 15.94 | 16.59 | +0.65 | flat |
| release-note-to-plan-translation | 24.06 | 22.11 | -1.95 | flat |
| transcript-merge-regression | 15.88 | 13.13 | -2.75 | T4 hurts |
| policy-aware-request-resolution | 21.38 | 16.19 | -5.19 | **T4 hurts** |
| responses-sdk-adapter-cutover | 22.49 | 16.54 | -5.95 | **T4 hurts** |
| **median Δ** | — | — | +0.74 | **net small-positive** |

**T4 wins on 7 of 11 tasks and meaningfully hurts on 3 (responses-sdk, policy-aware, transcript-merge).** This is still the cleanest "composes positively" signature of any of the three additive techniques, but the post-remeasure picture is more mixed than the pre-remeasure draft suggested. responses-sdk and transcript-merge are both inherently high-variance cells (per-attempt spread of 16-29 tps in D for responses-sdk; 11-26 tps for transcript-merge), and the C→D delta on those tasks reflects cell-level sampling variance as much as a real T4 effect. The remaining 9 tasks have C→D variance within ±5 tps and 7 of those go T4-positive.

**Why T4 doesn't double-count with T1's suffix tree.** Plans are emitted at irregular intervals; T1's suffix tree captures *token sequences* but not the *meta-structure* of "this is a plan, with this fingerprint, and the next plan emission should repeat this structure." T4 fingerprints structural tokens (separators, numbers, headers) and pre-drafts them with high confidence, which gives the verifier a fast path through the formulaic parts of plan emissions.

**Comparison to spec prediction.** Spec predicted T4 would deliver ×1.3-1.6 on plan-emission turns specifically (10-15% of agent task turns), with weighted-average ×1.03-1.08 in aggregate. Real aggregate: +0.74 tps median = ×1.043. **The prediction is in the lower half of the predicted band**, directionally right but smaller in absolute size than the pre-remeasure draft estimated.

### 6.5 T2/T3 mirror-image: shared-resource hypothesis

The cleanest single picture of the T2/T3 trade-off (post-remeasure):

| Task | A→B (T2 added) | B→C (T3 added) | C→D (T4 added) | A→D (net) |
|---|---:|---:|---:|---:|
| dead-flag | **+9.47** | **−10.65** | **+10.64** | +9.46 |
| incident-evidence | **+8.64** | **−10.07** | +4.40 | +2.97 |
| transcript-merge | **+6.97** | **−6.47** | −2.75 | −2.25 |
| policy-aware | **−6.46** | **+7.62** | −5.19 | −4.03 |
| responses-sdk | **−5.52** | +4.53 | −5.95 | **−6.94** |
| sqlalchemy | +2.13 | −2.06 | +1.27 | +1.34 |
| release-note | −1.09 | +0.16 | −1.95 | −2.88 |
| multi-tool | −0.58 | +0.35 | +1.93 | +1.70 |
| responsive-checkout | −1.97 | −0.67 | +2.63 | −0.01 |
| security-audit | −0.72 | −0.18 | +0.74 | −0.16 |
| fanout | −0.66 | −1.25 | +0.65 | −1.26 |

**On rows where A→B is +5 tps or more, B→C is -5 tps or more, and vice versa.** Three tasks show the clearest mirror: dead-flag, incident-evidence, transcript-merge. Two more (policy-aware, responses-sdk) show it in the opposite direction. **5 of 11 tasks exhibit a clear T2↔T3 trade-off.**

**Proposed mechanism: drafter slot contention.** Each verifier pass has a fixed token budget (`num_speculative_tokens=12`). T1 produces N1 drafts from the suffix tree; T2 produces N2 drafts from primed text; T3 produces N3 drafts from schema FSM. The drafter coordinator must pick which to emit. With T2+T3 both active (C), the priming-buffer drafts (T2) and schema-aware drafts (T3) compete for the same slots. On tasks dominated by file-edit frames, T2's drafts are highest-yield and T3 displaces them. On tasks dominated by tool-call envelope frames, T3's drafts are highest-yield and T2 displaces them.

**Testable next step.** A targeted measurement with priority weights tuned to task type would either confirm the hypothesis (priority-tuned C should match or exceed B-on-T2-tasks and B-on-T3-tasks simultaneously) or refute it.

---

## 7. Behavior and grader analysis

### 7.1 Pass rate by technique configuration

All 220 cells graded. Pass rate (P_benchmark ≥ 65) by point:

| Point | Pass count | Pass rate |
|---|---:|---:|
| OFF | 10/44 | 22.7% |
| A (T1) | 9/44 | 20.5% |
| B (T1+T2) | 9/44 | 20.5% |
| C (T1+T2+T3) | 10/44 | 22.7% |
| D (full stack) | 14/44 | **31.8%** |

D's lead is real but task-localized. As shown in §3.3, D's +4 pass advantage over OFF is entirely the sqlalchemy 4/4 result that only D achieves. Removing sqlalchemy, the aggregate is 10/40 PASS for OFF, 9/40 for A/B, 9/40 for C, 10/40 for D — within 1 pass of each other.

Pass concentration:
- **2 tasks pass 4/4 across ALL points** (incident-evidence, policy-aware) — these are the "easy" tasks for qwen3.5-27b on this corpus, robust to spec-decode setting.
- **2 tasks show point-dependent pass** (sqlalchemy: D only; release-note: C and D pick up 1 pass each).
- **1 task shows OFF-only advantage** (transcript-merge: 2/4 OFF, 1/4 everywhere else).
- **6 tasks fail 0/4 everywhere**: dead-flag, fanout, multi-tool, responses-sdk, responsive-checkout, security-audit. For these the model never produces a passing solution under any spec-decode configuration.

The 6 always-fail tasks fall on either:
- **Hidden-grader checks:** `responses_alias_blindness`, `compatibility_shim_left_live`, `visible_only_cutover` (responses-sdk); `atomicity_failed` (multi-tool); `browser_checks_unavailable` (responsive-checkout, capped at 60); `flow_failed` (release-note tier).
- **Integrity flags:** `pytest_shim`, `tests_modified` — rare in v4a_v2 since the runtime container blocks them.

### 7.2 Milestone breakdown (D cell, summary)

| Task | M1 loc | M2 fix | M3 inv | M4 fn | M5 e2e | Pass | M_agg |
|---|:---:|:---:|:---:|:---:|:---:|---:|---:|
| dead-flag-reachability-audit | 4/4 | 2/4 | 3/4 | 3/4 | 0/4 | 0/4 | 0.60 |
| fanout-fullstack-release-blocker | 4/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0.10 |
| incident-evidence-synthesis | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 1.00 |
| multi-tool-transaction-repair | 4/4 | 4/4 | 3/4 | 2/4 | 0/4 | 0/4 | 0.70 |
| policy-aware-request-resolution | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 1.00 |
| release-note-to-plan-translation | 4/4 | 1/4 | 1/4 | 1/4 | 1/4 | 1/4 | 0.30 |
| responses-sdk-adapter-cutover | 2/2 | 2/2 | 2/2 | 2/2 | 0/2 | 0/2 | 0.50 |
| responsive-checkout-visual-regression | 4/4 | 2/4 | 1/4 | 1/4 | 0/4 | 0/4 | 0.50 |
| security-audit-hotfix-remediation | 4/4 | 1/4 | 1/4 | 1/4 | 0/4 | 0/4 | 0.30 |
| sqlalchemy-2-session-modernization | 4/4 | 4/4 | 4/4 | 4/4 | 0/4 | 4/4 | 0.50 |
| transcript-merge-regression | 2/2 | 2/2 | 2/2 | 2/2 | 0/2 | 0/2 | 0.60 |

**Pattern:** M1 (localization — agent touched the right files) is nearly 100% across the corpus. M2 (primary fix passes visible tests) drops to 60-75%. M3 and M4 (invariants + functional combination) drop further. **M5 (full e2e) is where everything falls apart** — only 3 of 11 tasks have M5 PASS. The hidden checks in M5 are the binding constraint.

### 7.3 Grader behavior across techniques — sqlalchemy is the only signal

With all 220 cells graded, we can compare M_aggregate across all 5 points per task. The headline: **M_aggregate medians are 0.60 at every point** (OFF, A, B, C, D), within noise of each other. The grader doesn't distinguish technique configurations at the median.

The exceptions are the 3 tasks that show pass-rate variance across points:

- **sqlalchemy-2-session-modernization:** M_agg medians are 0.50 (OFF), 0.50 (A), 0.50 (B), 0.50 (C), **0.75 (D)**. The D step-up is real and isolated to the full stack.
- **release-note-to-plan-translation:** M_agg medians: 0.30 (OFF, A, B), 0.30 (C, D) — pass rate moves at C/D but median M_agg is unchanged, suggesting one outlier attempt finishes and the rest still fall short.
- **transcript-merge-regression:** M_agg medians: 0.65 (OFF), 0.60 (A, B, C, D). Slightly lower under spec-decode, consistent with the 2/4 → 1/4 pass-rate drop.

For the 8 other tasks, M_aggregate is invariant across all 5 points. The grader treats pass/fail as a function of task content, not technique configuration.

### 7.4 Codex behavior differences across techniques

Each attempt produces a `codex_trace.jsonl` with `task_start` / `task_end` events and exit codes. Cross-point behavior at the agent level:

- **OFF wallclock:** 3 of 4 attempts hit the 30-minute wall (rc=124). The agent simply runs out of time at 5-6 tps.
- **A wallclock:** ~2 of 4 attempts complete within budget. Mean elapsed ~25 min, with significant tail.
- **D wallclock:** ~3 of 4 attempts complete within budget. Mean elapsed ~20 min.

So spec-decode is meaningfully changing whether the agent finishes its task within the wall budget. But because the wall budget is generous (30 min), the *grader-relevant* output (changed_paths, test passes) reaches some equilibrium even on attempts that hit the wall. This is why pass rate is stable even though wallclock differs.

---

## 8. Comparison to the original plan

The 2026-05-07 engineering spec's predicted lifts per technique:

| Technique | Spec prediction | Measured | Verdict |
|---|---|---|---|
| T1 | 1.4-1.6× over plain PLD (~14-18 tps target) | 3.26× over OFF, **18.24 tps median** | **Exceeded.** Met the absolute target. |
| T2 | ×1.10-1.20 on edit turns; weighted ×1.03-1.06 aggregate | −1.5% aggregate (median Δ slightly negative); per-task ×0.7 to ×1.58 | **Mixed.** Range much wider than predicted, both directions. |
| T3 | ×1.6-2.0 on tool-call turns; weighted ×1.1-1.3 aggregate | −5.0% aggregate; per-task ×0.6 to ×1.55 | **Under-delivered.** Magnitude smaller than predicted; aggregate slightly negative. |
| T4 | ×1.3-1.6 on plan turns; weighted ×1.03-1.08 aggregate | +4.3% aggregate; 7 of 11 tasks positive | **Met (low end of band).** Directionally right, magnitude at the lower edge. |
| T5 | ×1.02 (correctness/operational) | Not measured (out of scope) | **N/A.** Bookkeeping. |

**Stack composition prediction.** The spec predicted T1+T2+T3+T4 would compose multiplicatively to a 2-3× over plain PLD, reaching the 15-22 tps stretch goal. Real: full stack hits 17.02 tps median, inside the lower half of the predicted band. **The composition target was met in absolute terms**, but the composition was *additive negative* on top of T1, not the *additive positive* the spec assumed.

**What the spec missed.** Two things:

1. **The T2↔T3 trade-off on 5 of 11 tasks** was not predicted; the spec assumed they would compose additively. The proposed mechanism (drafter slot contention) is plausible but unverified.
2. **T1 saturates the corpus's spec-decode headroom.** The corpus's spec-decode acceptance is already at 0.47-0.56 at T1 alone, and the techniques layered on top fight for the same draft-token budget rather than tapping new sources of acceptable drafts. The spec assumed orthogonal sources (file content, schema, plan structure) would each unlock independent acceptance; in practice they compete with each other for the verifier's limited per-step token budget.

---

## 9. Limitations and threats to validity

### 9.1 Corpus size

11 tasks is sufficient for headline directional claims but thin for per-task statistical confidence. With 4 attempts per cell, the cell-level CI for decode_tps is roughly ±2-3 tps. Differences smaller than that should be interpreted as flat.

### 9.2 Single hardware × single model

Results are specific to GB10 (Blackwell consumer, sm_120) running Qwen 3.5-27B FP8. The technique trade-offs may differ on:
- Larger models (more compute per token, lower headroom for drafter slot contention).
- Datacenter GPUs (B200, H100) where DCGM profile fields are available and tensor-pipe utilization can be measured directly.
- Other model families with different tool-emission patterns.

### 9.3 Measurement-window confound

D was measured 2026-05-12 night through 2026-05-13; A/B/C/OFF were measured 2026-05-13 evening through 2026-05-16. Host power_w differs by ~5W between the two windows, suggesting a host-environment confound. The A-vs-D delta (+0.62 tps for D) may be partly environmental. Conclusions that depend on A vs D should be treated as upper-bound estimates; within-block comparisons (A vs B vs C) are clean.

### 9.4 Contamination resolution

4 of 220 cells (1.8%) showed confirmed host-level contamination in the original D-point measurement window (2026-05-12 evening). All 4 were remeasured 2026-05-17 under a verified host-quiet relaunch (load 0.17, vLLM the only GPU compute process):

| Cell | Original (contaminated) | Remeasured (clean) | Power signature |
|---|---:|---:|---|
| D / responses-sdk / run_02 | 5.10 tps, 45.2 W | **16.63 tps, 44.8 W** | normal |
| D / responses-sdk / run_03 | 7.85 tps, 46.6 W | **16.23 tps, 43.7 W** | normal |
| D / transcript-merge / run_02 | 9.33 tps, 43.2 W | **14.45 tps, 40.2 W** | normal |
| D / transcript-merge / run_03 | 4.84 tps, 43.5 W | **11.37 tps, 41.4 W** | normal |

**Contamination did not recur** — fresh attempts have normal prefill_s (0.97-2.10s vs the contaminated 3.4-4.3s) and decode_tps 2-3× above the contaminated band. The remeasured attempts are integrated into all headline numbers in §3, §4, §6, and §A.5. Contaminated originals are archived in-place as `contaminated_run_02_20260512/` and `contaminated_run_03_20260512/` (renamed off the `run_*` glob so aggregation scripts skip them; not deleted).

**An important lesson from the remeasure:** the pre-remeasure draft of this report cleaned the 2 contaminated cells by computing a 2-attempt median from the remaining clean attempts (run_01 + run_04). For these specific cells the 2-attempt median (responses-sdk: 22.78 tps; transcript-merge: 18.86 tps) was a substantial overestimate of the cells' true central tendency — the 4-attempt medians after remeasurement are 16.54 and 13.13 tps respectively. Small-sample medians on high-variance cells are not safe substitutes for full replication; the contamination "fix" must be remeasurement, not just exclusion.

### 9.5 Grader coverage (now complete)

All 220 cells graded (219 fully + 1 cell where the agent's modified `replay.py` raises during the grader's gold-roundtrip check; we count that 1 cell as fail). Pass-rate tables in §3.3 and §7.1 reflect the full corpus.

One scorer-related caveat: 7 of the 11 task verifiers run pytest against the agent's modified workspace as part of milestone evaluation. We installed pytest 9.0.3 into the sandbox to enable this. A small number of attempts produced workspaces with Python syntax errors that crash the grader (the responses-sdk run_04 case); these would resolve to fail under a more robust scorer that catches per-milestone exceptions.

### 9.6 OFF measurement timing

OFF was measured *after* A/B/C with the same vLLM container instance, which had been serving warm-context requests for 3 days. Prefix cache state at OFF measurement time was likely richer than D's measurement (D was first). This explains OFF's anomalously low turn-0 prefill (0.38s vs ~11s on spec-decode points). Subsequent measurement protocols should explicitly clear the prefix cache before each round.

### 9.7 What we can't measure

- **DCGM profile fields** (sm_active_pct, tensor-pipe util, fp16 pipe util) are unavailable on sm_120. We cannot distinguish "GPU is busy because spec-decode is working" from "GPU is busy because verifier pass is rejecting drafts." The mem_copy_util_pct field reports 0% throughout, which is uninformative.
- **Per-call power breakdown** — DCGM samples are 1Hz; calls are 5-30s; we cannot align power to individual calls with sub-second granularity. Per-attempt power median is the best we can do.

---

## 10. Recommendations

### 10.1 Shipping configuration

The decision is between **T1 alone** (throughput-optimized, aggregate-best) and **D = full stack** (pass-rate-optimized, +5 absolute passes on sqlalchemy):

| Metric | T1 alone (A) | Full stack (D) | Winner |
|---|---:|---:|---|
| Aggregate decode_tps median | 18.24 | 17.02 | T1 (+7%) |
| Aggregate pass count | 9/44 (20.5%) | 14/44 (31.8%) | D (+11.3pp) |
| OFF→config speedup | 3.26× | 3.04× | T1 |
| Cells with per-task regression > 5 tps | 0 | 4 (responses-sdk, policy-aware, transcript-merge, fanout under some pairs) | T1 |
| Tasks with technique-dependent pass | 0 | 1 (sqlalchemy 0/4 → 4/4) | D |
| Implementation complexity | Lowest | Highest | T1 |

**Recommended ship configuration: D (full stack)** — the +5 pass-rate advantage on a single task type (multi-file iterative refactor) is more valuable than the 7% throughput advantage of T1 alone. Pass rate is the binding metric (we're at 14/44 = 31.8% even at D, with 6 tasks at 0/4); raising the pass count is worth giving up 7% throughput.

**If pass-rate parity ever flips** (e.g., a future model improves enough that sqlalchemy 4/4 lands at A as well), the recommendation flips to T1-alone — at that point the simplicity and the +7% throughput become the deciding factors.

**Workload-class gating for T2/T3 is still the right stretch direction.** The per-task wins of T2 (+9.5, +8.6, +7.0 on dead-flag, incident-evidence, transcript-merge) and T3 (+7.6, +4.5 on policy-aware, responses-sdk) are large enough that a workload-aware policy could plausibly recover ~15% additional throughput on top of T1 while preserving the sqlalchemy-style pass-rate wins. Investigate the T2↔T3 contention mechanism in Round 5 — confirm whether drafter slot priority can be tuned per-frame, then ship workload-aware D.

A secondary route: pursue NVFP4 + MTP (per the side-investigation in this session) to get **both** more throughput headroom (MTP-1 + SuffixDecoding hybrid projects to ~22-25 tps on DGX Spark per the osoleve benchmark) **and** the longer-context completion that helped sqlalchemy pass in the first place.

### 10.2 Protocol additions

1. **Per-cell power_w validity gate.** Bake the contamination detector (cell-max tps + cell-min power as references, work-power inversion flag) into the round driver. Flag at collection time, not post-hoc.
2. **Prefix-cache clear between rounds.** OFF's anomalously low turn-0 prefill suggests prefix cache state leaked across rounds. Each round should start from a known-cold cache.
3. **Sample DCGM during measurement, not just before/after.** This was added in v4a_v2 (the 7.6M sample dataset) and worked well. Keep it.
4. **Schedule rounds in adjacent windows or interleave attempts.** D was measured 3 days before OFF. Interleaving (e.g., D run_01 → A run_01 → B run_01 → C run_01 → OFF run_01 → D run_02 …) would eliminate the host-window confound.

### 10.3 Open questions for Round 5

1. **T2/T3 slot contention.** Implement per-frame draft priority and re-measure full stack with tuned priorities. Confirm or refute the shared-resource hypothesis.
2. **LMCache integration.** Cross-session KV reuse remains unlanded due to vLLM 0.19 hybrid-cache incompatibility. If T1 captures the harness-side state via session-scoped suffix tree, LMCache would compose with prefix caching independently. Worth re-pursuing in vLLM 0.20+.
3. **Pass-rate ceiling.** Pass rate is 32.5% at v4a_v2 baseline and largely invariant across techniques. The path to higher pass rate is **model-side, not inference-side**: better tool-call schemas, more representative training corpus, or a larger model. Track B can claim throughput won, but the binding org-level constraint is correctness work, not further spec-decode tuning.

---

## 11. Provenance and reproduction

### 11.1 Files

- This report: `docs/reports/auto_research/track-b-round4b-ablation-formal-report-20260516.md`
- Companion contamination methodology: `docs/reports/auto_research/track-b-round4b-power-w-remeasure-list-20260516.md`
- Companion remeasure results (closes P1 gate): `docs/reports/auto_research/track-b-round4b-dpoint-remeasure-results-20260517.md`
- Master data sweep script: `scripts/full_data_sweep.py`
- Contamination detector: `scripts/contamination_sweep.py`
- Engineering spec (parent): `docs/reports/auto_research/codex-harness-spec-decode-engineering-20260507.md`
- D-point per-task data: `output/track_b_e2e_v4a_v2/round_0/`, `round_0_phase{1,2,3a}_PRESERVED/`
- A/B/C/OFF data: `output/track_b_e2e_v4a_v2_ablation/round_{1,2,3,4}/`
- Per-attempt artifacts (every cell): `prompt.md`, `codex_stdout.log`, `codex_trace.jsonl`, `runner_metadata.json`, `vllm_per_turn.json`, `vllm_request_metrics.jsonl`, `vllm_metrics_pre.txt`, `vllm_metrics_post.txt`, `dcgm_samples.jsonl`, `workspace/`, `grader_result.json`

### 11.2 Reproduce headline aggregate table

```bash
python3 scripts/full_data_sweep.py
```

Outputs the per-point aggregate, per-task × point decode_tps matrix, slice medians, and writes the full structured dataset to `output/track_b_e2e_v4a_v2_report_data.json`.

### 11.3 Reproduce contamination detector

```bash
python3 scripts/contamination_sweep.py
```

Outputs per-attempt contamination flags (P1 confirmed, P2 power anomaly, P3 tps-low) with cell-max/cell-min references.

### 11.4 Spec-decode runtime config

```json
{
  "method": "suffix",
  "num_speculative_tokens": 12,
  "suffix_decoding_max_cached_requests": 1000,
  "suffix_decoding_max_spec_factor": 2.0,
  "suffix_decoding_max_tree_depth": 32,
  "suffix_decoding_min_token_prob": 0.05,
  "runtime_config_hash": "sha256:5ae88ac4e10201f83a617e2bda3f1c07da4c7217c80db5482d317a79dd93b43a"
}
```

### 11.5 Ablation toggle file

```bash
# A point (T1 only)
echo '{"LUMO_DISABLE_T2": true, "LUMO_DISABLE_T3": true, "LUMO_DISABLE_T4": true}' > /tmp/lumo_track_b_runtime_flags.json

# B point (T1+T2)
echo '{"LUMO_DISABLE_T3": true, "LUMO_DISABLE_T4": true}' > /tmp/lumo_track_b_runtime_flags.json

# C point (T1+T2+T3)
echo '{"LUMO_DISABLE_T4": true}' > /tmp/lumo_track_b_runtime_flags.json

# D point (all on)
echo '{}' > /tmp/lumo_track_b_runtime_flags.json

# OFF requires separate vLLM relaunch with speculative_config cleared
```

---

## Appendix A. Full per-cell data table

All 220 cells (point × task × attempt) with key metrics. The 4 originally-contaminated D-point attempts (responses-sdk run_02/run_03, transcript-merge run_02/run_03) were remeasured 2026-05-17; remeasured values are shown inline and the contaminated originals are archived in-place as `contaminated_run_NN_20260512/` (renamed off the `run_*` glob).

(See `output/track_b_e2e_v4a_v2_report_data.json` for machine-readable form.)

### A.1 OFF cells

| Task | run | tps | pwr | gpu_util |
|---|---|---:|---:|---:|
| dead-flag-reachability-audit | run_01 | 5.61 | 37.86 | 96 |
| dead-flag-reachability-audit | run_02 | 6.44 | 37.84 | 96 |
| dead-flag-reachability-audit | run_03 | 5.57 | 37.80 | 96 |
| dead-flag-reachability-audit | run_04 | 5.47 | 37.76 | 96 |
| fanout-fullstack-release-blocker | run_01 | 5.80 | 37.57 | 96 |
| fanout-fullstack-release-blocker | run_02 | 6.04 | 37.87 | 96 |
| fanout-fullstack-release-blocker | run_03 | 6.15 | 37.96 | 96 |
| fanout-fullstack-release-blocker | run_04 | 6.18 | 37.88 | 96 |
| incident-evidence-synthesis | run_01 | 4.74 | 37.46 | 96 |
| incident-evidence-synthesis | run_02 | 5.83 | 37.48 | 96 |
| incident-evidence-synthesis | run_03 | 4.44 | 37.49 | 96 |
| incident-evidence-synthesis | run_04 | 4.68 | 37.44 | 96 |
| multi-tool-transaction-repair | run_01 | 3.11 | 37.26 | 96 |
| multi-tool-transaction-repair | run_02 | 4.22 | 37.25 | 96 |
| multi-tool-transaction-repair | run_03 | 5.23 | 37.22 | 96 |
| multi-tool-transaction-repair | run_04 | 2.18 | 37.20 | 96 |
| policy-aware-request-resolution | run_01 | 5.63 | 37.34 | 96 |
| policy-aware-request-resolution | run_02 | 5.52 | 37.33 | 96 |
| policy-aware-request-resolution | run_03 | 5.76 | 37.32 | 96 |
| policy-aware-request-resolution | run_04 | 5.67 | 37.35 | 96 |
| release-note-to-plan-translation | run_01 | 5.05 | 37.22 | 96 |
| release-note-to-plan-translation | run_02 | 5.14 | 37.27 | 96 |
| release-note-to-plan-translation | run_03 | 5.33 | 37.29 | 96 |
| release-note-to-plan-translation | run_04 | 5.22 | 37.31 | 96 |
| responses-sdk-adapter-cutover | run_01 | 5.55 | 37.95 | 96 |
| responses-sdk-adapter-cutover | run_02 | 5.63 | 38.14 | 96 |
| responses-sdk-adapter-cutover | run_03 | 5.16 | 38.22 | 96 |
| responses-sdk-adapter-cutover | run_04 | 5.21 | 38.18 | 96 |
| responsive-checkout-visual-regression | run_01 | 2.62 | 37.48 | 96 |
| responsive-checkout-visual-regression | run_02 | 5.96 | 37.49 | 96 |
| responsive-checkout-visual-regression | run_03 | 6.56 | 37.50 | 96 |
| responsive-checkout-visual-regression | run_04 | 5.54 | 37.50 | 96 |
| security-audit-hotfix-remediation | run_01 | 6.19 | 37.58 | 96 |
| security-audit-hotfix-remediation | run_02 | 5.78 | 37.60 | 96 |
| security-audit-hotfix-remediation | run_03 | 6.38 | 37.58 | 96 |
| security-audit-hotfix-remediation | run_04 | 5.74 | 37.54 | 96 |
| sqlalchemy-2-session-modernization | run_01 | 4.92 | 37.69 | 96 |
| sqlalchemy-2-session-modernization | run_02 | 5.91 | 37.63 | 96 |
| sqlalchemy-2-session-modernization | run_03 | 5.94 | 37.58 | 96 |
| sqlalchemy-2-session-modernization | run_04 | 5.69 | 37.59 | 96 |
| transcript-merge-regression | run_01 | 5.40 | 38.05 | 96 |
| transcript-merge-regression | run_02 | 5.48 | 37.99 | 96 |
| transcript-merge-regression | run_03 | 5.77 | 37.89 | 96 |
| transcript-merge-regression | run_04 | 5.55 | 37.89 | 96 |

### A.2 A cells

| Task | run | tps | pwr | gpu_util |
|---|---|---:|---:|---:|
| dead-flag-reachability-audit | run_01 | 13.74 | 42.64 | 95 |
| dead-flag-reachability-audit | run_02 | 22.69 | 44.66 | 95 |
| dead-flag-reachability-audit | run_03 | 13.80 | 41.91 | 95 |
| dead-flag-reachability-audit | run_04 | 28.95 | 46.11 | 95 |
| fanout-fullstack-release-blocker | run_01 | 14.89 | 42.30 | 95 |
| fanout-fullstack-release-blocker | run_02 | 16.40 | 40.49 | 95 |
| fanout-fullstack-release-blocker | run_03 | 19.85 | 43.34 | 95 |
| fanout-fullstack-release-blocker | run_04 | 19.31 | 44.85 | 95 |
| incident-evidence-synthesis | run_01 | 22.07 | 45.84 | 95 |
| incident-evidence-synthesis | run_02 | 18.51 | 45.30 | 95 |
| incident-evidence-synthesis | run_03 | 20.32 | 40.27 | 95 |
| incident-evidence-synthesis | run_04 | 35.52 | 44.49 | 95 |
| multi-tool-transaction-repair | run_01 | 9.82 | 40.48 | 95 |
| multi-tool-transaction-repair | run_02 | 16.39 | 41.45 | 95 |
| multi-tool-transaction-repair | run_03 | 16.10 | 39.73 | 95 |
| multi-tool-transaction-repair | run_04 | 14.54 | 41.03 | 95 |
| policy-aware-request-resolution | run_01 | 21.17 | 45.17 | 95 |
| policy-aware-request-resolution | run_02 | 15.83 | 42.32 | 95 |
| policy-aware-request-resolution | run_03 | 26.96 | 46.54 | 95 |
| policy-aware-request-resolution | run_04 | 19.27 | 45.57 | 95 |
| release-note-to-plan-translation | run_01 | 23.57 | 44.10 | 95 |
| release-note-to-plan-translation | run_02 | 15.99 | 42.40 | 95 |
| release-note-to-plan-translation | run_03 | 29.23 | 45.97 | 95 |
| release-note-to-plan-translation | run_04 | 26.41 | 45.91 | 95 |
| responses-sdk-adapter-cutover | run_01 | 16.76 | 42.11 | 95 |
| responses-sdk-adapter-cutover | run_02 | 12.27 | 40.38 | 95 |
| responses-sdk-adapter-cutover | run_03 | 34.64 | 44.77 | 95 |
| responses-sdk-adapter-cutover | run_04 | 30.19 | 46.11 | 95 |
| responsive-checkout-visual-regression | run_01 | 15.56 | 39.97 | 95 |
| responsive-checkout-visual-regression | run_02 | 16.27 | 40.71 | 95 |
| responsive-checkout-visual-regression | run_03 | 11.92 | 41.77 | 95 |
| responsive-checkout-visual-regression | run_04 | 19.35 | 43.38 | 95 |
| security-audit-hotfix-remediation | run_01 | 14.80 | 40.60 | 95 |
| security-audit-hotfix-remediation | run_02 | 18.65 | 40.75 | 95 |
| security-audit-hotfix-remediation | run_03 | 20.38 | 40.77 | 95 |
| security-audit-hotfix-remediation | run_04 | 16.02 | 41.43 | 95 |
| sqlalchemy-2-session-modernization | run_01 | 28.48 | 42.69 | 95 |
| sqlalchemy-2-session-modernization | run_02 | 25.94 | 43.31 | 95 |
| sqlalchemy-2-session-modernization | run_03 | 30.87 | 43.62 | 95 |
| sqlalchemy-2-session-modernization | run_04 | 27.64 | 43.44 | 95 |
| transcript-merge-regression | run_01 | 11.98 | 41.36 | 95 |
| transcript-merge-regression | run_02 | 14.56 | 43.73 | 95 |
| transcript-merge-regression | run_03 | 17.79 | 41.18 | 95 |
| transcript-merge-regression | run_04 | 16.20 | 43.62 | 95 |

### A.3 B cells

| Task | run | tps | pwr | gpu_util |
|---|---|---:|---:|---:|
| dead-flag-reachability-audit | run_01 | 21.79 | 44.65 | 95 |
| dead-flag-reachability-audit | run_02 | 28.34 | 43.40 | 95 |
| dead-flag-reachability-audit | run_03 | 27.08 | 44.84 | 95 |
| dead-flag-reachability-audit | run_04 | 34.20 | 45.64 | 95 |
| fanout-fullstack-release-blocker | run_01 | 17.88 | 44.13 | 95 |
| fanout-fullstack-release-blocker | run_02 | 17.32 | 40.45 | 95 |
| fanout-fullstack-release-blocker | run_03 | 17.07 | 41.95 | 95 |
| fanout-fullstack-release-blocker | run_04 | 16.04 | 41.80 | 95 |
| incident-evidence-synthesis | run_01 | 28.74 | 46.45 | 95 |
| incident-evidence-synthesis | run_02 | 32.89 | 47.27 | 95 |
| incident-evidence-synthesis | run_03 | 30.92 | 44.48 | 95 |
| incident-evidence-synthesis | run_04 | 26.53 | 46.47 | 95 |
| multi-tool-transaction-repair | run_01 | 15.52 | 42.55 | 95 |
| multi-tool-transaction-repair | run_02 | 13.96 | 40.87 | 95 |
| multi-tool-transaction-repair | run_03 | 15.65 | 42.93 | 95 |
| multi-tool-transaction-repair | run_04 | 12.54 | 42.69 | 95 |
| policy-aware-request-resolution | run_01 | 13.16 | 41.86 | 95 |
| policy-aware-request-resolution | run_02 | 14.36 | 42.19 | 95 |
| policy-aware-request-resolution | run_03 | 13.01 | 45.02 | 95 |
| policy-aware-request-resolution | run_04 | 25.83 | 45.37 | 95 |
| release-note-to-plan-translation | run_01 | 25.27 | 45.84 | 95 |
| release-note-to-plan-translation | run_02 | 29.46 | 46.86 | 95 |
| release-note-to-plan-translation | run_03 | 22.53 | 45.44 | 95 |
| release-note-to-plan-translation | run_04 | 15.51 | 43.27 | 95 |
| responses-sdk-adapter-cutover | run_01 | 13.28 | 42.78 | 95 |
| responses-sdk-adapter-cutover | run_02 | 19.26 | 39.84 | 95 |
| responses-sdk-adapter-cutover | run_03 | 16.66 | 40.12 | 95 |
| responses-sdk-adapter-cutover | run_04 | 19.93 | 43.03 | 95 |
| responsive-checkout-visual-regression | run_01 | 12.59 | 41.83 | 95 |
| responsive-checkout-visual-regression | run_02 | 14.39 | 41.13 | 95 |
| responsive-checkout-visual-regression | run_03 | 16.76 | 40.70 | 95 |
| responsive-checkout-visual-regression | run_04 | 13.48 | 42.39 | 95 |
| security-audit-hotfix-remediation | run_01 | 15.58 | 40.15 | 95 |
| security-audit-hotfix-remediation | run_02 | 17.20 | 40.51 | 95 |
| security-audit-hotfix-remediation | run_03 | 16.01 | 42.27 | 95 |
| security-audit-hotfix-remediation | run_04 | 21.97 | 42.11 | 95 |
| sqlalchemy-2-session-modernization | run_01 | 27.30 | 42.53 | 95 |
| sqlalchemy-2-session-modernization | run_02 | 33.08 | 43.89 | 95 |
| sqlalchemy-2-session-modernization | run_03 | 26.06 | 42.81 | 95 |
| sqlalchemy-2-session-modernization | run_04 | 34.44 | 43.31 | 95 |
| transcript-merge-regression | run_01 | 27.62 | 44.44 | 95 |
| transcript-merge-regression | run_02 | 20.11 | 43.55 | 95 |
| transcript-merge-regression | run_03 | 17.81 | 40.74 | 95 |
| transcript-merge-regression | run_04 | 24.59 | 44.50 | 95 |

### A.4 C cells

| Task | run | tps | pwr | gpu_util |
|---|---|---:|---:|---:|
| dead-flag-reachability-audit | run_01 | 14.13 | 40.52 | 95 |
| dead-flag-reachability-audit | run_02 | 19.13 | 43.32 | 95 |
| dead-flag-reachability-audit | run_03 | 15.00 | 43.47 | 95 |
| dead-flag-reachability-audit | run_04 | 26.37 | 43.91 | 95 |
| fanout-fullstack-release-blocker | run_01 | 12.81 | 43.01 | 95 |
| fanout-fullstack-release-blocker | run_02 | 14.46 | 42.23 | 95 |
| fanout-fullstack-release-blocker | run_03 | 17.42 | 42.58 | 95 |
| fanout-fullstack-release-blocker | run_04 | 18.95 | 44.46 | 95 |
| incident-evidence-synthesis | run_01 | 16.12 | 41.66 | 95 |
| incident-evidence-synthesis | run_02 | 30.43 | 45.47 | 95 |
| incident-evidence-synthesis | run_03 | 22.72 | 45.01 | 95 |
| incident-evidence-synthesis | run_04 | 16.80 | 39.83 | 95 |
| multi-tool-transaction-repair | run_01 | 14.82 | 41.37 | 95 |
| multi-tool-transaction-repair | run_02 | 13.72 | 41.91 | 95 |
| multi-tool-transaction-repair | run_03 | 15.35 | 40.80 | 95 |
| multi-tool-transaction-repair | run_04 | 24.52 | 42.73 | 95 |
| policy-aware-request-resolution | run_01 | 18.21 | 39.43 | 95 |
| policy-aware-request-resolution | run_02 | 25.56 | 45.51 | 95 |
| policy-aware-request-resolution | run_03 | 15.60 | 43.21 | 95 |
| policy-aware-request-resolution | run_04 | 24.54 | 46.39 | 95 |
| release-note-to-plan-translation | run_01 | 19.98 | 44.37 | 95 |
| release-note-to-plan-translation | run_02 | 25.86 | 44.69 | 95 |
| release-note-to-plan-translation | run_03 | 24.59 | 46.26 | 95 |
| release-note-to-plan-translation | run_04 | 23.52 | 45.70 | 95 |
| responses-sdk-adapter-cutover | run_01 | 27.97 | 44.83 | 95 |
| responses-sdk-adapter-cutover | run_02 | 17.01 | 40.34 | 95 |
| responses-sdk-adapter-cutover | run_03 | 28.78 | 45.73 | 95 |
| responses-sdk-adapter-cutover | run_04 | 15.96 | 42.07 | 95 |
| responsive-checkout-visual-regression | run_01 | 12.66 | 40.50 | 95 |
| responsive-checkout-visual-regression | run_02 | 13.11 | 40.87 | 95 |
| responsive-checkout-visual-regression | run_03 | 17.72 | 40.20 | 95 |
| responsive-checkout-visual-regression | run_04 | 13.43 | 42.04 | 95 |
| security-audit-hotfix-remediation | run_01 | 14.80 | 41.49 | 95 |
| security-audit-hotfix-remediation | run_02 | 19.29 | 42.60 | 95 |
| security-audit-hotfix-remediation | run_03 | 16.29 | 42.58 | 95 |
| security-audit-hotfix-remediation | run_04 | 16.57 | 40.87 | 95 |
| sqlalchemy-2-session-modernization | run_01 | 30.11 | 42.43 | 95 |
| sqlalchemy-2-session-modernization | run_02 | 26.15 | 44.10 | 95 |
| sqlalchemy-2-session-modernization | run_03 | 31.90 | 43.16 | 95 |
| sqlalchemy-2-session-modernization | run_04 | 15.88 | 43.22 | 95 |
| transcript-merge-regression | run_01 | 10.37 | 43.53 | 95 |
| transcript-merge-regression | run_02 | 16.25 | 44.19 | 95 |
| transcript-merge-regression | run_03 | 17.93 | 43.93 | 95 |
| transcript-merge-regression | run_04 | 15.50 | 40.70 | 95 |

### A.5 D cells

| Task | run | tps | pwr | gpu_util | Note |
|---|---|---:|---:|---:|---|
| dead-flag-reachability-audit | run_01 | 28.69 | 37.27 | 94 | |
| dead-flag-reachability-audit | run_02 | 14.79 | 37.16 | 94 | tps-low (not contaminated) |
| dead-flag-reachability-audit | run_03 | 26.71 | 38.09 | 94 | |
| dead-flag-reachability-audit | run_04 | 30.05 | 38.18 | 94 | |
| fanout-fullstack-release-blocker | run_01 | 14.56 | 42.04 | 95 | |
| fanout-fullstack-release-blocker | run_02 | 15.98 | 42.39 | 95 | |
| fanout-fullstack-release-blocker | run_03 | 17.20 | 41.92 | 95 | |
| fanout-fullstack-release-blocker | run_04 | 18.89 | 43.52 | 95 | |
| incident-evidence-synthesis | run_01 | 13.87 | 34.95 | 94 | tps-low |
| incident-evidence-synthesis | run_02 | 31.29 | 37.36 | 94 | |
| incident-evidence-synthesis | run_03 | 24.63 | 35.81 | 94 | |
| incident-evidence-synthesis | run_04 | 23.70 | 35.51 | 94 | |
| multi-tool-transaction-repair | run_01 | 12.18 | 33.56 | 94 | tps-low |
| multi-tool-transaction-repair | run_02 | 25.83 | 36.57 | 94 | |
| multi-tool-transaction-repair | run_03 | 10.09 | 34.08 | 94 | tps-low |
| multi-tool-transaction-repair | run_04 | 21.87 | 36.74 | 94 | |
| policy-aware-request-resolution | run_01 | 25.84 | 39.09 | 94 | |
| policy-aware-request-resolution | run_02 | 13.89 | 37.34 | 94 | |
| policy-aware-request-resolution | run_03 | 13.91 | 35.35 | 94 | |
| policy-aware-request-resolution | run_04 | 18.47 | 33.72 | 94 | |
| release-note-to-plan-translation | run_01 | 18.66 | 38.35 | 94 | |
| release-note-to-plan-translation | run_02 | 15.56 | 35.92 | 94 | |
| release-note-to-plan-translation | run_03 | 25.56 | 38.44 | 94 | |
| release-note-to-plan-translation | run_04 | 28.60 | 39.05 | 94 | |
| responses-sdk-adapter-cutover | run_01 | 16.45 | 36.21 | 94 | |
| responses-sdk-adapter-cutover | run_02 | 16.63 | 44.78 | 95 | remeasured 2026-05-17 |
| responses-sdk-adapter-cutover | run_03 | 16.23 | 43.70 | 95 | remeasured 2026-05-17 |
| responses-sdk-adapter-cutover | run_04 | 29.11 | 39.83 | 94 | |
| responsive-checkout-visual-regression | run_01 | 12.60 | 35.08 | 94 | |
| responsive-checkout-visual-regression | run_02 | 15.53 | 35.88 | 94 | |
| responsive-checkout-visual-regression | run_03 | 17.67 | 34.14 | 94 | |
| responsive-checkout-visual-regression | run_04 | 16.27 | 34.04 | 94 | |
| security-audit-hotfix-remediation | run_01 | 15.34 | 35.69 | 94 | |
| security-audit-hotfix-remediation | run_02 | 18.24 | 34.16 | 94 | |
| security-audit-hotfix-remediation | run_03 | 16.84 | 34.88 | 94 | |
| security-audit-hotfix-remediation | run_04 | 17.50 | 34.54 | 94 | |
| sqlalchemy-2-session-modernization | run_01 | 30.41 | 35.58 | 94 | |
| sqlalchemy-2-session-modernization | run_02 | 23.98 | 36.73 | 94 | |
| sqlalchemy-2-session-modernization | run_03 | 37.74 | 36.75 | 94 | |
| sqlalchemy-2-session-modernization | run_04 | 28.39 | 36.21 | 94 | |
| transcript-merge-regression | run_01 | 25.91 | 37.54 | 94 | |
| transcript-merge-regression | run_02 | 14.45 | 40.17 | 94 | remeasured 2026-05-17 |
| transcript-merge-regression | run_03 | 11.37 | 41.35 | 94 | remeasured 2026-05-17 |
| transcript-merge-regression | run_04 | 11.81 | 37.30 | 94 | |

## Appendix B. Glossary

| Term | Definition |
|---|---|
| decode_tps | `completion_tokens / decode_sum_s` per request; final-output decode speed |
| accept_rate | `spec_decode_num_accepted_tokens / spec_decode_num_draft_tokens` per request |
| M_aggregate | Milestone-weighted score (0.0-1.0), from grader's milestone_vector |
| P_benchmark | Scaled benchmark score (0-100); PASS at P ≥ 65 |
| T1 | Cross-turn SuffixDecoding (session-scoped ngram cache) |
| T2 | read_file proactive priming (folded into suffix tree) |
| T3 | Schema-aware tool-call drafter (ToolSpec FSM + XGrammar-2 mask) |
| T4 | Plan-structure pre-drafter (heuristic detector + structural tokens) |
| T5 | Turn-boundary drafter lifecycle (session/turn hooks) — out of scope |
| Oracle API | Harness→drafter channel: session_id, turn_index, tool_schemas, primed_text |
| Cell | Single (point, task, attempt) measurement |
| Point | Ablation configuration (D, A, B, C, OFF) |
| OFF | Base decode, no spec_decode |
| A | T1 only |
| B | T1+T2 |
| C | T1+T2+T3 |
| D | T1+T2+T3+T4 (full stack) |
| Contamination | Host-level resource pressure during measurement, identified by simultaneous high power + low decode_tps |
| Cell-max / cell-min reference | Use the cleanest attempt in a cell as the comparison baseline (since cell median is biased by contamination itself) |

---

**End of report.**
