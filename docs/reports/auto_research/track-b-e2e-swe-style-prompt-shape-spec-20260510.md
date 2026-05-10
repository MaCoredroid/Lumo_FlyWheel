# Track B E2E — SWE-Style Prompt Shape Fix

Generated: 2026-05-10
Status: **RETRACTED 2026-05-10** — see §11. Original spec preserved §§1–10 for audit.

Companion to:
- `track-b-e2e-agentic-saturation-plan-20260508-v2.md` (parent saturation plan; this changes the §3 sample shape)
- `track-b-round3-e2e-v3-closeout-20260510.md` (Round 3 −12.5% baseline — preserved as the "v3 / bundle-in-prompt" reference)
- `codex-harness-spec-decode-engineering-20260507.md` (engineering spec; Round 4 should run against the new shape)

## 1. Hypothesis

The Round 3 v3 measurement found decode is ~10% of wallclock on the 13-task CNB-55 sample. Cross-checked against the SuffixDecoding paper's published SWE-Bench Verified results, decode there is "the majority of the time across all SWE-Bench tasks, dominating both prefilling and agentic actions." This is a workload-shape divergence, not a model or hardware artifact:

- **CNB-55 v3 (current):** workspace bundle (`AGENTS.md`, `release_notes/`, `repo_inventory/`, scenario_variant, optional release_context / incident_context) is **materialized into the request prompt** by `scripts/measure_track_b_real_content_task.py`'s `_build_prompt`. Initial prompt is ~5000+ tokens; every turn re-prefills the bundle plus growing transcript. **Prefill dominates wallclock.**
- **SWE-Bench-style:** small initial prompt (issue text, ~500-1500 tokens). Agent discovers files on-demand via its own `read_file` / `bash` / `grep` tool calls. **Decode dominates wallclock.**

The bundle-in-prompt construction was an artifact of CNB-55's design choice to materialize all context inline for reproducibility — the workspace files are pinned via `manifest_locked` per `family.yaml`. That gives deterministic test inputs, but it also caps decode share at ~10% in a way that doesn't reflect how real Codex SWE workflows actually run.

This spec converts the per-task prompt construction from bundle-inline to bundle-on-disk while preserving every other test content guarantee (task instruction, workspace contents byte-equality, grader, milestone weights, B-1/B-2/B-3 gates, integrity rules).

## 2. What changes

**One module:** the prompt builder used by the e2e runner. Specifically:
- `scripts/measure_track_b_real_content_task.py::_build_prompt` (the legacy CNB-55 workload builder) and
- the analogous prompt construction inside `scripts/run_track_b_e2e_task.py` if it has its own copy.

**One CLI invocation surface:** Codex CLI is launched with the materialized workspace as its working directory. The path hint and the canonical task instruction are the only inline content.

**Nothing else changes:**
- Workspace bundle file contents — same bytes, just relocated from prompt to disk.
- Canonical task instruction text — same wording from `benchmark_blueprints/families/<family>/task_spec.md`.
- Family graders, milestone weights, integrity rules — untouched.
- B-1/B-2/B-3 schema-strict equivalence gates — untouched.
- Codex CLI tool inventory (`read_file`, `apply_patch`, `bash`, etc.) — untouched.
- Sample membership (the 13 families × `v1-clean-baseline`) — unchanged.

### 2.1 New prompt shape

```
You are working in a sandboxed workspace at /tmp/track_b_e2e/<run_id>/<family>/<variant>/workspace/.

Top-level directories present in the workspace:
- AGENTS.md  (one-line: "agent guidance lives here")
- release_notes/        (directory)
- repo_inventory/       (directory)
- .scenario_variant     (file)
- bin/                  (directory; family CLI is here)
- tests/                (directory)
[... other top-level entries from `ls workspace_bundle/<variant>/` ...]

<canonical task instruction from task_spec.md>

Use your read_file / bash / apply_patch tools to read whatever workspace
context you need before producing your structured output.
```

The "top-level directories" listing is generated automatically by enumerating `workspace_bundle/<variant>/` at runtime — one line per top-level entry, **directories listed without contents**, files only listed by name and 1-line type hint (no body). This is roughly equivalent to what the agent would see if it ran `ls -la` itself, hoisted into the prompt to bound the discovery loop. It mirrors how SWE-Bench prompts hint at the existence of files without dumping their contents.

Initial prompt size target: **≤ 1500 tokens** (vs current ≥ 5000). For families with very long task instructions, the canonical text is preserved verbatim — the savings come from removing pre-loaded file bodies, not from compressing the instruction.

### 2.2 Workspace materialization

Per task run, the runner creates a writeable copy of `benchmark_blueprints/families/<family>/workspace_bundle/<variant>/` at `/tmp/track_b_e2e/<run_id>/<family>/<variant>/workspace/`. The copy preserves file bytes (verified against `manifest_lock.json`'s content_hash). Codex CLI is launched with `--cwd` pointing at the copy. Agent edits operate on the copy. The grader runs against the copy after task completion.

### 2.3 Truthful-measurement contract addition

Add **rule 16** to the §8 attestation in the saturation plan:

> **rule_16_workspace_bundle_hash_match** — the materialized workspace's recursive content hash must equal the family's `manifest_lock.json` content_hash. Hard fail if mismatch.

This guarantees that "the agent saw the same workspace files we put in the prompt before" — content equivalence is preserved across the prompt-shape change.

## 3. Target estimate (after change, on the same 13-task sample)

Per-task wallclock reconstruction projected from the v3 baseline plus published SWE-Bench shape:

| Component | v3 / bundle-in-prompt (measured) | v4 / SWE-style projected | Direction |
|---|---:|---:|---|
| Initial prefill (turn 0) | ~15-20 s (5000+ tok) | **~3-5 s** (≤1500 tok) | ↓ (small initial prompt) |
| Per-turn prefill across loop | ~20-35 s | **~7-15 s** (smaller bundle weight on each re-prefill; partially offset by more turns) | ↓ |
| Tool execution wait | ~10-30 s (1-3 tool calls) | **~30-60 s** (5-10 tool calls — agent actively reads files) | ↑ |
| Decode | ~5-10 s (~170 tokens out, decode_tps ~33) | **~25-50 s** (~700-1500 tokens out — agent emits more reasoning between tool calls) | ↑ |
| Codex CLI / proxy / retry overhead | ~10-20 s | ~10-20 s | flat |
| **Total wallclock** | ~95-110 s | **~80-150 s** | similar magnitude |
| **Decode share** | **~10%** | **~30-45%** | **major shift** |

**Target: after change, decode share ≥ 25% as a lower bound, 30-45% as the realistic range, on the same 13-task sample.** A measurement that comes in below 25% means the change didn't take or some families are still bundle-dominated and need a per-family fix.

**Reasoning behind the numbers:**

- **Decode time scales with decode tokens output.** Decode tokens grow because the agent emits more reasoning text between tool calls — "I should read AGENTS.md first to understand the task" → tool_call → result → "Now I need to look at release_notes/v1.md" → tool_call → result → ... With ~5-10 tool calls per task (vs 1-3 currently) and ~50-150 reasoning tokens between each, total decode tokens roughly 4-8× the current 170.
- **Prefill time drops sharply on turn 0** (where bundle pre-loading currently inserts ~5000 tokens) and grows more linearly across the agent loop. With prefix caching enabled, the dominant cost is the unique-per-turn delta, not the cumulative prompt.
- **Tool execution wait grows** because read_file and bash calls are real subprocess work. Each `read_file` on a small workspace file is ~50-200 ms; on a larger file ~500-1500 ms. 5-10 tool calls × ~3-6 s averaged = 15-60 s. The upper end is tool-exec-bound territory.
- **Total wallclock should land in similar magnitude** to current — same task, same model, same answer expected. The ratio shifts internally.

Compared to the SuffixDecoding paper's "decode dominates the majority of SWE-Bench wallclock" claim, our 30-45% projection is conservative because:
- CNB-55 task instructions are still longer than typical SWE-Bench issue text.
- Some families (e.g., `release-note-to-plan-translation`) have inherently long structured outputs that emit a lot of decode regardless of context-discovery shape.
- We're not removing the workspace listing entirely — that 200-500 token dirlist is a small inline reduction vs full bundle.

If we wanted to push further toward 50%+ decode share (closer to SWE-Bench's reported shape), the lever would be **stripping the dirlist hint and letting the agent run `ls` itself** — but that adds a turn per task and the realism gain is small. Recommend keeping the dirlist hint as a reasonable middle ground.

## 4. Acceptance criteria

A single round (no full Round 4 sweep) validates the change. The acceptance criteria, in order:

1. **Workspace correctness** — for each task, the materialized workspace's recursive content hash equals the family's `manifest_lock.json` content_hash. (Truthful-measurement rule 16.)
2. **Initial prompt size** — recorded `prompt_tokens` for turn 0 across the 13 tasks: median ≤ 1500, p95 ≤ 2500. (Bundle removal is real.)
3. **Total wallclock magnitude** — median wallclock within ±25% of the v3 baseline (95.44 s). I.e., 71-119 s. Tasks landing outside this band are flagged for per-family review.
4. **Decode share** — aggregate decode share across the 13 tasks ≥ 25%. (Headline target; 30-45% expected.)
5. **Task completion preserved** — same 12-13 of 13 tasks reach `task_end` with `exit_code == 0` as v3. Round-level rule: ≥ 12 of 13 must complete. If a family that completed in v3 fails in v4, that family is flagged for per-family fallback.
6. **Milestone scoring drift bound** — aggregate milestone score across the 13 tasks within ±10% of v3's milestone aggregate. (Same task, same answer expected; meaningful drift indicates the prompt change broke task semantics, not workload shape.)
7. **B-1 / B-2 / B-3 quality gates** — same status as v3 (all pass at exit-code contract; B-1/B-2/B-3 strict-evaluator pending separately, not gated by this change).

If criteria 4 fails (decode share < 25%) **but** criteria 1-3, 5-7 all pass, that's a workload-shape investigation finding — the bundle-in-prompt was not actually the dominant prefill driver, and the next investigation should look at per-turn transcript growth instead of initial prompt.

If criteria 5 or 6 fails on more than 1-2 families, that's a real test-content regression and per-family fallback is required for the affected families.

## 5. Per-family fallback policy

Some families may rely on bundle-in-prompt for legitimate reasons (e.g., a task that requires the agent to reason about release notes globally before deciding what to action). For these:

- Flag the family in `output/track_b_e2e_v4/round_0/family_fallbacks.json` with `bundle_inline=true` and a one-line rationale.
- Use the v3 prompt shape for that family.
- Continue using the v4 shape for the remaining families.
- Round summary aggregates note the cohort split.

This preserves test intent for families where the bundle is the test, while shifting the bulk of the sample to a more representative shape.

## 6. Cheap preflight (per the saturation plan §7.3)

Before the v4 sweep, add two preflight checks:

| Check | Command | Pass criterion |
|---|---|---|
| Workspace materialization works | `python scripts/preflight_track_b_e2e.py --check workspace_materialization` | Workspaces created for all 13 families; recursive hash equals `manifest_lock.json` content_hash |
| Initial prompt size budget | `python scripts/preflight_track_b_e2e.py --check initial_prompt_tokens_v4` | Median initial prompt across 13 families ≤ 1500 tokens; p95 ≤ 2500 |

If either fails, the v4 sweep does not run. Hard gate.

## 7. Sequencing

This change should land before Round 4 (MTP test, harness oracle reasoning-regime tuning, or any other technique evaluation):

1. **Phase 1 — implementation** (≤ 0.5 day). New `_build_prompt_v4` builder; runner accepts `--prompt-shape v3|v4` flag (default v3 for now to preserve baseline). Workspace materialization helper. Truthful-measurement rule 16.
2. **Phase 2 — single-task smoke** (≤ 30 min). Run `transcript-merge-regression/v1` (smallest task) under v4. Verify all acceptance criteria 1-7 individually for this single task.
3. **Phase 3 — 13-task sweep, single attempt each** (≤ 30 min). Verify aggregate criteria 4, 5, 6 across the sample.
4. **Phase 4 — full Round 4 baseline** (≤ 30 min). 13 tasks × 4 repeats with proxy capture, real runtime hash, GPU memory snapshots — same shape as Round 3 v3 measurement, but under v4 prompt shape. Lands at `output/track_b_e2e_v4/round_0/`.
5. **Phase 5 — promote v4 as the canonical sample for Round 4+**. v3 baseline preserved at `output/track_b_e2e_v3/`. The auto-research-loop default switches to v4. The codex-harness-spec-decode engineering spec's regime-share assumptions get re-verified against v4 numbers (the §6.5 diagnosis taxonomy thresholds may need recalibration if regime distribution shifts substantially — likely tool-call drops from 89% to 60%, reasoning grows from 11% to 30%+).

The whole change is roughly 2 hours of implementation + 1.5 hours of measurement + 1 hour of doc updates. Cheap.

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Codex CLI doesn't honor `--cwd` correctly (relative-path bugs we hit in Round 3) | Use absolute paths everywhere; reuse the `task_dir.resolve()` fix from `ae6ff3a` |
| Some tasks regress in milestone score because the agent doesn't discover the right files | Per-family fallback policy (§5); keep v3 shape for affected families |
| Total wallclock blows up (agent does too many discovery turns, hits Codex's iteration limit) | Cap added: if any task exceeds 1.5× v3 wallclock, abort that task and flag for review |
| Decode share doesn't actually move (workload-shape hypothesis was wrong) | Documented as a real research finding (§4 acceptance criteria 4 failure handling); next investigation looks at per-turn transcript growth instead |
| Sample hash changes invalidate v3-vs-v4 comparison | Expected and intentional; v4 is a sibling sample. v3 baseline preserved. Round-over-round comparisons within v4 are well-defined; cross-shape comparisons are not. |
| Workspace materialization adds slow tmpdir copy overhead | Use hardlinks where possible; materialize once per round (not per attempt) and snapshot per attempt via `cp -al` |
| Per-family discovery patterns vary so much that aggregate decode share is misleading | Report per-family decode share alongside aggregate; flag families with extreme outliers |

## 9. What this does and doesn't claim

**This claims:** a specific test-fixture choice in CNB-55 (workspace-bundle-in-prompt) inflates prefill share in a way that's not representative of real Codex SWE workflows, and a small implementation change brings the measured shape closer to published SWE-Bench profiles without changing test content.

**This does not claim:** that v4 wallclock will be smaller than v3. The agent does more work in v4 (discovery via tools instead of reading from prompt), and total wallclock could be larger, smaller, or similar depending on per-family characteristics. The headline metric shift is **decode share**, not absolute wallclock.

**This does not claim:** the existing Round 1-3 wins are wrong. Round 3's −12.5% wallclock reduction on the v3 sample is a valid measurement against a valid baseline; v4 produces a different sample with a different baseline, and the same techniques will likely show a larger relative wallclock reduction there because decode is a bigger share. We may rerun Round 1-3's ablation under v4 in a follow-up to compute the more realistic technique-attribution numbers.

## 10. Decisions

Both gating questions decided 2026-05-10 by Mark:

1. **Canonical sample switches to v4. ✅ DECIDED.** v3 baseline at `output/track_b_e2e_v3/round_3/` is preserved unchanged as the frozen reference for Round 1-3 historical comparisons. All Round 4+ measurements run against v4. The auto-research-loop default shifts to v4 after Phase 4 completes.
2. **Dirlist hint vs no-hint — DEFERRED to one-task measurement.** Phase 1 implements the **dirlist hint** variant per §2.1 (recommended default). Phase 2's single-task smoke runs both variants on `transcript-merge-regression/v1` (smallest task, cheapest to measure twice), captures decode-share + total wallclock + milestone score for each. Mark reviews the two-variant comparison before Phase 3's full 13-task sweep commits to one shape.

### 10.1 Phase 2 — two-variant smoke spec

`transcript-merge-regression/v1` runs through `run_track_b_e2e_task.py` four times under controlled conditions:

| Run | Prompt shape | Purpose |
|---|---|---|
| `smoke_v3_baseline` | v3 (current bundle-inline) | Baseline reference; matches existing v3 runs for sanity |
| `smoke_v4_dirlist` | v4 with §2.1 dirlist hint | Recommended default |
| `smoke_v4_nohint` | v4 with no dirlist hint, only canonical task instruction + cwd path | SWE-Bench-pure variant |
| `smoke_v4_dirlist_repeat` | v4 with §2.1 dirlist hint (second attempt) | Variance check on the recommended default |

Each run captures: `prompt_tokens` (turn 0), total wallclock, decode_sum_s, decode share, generation_tokens, accepted_per_draft_token (per regime), milestone score, exit code, turn count.

**Decision criteria the variant comparison feeds:**

- If `smoke_v4_dirlist` and `smoke_v4_nohint` produce within ±5% wallclock and within ±5 percentage-point decode share of each other, dirlist hint wins (it preserves more test intent at no realism cost).
- If `smoke_v4_nohint` shows materially higher decode share (e.g., +10 pp) at acceptable wallclock and acceptable milestone score, no-hint wins (pure SWE-Bench shape, worth the test-intent compromise).
- If `smoke_v4_nohint` regresses milestone score on this single task (the agent can't find the right files), dirlist hint wins by safety.
- If both v4 variants regress milestone score vs `smoke_v3_baseline`, the spec needs revision before any 13-task sweep — the workspace materialization or path hint is broken.

### 10.2 Sequencing after Phase 2 decision

Phase 3 (single-attempt 13-task sweep) and Phase 4 (full 13×4 baseline) only commit to one prompt shape after Phase 2's two-variant smoke is reviewed. The v4 builder accepts `--prompt-shape v4-dirlist | v4-nohint` so the chosen variant is a runtime flag, not an irreversible code path. v3 builder remains available via `--prompt-shape v3` indefinitely.

Phase 1 implementation work is unblocked by these decisions and can start now. Phase 2 measurements report to Mark before Phase 3 runs.

## 11. Retraction and revision (2026-05-10, post-measurement)

§§1–10 above are preserved unchanged for audit-trail integrity. This section retracts the §1 hypothesis after a direct measurement of the existing Round 3 v3 capture artifacts (no new instrumentation needed) and reframes Phase 1 around the actual prefill driver.

### 11.1 What §1 claimed (under-the-table)

§1's working model:
- e2e runner materializes the workspace bundle into the request prompt
- Initial prompt is "~5000+ tokens", every turn re-prefills that bundle plus growing transcript
- Bundle removal would drop initial prompt to ≤1500 tokens and shift decode share from ~10% to 30–45%

This conflated **two different code paths** with the same `_build_prompt` name:
- `scripts/measure_track_b_real_content_task.py::_build_prompt` (legacy CNB-55 *warm-only* probe) — does inline `release_notes/` + `repo_inventory/` files, ~5000+ tokens. **Not used by the e2e sweep.**
- `scripts/run_track_b_e2e_task.py::_write_prompt` (the actual e2e runner) — only writes `AGENTS.md` + `.scenario_variant`. The real `prompt.md` for `transcript-merge-regression/v1` in `output/track_b_e2e_v3/round_3/.../run_01/prompt.md` is **498 bytes / ~150 tokens**, not 5000.

§1's premise was wrong from the start; the e2e runner was already close to the proposed v4 shape before the spec was written.

### 11.2 What the v3 capture actually shows

Sources used (no new measurement needed): `/tmp/track_b_e2e_proxy_capture/request_metrics.jsonl` (per-turn proxy capture, 30 turns inside the v3 sweep window) and `output/track_b_e2e_v3/round_3/*/run_*/vllm_metrics_{pre,post}.txt` (per-task vLLM Prometheus diffs, all 52 runs).

| Metric | Measured | Spec assumed |
|---|---:|---:|
| Decode share (aggregate over v3 sweep window, 30 turns) | **8.1%** | ~10% (close, but for the wrong reason) |
| Median per-turn `prompt_tokens` | **69,516** | ≥5000 (from §1, actual is ~14× larger) |
| Median per-turn `completion_tokens` | **102** | not stated |
| Σ prefill_s in window | **1,629.5 s** | not stated |
| Σ decode_s in window | **143.9 s** | not stated |
| Token-level prefix-cache hit rate, all 52 runs | **34.8%** | not stated |
| Codex turns per run (median across 52 runs) | **1** | spec implied ~3 (current) → 5–10 (v4) |
| Runs that produced 0 output tokens (Codex 0.128.0 zero-token quirk) | **34/52 (65%)** | not addressed |

Per-run cache-hit pattern (from per-task `vllm_metrics_{pre,post}.txt` diffs):

| Run shape | Hit rate | Mechanism |
|---|---:|---|
| Single-turn run (1 request, 69 K prompt) | 0.0% | turn 1 always cold — the runner calls `--reset-prefix-cache` per task |
| Two-turn run (139 K prompt) | ~49.5% | turn 2 fully hits turn 1's cached prefix |
| Three-turn run (209 K prompt) | ~66% | turns 2+3 hit |
| Four-turn run (279 K prompt) | ~74% | turns 2–4 hit |
| Five-turn run (348 K prompt) | ~79% | turns 2–5 hit |

Each turn after the first adds **only ~1.8 K cold-compute tokens** (`local_compute` delta per extra turn ≈ 1.8 K vs `local_cache_hit` delta ≈ 68.7 K). The cache is healthy — the agent transcript barely grows across turns.

**The actual prefill driver is the ~70 K-token codex CLI initial system prompt** (tool inventory, MCP descriptions, sandbox/model-provider config, plus our 150-token user message) that codex sends on turn 1. At ~770 tok/s prefill throughput that's ~90 s, paid once per task because the runner resets the prefix cache before every task.

### 11.3 Why v4 prompt-shape change cannot move decode share

The v4-dirlist variant changes `prompt.md` from 150 tokens → ~250–500 tokens (1 line per top-level entry × 6–12 entries). The v4-nohint variant keeps it at ~150 tokens. Either way, the actual per-turn prompt the model sees is dominated by codex's ~70 K-token system prompt, which neither variant changes. The expected wallclock or decode-share movement is < 1 % — well below measurement noise.

### 11.4 Real levers (priority order, replacing §3's projection)

1. **Trim codex CLI's initial system prompt (highest leverage).** ~70 K tokens of tool definitions, MCP server descriptions, sandbox/model-provider docs and config that prefill at ~770 tok/s = ~90 s of pure overhead per task before the agent does anything. Investigate which codex 0.128.0 flags / config knobs reduce the inventory (e.g. `-c skill_set=...`, fewer MCP servers, narrower model-provider definition). Even a 50 % reduction = ~45 s/task wallclock saved across all 13 tasks.
2. **Stop resetting prefix cache between tasks.** `scripts/run_track_b_e2e_task.py:569` posts `/reset_prefix_cache` per task. With reset removed, task N's turn 1 would hit task N-1's cached codex system prompt → first-turn prefill drops from ~90 s to <2 s for tasks 2..N. The "isolation" rationale is real but the cost is ~52 × 88 s = ~76 minutes per sweep. Consider a single per-round reset, or per-cohort reset.
3. **Fix Codex 0.128.0 zero-token quirk.** 34/52 (65 %) v3 runs sent the request, vLLM spent ~90 s on prefill, codex aborted before getting tokens back. These runs still pay full prefill cost but produce no usable output. The runner has `--zero-token-retries` already (`run_track_b_e2e_task.py:768`) but it defaults to 0; setting it to 3 for the next sweep would roughly 2× the effective sample density.
4. **Codex transcript compaction (lowest priority).** Per-turn transcript growth is ~1.8 K tokens — small relative to the 70 K initial system prompt and the ~68.7 K cached prefix per warm turn. Compacting the transcript saves at most a few seconds per multi-turn task; not the bottleneck.

### 11.5 Phase reframing

| Original phase | Status | Replacement |
|---|---|---|
| Phase 1 — implement v4 builder | **WITHDRAWN** | v4 prompt shape doesn't address the bottleneck. No code change. |
| Phase 2 — single-task v3-vs-v4-dirlist-vs-v4-nohint smoke | **WITHDRAWN** | Same. |
| Phase 3 — 13-task v4 sweep | **WITHDRAWN** | Same. |
| Phase 4 — full Round 4 v4 baseline | **WITHDRAWN** | Same. |
| Phase 5 — promote v4 as canonical | **WITHDRAWN** | v3 sample shape stays canonical. |

New Phase 1 — **Codex initial-system-prompt forensics** (≤ 1 hour). Capture the verbatim `/v1/responses` `input` field of one fresh `codex exec` call (e.g. via proxy log + `LUMO_TRACK_B_LOG_REQUEST_BODY=1` env var on the proxy) for `transcript-merge-regression/v1`. Inventory what's in the 70 K — tool descriptions, MCP server definitions, model-provider config, sandbox docs. Identify which codex flags or config blocks would shrink it. Output: list of codex-config knobs to A/B in Phase 2.

New Phase 2 — **Codex-config A/B on one task** (≤ 30 min). Run `transcript-merge-regression/v1` twice: baseline (current 70 K system prompt) vs reduced-inventory (smallest viable codex config). Capture initial-prompt size, first-turn prefill_s, decode share. Decision criterion: reduced config produces same exit code and ≥30 % first-turn prefill reduction → adopt for sweep.

New Phase 3 — **Cache-reset policy A/B on the 13-task sample** (≤ 30 min). Two single-attempt sweeps under the Phase-2-chosen config: per-task reset (current) vs single-reset-at-sweep-start. Compare aggregate wallclock and aggregate cache hit rate. Decision criterion: single-reset preserves correctness on all 13 → adopt.

New Phase 4 — **Zero-token retry default change** (≤ 5 min code, included in Phase 5). Set runner default `--zero-token-retries=3` so 65 % of runs aren't wasted prefill.

New Phase 5 — **Round 4 baseline under the revised configuration** (≤ 30 min). 13 × 4 sweep with: reduced codex config (Phase 2 winner), single-reset cache policy (Phase 3 winner), zero-token-retries=3, otherwise identical to v3 collection. Lands at `output/track_b_e2e_v3.1/round_4/` (still v3 *prompt shape*, but with the configuration changes that actually move decode share).

### 11.6 Audit notes

- The "decode is ~10 %" figure in `track-b-round3-e2e-v3-closeout-20260510.md:83` was a **back-calculation** from the gap between microbench acceleration and end-to-end speedup, not a direct measurement. The 8.1 % direct measurement above corroborates it from the other direction.
- Per-turn metrics in `output/track_b_e2e_v3/round_3/.../vllm_per_turn.json` are all `deferred:true` — that artifact was never populated for the v3 sweep. Future sweeps should not defer this; the proxy capture file is the working source.
- The `--reset-prefix-cache` policy was inherited from the warm-only probe in `measure_track_b_real_content_task.py` (where isolation is the right call because each call is one independent measurement). It was carried over into the agent runner without the same justification, and it imposes a ~90 s/task tax that dominates the sweep wallclock.
