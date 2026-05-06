# Codex-Harness-Coupled Speculative Decoding — Engineering Spec

Generated: 2026-05-07
Revised: 2026-05-06 (post-reproduction findings)

Companion to:
- `l0-warm-decode-quality-bounded-track-20260505.md` (Track B parent spec)
- `track-b-real-workload-5x-audit-20260506.md` (51-candidate audit)
- `track-b-candidate-051-validation-20260506.md` (c1 validation recheck for 051)
- `track-b-spec-decode-salvage-20260506.md` (c1 salvage attempts for 020, 025, 028, 051)
- `track-b-concurrency-measurement-audit-20260506.md` (warm_concurrency measurement audit)

## Status update — 2026-05-06 reproduction findings

The original framing of this spec ("build on candidate 051's 2.28× win") **does not survive c1 reproduction**. Three load-bearing facts have changed:

1. **Candidate 051's 17.087 tok/s was a c4 measurement artifact, not a stable 2.28× win.** Five concurrency-1 repeats measured `7.658967-7.713841 tok/s` (mean `7.677189`, std `0.022`), all below the `9.0 tok/s` 20% gate. The original c4 run included a 4096-token cap-hit warm completion that inflated the batched aggregate; later c4 reruns also failed (`7.561378 tok/s`). B-1/B-2/B-3 do pass at c1 — correctness is fine — but speed is not reproducible. See `track-b-candidate-051-validation-20260506.md`.
2. **Candidates 020 and 025 crash vLLM EngineCore at c1** with `AssertionError: num_required_blocks N < len(req_blocks) N+1`. This matches **vLLM PR #39562** (open, not merged) — a KV-block allocator race triggered by the combination of speculative decoding + prefix caching + dynamic draft length (`prompt_lookup_min != prompt_lookup_max`). Candidate 028 doesn't crash but only reaches `8.32-8.45 tok/s` (mean `8.36`) — still below the `9.0 tok/s` threshold. See `track-b-spec-decode-salvage-20260506.md`.
3. **Realistic c1 ceiling for ngram on Qwen3 hybrid + GB10 is 1.02-1.20×, not 2×.** Online research (May 2026) finds no published benchmark of this hardware × model × workload triple where ngram alone clears 1.20× at c1. The published 2-4× speed-decode numbers are dominated by EAGLE-3, SuffixDecoding, or Medusa — not vanilla PLD/ngram — and most assume non-hybrid attention.

**Implications for this spec:**

- The "candidate 051 baseline = 17.1 tok/s" anchor used throughout the original draft has been replaced with **`7.5 tok/s` baseline (vanilla decode) ≈ `7.7 tok/s` ngram-PLD c1 (pre-fix)**. Headline targets recalibrated accordingly.
- A new **Step 0** is added before any technique work: fix the EngineCore KV allocator crash (apply PR #39562 patch OR enforce `prompt_lookup_min == prompt_lookup_max` OR disable prefix caching during spec decode). Without this, candidates 020/025-class configs cannot even be measured.
- The 9.0 tok/s 20% threshold was calibrated against an inflated baseline; the techniques in this spec (especially Techniques 1+2+3 composed) are now the load-bearing path to clearing it, not an enhancement on top of an existing 2.28× win.
- The `Cross-turn ngram (Technique 1)` path is now the most likely **first** technique to clear the threshold at c1 — SuffixDecoding's published ~2-3× on agent traces is the closest published precedent for what we need.

The rest of this spec keeps the technique inventory but rewires the goals, the composition math, and the implementation sequence around these findings.

## Why this spec exists

Track B Round 1 (Eagle-3 + PLD hybrid speculative decoding) was specified generically — "speculative decoding works across LLM workloads." Vanilla `ngram` PLD on this hardware × model at c1 hits **at most 1.05-1.20× over baseline**, well below the 1.20× acceptance gate when measured fairly. The `2.28×` originally reported for candidate 051 was a c4 measurement artifact (one 4096-token cap-hit warm completion in a batched aggregate, not reproducible at c1).

**The Codex-harness-specific opportunity is to push acceptance rate substantially higher by exposing harness state to the drafter** — recent literature (Dec 2025 - May 2026) confirms this is a real research direction with published precedents and 2-5× wins on agent traces. This spec defines the engineering work to ship harness-coupled speculative decoding as the path that actually clears the Track B speed gate on this hardware.

**This is not novel as a pattern.** Cursor's "speculative edits" (production, 13× over vanilla 70B baseline, ~1000 tok/s on Fireworks) is the strongest existence proof for harness-state-feeding-the-drafter. AgentInfer (Lin et al., arXiv:2512.18337, Dec 2025) is the canonical co-design framing. ToolSpec (Xia et al., arXiv:2604.13519, April 2026) covered the schema-aware tool-call piece. **What's open territory in the literature is** (a) proactive priming from out-of-prompt agent context (read_file outputs not yet quoted), (b) turn-boundary drafter lifecycle management, and (c) token-level plan-structure pre-drafting. This spec focuses engineering effort there.

## Goals and non-goals

**Goals (recalibrated 2026-05-06):**
- Push warm-cache decode tok/s from the c1 ngram-PLD baseline (`~7.7 tok/s`, essentially flat with vanilla `7.5`) to **11-15 tok/s sustained** (1.5-2.0× baseline) on the heavy agent workload — clearing the Track B `9.0 tok/s` 20% gate with margin and matching SuffixDecoding's published 2-3× on agent traces.
- Stretch goal: **17-22 tok/s sustained** (2.3-3.0× baseline) when Techniques 1+2+3 compose on cache-rich Codex traces. This is the territory the original (artifactual) 17.1 tok/s number suggested; here it's earned, not an accident.
- Stay within the v0.6 weight-immutability constraint — same FP8 weights, same model.
- Stay within the v0.7 quality-preservation constraint — output distribution mathematically identical (rejection sampling theorem) or quality-bounded with B-1/B-2/B-3 gate.
- Compose multiplicatively with prefix caching + LMCache (Round 0) so combined ceiling on cache-hit turns reaches 3-5×.

**Non-goals:**
- Replacing the agent harness (Codex CLI / Claude Code-style); this spec couples to whichever harness is in use, not to a specific implementation.
- General-purpose chatbot inference optimization — the wins here depend on agent-workload structure.
- Changing the model architecture or weight format.
- Multi-tenant serving — we assume concurrency=1, both because the EngineCore KV allocator crash (PR #39562) blocks c2+ for many configs AND because c4 measurements turned out to be unreliable as a candidate acceptance basis (see `track-b-concurrency-measurement-audit-20260506.md`). When PR #39562 lands and the measurement protocol is hardened, the design generalizes to c2+.

## Architecture overview

The architecture co-designs the agent harness and the inference layer. The harness (Codex CLI / Claude Code) emits structured information that crosses the inference boundary into the drafter:

```
                                    ┌────────────────────────────────┐
                                    │  Codex Harness (per agent task) │
                                    │ ┌────────────────────────────┐ │
                                    │ │ Tool registry              │ │
                                    │ │   - tool schemas (JSON)    │ │
                                    │ │   - tool name allow-list   │ │
                                    │ │ Turn boundary signals      │ │
                                    │ │   - new_task / new_turn    │ │
                                    │ │   - tool_invoked / done    │ │
                                    │ │ File access events         │ │
                                    │ │   - read_file <path>       │ │
                                    │ │   - write_file <path>      │ │
                                    │ └────────────────────────────┘ │
                                    └─────────────┬──────────────────┘
                                                  │
                                                  │  HARNESS ORACLE API
                                                  │  (new in this spec)
                                                  │
                ┌─────────────────────────────────┴─────────────────────────────┐
                │                  Drafter Coordinator                          │
                │ ┌────────────────────────────────────────────────────────────┐ │
                │ │ DrafterMixer                                               │ │
                │ │   1. Cross-turn ngram cache (build on SuffixDecoding)     │ │
                │ │   2. Read-file priming buffer (NEW, open territory)        │ │
                │ │   3. Schema-aware tool drafter (build on ToolSpec/XGrammar)│ │
                │ │   4. Plan-structure pre-drafter (NEW, open territory)      │ │
                │ │   5. Turn-boundary lifecycle controller (NEW)              │ │
                │ └────────────────────────────────────────────────────────────┘ │
                └────────────────────────────┬──────────────────────────────────┘
                                             │
                                             │  draft tokens (k=4-8)
                                             │
                                             ▼
                               ┌─────────────────────────────┐
                               │   vLLM target model verifier │
                               │   (Qwen 3.x 27B FP8, FA2)    │
                               │   rejection sampling →       │
                               │   accepted token sequence    │
                               └─────────────────────────────┘
```

The HARNESS ORACLE API is the load-bearing piece. It carries structured information from the harness to the drafter that vanilla vLLM's spec_decode never sees.

## Five techniques (per-technique spec)

Each technique below has: prior-art alignment, mutation surface, what to do, what to measure.

### Technique 1: Cross-turn ngram cache (BUILD ON SUFFIXDECODING)

**Prior art alignment.** SuffixDecoding (He et al., NeurIPS 2025 spotlight, arXiv:2411.04975) is the published precedent. It builds a suffix tree across prompts AND prior outputs, reports up to 5.3× on agentic SWE-Bench / Text-to-SQL workloads — explicitly motivated by "agentic frameworks submit repetitive inference requests." Production-deployed in Snowflake's ArcticInference (vLLM). AgentSAM (in AgentInfer, arXiv:2512.18337) is the cross-session generalization.

**Mutation surface.**
- `vllm/v1/spec_decode/ngram_proposer.py` — extend to maintain a per-session suffix tree spanning all prior turns of a Codex agent task.
- `vllm/engine/protocol.py` — add `session_id` field to request envelope so the drafter can route lookups to the correct session's tree.
- Suffix tree memory budget: 50-200 MB per session (configurable); evict via LRU on `session_close` signal from the harness.

**What to do (engineering steps):**
1. Pull SuffixDecoding's reference implementation (Snowflake ArcticInference fork of vLLM has it production-grade); fork or rebase onto our vLLM version.
2. Add `session_id` to the OpenAI-completions-compatible API request shape; the harness sets one session_id per agent task and maintains it across all turns of that task.
3. Wire `session_id` through vLLM's request scheduler to the drafter's lookup path.
4. On `session_close`, drop the session's suffix tree (memory reclaim).

**What to measure:**
- Per-turn draft acceptance rate, broken down by turn type (edit / tool-call / reasoning / plan).
- Suffix-tree size growth across turns (memory budget validation).
- Cross-turn hit rate: fraction of draft proposals that match patterns from PRIOR turns (not the current turn's prompt).
- Compared to baseline (PLD with prompt-only ngram): expected absolute increase of 10-25 percentage points in cross-turn-relevant turn types.

**Expected lift over c1 PLD baseline (`~7.7 tok/s`):** PLD captures within-prompt repetition; cross-turn captures the rest. SuffixDecoding's published numbers on agent traces (SWE-Bench, Text-to-SQL) are 2-3× over base decode, of which ~1.5-2.0× is the increment over plain PLD. On a 20-turn Codex task with high cross-turn echo, expected aggregate lift is **1.4-1.6× over plain PLD** at c1, putting us in `11-13 tok/s` territory. **This is the technique most likely to single-handedly clear the `9.0 tok/s` gate.**

---

### Technique 2: Read_file proactive priming (OPEN TERRITORY)

**Prior art alignment.** No published paper does exactly this. Closest precedent: REST (He et al., NAACL 2024, arXiv:2311.08252) does retrieval-based spec decode by retrieving from an external datastore at inference time, but retrieves based on the CURRENT suffix at decode time — not "proactively prime when the harness reveals likely-relevant content." Nichols et al. (arXiv:2512.15834, Dec 2025) hint at the idea by recommending a "tool cache API endpoint" so LM providers expose harness-aware optimizations to callers, but don't elaborate. **This is a real open gap.**

**Mutation surface.**
- New harness oracle hook: `prime_drafter_with_text(text: str, source_tag: str, ttl_turns: int)`. The Codex harness calls this when:
  - Agent invokes `read_file <path>` and receives content (prime with file content; tag = `file:<path>`; ttl = until file modified).
  - Tool returns a long structured output (prime with tool output; tag = `tool:<name>:<call_id>`; ttl = N turns).
  - Agent receives a long observation (prime; tag = `observation:<step>`; ttl = N turns).
- Drafter integration: the primed text is fed into the same suffix-tree/ngram structure as the cross-turn cache (Technique 1), with provenance tags so the drafter can score lookups by source.

**What to do:**
1. Add the `prime_drafter_with_text` API endpoint to vLLM's request shape (likely as an OpenAI-extension field on session-aware requests).
2. Add the priming-buffer ingestion path in the drafter coordinator — text from primings is tokenized and folded into the session's suffix tree.
3. Add provenance scoring: when the drafter has multiple lookup matches, prefer matches from primed file content if the agent's recent context references that file (heuristic: "agent emitted the file path within the last 1024 tokens").
4. Hook the Codex harness's `read_file` tool result handler to call `prime_drafter_with_text` automatically.

**Risk: cache poisoning.** SuffixDecoding paper notes long unique strings can pollute the lookup table. Limit primed-text size per call (e.g., 64 KB cap); evict aggressively on `session_close`.

**What to measure:**
- Per-turn acceptance lift on edit/echo turns where the agent rewrites recently-read code: with vs without proactive priming.
- Memory cost: priming buffer size growth as agent task progresses.
- False-positive draft rate: primed content that the drafter proposes but the verifier rejects (should stay low; high values indicate noise pollution).
- Agent task wallclock with priming enabled vs disabled — the headline number.

**Expected lift over Technique 1:** specifically on turns where the agent edits content from a file it just read. For a Codex task that's 30-40% file-edit turns, expected aggregate lift is 1.10-1.20× over Technique 1 alone (i.e., turns Technique 1's `~11-13 tok/s` into `~12-15 tok/s` on edit-heavy traces).

---

### Technique 3: Schema-aware tool-call drafter (BUILD ON TOOLSPEC + XGRAMMAR-2)

**Prior art alignment.** ToolSpec (Xia et al., arXiv:2604.13519, April 2026) is the published version of this technique — finite-state machine alternates between deterministic schema-token filling and speculative generation for variable JSON fields, plus retrieval of historical tool invocations as drafts. **4.2× speedup reported.** XGrammar-2 (MLC, May 2026) integrates constrained decoding with speculative decoding via `traverse_draft_tree` — production-integrated into SGLang, vLLM, TensorRT-LLM, MLC-LLM.

**Mutation surface.**
- `vllm/v1/spec_decode/schema_aware_proposer.py` — new proposer that:
  - Receives tool schemas via the harness oracle API (`set_tool_schemas`).
  - When generation enters a tool-call frame (detected by agent harness signal `tool_call_started`), the schema-aware proposer drafts the JSON skeleton tokens with high confidence (`{"name":` is forced; `"name"` value is one of the tool names in the schema; etc.).
  - For free-text argument values, falls back to PLD-style lookup over recent context (file paths, identifiers in the prompt).
- XGrammar-2 integration: when the drafter proposes a token, XGrammar's mask is applied AT DRAFT TIME (not just at verify time). Forced tokens skip sampling entirely.

**What to do:**
1. Pull XGrammar-2 if not already present in our vLLM build (per audit, xgrammar IS present).
2. Implement the schema-aware proposer per ToolSpec's FSM design — straightforward port from the published code if available; otherwise re-implement from the paper's pseudocode.
3. Harness oracle hook: `set_tool_schemas(schemas: list[dict])` called once per session.
4. Harness oracle hook: `tool_call_started()` and `tool_call_finished()` signals so the drafter knows when to switch to schema-aware mode vs back to general PLD.

**What to measure:**
- Acceptance rate on tool-call-emission turns specifically (separate from the aggregate).
- Forced-token skip rate (XGrammar-2 should report this).
- Output schema validity (target 100%).
- Per-tool-call wall time before vs after.

**Expected lift on tool-call turns specifically:** 1.6-2.0× over Technique 1+2 baseline. ToolSpec's reported 4.2× is on a pure tool-call benchmark; in mixed agent traffic our weighted-average lift is more modest because tool calls are 20-30% of turns. **Caveat:** vLLM Issue #40875 documents `prompt_lookup_min=2` corrupting tool-call XML on Qwen3 — Technique 3's schema-aware drafter must use `prompt_lookup_min >= 3` for free-text fallback or use the FSM forced-token path exclusively for the structured frame.

---

### Technique 4: Plan-structure token-level pre-drafting (OPEN TERRITORY)

**Prior art alignment.** Closest: Speculative Actions (Ye et al., ICLR 2026 oral, arXiv:2510.04371) — fast Speculator predicts likely next AGENT ACTIONS, executes in parallel, rolls back on mismatch. Not the same problem: Speculative Actions speculates at action GRANULARITY (one tool call as a unit); we need TOKEN-granularity inside a plan/TODO emission. Dynamic Speculative Planning (arXiv:2509.01920) and Interactive Speculative Planning (arXiv:2410.00079) are related but also action-granularity. **Token-level plan-structure speculation appears to be a genuine gap.**

**Mutation surface.**
- New plan-structure detector: when the agent emits a structured plan/TODO list, the drafter recognizes the structure (markdown list pattern, numbered steps, structured headers).
- Per-task plan template registry: when the agent emits its FIRST plan in a task, the drafter fingerprints the structure (number of steps, header pattern, separator style). On SUBSEQUENT plan emissions in the same task (e.g., "updated plan after step 1"), the drafter pre-drafts the structural tokens (numbers, separators, headers) with high confidence.
- Harness oracle hook: `plan_emitted(structure_fingerprint: dict)` called by the harness when it detects the agent emitted a plan; or auto-detected by the drafter via heuristic.

**What to do:**
1. Heuristic plan detector — regex over recent decoded tokens looking for structured patterns: numbered lists (`1.`, `2.`, etc.), bullet structures (`- `, `* `), markdown headers (`### `, `## `), checklist items (`- [ ]`, `- [x]`).
2. Plan-structure fingerprint: extract structural tokens (the numbers, separators, headers — NOT the plan content).
3. On detected plan emission, register the fingerprint in the session's plan registry.
4. On subsequent decoder calls, if context suggests another plan emission (e.g., agent recently wrote "updated plan:" or "revised plan:"), the drafter proposes the structural tokens as drafts.
5. Verifier accepts/rejects normally; high acceptance because structural tokens repeat across emissions.

**Risk: false-positive plan detection.** A markdown list that's NOT a plan (e.g., agent quoting a file's content) would trigger the detector. Mitigation: scoring threshold + require at least 3 prior plan emissions before activating the pre-drafter for this session.

**What to measure:**
- Plan detection precision/recall on annotated agent traces.
- Acceptance rate on plan-update turns specifically.
- Wallclock per plan-emission turn before vs after.

**Expected lift on plan-emission turns specifically:** 1.3-1.6× since the structural tokens are highly predictable. Plan-emission turns are ~10-15% of agent task turns, so weighted-average lift is modest (1.03-1.08× e2e on top of Technique 1+2+3 — measured against the c1 PLD baseline, this is the difference between landing at `~13 tok/s` vs `~14 tok/s`).

---

### Technique 5: Turn-boundary drafter lifecycle (OPEN TERRITORY)

**Prior art alignment.** No published paper treats drafter lifecycle as a first-class concept. SGLang's RadixAttention (Zheng et al., NeurIPS 2024) and vLLM's APC are session/multi-turn aware for KV CACHE, but for KV cache, not for drafter state. AgentSched (in AgentInfer) is the closest published "agent-turn-aware scheduler", but it's about scheduling, not drafter. **Genuine gap on the drafter side.**

**Mutation surface.**
- New harness oracle hooks: `session_open(session_id)`, `session_close(session_id)`, `turn_open(turn_index)`, `turn_close(turn_index)`.
- Drafter coordinator implements lifecycle handlers:
  - `session_open`: allocate fresh suffix tree, plan registry, priming buffer.
  - `turn_open`: snapshot current drafter state (so a hypothetical rollback could restore).
  - `turn_close`: commit the turn's tokens to the persistent session state; clear ephemeral within-turn buffers.
  - `session_close`: free all session-scoped state; garbage-collect.
- Default policy if harness doesn't emit lifecycle signals: timer-based eviction on session-id idle (e.g., 5 min no requests = session_close implicit).

**What to do:**
1. Add the lifecycle hooks to vLLM's API extension surface.
2. Implement state snapshot/commit in the drafter coordinator.
3. Update Codex harness to emit `session_open` per agent task, `turn_open`/`turn_close` per agent turn, `session_close` on task completion.
4. Default fallback for harnesses that don't emit signals: idle-timer-based.

**What to measure:**
- Memory consumption over a multi-task agent session (10 sequential agent tasks with proper lifecycle vs no lifecycle = unbounded growth).
- Drafter cache pollution: false-positive draft rate when prior session's state leaks into a new session (should be zero with proper lifecycle).
- Latency of session_open / turn_open hooks (should be <10 ms).

**Expected lift:** marginal in absolute throughput (~1.02× e2e at most; mainly avoids cross-session contamination). The bigger wins are correctness/operational: bounded memory growth, no cross-session draft pollution. **This technique is foundational for Techniques 1-4 to work CORRECTLY in production multi-agent-task settings.**

---

## Composition (the harness-coupled-spec-decode bundle)

The five techniques compose multiplicatively on different turn types. Baseline is the c1 ngram-PLD reproduced number (`~7.7 tok/s`), not the artifactual c4 number.

| Turn type | Frequency | Baseline (c1 PLD) | Technique 1 | + 2 | + 3 | + 4 | + 5 |
|---|---|---|---|---|---|---|---|
| Code edit / rewrite | ~30% | 7.7 | ×1.50 → 11.6 | ×1.20 → 13.9 | n/a | n/a | ×1.0 |
| Tool call emission | ~25% | 7.7 | ×1.20 → 9.2 | ×1.05 → 9.7 | ×1.70 → **16.5** | n/a | ×1.0 |
| Plan / status update | ~15% | 7.7 | ×1.30 → 10.0 | ×1.05 → 10.5 | n/a | ×1.40 → **14.7** | ×1.0 |
| Free-form reasoning | ~30% | 7.7 | ×1.05 → 8.1 | ×1.0 | n/a | n/a | ×1.0 |

**Workload-weighted average target:** ~11-15 tok/s sustained (1.5-2.0× over baseline 7.5), clearing the `9.0 tok/s` gate. Stretch on cache-rich Codex traces: ~17-22 tok/s when Techniques 1+2+3 land cleanly. Combined with Round 0 (prefix cache + LMCache, 2-3× on cache-hit prefill): **e2e on cache-hit turns: 3-5×.**

**Note on the recalibration.** The original draft assumed candidate 051's 17.1 tok/s as a free starting point and described the techniques as multiplicative on top of it. Reproduction shows that number was a measurement artifact; the techniques are now what gets us to the speed target, not a topping. SuffixDecoding alone (Technique 1) on agent traces is published at 2-3×; that's the most likely first-arrival path to clearing the gate.

## Integration with vLLM and Codex harness

### vLLM-side integration (extension layer)

- New module `vllm/v1/spec_decode/harness_coupled/` containing:
  - `oracle_api.py` — REST/IPC endpoint for harness oracle hooks
  - `drafter_coordinator.py` — multiplexes the five drafters
  - `session_state.py` — per-session state container (suffix tree, priming buffer, plan registry, schemas)
  - `lifecycle.py` — session/turn open/close handlers
- API surface exposed via vLLM's OpenAI-compatible API as extension fields:
  ```json
  {
    "model": "qwen-3.x-27b-fp8",
    "session_id": "agent-task-abc123",
    "harness_oracle": {
      "tool_schemas": [...],
      "primed_text": [{"text": "<file content>", "tag": "file:src/foo.py", "ttl_turns": 5}],
      "turn_event": "open",
      "plan_emitted_fingerprint": null
    },
    "messages": [...]
  }
  ```
- Backward-compatible: vLLM clients without `harness_oracle` field get plain spec_decode.

### Codex harness side integration

- New module in the harness: `harness_oracle_client.py` that:
  - Maintains `session_id` per agent task.
  - Calls `set_tool_schemas` once per session at task open.
  - Calls `prime_drafter_with_text` on `read_file` tool returns and other long observations.
  - Emits `turn_open`/`turn_close` around each turn.
  - Emits `session_close` on task completion.
- Configurable disable: harness operates correctly without the oracle (vLLM just falls back to vanilla spec_decode).

## Measurement plan

Three measurement tracks, ablation-friendly so per-technique contribution is isolatable.

### Track 1: per-technique micro-benchmarks

For each of the 5 techniques, a focused benchmark on a workload slice where it should help:

| Technique | Slice | Metric | Target |
|---|---|---|---|
| 1: cross-turn ngram | 20-turn agent task with high cross-turn echo | Draft acceptance rate per turn | +10-25 pp over plain PLD on echo-heavy turns |
| 2: read_file priming | Edit-after-read sequences (read_file then edit) | Acceptance rate on edit turn | +15-30 pp over Technique 1 alone |
| 3: schema-aware tool drafter | Tool-call-only synthetic turns | Tool-call wall-time | 1.6-2.0× faster |
| 4: plan-structure pre-drafter | Plan-update turns after a plan was emitted earlier | Acceptance on plan structural tokens | >85% on structural tokens, >50% on freely-emitted plan content |
| 5: turn-boundary lifecycle | 10 sequential agent tasks | Memory consumption growth | Bounded (each session_close frees state); no cross-session contamination |

### Track 2: ablation on the heavy agent workload

Run the SAME workload trace (`responses-sdk-adapter-cutover-heavy/seed_trace_v5.jsonl`) under each combination:

- A: candidate 051 baseline (PLD only)
- B: A + Technique 1 (cross-turn ngram)
- C: B + Technique 2 (read_file priming)
- D: C + Technique 3 (schema-aware tool drafter)
- E: D + Technique 4 (plan-structure pre-drafter)
- F: E + Technique 5 (turn-boundary lifecycle, mainly correctness)

Each combination measured on the same trace. Report tok/s, draft acceptance rate breakdown, B-1/B-2/B-3 quality gate results.

### Track 3: end-to-end agent task wallclock

Run a real Codex agent task (e.g., "fix this bug in the codebase") under each combination. Measure:

- End-to-end task wallclock
- Per-turn p50/p95 latency
- Token count
- Quality (does the task complete correctly? B-2 behavioral verifies)

## Failure modes and mitigation (informed by prior research)

| Failure mode | Source | Mitigation |
|---|---|---|
| Naive speculation increases COST even when it saves latency | DSP (arXiv:2509.01920) | Track Pareto frontier (latency × compute × draft-cost). Reject techniques whose net compute cost exceeds latency saved by the operator's chosen ratio. |
| Error compounding when speculative actions execute on wrong premises | Sherlock (arXiv:2511.00330) | This system speculates at TOKEN level under rejection sampling — error compounding is bounded by the verifier. No mitigation needed for token-level; for action-level (out of scope) would need rollback. |
| PLD inconsistency vs standard generation if not implemented carefully | HuggingFace issue #30448 | Verifier MUST be exact; rejection sampling must use the target model's logits, not draft logits. B-1 distributional KL near-zero gate catches violations. |
| Cache poisoning by long unique strings (Technique 1, 2) | SuffixDecoding paper | Per-priming size cap (64 KB); per-session lifetime cap; aggressive eviction on `session_close`. |
| Mask invalidation when draft tree branches outside grammar (Technique 3) | XGrammar-2 paper | Use XGrammar-2's `traverse_draft_tree` API which validates each draft branch against the grammar at draft time; reject mid-draft branches that exit grammar. |
| Drafter state leaks across sessions (Technique 5) | Open territory; no published precedent | Lifecycle hooks; idle-timer fallback; per-session state container with explicit ownership. |
| False-positive plan detection (Technique 4) | Open territory | Heuristic threshold + minimum-3-prior-plan-emissions before activating. |

## Implementation sequence

The sequence is now **fix-spec-decode-stability-first, then build harness-coupled techniques, then measure**. Steps 0a/0b are new prerequisites surfaced by the 2026-05-06 reproduction findings.

| Step | Output | Dependency | Notes |
|---|---|---|---|
| **0a. Fix vLLM EngineCore KV allocator crash** | `output/spec_decode_concurrency_fix_validation.md` | nothing | **Blocker.** Apply PR #39562 patch OR enforce `prompt_lookup_min == prompt_lookup_max` OR disable prefix caching for spec decode. Without this, candidates 020/025-class configs cannot even be measured. Validate by re-running 020/025 c1 reproduction without crashing. |
| **0b. Lock down measurement protocol** | `output/spec_decode_measurement_protocol.md` | 0a | Median-of-3-or-5 c1 runs. Generated-token-volume guard (flag any run with output >= cap). Same `warm_concurrency` between baseline and candidate. Per `track-b-concurrency-measurement-audit-20260506.md`. |
| **0c. Establish honest c1 baseline** | `output/spec_decode_c1_baseline.md` | 0a, 0b | Re-measure baseline (vanilla decode, no spec decode) and PLD (`prompt_lookup_min=7, prompt_lookup_max=8` matching 051) under the new protocol. Expected: baseline ≈ 7.5, PLD ≈ 7.7. This is the floor the techniques must beat. |
| 1. Round 0 — install LMCache + verify | `output/round_0_lmcache.md` | nothing (parallel) | Independent prerequisite for the combined 3-5× cache-hit target. |
| 2. Pull SuffixDecoding from Snowflake ArcticInference | vLLM fork or rebase | 0a | Foundation for Technique 1; production-grade code. SuffixDecoding alone is published at 2-3× on agent traces — most likely first-arrival to clearing the speed gate. |
| 3. Implement harness oracle API (vLLM extension) | `vllm/v1/spec_decode/harness_coupled/oracle_api.py` | step 2 | Non-breaking; backward compatible. |
| 4. Wire Technique 1 (cross-turn ngram) | `drafter_coordinator.py` updated | steps 2-3 | Build on SuffixDecoding. **Gate after this step**: re-measure under protocol 0b; if Technique 1 alone clears `9.0 tok/s` aggregate, ship-it path opens. |
| 5. Wire Technique 5 (lifecycle) | `lifecycle.py` | step 3 | Foundational; do early to bound memory while iterating. |
| 6. Implement Technique 2 (read_file priming) | priming buffer integrated | steps 4, 5 | Open territory; novel piece. |
| 7. Pull XGrammar-2; verify availability | sanity check | nothing (parallel) | xgrammar already present per audit. |
| 8. Implement Technique 3 (schema-aware tool drafter) | port ToolSpec FSM | step 7 | Build on ToolSpec + XGrammar-2. |
| 9. Implement Technique 4 (plan-structure pre-drafter) | plan detector + registry | step 5 | Open territory; novel piece. |
| 10. Codex harness side integration | `harness_oracle_client.py` | steps 3-9 | Harness emits oracle calls. |
| 11. Run measurement plan Track 1 (per-technique) | per-technique reports | all techniques wired | Ablation data. |
| 12. Run measurement plan Track 2 (heavy-workload ablation) | combined report | step 11 | Composition data. |
| 13. Run measurement plan Track 3 (real Codex task) | wallclock report | step 12 | End-to-end signal. |
| 14. Closeout report + recommendations | `docs/reports/auto_research/codex-harness-spec-decode-closeout.md` | step 13 | Decision: ship vs iterate. |

### Step 0a workaround details

The `AssertionError: num_required_blocks N < len(req_blocks) N+1` triggers when the speculative drafter requests more blocks than the KV allocator pre-reserved for the request. Three workarounds are available; pick the one that interferes least with techniques downstream:

1. **Preferred — apply PR #39562 patch.** Open PR (not merged as of 2026-05-06). Patches the allocator to handle dynamic draft length without the assertion. Risk: maintenance cost of carrying an unmerged patch.
2. **Constraint — `prompt_lookup_min == prompt_lookup_max`.** Eliminates dynamic draft-length variance, sidestepping the allocator's assumption. Acceptable for ngram-PLD; more restrictive for SuffixDecoding (Technique 1) and not viable for Techniques 3/4 which expect variable-length drafts. Use as a stop-gap for Technique-1 bring-up only.
3. **Disable prefix caching during spec decode.** Sidesteps the interaction. Costs us Round-0 prefix-cache wins on the same request — unacceptable for the combined 3-5× cache-hit target. Use only for isolation testing.

See also vLLM Issue #39273 (GDN ngram corruption — separate but related; ensures we don't ship `prompt_lookup_min < 3` configs on Qwen3 hybrid) and Issue #40875 (`prompt_lookup_min=2` corrupts tool-call XML on Qwen3 — affects Technique 3).

## Open questions

1. **vLLM fork or upstream contribution?** Techniques 2, 4, 5 are open territory and would be publishable contributions. Decision: fork for now (controlled iteration), upstream the non-novel pieces (1, 3) once stable.
2. **PR #39562 carrying cost.** If we apply the unmerged patch, we own the rebase burden until upstream merges. Track the PR; switch to upstream the moment it lands. Maintain the `prompt_lookup_min == prompt_lookup_max` fallback as a one-line config flip in case the patch needs to be reverted.
3. **Hybrid attention + spec decode at c1.** Verified correctness at c1 in the audit (B-1/B-2/B-3 pass for 051). Speed at c1 is the issue, not correctness. Multi-session state needs verification before shipping; covered by Technique 5 lifecycle work.
4. **Is the `9.0 tok/s` 20% gate the right threshold?** Calibrated against an inflated baseline in the original Track B spec. Two options: (a) keep `9.0` and let the techniques in this spec be the path to clearing it (current plan), or (b) lower to `~9.0` over the honest baseline (effectively unchanged) but explicitly note that vanilla ngram-PLD alone cannot clear it on this hardware. Recommend (a) — the techniques are the point.
5. **Concurrency generalization.** Once PR #39562 lands and the c4 measurement protocol is hardened, the design generalizes. Until then, c1 is the only acceptance shape. Track B parent spec should be updated to match.
6. **LMCache + harness oracle interaction.** LMCache (Round 0) caches KV across turns; the harness oracle caches drafter state across turns. They're orthogonal but both consume `session_id`. Verify they coexist.
7. **Should we evaluate SuffixDecoding standalone before building the harness oracle?** SuffixDecoding alone is published at 2-3× on agent traces. If Snowflake ArcticInference's drop-in achieves the gate alone, Techniques 2-4 become enhancement rather than required. Decision: gate after Step 4 — re-measure with SuffixDecoding alone before building 2-4.

## References

| Source | Status | Relevance |
|---|---|---|
| AgentInfer / Lin et al. — https://arxiv.org/abs/2512.18337 (Dec 2025) | Verified | Canonical co-design framing for harness + inference |
| SuffixDecoding / He et al. — https://arxiv.org/abs/2411.04975 (NeurIPS 2025 spotlight) | Verified | Cross-turn ngram drafter; foundation for Technique 1 |
| ToolSpec / Xia et al. — https://arxiv.org/abs/2604.13519 (April 2026) | Verified | Schema-aware tool-call drafter with FSM; foundation for Technique 3 |
| REST / He, Zhong, Cai, Lee, He — https://arxiv.org/abs/2311.08252 (NAACL 2024) | Verified | Retrieval-based spec decode; closest precedent for proactive priming |
| Optimizing Agentic LM Inference via Speculative Tool Calls / Nichols et al. — https://arxiv.org/abs/2512.15834 (Dec 2025) | Verified | Tool-cache-API recommendation; closest hint at Technique 2 |
| Cursor speculative edits — https://fireworks.ai/blog/cursor and https://cursor.com/blog/instant-apply | Verified | Production existence proof: harness-side knowledge driving drafter, 13× speedup |
| XGrammar-2 — https://blog.mlc.ai/2026/05/04/xgrammar-2-fast-customizable-structured-generation | Verified | Constrained + spec decode integration; `traverse_draft_tree` |
| Speculative Actions / Ye et al. — https://arxiv.org/abs/2510.04371 (ICLR 2026 oral) | Verified | Action-level speculation; precedent for Technique 4 (different granularity) |
| Dynamic Speculative Planning / Guan et al. — https://arxiv.org/abs/2509.01920 | Verified | Pareto-frontier framing for speculation cost vs latency |
| Sherlock / Microsoft Research — https://arxiv.org/abs/2511.00330 (Nov 2025) | Verified | Speculative agent execution with rollback; failure-mode reference |
| PLD / Saxena — https://github.com/apoorvumang/prompt-lookup-decoding | Verified | Original PLD; baseline candidate 051 used this |
| PLD+ — https://arxiv.org/abs/2412.01447 | Verified | Extended PLD with hidden-state signals |
| CDSL / Nakshatri et al. — https://arxiv.org/abs/2412.10418 (NAACL 2025) | Verified | Constrained + speculative + task-reward; tangential precedent |
| Snowflake ArcticInference SuffixDecoding deployment | Verified (blog) | Production-grade implementation we can fork |
| vLLM APC + spec_decode docs — https://docs.vllm.ai/ | Verified | Integration target |
| SGLang RadixAttention — Zheng et al., NeurIPS 2024 | Verified | Closest precedent for session-aware KV; not drafter-aware |
| vLLM PR #39562 (KV allocator assertion fix) — https://github.com/vllm-project/vllm/pull/39562 | Verified, OPEN | Canonical patch for `num_required_blocks N < len(req_blocks) N+1` crash hit by candidates 020, 025. Step 0a dependency. |
| vLLM Issue #39273 (GDN ngram corruption) — https://github.com/vllm-project/vllm/issues/39273 | Verified | Documents corruption when `prompt_lookup_min < 3` on Qwen3 hybrid-attention models. Constrains Technique 1 config. |
| vLLM Issue #40875 (`prompt_lookup_min=2` corrupts tool-call XML) — https://github.com/vllm-project/vllm/issues/40875 | Verified | Documents tool-call XML corruption at low ngram min on Qwen3. Constrains Technique 3 fallback config. |

---

*This spec defines the engineering work to take Track B Round 1 from a c1 ngram-PLD baseline (`~7.7 tok/s`, essentially flat with vanilla decode) past the `9.0 tok/s` 20% gate and into the `11-15 tok/s` sustained range (1.5-2.0× baseline; 3-5× combined with Round 0 prefix cache + LMCache on cache-hit turns). The original "candidate 051's 2.28×" framing was a c4 measurement artifact and has been retired. Two of the five techniques have published precedent we build on (SuffixDecoding for Technique 1, ToolSpec/XGrammar-2 for Technique 3); three are open research territory and may be publishable contributions in their own right. Step 0a (vLLM PR #39562 patch or `prompt_lookup_min == prompt_lookup_max` workaround) is a hard prerequisite — without it, candidates 020/025-class configs crash EngineCore at c1 and cannot even be measured.*
