# Track B E2E — Round 4a Measurement-Protocol Fix

Generated: 2026-05-10
Status: active spec, pre-implementation

Companion to:
- `track-b-e2e-swe-style-prompt-shape-spec-20260510.md` (retracted v4 prompt-shape spec; this doc supersedes its §§1–10 with the corrected diagnosis from its §11)
- `track-b-e2e-agentic-saturation-plan-20260508-v2.md` (parent saturation plan; this adds rules 17/18/19 to its §8 truthful-measurement contract)
- `track-b-round3-e2e-v3-closeout-20260510.md` (v3 baseline; supersedes the v3 baseline once Round 4a baseline lands)

## 1. Predicate finding (from retracted v4 spec §11)

The retracted v4 spec measured the v3 sweep's per-turn proxy capture directly and found:

| Bucket | Turns | Median prompt_tokens | Σ prefill_s | Σ decode_s | Decode share |
|---|---:|---:|---:|---:|---:|
| Aggregate | 30 | 69,516 | 1,629.5 | 143.9 | 8.1% |
| tool-call regime | 30 | 69,516 | 1,718.3 | 144.4 | 7.8% |
| reasoning regime | 4 | 69,672 | 2.4 | 11.5 | 82.7% |

Three load-bearing facts from that measurement:

1. **Per-turn prompt_tokens median = 69,516** (not 5000+ as the v4 spec hypothesized). The workspace bundle contributes ~150 tokens; the rest is **Codex CLI's task-agnostic system prompt** (tool definitions, MCP server descriptions, sandbox policy, model-provider config).
2. **Token-level prefix-cache hit rate = 34.8%** — fully explained by the runner's per-task `--reset-prefix-cache` policy. Turn 1 is always 100% cold (~90 s prefill on the 70K prompt). Turns 2+ are ~98% warm (~1.8K cold tail tokens added per turn, agent transcript barely grows).
3. **65% of v3 runs hit Codex 0.128.0's zero-token quirk.** Codex sends request, vLLM spends ~90 s on cold prefill, Codex aborts client-side before vLLM emits the first SSE chunk. vLLM pays full prefill cost; reports `usage: 0 tokens`. Runner's `--zero-token-retries` flag wired but defaults to 0.

The dominant per-task wallclock cost is **a single cold turn-1 prefill on the ~70K-token Codex system-prompt prefix**, amplified by the 65%-incidence client-side timeout that pays full prefill for zero output. The retracted v4 spec proposed addressing prefill via prompt-shape change; that cannot fire because the e2e runner already produced ~150-token user prompts and the dominant cost is upstream of the user prompt entirely.

## 2. Goal and non-goals

**Goal:** establish a measurement protocol where the cold turn-1 prefill cost is amortized (approximating production cache warmth), the zero-token quirk no longer dominates variance, and the static system prompt is recorded for visibility — without removing or modifying any of its content.

**Non-goals:**
- System-prompt trimming. The 70K is recorded, not optimized. (Decision Q2 below.)
- Workspace bundle changes. The v3 prompt shape stays — `_write_prompt` continues to write only `AGENTS.md` + `.scenario_variant` (~150 tokens).
- Sample membership changes. Same 13 families × `v1-clean-baseline`.
- Model or runtime config changes. Same vLLM container `lumo-vllm-track-b-suffix`, same `runtime_config_hash` from Round 3.

## 3. Decisions (recorded 2026-05-10 by Mark)

| # | Question | Decision |
|---|---|---|
| Q1 | Cold turn-1 prefill: keep, drop, or warmup-pass? | **Warmup-pass** that pre-warms the task-agnostic Codex system-prompt prefix only. Cache retained across tasks for the static portion; per-task content tail still pays its own prefill on turn 1. |
| Q2 | Trim the Codex system prompt? | **No.** Record per-section token counts as part of measurement instead — visibility, not optimization. |
| Q3 | Investigate Codex CLI request timeout config? | **Yes**, in parallel with warmup-pass. Bump as defense-in-depth so the timeout never fires even on edge-case slow prefills. |
| Q4 | Flip `--zero-token-retries` default? | **Yes**, from 0 to 3 in the round driver. Belt-and-suspenders after warmup-pass eliminates the underlying cause. |

## 4. Zero-token quirk explanation

Mechanism, end to end:

1. Codex `POST /v1/responses` (streaming).
2. vLLM accepts, starts prefill on the ~70K-token prompt — ~90 s cold.
3. Before vLLM emits the first SSE chunk, **Codex aborts client-side** (HTTP read-timeout or SSE buffering edge case; not yet reproduced cleanly).
4. vLLM has no signal Codex disconnected until first-byte time, so it pays the full prefill anyway.
5. Codex reports `turn.completed` with `usage: 0 tokens`. From the proxy: a request went out, a response came back, no tokens generated.

Root-cause fix: eliminate the slow prefill (§5 warmup-pass). Defense-in-depth: bump Codex's request timeout (§7.3) and flip the retry default (§7.5).

## 5. Warmup-pass design

**Goal:** make turn 1 of every task hit the prefix cache for the system-prompt prefix (the ~69K task-agnostic content), so the only cold prefill on turn 1 is the per-task tail (~150-300 tokens of `AGENTS.md` + `.scenario_variant` + first user message).

**Implementation:**

1. **Capture the static system-prompt prefix once per round.** At round start (before any task measurement), the proxy captures one Codex `/v1/responses` request payload, extracts the system-prompt portion (everything up to the first user message), and saves it to `output/track_b_e2e_v4a/round_<N>/codex_system_prompt.json` with token count and a content hash.

2. **Warmup-pass executor — new helper `scripts/run_track_b_e2e_warmup.py`.** POSTs the captured system-prompt prefix + a one-token user message (e.g., `"ok"`) to vLLM `/v1/responses`. Completion produces ≤5 tokens. After this fires, vLLM's prefix cache contains the full system-prompt prefix.

3. **Per-task measurement protocol changes.** Replace the existing `--reset-prefix-cache` only sequence with:
   - `POST /reset_prefix_cache` (truthful-measurement rule 5, unchanged) — clears all KV cache state.
   - Call warmup-pass executor — re-warms the system-prompt prefix into KV cache.
   - Run the actual Codex task — turn 1's prompt has ~100% cache hit on the system-prompt prefix; only the per-task ~150-300 token tail re-prefills cold.

4. **Cache state verification.** After warmup-pass executes, the runner pulls vLLM `/metrics`, computes `prefix_cache_hits / prefix_cache_queries` for the warmup pass, and asserts ≥ 95% hit rate on the **second** warmup invocation (first warmup primes the cache; second warmup verifies). If verification fails, the round is aborted with a measurement-protocol error.

**Why pre-warm and not just stop resetting:** the cache reset is in the truthful-measurement contract (saturation plan §8 rule 5) because cross-task cache contamination would let an agent cherry-pick wins by ordering tasks to inherit favorable cache state. Warmup-pass preserves the per-task isolation guarantee for the **task-specific tail** (the small portion that changes per task) while restoring the **static system-prompt prefix** to a known-warm state. Round-over-round comparison stays valid.

## 6. Codex system-prompt decomposition (measurement, not trimming)

New artifact per round: `output/track_b_e2e_v4a/round_<N>/codex_system_prompt_decomposition.json`

Schema:

```json
{
  "schema": "lumo.track_b.codex_system_prompt_decomposition.v1",
  "round": 4,
  "ts": "2026-05-10T...",
  "runtime_config_hash": "sha256:...",
  "codex_version": "0.128.0",
  "total_tokens": 69516,
  "total_chars": 287342,
  "content_hash": "sha256:...",
  "sections": [
    {"name": "system_role_preamble", "start_offset": 0, "end_offset": 1234, "tokens": 312},
    {"name": "tool_definitions", "start_offset": 1234, "end_offset": 192847, "tokens": 48201},
    {"name": "mcp_server_descriptions", "start_offset": 192847, "end_offset": 234109, "tokens": 10324},
    {"name": "sandbox_policy", "start_offset": 234109, "end_offset": 248732, "tokens": 3651},
    {"name": "model_provider_config", "start_offset": 248732, "end_offset": 287342, "tokens": 7028}
  ]
}
```

Section detection parses the captured system-prompt text against Codex CLI's section markers (XML-like tags or comment delimiters Codex emits — concrete markers identified during Phase 1 by direct inspection of one captured payload). Each section gets a token count via the same tokenizer the model uses (Qwen 3.5 tokenizer accessible via vLLM's tokenizer endpoint or a local `tokenizers`-library load).

**No content is removed or modified.** This is purely instrumentation. The decomposition lets us see, round-over-round, what's eating the 70K — and gives a future operator the data to make a trim decision if they want, without committing to one now.

## 7. Truthful-measurement contract additions

Per the saturation plan §8 numbering, three new rules:

| # | Rule | How to verify | Failure handling |
|---|---|---|---|
| 17 | Warmup-pass executed before each task attempt | Per-attempt `runner_metadata.json` records `warmup_pass_executed_at: <ts>`, `warmup_pass_cache_hit_rate: ≥0.95`, `warmup_pass_first_byte_ms: <N>` | Hard fail if absent or hit rate < 0.95; rerun |
| 18 | System-prompt decomposition recorded for round | `output/track_b_e2e_v4a/round_<N>/codex_system_prompt_decomposition.json` exists and validates against `lumo.track_b.codex_system_prompt_decomposition.v1` | Hard fail if absent at round close |
| 19 | Static system-prompt content_hash stable across round | All per-attempt warmup-pass calls record the same `system_prompt_content_hash`; mismatches indicate Codex CLI version drift mid-round | Round-level hard fail if any attempt's hash differs from the round's first |

Existing rule 5 ("Cache state reset before each run") amended: "Cache state reset, **then warmup-pass executed**, before each run. Cache reset clears all KV state; warmup-pass restores static system-prompt prefix to cache before per-task measurement begins."

## 8. Codex timeout investigation (parallel sub-step Q3)

In parallel with §5 implementation:

1. **Locate Codex CLI's request timeout config.** Inspect `~/.codex/config.toml`, `codex --help` flags, Codex environment variables. The Rust source under `codex-rs/core/src/client.rs` (read in earlier patch-surface audit) has the HTTP client; check its timeout setting.
2. **Bump it to a safe ceiling** — recommend 300 s (5 min) as defense-in-depth, well above the worst-case 90 s cold prefill we currently see plus headroom.
3. **Document the discovered config knob** in `docs/runbook/track-b-codex-timeout-config.md`.
4. **Verify timeout fix alone reduces zero-token rate.** Measure on `transcript-merge-regression/v1` with timeout bumped, no warmup-pass. Distinguishes the two contributing causes.

The combined warmup-pass + timeout-bump should drive zero-token rate to near-zero. Either alone may be sufficient; warmup-pass is the cleaner root-cause fix because it eliminates the slow prefill that triggers the timeout in the first place.

## 9. Acceptance criteria

The Round 4a sweep is the same 13-task sample as v3 (sample hash unchanged), measured under the new protocol:

1. **Warmup-pass cache hit rate** — verified ≥ 95% per truthful-measurement rule 17.
2. **Zero-token rate** — drops from v3's 65% to **≤ 5%** of attempts. (Threshold conservative; expect near-zero.)
3. **Median wallclock** — no acceptance threshold (this is a measurement-protocol fix, not an optimization). Recorded for reference.
4. **Decode share** — recorded for reference. Likely jumps materially because the dominant ~90 s cold-prefill cost is removed; expect 30-60% range.
5. **Task correctness preserved** — all 12-13 tasks still complete with `exit_code == 0`; aggregate milestone score within ±5% of v3 aggregate.
6. **System-prompt decomposition recorded** — per truthful-measurement rule 18.
7. **System-prompt content_hash stable** — per truthful-measurement rule 19.
8. **Codex timeout config documented** — `docs/runbook/track-b-codex-timeout-config.md` exists with the discovered knob and chosen value.

## 10. Sequencing

| Phase | Work | Estimated time |
|---|---|---:|
| 1 | System-prompt capture + decomposition driver. Inspect a captured `/v1/responses` request, identify section markers, build `scripts/build_track_b_codex_system_prompt_decomposition.py`. Validate against schema. | ≤ 2 h |
| 2 | Warmup-pass executor. `scripts/run_track_b_e2e_warmup.py` + integration into `run_track_b_e2e_task.py` (call warmup before each Codex spawn). Truthful-measurement rules 17-19 enforced in `run_track_b_e2e_round.py`. | ≤ 1 h |
| 3 | Codex timeout config investigation (parallel with Phase 1). Locate the timeout knob, bump to 300 s, document at `docs/runbook/track-b-codex-timeout-config.md`. | ≤ 1 h |
| 4 | Single-task smoke. Run `transcript-merge-regression/v1` × 4 attempts under the new protocol. Verify rules 17-19, decode share, zero-token rate. | ≤ 30 min |
| 5 | Flip `--zero-token-retries` default 0 → 3 in `scripts/run_track_b_e2e_round.py` after Phase 4 confirms warmup-pass eliminates the underlying cause. | ≤ 5 min |
| 6 | Full Round 4a baseline. 13 tasks × 4 attempts. Lands at `output/track_b_e2e_v4a/round_0/`. **Supersedes v3 as the canonical baseline** for Round 4b+ (drafter work). | ≤ 30 min |
| 7 | Short post-Round-4a session summary. Documents corrected mechanism, protocol fix, re-measured decode share + zero-token rate, system-prompt decomposition, implication that Round 1-3 wins are lower-bound on production impact. | ≤ 30 min |

Total: roughly 5-6 hours serial. Cheaper than the v4 prompt-shape work would have been, and addresses the actual mechanism.

## 11. What this does and doesn't claim

**This claims:** the dominant per-task wallclock cost in v3 measurements was a single cold turn-1 prefill on the ~70K-token Codex system-prompt prefix, amplified by a 65%-incidence Codex client-side timeout that paid full prefill cost without producing tokens. A warmup-pass that pre-caches the static system-prompt portion + a Codex timeout bump + zero-token-retry default-on eliminates the cost in measurement and approximates production cache warmth.

**This does not claim:** that wallclock will drop by any specific percentage. The ~90 s cold turn-1 cost will largely disappear from per-task numbers, but rebaselining against the protocol-fixed measurement is the point — not an optimization headline.

**This does not claim:** that the Codex system prompt should be trimmed. §6 records its decomposition for visibility; trim decisions are deferred to a future investigation (or to operator judgment, given the per-section data this produces).

**This does not claim:** the existing Round 1-3 wins are wrong. The −12.5% wallclock reduction was a valid measurement against an inflated baseline. Translating to a warmup-protocol baseline likely shows the same techniques landing a larger relative wallclock improvement, because the denominator drops when cold-prefill cost is amortized. The Round 1-3 ablation may be rerun against the Round 4a baseline once the protocol is stable, to compute the more realistic technique-attribution numbers.

## 12. Production parallel

The warmup-pass approximates the cache-warmth condition production would naturally have. In a production deployment serving many Codex agent sessions, every session starts with the same ~70K system-prompt prefix → vLLM's prefix cache absorbs it permanently. Cold turn-1 cost amortizes to ~zero across the second and subsequent sessions. The measurement protocol's per-task `--reset-prefix-cache` fights this — warmup-pass restores it, on a per-task scope that still preserves the truthful-measurement isolation guarantee for the per-task content tail.

The Round 4a measurement is therefore closer to production reality than v3 was, while preserving the round-over-round comparability that the truthful-measurement contract was designed to ensure.

## 13. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Codex CLI version drift mid-round changes the system-prompt content_hash | Truthful-measurement rule 19 hard-fails the round if hashes don't match across attempts |
| Warmup-pass cache primes the wrong prefix (e.g., prior round's stale system prompt) | Phase 2 implementation captures the system prompt fresh per round; rule 18 records the round's decomposition with the new content hash |
| Codex client times out faster than 300 s in some configurations we don't know about | Phase 3 investigation locates the actual config knob and chosen value is documented; rule 17 enforces ≥95% cache hit rate so prefill is fast regardless |
| Reducing cold prefill changes the regime acceptance distribution that Round 4b+ baselines against | Expected and intentional; Round 4a is the new baseline for Round 4b drafter work. Re-measure regime acceptance on the v4a artifacts before any drafter intervention. |
| Warmup-pass adds ~few-second overhead per task | Acceptable. Even a 5 s overhead per task × 52 attempts = 260 s, vs the ~3360 s of cold-prefill cost it eliminates (52 × ~65 s effective average). Net wallclock saving is ~12-15 minutes per round. |
| The static system prompt isn't actually static (e.g., includes a session id or timestamp) | Phase 1's content_hash check across two captured requests detects this immediately. If true, the "task-agnostic prefix" portion is shorter than the full system prompt — rule 18 records only the genuinely-static portion. |

## 14. References

- `track-b-e2e-swe-style-prompt-shape-spec-20260510.md` (retracted v4 spec; §11 retraction documents the predicate measurement)
- `track-b-e2e-agentic-saturation-plan-20260508-v2.md` (parent plan; §8 truthful-measurement contract receives rules 17/18/19)
- `track-b-round3-e2e-v3-closeout-20260510.md` (v3 baseline; superseded as canonical baseline once Round 4a baseline lands)
- `codex-harness-spec-decode-engineering-20260507.md` (engineering spec; Round 4b drafter work runs against the Round 4a baseline)
- `scripts/run_track_b_e2e_task.py:50–56` (`_write_prompt` — confirms only AGENTS.md + .scenario_variant are inlined; ~150 tokens)
- `scripts/run_track_b_e2e_task.py:569` (`--reset-prefix-cache` policy — the source of the cold-turn-1 cost)
- `scripts/run_track_b_e2e_task.py:768` (`--zero-token-retries` — flag exists but default is 0)
- `/tmp/track_b_e2e_proxy_capture/request_metrics.jsonl` (proxy capture used for the §1 predicate measurement)

---

*This spec defines Round 4a as a measurement-protocol fix that eliminates the cold-turn-1 prefill cost and the Codex zero-token quirk's contribution to wallclock variance, without changing any test content. Round 4b (drafter work — MTP test, reasoning-regime intervention, etc.) runs against the Round 4a baseline once it lands.*
