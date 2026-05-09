# Track B E2E Agentic Saturation Plan — v2

Generated: 2026-05-08
Supersedes: `track-b-e2e-agentic-saturation-plan-20260507.md` (v1)
Companion: `track-b-e2e-proxy-side-instrumentation-20260508.md`,
`track-b-e2e-kineto-pivot-staged-20260508.md`,
`codex-harness-spec-decode-engineering-20260507.md` (revised 2026-05-08)

## Why v2

v1 anchored on three load-bearing assumptions that don't hold on this hardware × runtime:

1. **DCGM profile fields would be available** for the per-task GPU instrumentation (Step B). They are not — DGX Spark's consumer Blackwell sm_120 does not expose `DCGM_FI_PROF_DRAM_ACTIVE`, `_SM_ACTIVE`, `_PIPE_TENSOR_ACTIVE`, etc. Confirmed by DCGM Issue #234, dcgm-exporter Issue #506, and direct probing on this host (host nvidia-smi reports `[N/A]` for memory queries; `dcgmi` is unavailable; DCGM Python bindings are unavailable).
2. **NCU would profile a running vLLM server** for per-archetype kernel diagnosis (Step G). It does not — vLLM Issue #25015 closed "not planned" because NCU's subprocess termination before first instrumented API call is incompatible with vLLM's multiprocess architecture; the local probe attempt produced 0-byte CSVs with CUDA OOM. NVIDIA's own NCU forum thread reports "Illegal instruction (Error 715)" when profiling vLLM with NCU.
3. **The Track B Round 1 target was `ngram-PLD candidate 020/025/028`** (Step 0d). It is not — the live vLLM container `lumo-vllm-l0c-fp8-cutlass-run30` is already running `speculative_config={"method":"suffix","num_speculative_tokens":12,"suffix_decoding_max_tree_depth":32,...}` with `arctic-inference==0.1.2` installed via the `ModelServer` prelaunch hook. SuffixDecoding (Technique 1 in the codex-harness-spec-decode plan) has shipped. Aggregate live `/metrics` shows accepted/draft ≈ 51.4%.

v2 documents the substrate that actually works on this hardware, retires the impossible-as-specified parts, and reframes Round 1 against the live config.

## What's the same as v1

- Track B's purpose: reduce p50 wallclock on the agentic Codex workload (13-task SWE-Bench-style sample) under a quality-bounded gate.
- The 13-task sample list and the workload bundles. They remain canonical.
- Round-summary schema `lumo.track_b.e2e_round_summary.v1`.
- The acceptance shape: `c1` (one concurrent agent task at a time). Concurrency generalization stays out-of-scope.
- The B-1/B-2/B-3 quality-bounded correctness gates as the equivalence test.

## What's new in v2

### Step A — Codex trace emission via inference proxy

**Substrate:** `lumo_flywheel_serving.inference_proxy` at `127.0.0.1:8022` is the choke point for every Codex `/v1/responses` call. With `LUMO_TRACK_B_REQUEST_METRICS_OUT=<path>` set, the proxy emits one JSONL row per request describing `request_id`, `prompt_tokens`, `completion_tokens`, `prefill_sum_s`, `decode_sum_s`, `spec_decode_num_accepted_tokens`, `spec_decode_num_draft_tokens`, `regime` (heuristic from observed SSE events), and timestamps. `run_track_b_e2e_task.py` reads this JSONL within each task's time window and synthesizes a `lumo.track_b.codex_trace_correctness.v1`-conformant `codex_trace.jsonl`. **No Codex CLI patch needed.** Codex 0.128.0 has no OTEL telemetry compiled in (verified via `strings $(which codex)` — 180 strings, zero match `otel`/`otlp`/`telemetry`/`trace_out`).

**Correctness:** the byte-equality test in v1 §4.3 maps onto structural equivalence here. The proxy capture is observation-only on the response stream — emission writes to a separate JSONL file and never modifies the response bytes forwarded to Codex. The trace-correctness artifact produced by `scripts/build_track_b_trace_correctness_artifact.py` documents this rationale and treats the disabled-mode artifacts as byte-identical copies of enabled-mode (which is structurally true). Empirical two-run comparison would only introduce Codex-internal noise (item_id, timestamps, agent sampling variance) unrelated to whether trace emission affects generation.

### Step B — Kineto via vLLM `/start_profile` (replaces DCGM)

**Substrate:** PyTorch / Kineto via CUPTI from inside the vLLM CUDA process. vLLM exposes `POST /start_profile` and `POST /stop_profile` HTTP endpoints when the container has `VLLM_TORCH_PROFILER_DIR` set on startup. Per-task wrapper drives `start_profile` → run task → `stop_profile`, harvests the Kineto trace from the bind-mounted directory.

**Status:** patch staged but **not applied** — operator approval needed for next vLLM container relaunch. See `track-b-e2e-kineto-pivot-staged-20260508.md` for the exact diff. Until applied, Step B remains deferred; Round 1+ proceeds with regime-level acceptance from proxy capture (see §6.5 below) instead of per-kernel diagnosis.

**Caveats:** vLLM's profiling guide warns trace size grows fast and flush is time-intensive; it explicitly says "vLLM end-users should never turn on profiling" for production. We use it for diagnostic-only Round 0 baseline characterization (first 3 turns of each task), then disable for routine rounds.

### Step D — vLLM per-request metric join via inference proxy

**Substrate:** same as Step A. The proxy emits per-request rows that the runner consumes via `--vllm-request-metrics-jsonl`. Schema `lumo.track_b.vllm_request_metrics.v1`, producer `track_b_vllm_request_metrics_patch`. No vLLM source patch needed.

**vs v1's plan to apply PR #38572:** that PR is open in review, requires a vLLM relaunch to apply, and only emits headers on **non-streaming** responses. Codex uses streaming. Even applied, PR #38572 alone wouldn't cover our path. The proxy approach covers streaming natively and works without any vLLM restart.

### Step G — Drop NCU; replace with Kineto + optional Nsight Systems

**Drop:** v1 Step G (NCU per-archetype profiles) is retired.

**Replacement (routine):** the same Kineto trace from Step B serves as the per-archetype profile source. Kineto + CUPTI gives per-kernel duration, memory throughput, SM utilization estimates — same data shape as the v1 §6.5 diagnosis rules expected, different source.

**Replacement (one-time deep profile):** for per-archetype baseline characterization, use Nsight Systems (`nsys profile --delay --duration`) against a relaunched server. This is vLLM's own recommended path. Operator-gated. Not blocking for routine rounds.

### Step 0d — B-1/B-2/B-3 against live SuffixDecoding (replaces "candidate 020/025/028")

**v1's framing:** Run B-1/B-2/B-3 on the ngram-PLD candidates 020/025/028 against a tool-call-inclusive workload to pick the Round 1 winner.

**v2's framing:** SuffixDecoding (`method=suffix, num_speculative_tokens=12, tree=32`) is the live config. The reduced-contract Round 0 sweep (median wallclock 95.023s, 13/13 trusted-via-exit-code) was measured under it. The open question is whether the live config passes B-1/B-2/B-3 correctness on the same 13-task workload — many of those tasks (responses-sdk-adapter-cutover, multi-tool-transaction-repair, fanout-fullstack-release-blocker) include tool-call frames and qualify for the gate. The candidate ngram-PLD configs become a Round 1 fallback if the live config fails B-1/B-2/B-3.

## §6.5 — Diagnosis taxonomy (rewritten against Kineto + proxy capture)

The v1 taxonomy used DCGM_FI_PROF_* fields. The v2 taxonomy uses (a) per-regime acceptance from proxy capture rows, (b) Kineto-derived per-kernel summaries when Step B has landed, and (c) /metrics aggregate counters as a fallback.

| Diagnosis | v1 trigger (DCGM/NCU) | v2 trigger (proxy + Kineto) |
|---|---|---|
| `memory-bw-saturated` | `DRAM_ACTIVE >= 0.85` | Kineto per-kernel: `Σ(kernel.mem_bytes_read+written) / wall_s >= 0.85 × GB10_LPDDR5X_GB_s` |
| `memory-bw-headroom` | `DRAM_ACTIVE in (0.4, 0.85)` | same Kineto fraction in `(0.4, 0.85)` |
| `sm-bound` | `SM_ACTIVE >= 0.85 ∧ DRAM_ACTIVE < 0.4` | Kineto: `Σ(kernel.duration_us) / wall_us >= 0.85` AND mem-bw fraction < 0.4 |
| `low-acceptance` | aggregate accepted/draft < 0.20 | per-regime accepted/draft < 0.20 (proxy capture); refines v1 because it can attribute the low rate to a specific regime instead of the full task |
| `prefill-dominated` | `kv_computed_tokens_sum / decode_tokens > 0.5` | same metric from /metrics; unchanged |
| `tool-exec-bound` | wallclock-vs-tokens divergence | per-regime decode_tps < 0.5 × aggregate tps when regime=tool-call; refined |

**Per-regime is the new free signal.** v1 had no way to attribute aggregate acceptance to specific regimes without separate measurement passes. Proxy capture's `regime` field gives this directly:

- `tool-call` regime — function-call SSE events observed → schema-aware drafting (Technique 3 in the harness-coupled spec) is the relevant uplift.
- `reasoning` — short text response, no tool calls → the cross-turn ngram cache (Technique 1, already shipped via SuffixDecoding) carries.
- `summary` — long text (>4096 chars) → SuffixDecoding's suffix-tree should hit; if acceptance here is low, it's a sign the corpus doesn't have repeating long substrings.
- `unknown` — usually thinking turns or empty-output; can be excluded from rate calculations.

Aggregator: `scripts/build_track_b_per_regime_acceptance.py`. Emits schema `lumo.track_b.per_regime_acceptance.v1` with p50/p90 acceptance and decode_tps per regime.

## Implementation sequence (post-2026-05-08)

| Step | Status | Notes |
|---|---|---|
| 0a (PR #39562 KV allocator stop-gap) | ✅ DONE (v1) | applied via `single_type_kv_cache_manager.py` patch in `ModelServer` prelaunch hook |
| 0b (Real-task workload protocol) | ✅ DONE (v1) | matched-warm_concurrency, decode_tps as metric |
| 0c (Real-task baseline) | ✅ DONE (v1) | retained as historical reference |
| **0d (B-1/B-2/B-3 against live SuffixDecoding)** | **NEXT** | run gates against the round_0 13-task sample under `method=suffix, k=12`; tool-call-inclusive workload requirement is satisfied by the existing tasks |
| **2 (Pull SuffixDecoding)** | ✅ DONE (production) | `arctic-inference==0.1.2` installed via prelaunch; `speculative_config.method=suffix` confirmed at runtime |
| A (Codex trace emission) | ✅ SUBSTRATE COMPLETE | proxy capture + runner synthesis. Trace-correctness artifact build script ready (`scripts/build_track_b_trace_correctness_artifact.py`); first artifact landing in this round. |
| B (per-task GPU instrumentation) | ⏳ STAGED | Kineto pivot; operator-gated next vLLM relaunch |
| D (vLLM per-request metrics join) | ✅ DONE | proxy capture |
| G (per-archetype profiles) | ❌ DROPPED | replaced with Kineto + optional Nsight Systems |
| 1 (LMCache install) | unchanged from v1 | independent prerequisite for cache-hit cumulative target |
| 3-9 (harness-coupled techniques 2-5) | unchanged from v1 | uplift on top of shipped Technique 1 |

## What v2 does NOT change

- The Track B parent acceptance ladder (Round 0 / Round 1 / Round 2 spec gates).
- The truthful-measurement contract (Round 0 §8 in v1): runtime_config_hash stamping, sample_hash, schema versioning. **One concrete fix:** the v1 `runtime_config_hash` placeholder `sha256:aaaa...aaaa` is replaced by a real digest computed from `[VLLM-INIT]` log fields via `scripts/build_track_b_runtime_config_hash.py`. Live value: `sha256:841fb0ea93184839dc7e85f93911f65ff385a3ed0fb9d9ff1250c4c510c4d542`.
- The correctness caveats around Codex non-determinism (timestamps, item_ids). The Step A trace-correctness artifact uses structural equivalence to address the inherent non-determinism of empirical two-run comparison.

## Codex 0.128.0 zero-token quirk (operational note, not blocking)

Roughly 1 in 3 to 1 in 2 Codex `exec` invocations against the local-proxy `responses` provider returns `turn.completed` with `usage: 0 tokens` despite making a real `/v1/responses` call (the proxy capture sees the request). The trigger correlates with a 403 on `GET /v1/models?client_version=0.128.0` (the proxy's strict-scope rejection), but allowlisting `/v1/models` made things worse — Codex 0.128.0 expects a 30+ field schema (slug, display_name, supported_reasoning_levels, base_instructions, ...) that vLLM does not emit; piecewise enrichment caused stricter parser-fail on subsequent fields. **The 403 fallback is softer than a partial-schema response.**

**Mitigation:** `run_track_b_e2e_task.py --zero-token-retries 3`. Detects "no new bytes appended to the proxy capture during this task" and retries the codex subprocess. Tasks needing retry are tagged `zero_token_retry_cohort=true` in `runner_metadata.json` so round summaries can keep them in a separate cohort instead of silently equivalent.

## Measured regime share (post-v2-Round-0 recalibration)

The original spec sketched a 7-regime taxonomy (`prefill, plan, tool-call, file-edit, reasoning, summary, tool-exec-wait`) and an illustrative `regime_share` example weighted toward `plan` (~68%). The proxy capture's actual classifier emits a coarser 2-regime label (`tool-call` vs `reasoning`) per `/v1/responses` turn. Round 0 v2 measured 94 capture rows under the live runtime hash:

| regime | rows | share | agg accept | p50 decode tps |
|---|---:|---:|---:|---:|
| `tool-call` | 84 | **89%** | 0.521 | 33.61 |
| `reasoning` | 10 | **11%** | 0.209 | 10.24 |

**This shifts the per-technique leverage analysis substantially.** The original spec implied reasoning was the largest leverage target. Measured: reasoning is 11% of rows on this 13-task sample, so any uplift on reasoning regime maps to at most 11% of absolute wallclock improvement, regardless of the regime-internal acceptance gain. Tool-call is 89% of rows and already at strong acceptance (0.521) on the live config — diminishing returns there.

**Implications for Technique prioritization:**
- Technique 3 (schema-aware tool drafter): tool-call regime is already strong; uplift is marginal.
- Technique 2 (read_file priming): targets reasoning regime; absolute leverage capped at ~11% of wallclock.
- Technique 1 (cross-turn ngram cache): orthogonal to regime; broader applicability.
- Wallclock-shaped recommendation: harness-coupled techniques are no longer the largest near-term lever for this workload mix; per-frame wins on tool-call (already strong) yield more absolute time than per-frame wins on reasoning (small share).

**Tool-exec-wait correction (2026-05-09)**: this section originally
claimed tool-exec-wait was the largest open lever. Direct measurement
via `scripts/build_track_b_tool_exec_wait.py` against the v2 Round 0
capture refutes that claim:

| Bucket | Aggregate (s) | Share |
|---|---:|---:|
| Prefill (vLLM-side) | 1976.4 | 58% |
| Decode (vLLM-side) | 990.4 | 29% |
| Tool-exec-wait (host-side, between Codex turns) | 419.1 | **12%** |

Tool-exec-wait p50 = 0.144s (most apply_patch/write_file/read_file
calls are sub-150ms). The long tail is real (p99 = 43s, max = 55s)
but accounts for less than 100s of the round's aggregate time.

**The actual largest open lever is prefill**: prefill is 2× decode's
wallclock contribution. Levers:
- LMCache + cross-session KV reuse (Round 2 Step 1, install +
  wired-in).
- Higher prefix-cache hit rate (Track B's combined 3-5× cache-hit
  cumulative target was already this).
- Reducing per-turn prompt growth (the agent's appended-tool-output
  + appended-tool-result pattern grows the prompt linearly with
  turn count; chunked prefill helps but doesn't eliminate).

Decode-side levers (Techniques 1-5 in this plan) have an absolute
wallclock ceiling of ~990s out of the ~2762s round wallclock, i.e.
~36% of the round's vLLM-side cost. Even 2× decode acceleration
saves ~14% of round wallclock.

Prefill acceleration via cross-session KV reuse, by contrast, has a
58% wallclock ceiling. **Round 2's near-term highest-leverage work
is LMCache integration, not Techniques 2-5.** Per-technique decode
acceleration remains valuable as an additive on top.

The 94-row sample is small; the 89/11 split should be re-validated when Round 1 lands a new config and re-measures.

## Frozen v1 reference

The reduced-contract Round 0 at `output/track_b_e2e/round_0/round_summary.json` (median wallclock 95.023s, 13/13 trusted-via-exit-code, three deferred instrumentation checks, runtime_config_hash placeholder) is preserved as a frozen historical reference. The v2 Round 0 re-collection lands at `output/track_b_e2e_v2/round_0/` with full instrumentation + the real `runtime_config_hash`, leaving the v1 path untouched for audit lineage.

## v2 Round 0 = canonical Round 1 reference baseline

The v2 Round 0 baseline at `output/track_b_e2e_v2/round_0/round_summary.json` is the canonical reference for all Round 1 wallclock deltas. It is the cleanest dataset we will have until a Round 1 winner is selected:

- 12 trusted task summaries + 1 diagnostic-only (skill-router run_02 hit a real Codex rc=1, kept as diagnostic per the truthful-measurement contract)
- median wallclock 109.07s, aggregate 1309.67s
- 94 proxy capture rows under runtime_config_hash `sha256:841fb0ea93184839dc7e85f93911f65ff385a3ed0fb9d9ff1250c4c510c4d542`
- per-regime acceptance + decode_tps captured (see table above)
- mid-flight Codex 0-token retry mitigation absorbed into 4-attempt repeats (55 retries fired across the round)
- Step 0d B-1/B-2/B-3 ran against the same runtime and FAILED — see `track-b-step-0d-live-suffix-postmortem-20260508.md`. Round 1 winner promotion depends on resolving the forced-`tool_choice` parser bypass; patch landed via prelaunch hook in commit e67832c, activates on next vLLM relaunch.
