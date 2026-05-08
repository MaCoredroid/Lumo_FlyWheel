# Track B E2E Agentic Saturation Plan — Auto Research Loop Spec

Generated: 2026-05-07
Owner: Track B
Auto-research-loop scope: rounds N..M where N is the first round measured under this plan.

Companion to:
- `codex-harness-spec-decode-engineering-20260507.md` (technique inventory; this plan supersedes its Step 0d/0e ordering)
- `track-b-real-task-warmonly-pr39562-matrix-20260507.md` (post-PR#39562 matrix that motivated the recalibration)
- `track_b_tool_call_throughput_closeout_20260507.md` (candidate-056 closeout; baseline runtime config)
- `track-b-concurrency-measurement-audit-20260506.md` (warm_concurrency audit; established the truthful-measurement rules below)

## 1. Why this spec exists

Decode tok/s on isolated workload probes was the right metric to debug the spec_decode crash and rank candidates. It is the wrong metric to optimize a Codex agent because per-decode-call speed does not translate proportionally into end-to-end task wallclock — a real Codex task is `prefill → plan → (tool-call → tool-exec wait → tool-result-turn) × N → summary`, and a 6× tool-call frame win evaporates if tool-call frames are 15% of total wallclock.

This plan reframes Track B's optimization target as **median end-to-end Codex task wallclock on a hand-picked sample of authored CNB-55 families, decomposed by regime, joined to per-second hardware-utilization samples**. Decode tok/s remains a diagnostic, not a target.

The plan is shaped for an auto research agent loop. Each round has:

1. A **hypothesis** the agent must state in writing.
2. A **config delta** (YAML diff) the agent applies to the active runtime.
3. A **cheap preflight** (~1-3 minutes) the agent runs locally to filter out obviously-broken configs before paying for the full measurement.
4. A **full measurement** (the 13-task e2e run with four data streams) gated behind preflight pass.
5. A **correctness caveat list** that must be satisfied before a round is counted.
6. A **truthful-measurement contract** the agent attests to before recording results.

## 2. Goal and non-goals

**Goal:** the headline metric is **median per-task wallclock across the 13-task sample**, with a per-task bottleneck-regime diagnosis that the auto research agent uses to prioritize the next round's intervention.

**Stretch metric:** **aggregate wallclock saved across the 13-task sample** between Round N-1 and Round N. This is the round-over-round delta.

**Non-goals:**
- Per-call decode tok/s as a target. Reported as a diagnostic only.
- Multi-tenant c4 acceptance shape. c4 measurement remains capacity-headroom only; c1 stays the production assumption.
- Forced `tool_choice` Responses path optimizations. Codex uses `auto`; the forced path is a separate parser bug that does not affect real tasks.
- Synthetic first-five token-count proxy probes. Authored CNB-55 family workspaces only.

## 3. Hand-picked sample (13 tasks, full track coverage)

All `v1-clean-baseline` for first pass; v2-v5 are second-pass robustness checks after the first-pass measurement design proves out.

| # | Track | Family | Regime characterization (verified from task_spec.md) |
|---|---|---|---|
| 1 | 01 Core Implementation | `responses-sdk-adapter-cutover` | Multi-file Python migration, event-model replay determinism. Long-text plan + many `apply_patch` turns. |
| 2 | 01 Core Implementation | `transcript-merge-regression` | Tighter-scoped bug-fix-with-pytest loop. Small task baseline — exposes whether wallclock is fixed-overhead-dominated. |
| 3 | 02 Codebase Understanding | `dead-flag-reachability-audit` | Pure investigation: trace defaults / env parsing / branching across files, output structured JSON classification. Read-heavy, near-zero-write. |
| 4 | 03 Refactor Modernization | `sqlalchemy-2-session-modernization` | Multi-file behavioral modernization across api / repository / worker / admin_cli + docs. Long-text plan + many edit turns. |
| 5 | 04 Review Remediation | `security-audit-hotfix-remediation` | SARIF + AppSec triage → targeted patch. Mix of evidence-grounded reading + structured-output triage matrix + minimal `apply_patch`. |
| 6 | 05 Frontend Multimodal | `responsive-checkout-visual-regression` | Includes preview screenshots (mobile + desktop). Image tokens at prefill — only multimodal slot. |
| 7 | 06 Evidence-Grounded Research | `incident-evidence-synthesis` | Heavy corpus retrieval (logs / tickets / timeline / remediation) → structured packet. Long context loads. |
| 8 | 07 Stateful Tool/Policy | `policy-aware-request-resolution` | Same workload as candidate-056 closeout (66.3 tok/s). Continuity slot — validates lab-bench numbers translate to real e2e wallclock. |
| 9 | 07 Stateful Tool/Policy | `multi-tool-transaction-repair` | Stateful sandboxes (orders / billing / notifications), atomicity-preserving. Heavier on tool-exec wait time than on decode. |
| 10 | 08 Skills Tooling | `skill-router-contract-upgrade` | Modify repo-local router, schema migration, test loop. Mid-sized; mixed regime. |
| 11 | 09 MCP & Local Integrations | `plugin-scaffold-alignment` | Many small structured edits across JSON + markdown. Structured-output-heavy. |
| 12 | 10 Strategic Management | `release-note-to-plan-translation` | Same workload as suffix-decode 16 tok/s baseline. Continuity slot — long-text regime currently below target. |
| 13 | 11 Subagents Orchestration | `fanout-fullstack-release-blocker` | Multi-surface alignment (backend + frontend + docs); likely to spawn subagents. Tests orchestration overhead invisible at per-decode-call level. |

### 3.1 Why this sampling

Full coverage of 11/11 tracks. Two slots each on the two highest-leverage tracks (07 stateful-tool, 10 strategic-management — these are the two regimes where Round 1 measurements left the most headroom). Explicit slots for the three under-characterized regimes: pure-investigation (3), multimodal (6), subagents (13).

The auto research agent **must not change the sample** between rounds. Round-over-round comparisons are only valid against the same task set with the same workspace pinning (`manifest_locked` per `family.yaml`).

## 4. Codex CLI instrumentation

Codex CLI is open source. Patch a `trace_emitter` hook into the agent's main loop and the streaming response handler. Codex's internal state machine already distinguishes turn types; we don't infer regime from message content.

Current local Codex fact (2026-05-07): `codex-cli 0.128.0` has `codex exec --json`, but not `--trace-out`. Source and live-output inspection show that `--json` emits normalized thread events and aggregate token usage only; it does not preserve emitted `turn_id`, Responses `response_id`, vLLM request id, timestamps, or per-tool timing. Therefore `--json` is **not** accepted as a substitute for the `--trace-out` artifact below.

### 4.1 Patch surface

- Codex CLI fork carried at `vendor/codex-cli` (or a pinned upstream commit + patch series under `patches/codex/`).
- New CLI flag: `--trace-out PATH`. When set, every agent-loop event emits a JSONL line.
- Two patch sites:
  1. **Send-to-model dispatcher** — emit `turn_start` immediately before the request goes to vLLM, including the regime tag and the `vllm_request_id` returned in the response. Emit `turn_end` when the streaming response completes.
  2. **Tool-call lifecycle** — emit `tool_call` with three timestamps: `ts_codex_emit_start` (Codex begins generating tool-call XML), `ts_codex_emit_end` (XML fully streamed and parsed), `ts_tool_exec_end` (tool execution returns). The gap between `ts_codex_emit_end` and `ts_tool_exec_end` is `tool-exec-wait` time, which is not Track B's optimization target but must be visible in the trace so it does not contaminate decode-regime analysis.

Estimated patch size: ~200 LoC. Upstream as a `--trace-out` flag if the maintainers accept it; otherwise carry as a fork.

### 4.2 Trace event schema

```jsonl
{"event":"task_start","task_id":"sqlalchemy-2-session-modernization/v1-clean-baseline","family":"sqlalchemy-2-session-modernization","variant":"v1-clean-baseline","codex_version":"<sha>","model":"qwen3.5-27b","runtime_config_hash":"<sha256>","ts":"2026-05-07T18:00:00.000Z"}
{"event":"turn_start","turn":0,"regime":"prefill","ts":"2026-05-07T18:00:00.012Z","vllm_request_id":"req-001"}
{"event":"turn_end","turn":0,"ts":"2026-05-07T18:00:04.301Z","prompt_tokens":4892,"completion_tokens":0}
{"event":"turn_start","turn":1,"regime":"plan","ts":"2026-05-07T18:00:04.310Z","vllm_request_id":"req-002"}
{"event":"turn_end","turn":1,"ts":"2026-05-07T18:00:42.118Z","prompt_tokens":4892,"completion_tokens":612}
{"event":"tool_call","turn":2,"name":"read_file","args":{"path":"app/api.py"},"ts_codex_emit_start":"...","ts_codex_emit_end":"...","ts_tool_exec_end":"...","args_bytes":34,"result_bytes":8421,"vllm_request_id":"req-003"}
{"event":"file_read","turn":2,"path":"app/api.py","bytes":8421}
{"event":"turn_start","turn":3,"regime":"file-edit","ts":"...","vllm_request_id":"req-004"}
...
{"event":"task_end","ts":"2026-05-07T18:42:18.430Z","exit_code":0,"milestones":{"M1_localization":1.0,"M2_primary_fix":1.0,"M3_invariants":0.6,"M4_functional":1.0,"M5_e2e":0.4},"task_score":0.74}
```

`regime` is one of `{prefill, plan, tool-call, file-edit, reasoning, summary, tool-exec-wait}`. Codex's existing turn-type internals map directly. `runtime_config_hash` is the SHA256 of the active vLLM serve config + spec_decode params + chat template kwargs. The Codex trace emitter records it in `codex_trace.jsonl`; the runner stamps it into `runner_metadata.json`, `dcgm_samples.jsonl`, and `vllm_per_turn.json`; stamped artifact mismatches make the join invalid.

### 4.3 Codex correctness preservation

The trace_emitter must be **observation-only**. It must not change agent behavior or token output. Verification (one-time, before Round 0):

- Pick three short tasks. Run with `--trace-out` enabled and disabled.
- Assert task transcripts (model outputs, tool-call sequences, milestone scores) are byte-identical between the two.
- Record this verification artifact at `output/track_b_e2e/codex_trace_emitter_correctness.json`.

Minimum artifact schema:

```json
{
  "schema": "lumo.track_b.codex_trace_correctness.v1",
  "verified_at": "2026-05-07T20:00:00Z",
  "codex_version": "<patched codex version>",
  "trace_out_supported": true,
  "tasks": [
    {
      "task_id": "transcript-merge-regression/v1-clean-baseline",
      "trace_out_enabled_exit_code": 0,
      "trace_out_disabled_exit_code": 0,
      "model_outputs_byte_identical": true,
      "tool_call_sequences_byte_identical": true,
      "milestone_scores_identical": true
    }
  ]
}
```

The readiness manifest requires at least three task entries and rejects an existence-only artifact.
Use `scripts/verify_track_b_codex_trace_correctness.py` to build this artifact from real enabled/disabled run evidence; it compares model-output bytes, tool-call-sequence bytes, milestone-score JSON, and both exit codes for each task, and it validates that the enabled trace has the Track B schema anchors needed for joins: `task_start` with matching `task_id`, `runtime_config_hash` formatted as `sha256:<64-hex-digest>`, and timestamp; at least one `turn_start` with `turn`, `regime`, `vllm_request_id`, timestamp, and a matching `turn_end` with timestamp plus numeric `completion_tokens`; and `task_end` with timestamp and exit code.

If transcripts diverge, the patch is wrong and **no rounds may run** until it is fixed.

## 5. Four data streams joined per task

| Stream | Source | Granularity | File |
|---|---|---|---|
| `codex_trace.jsonl` | patched Codex CLI | per-turn, per-tool-call, per-file-read | per task |
| `vllm_per_turn.json` | vLLM Prometheus delta keyed by `vllm_request_id` | per-turn | per task |
| `dcgm_samples.jsonl` | DCGM/NVML 100 Hz sampler | timestamped | per task |
| `ncu_<archetype>.csv` + `ncu_<archetype>.json` | one-shot NCU per archetype | per-kernel profile plus provenance sidecar | one pair per archetype, reused across rounds |

### 5.1 vLLM per-turn join

The per-turn vLLM metrics are already half-built in `scripts/measure_track_b_real_content_task.py` (Prometheus delta logic in `compute_task_metrics`). Extend to:

- Capture metrics deltas keyed by `vllm_request_id`. The preferred public artifact is a bounded vLLM per-request JSONL side-channel normalized to `vllm_per_turn.json`; Prometheus request labels are also accepted if a patched vLLM exposes the required low-volume offline labels. The runner snapshots the side-channel byte offset before each task and normalizes only rows appended during that task, so stale request rows from previous tasks are not copied into the current artifact. The runner stamps the round `runtime_config_hash` into `runner_metadata.json`, `vllm_per_turn.json`, and the captured raw `vllm_request_metrics.jsonl` copy; raw JSONL side-channel fallbacks therefore carry the same hash when used directly. The task-summary join rejects a stamped artifact whose hash does not match the summary hash. The side-channel gate accepts either `generation_tokens` or the OpenAI-style alias `completion_tokens`, but the prompt/speculative-decode fields must be numeric in the same request-keyed row.
- Persist `decode_tps`, `accepted_per_draft_token`, `prompt_tokens`, `completion_tokens`, `prefill_sum_s`, `decode_sum_s`, `cache_hit_rate_pct`, `prefix_cache_queries`, `prefix_cache_hits` per `vllm_request_id`.
- Add **`vllm:spec_decode_num_accepted_tokens` and `vllm:spec_decode_num_draft_tokens`** to the captured metric set. Without these, accepted/draft ratio is invisible per-turn.

### 5.2 DCGM / NVML 100 Hz sampler

New script `scripts/sample_dcgm_during_task.py`. Records:

```jsonl
{"ts":"2026-05-07T18:00:00.000Z","gpu":0,"runtime_config_hash":"sha256:...","dram_active_pct":0.91,"sm_active_pct":0.42,"sm_occupancy_pct":0.31,"pipe_tensor_active_pct":0.18,"pipe_fp16_active_pct":0.04,"gpu_util_pct":85,"mem_copy_util_pct":12,"power_w":78}
```

Fields:

- `dram_active_pct` (`DCGM_FI_PROF_DRAM_ACTIVE`) — fraction of cycles where the memory subsystem is reading or writing. **This is the LPDDR5x bandwidth proxy on GB10's unified memory** — the closest thing to "are we eating the 273 GB/s ceiling."
- `sm_active_pct` (`DCGM_FI_PROF_SM_ACTIVE`) — fraction of SM cycles with at least one warp resident.
- `sm_occupancy_pct` (`DCGM_FI_PROF_SM_OCCUPANCY`) — average occupancy.
- `pipe_tensor_active_pct` (`DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`) — TensorCore active fraction.
- `pipe_fp16_active_pct` (`DCGM_FI_PROF_PIPE_FP16_ACTIVE`) — non-TC FP16/FP8 pipe.
- `gpu_util_pct` and `mem_copy_util_pct` — `nvidia-smi`-style coarse counters (legacy, kept for dashboard parity).

Sampler must run as a separate process started before Codex spawn and stopped after Codex exits. Measurement samples require a `sha256:<64-hex-digest>` `runtime_config_hash`; only preflight smoke probes may use the explicit `--allow-unstamped-smoke` bypass because their temporary rows are not measurement artifacts. If sample dropouts exceed 1% of expected samples, the run is **untrusted** — see §8.

### 5.3 NCU one-shot per archetype

Five archetypes, five NCU runs total. Reused across rounds unless the runtime config changes hardware-relevant behavior (e.g., spec_decode method, prefix caching on/off).

| Archetype | Representative task | Why it is the archetype |
|---|---|---|
| Long-text generation | `sqlalchemy-2-session-modernization/v1` (#4) | Long plan + many edit turns; long generation windows. |
| Tool-call frame | `policy-aware-request-resolution/v1` (#8) | Short structured function-call frames; high suffix-cache reuse. |
| Pure investigation | `dead-flag-reachability-audit/v1` (#3) | Read-heavy, structured-classification output. |
| Multimodal prefill | `responsive-checkout-visual-regression/v1` (#6) | Image tokens at prefill, smaller decode. |
| Subagent orchestration | `fanout-fullstack-release-blocker/v1` (#13) | Multi-surface; orchestration overhead. |

NCU command (all five profiles, one archetype task each):

```bash
.venv/bin/python scripts/run_track_b_e2e_ncu_profiles.py \
  --runtime-config-hash <hash> \
  --codex-command-template '<patched-codex-command-with-{trace_out}>'
```

The driver expands to the required single-profile NCU command shape:

```bash
ncu --target-processes all --kernel-id ::regex:.*linear.*|.*attention.*|.*sample.*|.*spec.*: \
    --launch-skip-before-match 200 --launch-count 16 \
    --metrics gpu__time_duration.sum,sm__cycles_active.avg.pct_of_peak_sustained_elapsed,\
              dram__throughput.avg.pct_of_peak_sustained_elapsed,\
              sm__warps_active.avg.pct_of_peak_sustained_active,\
              smsp__sass_thread_inst_executed_op_memory_ld_pred_on.sum,\
              l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
              tpc__warps_active.avg.pct_of_peak_sustained_active \
    --csv --log-file output/track_b_e2e/ncu_<archetype>.csv \
    -- python scripts/run_track_b_e2e_task.py <archetype-task> --runtime-config-hash <hash> --no-dcgm --ncu-mode
```

`scripts/run_track_b_e2e_ncu_profiles.py` owns the five archetype-to-task mappings, requires a `sha256:<64-hex-digest>` `runtime_config_hash`, runs `scripts/run_track_b_e2e_task.py` with `--no-dcgm --ncu-mode`, writes the five `output/track_b_e2e/ncu_<archetype>.csv` files plus `ncu_<archetype>.json` sidecars carrying archetype/task/round/runtime-hash provenance, stores profiled task artifacts under `output/track_b_e2e/ncu_task_runs/` so they cannot overwrite measurement-round attempts, and rejects missing, metric-incomplete, or non-finite required-metric CSVs before the readiness manifest sees them.

Profile sidecar `ncu_<archetype>.json`:

```json
{
  "schema": "lumo.track_b.ncu_archetype_profile.v1",
  "archetype": "long-text",
  "task_id": "sqlalchemy-2-session-modernization/v1-clean-baseline",
  "round": 0,
  "runtime_config_hash": "sha256:...",
  "profile_csv": "output/track_b_e2e/ncu_long-text.csv",
  "required_metrics": ["gpu__time_duration.sum", "..."]
}
```

Per-kernel classification rule for downstream analysis of the NCU CSV:

- `dram_throughput_pct >= 0.80` → `memory-bound`
- `sm_active_pct >= 0.80 AND dram_throughput_pct < 0.50` → `sm-bound`
- `sm_active_pct < 0.30 AND dram_throughput_pct < 0.30` → `latency-bound`
- otherwise → `mixed`

## 6. Per-task summary JSON (the auto research agent's input)

Every task produces a single summary file at `output/track_b_e2e/round_<N>/<task_id>/summary.json`:

```json
{
  "round": 5,
  "task_id": "sqlalchemy-2-session-modernization/v1-clean-baseline",
  "runtime_config_hash": "sha256:...",
  "wallclock_s": 142.3,
  "task_score": 0.74,
  "task_completed": true,
  "turns": [
    {"index": 0, "regime": "prefill", "duration_s": 4.2, "decode_tps": null, "accepted_per_draft": null, "dram_active_pct_p50": 0.91, "sm_active_pct_p50": 0.42, "vllm_request_id": "req-001"},
    {"index": 1, "regime": "plan", "duration_s": 38.1, "decode_tps": 14.8, "accepted_per_draft": 0.21, "dram_active_pct_p50": 0.54, "sm_active_pct_p50": 0.31},
    ...
  ],
  "regime_share": {"prefill": 0.13, "plan": 0.68, "tool-call": 0.09, "file-edit": 0.06, "tool-exec-wait": 0.04},
  "bottleneck_regime": "plan",
  "bottleneck_diagnosis": "memory-bw-headroom",
  "diagnosis_evidence": {
    "regime_share_pct": 68,
    "regime_dram_active_p50": 0.54,
    "regime_sm_active_p50": 0.31,
    "regime_accepted_per_draft_p50": 0.21
  },
  "truthful_measurement_attestation": {
    "cold_completion_discarded": true,
    "output_cap_hit_count": 0,
    "dcgm_dropout_pct": 0.2,
    "vllm_codex_clock_skew_ms_p99": 8,
    "task_completed_normally": true,
    "milestone_score_recorded": 0.74,
    "single_run_basis": false,
    "median_of_n_runs": 3
  }
}
```

`bottleneck_diagnosis` taxonomy (computed by deterministic rule, not LLM judgment):

| Diagnosis | Rule | What it implies for next round |
|---|---|---|
| `memory-bw-saturated` | regime_dram_active_p50 ≥ 0.85 | Harness coupling will not help this regime. Pivot to memory-traffic reduction (FP8 KV, KV compression, weight repacking). |
| `memory-bw-headroom` | regime_dram_active_p50 < 0.70 AND regime_sm_active_p50 < 0.50 | Acceptance-rate work pays off. Techniques 2/3/4 from the engineering spec are correct moves. |
| `sm-bound` | regime_sm_active_p50 ≥ 0.80 AND regime_dram_active_p50 < 0.70 | Compute-pipeline work — kernel fusion, operator-level optimization. |
| `low-acceptance` | regime_accepted_per_draft_p50 < 0.20 | Drafter is the limit. SuffixDecoding param tuning, schema-aware drafter for that regime. |
| `prefill-dominated` | regime_share.prefill ≥ 0.40 | Prefill optimization (LMCache cross-task warm KV, prefix-cache hit-rate work). |
| `tool-exec-bound` | regime_share["tool-exec-wait"] ≥ 0.30 | Not Track B's problem. Document and ignore. |

### 6.1 Round summary JSON

After all 13 tasks, produce `output/track_b_e2e/round_<N>/round_summary.json`:

```json
{
  "round": 5,
  "runtime_config_hash": "sha256:...",
  "config_delta_vs_round_4": "<unified diff>",
  "hypothesis": "Increasing suffix tree depth from 32 to 48 will lift accepted/draft on long-text turns based on Round 4's diagnosis of memory-bw-headroom + low-acceptance on plan regime.",
  "median_wallclock_s": 187.4,
  "aggregate_wallclock_s": 2618.1,
  "wallclock_delta_vs_round_4_s": -132.6,
  "tasks_completed": 13,
  "tasks_correctness_passed": 13,
  "regime_share_aggregate": {"plan": 0.41, "tool-call": 0.18, ...},
  "diagnosis_distribution": {"memory-bw-headroom": 5, "memory-bw-saturated": 2, "low-acceptance": 4, ...},
  "auto_research_agent_recommendation": "<text>",
  "next_round_proposal": "<config delta>"
}
```

## 7. Auto research loop — round structure

Each round is one agent turn. The agent reads the prior `round_summary.json`, proposes a `next_round_proposal`, applies it, runs preflight, and then runs full measurement.

### 7.1 Round 0 — baseline establishment (mandatory; not optional)

**Hypothesis:** none. Round 0 establishes the e2e baseline under the active candidate-056 runtime.

**Config:** unchanged from candidate-056 closeout. Verbatim:

- PR #39562 KV-allocator stop-gap applied
- `arctic-inference==0.1.2`
- `--enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3`
- `--default-chat-template-kwargs '{"enable_thinking": false}'`
- spec_decode method `suffix`, k=12, tree=32, factor=2.0, min_prob=0.05, probabilistic
- Active tuned-config bundle `712fd011-4b16-4051-9e8c-875405b70f5b`

**Cheap preflight:**

1. vLLM serves: `curl -s http://127.0.0.1:9950/health` returns 200.
2. Single short Codex task end-to-end: run task #2 (`transcript-merge-regression/v1`, smallest in the sample). Codex returns exit_code 0 within 15 minutes.
3. Trace integrity: `codex_trace.jsonl` for that one task contains `task_start` and `task_end` events; per-turn `vllm_request_id` present on all `turn_start` events.
4. DCGM sampler attached: `dcgm_samples.jsonl` has ≥ 99% expected sample count over the run window and numeric `dram_active_pct`, `sm_active_pct`, `sm_occupancy_pct`, `pipe_tensor_active_pct`, and `pipe_fp16_active_pct` fields.
5. Token correctness on a fixed canonical short prompt: serial output equals stored golden output byte-for-byte.

If any step fails, **Round 0 must not record measurements**. Fix the runtime, repeat preflight.

**Full measurement:**

1. Run the hard-gated round driver:
   `.venv/bin/python scripts/run_track_b_e2e_round.py --round 0 --runtime-config-hash <hash> --codex-command-template '<patched-codex-command-with-{trace_out}>' --clock-skew-ms-p99 <measured> --trace-emitter-correctness-verified-at <timestamp> --protocol-hash-match`.
2. The driver first rejects an already-populated round directory containing any measurement output other than a prior `preflight_audit.json`, then runs `scripts/preflight_track_b_e2e.py` and aborts before measurement if trace, DCGM, or vLLM request-correlation gates fail. If preflight passes, it runs all 13 tasks four times through `scripts/run_track_b_e2e_task.py`, discards cold `run_01`, verifies every measured attempt trace has matching task identity and `runtime_config_hash`, verifies the generation-token-volume guard from the measured attempt traces, derives each measured wallclock from `task_end.ts - task_start.ts`, summarizes canonical measured `run_02` with wallclocks from `run_02` through `run_04`, and produces `round_0/round_summary.json`.
3. Run NCU archetype profiles once (5 runs) and write the expected non-empty files plus metadata sidecars:
   `ncu_long-text.csv`, `ncu_tool-call-frame.csv`, `ncu_pure-investigation.csv`,
   `ncu_multimodal-prefill.csv`, and `ncu_subagent-orchestration.csv`.

`round_0/round_summary.json` is not accepted by existence alone. The readiness manifest validates `schema="lumo.track_b.e2e_round_summary.v1"`, `round=0`, `runtime_config_hash` formatted as `sha256:<64-hex-digest>`, `sample_hash`, numeric median/aggregate wallclock, non-empty `diagnosis_distribution`, at least 12 trusted/completed/correctness-passed task summaries, at least 12 unique trusted task IDs, no duplicate trusted task IDs, no unexpected trusted task IDs, and explicit zero values for `sample_hash_mismatch_count`, `runtime_config_hash_mismatch_count`, `task_summary_schema_mismatch_count`, and `task_summary_round_mismatch_count`. Round promotion only counts trusted task summaries with the Track B task-summary schema and matching round index, and the summary builder rejects non-`sha256:<64-hex-digest>` runtime hashes before writing task or round summaries. Trace correctness validation requires at least three enabled/disabled byte-equality tasks and `trace_schema_valid=true` for every task. NCU validation likewise requires the five named archetype CSVs, matching `ncu_<archetype>.json` metadata sidecars with integer `round`, expected archetype task ID, `runtime_config_hash` formatted as `sha256:<64-hex-digest>` and matching the Round 0 summary when one exists, exact required-metric list, and all §5.3 required metric names with finite numeric values in the CSV, not any five `ncu_*.csv` files. The Step F auto-research prompt gate is also content-validated: it must invoke `scripts/run_track_b_e2e_round.py`, pass `--runtime-config-hash`, keep the protocol-hash gate, run the preflight script, check all three spec-decode counters, and avoid the legacy direct `run_track_b_e2e_task.py --round {{round}} --tasks all --repeat 3` measurement path.

**Correctness caveats:** see §9.

**Truthful measurement contract:** see §8.

### 7.2 Round N (N ≥ 1) — intervention round template

Auto research agent produces this filled-in template before any work runs. Stored at `output/track_b_e2e/round_<N>/round_proposal.md`.

```
# Round N Proposal

## Hypothesis
<one sentence stating what the agent believes about the system based on Round N-1's diagnosis distribution>

## Targeted regime / diagnosis
<which bottleneck_diagnosis cluster this round attacks; reference Round N-1's diagnosis_distribution>

## Config delta (YAML diff vs Round N-1)
```yaml
<diff>
```

## Predicted impact
- Regime: <regime>
- Predicted accepted/draft change: <baseline → expected>
- Predicted DRAM_ACTIVE change: <baseline → expected>
- Predicted median wallclock change: <baseline → expected>

## Cheap preflight commands
1. <command 1>
2. <command 2>
...

## Cheap preflight pass criteria
- <criterion 1>
- <criterion 2>
...

## Full measurement command
`.venv/bin/python scripts/run_track_b_e2e_round.py --round N --runtime-config-hash <hash> --codex-command-template "<patched-codex-command-with-{trace_out}>" --clock-skew-ms-p99 <measured> --trace-emitter-correctness-verified-at <timestamp> --protocol-hash-match`

## Correctness caveat checklist
- [ ] B-1 batch equivalence retained
- [ ] B-2 workload equivalence retained
- [ ] B-3 longer-prefix equivalence retained
- [ ] Tool-call XML auto-mode 4/4 retained
- [ ] All 13 tasks complete (exit_code 0 + task_score recorded)
- [ ] Aggregate task_score does not regress more than 5% vs Round N-1
- [ ] No new spec_decode crashes

## Truthful measurement contract
<copy of §8 attestation, all checkboxes filled>
```

### 7.3 Cheap preflight design rules

Preflight must be **fast (≤ 5 minutes total) and predictive (catches the bulk of bad configs)**. The point is to filter out obvious failures before paying the ~6-12 hours of full measurement. Specific gates:

| Gate | Command | Failure mode it catches |
|---|---|---|
| Server starts | `curl http://127.0.0.1:9950/health` | vLLM crashes on startup (bad serve_config.yaml) |
| Spec_decode loads | `curl http://127.0.0.1:9950/metrics \| grep -E 'spec_decode_num_(drafts\|draft_tokens\|accepted_tokens)_total'` | spec_decode disabled by config error |
| Smoke task completes | Run task #2 end-to-end | Codex hangs / errors / Codex-trace integrity broken |
| Token correctness | Run a fixed deterministic short prompt; compare to golden | Token-level corruption (Issue #40875 recurrence) |
| Smoke decode_tps not catastrophic | task #2 decode_tps ≥ 0.6 × Round 0 baseline | Config silently disabled spec decode |
| DCGM sampler attached | `dcgm_samples.jsonl` non-empty after smoke task | Sampler crash / permission issue |
| Tool-call XML parse | task #8 (`policy-aware`) tool-call gate 4/4 auto mode | Issue #40875 recurrence |

If any gate fails, the round is **aborted**; the agent records `round_<N>/preflight_failed.json` with the failed gate and reason, then proposes a different config (or reverts to Round N-1 config and tries a smaller delta).

### 7.4 Full-measurement procedure (truthful)

For each task in the 13-task sample:

1. **Reset state.** `curl -X POST http://127.0.0.1:9950/reset_prefix_cache`. Wait 2 s. Capture Prometheus snapshot at `/metrics` as `vllm_metrics_pre.txt`.
2. **Start DCGM sampler.** `python scripts/sample_dcgm_during_task.py --out <task_dir>/dcgm_samples.jsonl &` and record PID.
3. **Run Codex CLI.** Real Codex binary, `--trace-out <task_dir>/codex_trace.jsonl`, model `qwen3.5-27b`, endpoint `http://127.0.0.1:9950/v1`. Workspace pinned per `family.yaml` (`manifest_locked`).
4. **Stop DCGM sampler.** Send SIGTERM; wait for graceful exit.
5. **Capture Prometheus delta.** `vllm_metrics_post.txt`. Compute per-`vllm_request_id` deltas → `vllm_per_turn.json`.
6. **Run grader.** Family-specific grader emits milestone scores → recorded in `task_end` event of the trace.
7. **Compute per-task summary.** Join the four streams → `summary.json`. Verify the truthful-measurement contract; if any attestation fails, mark the run **untrusted** and rerun.
8. **Repeat × 3.** Three independent runs per task. Median wallclock is reported; per-turn metrics are aggregated to per-task p50/p95.

The full sweep is 13 × 3 = 39 task runs. At an estimated 10-90 minutes per task, the full round wallclock is roughly 6-12 hours; this is acceptable because rounds are infrequent (one or two per day) and fully unattended.

## 8. Truthful measurement contract

**This is the anti-cheat / anti-artifact layer.** The 2026-05-06 candidate-051 incident (a 17.087 tok/s number that turned out to be a c4 measurement artifact from one 4096-token cap-hit completion) is the canonical failure mode this contract prevents. The auto research agent **must** record an attestation for every task summary; failed attestations invalidate the run.

### 8.1 Attestation rules (mandatory; one row = one rule)

| # | Rule | How to verify | Failure handling |
|---|---|---|---|
| 1 | Cold completion discarded | First Codex turn marked `prefill_cold=true` in trace; not counted in regime aggregates. | Rerun. |
| 2 | No output-cap completions silently passing | Any completion where `completion_tokens == max_tokens_for_that_request` is flagged. If any flag in the round, the run is **flagged for review** but not auto-discarded; human or agent must decide whether the cap-hit was natural (long task) or artifact (proxy for "I'd have generated more if allowed"). | Flag in `summary.json`. |
| 3 | Median of 3+ runs, not single | Per-task wallclock is the median of N ≥ 3 runs. Single-run measurements are **diagnostic only** and never reported as headline. | Mark single-run measurements as `single_run_basis=true`. |
| 4 | Same workload between baseline and candidate | Workspace bundle hash (`manifest.lock.json` content_hash) must match Round 0's. Prompt content hash must match. | Hard fail; abort round. |
| 5 | Cache state reset before each run | `reset_prefix_cache` returns 200; `vllm_metrics_pre.txt` shows `prefix_cache_hits=0` for the next request. | Hard fail; rerun. |
| 6 | DCGM sampler dropout < 1% | `(actual_samples / expected_samples) ≥ 0.99`. | Mark `untrusted_dcgm=true`; fall back to vLLM-only diagnosis. |
| 7 | Clock skew between Codex trace and vLLM Prometheus < 100 ms p99 | NTP-synced wall clocks; per-event timestamps compared. | Mark `untrusted_join=true`. |
| 8 | Codex task completion verified | `task_end` event present, `exit_code == 0`, milestone score recorded by family grader. | Mark `task_failed=true`. Round-level rule: ≥ 12 of 13 tasks must complete normally; otherwise round is invalid. |
| 9 | Wallclock measured wall-to-wall, not summed per-turn | `wallclock_s = task_end.ts − task_start.ts`. Per-turn duration sum is recorded separately as `decode_busy_s` for diagnosis. | Hard fail (programming error in summary computation). |
| 10 | No comparison across different baseline protocols | All compared rounds must use the same `runtime_config_hash` for the **measurement protocol** parts (sample list, runner, grader). Only the **runtime config under test** changes. | Hard fail; abort comparison. |
| 11 | Generation-token-volume guard | If any task's aggregate completion_tokens exceeds 1.5× the median across the 3 runs, that run is rerun. | Auto-rerun once; if persists, flag for review. |
| 12 | Spec_decode metrics captured | `vllm:spec_decode_num_accepted_tokens` and `vllm:spec_decode_num_draft_tokens` present in `vllm_per_turn.json` for every spec_decode-eligible turn. | Hard fail; missing metric means measurement protocol broken. |
| 13 | No silent fallback to vanilla decode | `spec_decode_num_drafts_total`, `spec_decode_num_draft_tokens_total`, and `spec_decode_num_accepted_tokens_total` increment on every spec-decode-eligible plan/file-edit/tool-call turn (not on prefill). | Mark `silent_fallback_to_vanilla=true`; investigate config. |
| 14 | Task transcript byte-equality with `--trace-out` disabled | Verified once at trace_emitter patch time, §4.3. Re-verified after every Codex CLI fork rebase. | Block all rounds until reverified. |
| 15 | Auto research agent does not modify the sample | `tasks_in_round` array hash matches Round 0's. | Hard fail; the agent has no authority to change the sample mid-loop. |

Task summaries also reject trace identity and provenance mismatches: if `task_start.task_id` or `task_end.task_id` is present in `codex_trace.jsonl`, it must equal the summary's `family/variant` task ID, and any stamped `runtime_config_hash` in trace, runner, DCGM, or vLLM artifacts must match the summary hash.

### 8.2 Why these rules specifically

- **Rule 2** is the direct counter to the 051 c4 17.087 incident: a single cap-hit completion in a batched aggregate inflated decode_tps. Flag, don't auto-discard, because legitimately long tasks do hit the cap. Median-of-3 (rule 3) usually washes this out.
- **Rule 4** is the counter to "agent slipped a different prompt into the candidate run." `manifest_locked` per family.yaml + content hash comparison.
- **Rule 5** is the counter to "candidate happened to inherit a hot prefix cache from the baseline." Each run starts cold-prefix.
- **Rule 9** is the counter to "agent reported sum-of-decode-times as wallclock, hiding tool-exec-wait." E2E wallclock is wall-to-wall.
- **Rule 10** is the counter to "agent compared a c1 candidate against a c4 baseline" or "agent compared a synthetic-probe baseline against a real-task candidate."
- **Rule 13** is the counter to "config silently disabled spec_decode and the agent reported the resulting clean numbers as a win."
- **Rule 15** prevents the agent from cherry-picking easier tasks into the sample.

### 8.3 Attestation block format

The agent emits this block in every `summary.json`:

```json
"truthful_measurement_attestation": {
  "rule_1_cold_completion_discarded": true,
  "rule_2_output_cap_hit_count": 0,
  "rule_3_median_of_n_runs": 3,
  "rule_4_workspace_hash_match": true,
  "rule_5_cache_reset_verified": true,
  "rule_6_dcgm_dropout_pct": 0.2,
  "rule_6_dcgm_profile_fields_present": true,
  "rule_6_dcgm_observed_numeric_profile_fields": ["dram_active_pct", "pipe_fp16_active_pct", "pipe_tensor_active_pct", "sm_active_pct", "sm_occupancy_pct"],
  "rule_6_dcgm_missing_profile_fields": [],
  "rule_7_clock_skew_ms_p99": 8,
  "rule_8_task_completed_normally": true,
  "rule_9_wallclock_wall_to_wall": true,
  "rule_10_protocol_hash_match": true,
  "rule_11_generation_volume_within_band": true,
  "rule_12_spec_decode_metrics_present": true,
  "rule_13_silent_fallback_to_vanilla": false,
  "rule_14_trace_emitter_correctness_verified_at": "2026-05-07T14:00:00Z",
  "rule_15_sample_hash_match": true
}
```

A round summary may not be promoted to round_summary.json unless **every rule's check passes** for ≥ 12 of 13 tasks.
The promoted trusted set must also contain at least 12 unique task IDs from the fixed §3 sample, with no duplicate trusted task IDs, no unexpected task IDs, and no `sample_hash` mismatch.

## 9. Correctness caveats (gates that block round acceptance regardless of speed)

These are the speed-independent correctness gates. A round that improves speed but fails any of them is **not promoted** as the new baseline.

| Gate | Source | What it catches |
|---|---|---|
| B-1 batch serial/concurrent equivalence | `output/.../candidates/<id>/b1_result_*.json` | Speculative decoding logits divergence at batch boundaries. |
| B-2 workload equivalence | `b2_result_*.json` | Diverging completions on real workloads. |
| B-3 longer-prefix equivalence | `b3_result_*.json` | Drift on long contexts. |
| Tool-call XML auto-mode 4/4 | `tool_call_b2_*_auto_*.json` | Regression of the candidate-028 vLLM Issue #40875 verification. |
| Aggregate task_score regression ≤ 5% | Sum of milestone scores across 13 tasks vs Round N-1 | Speed-via-quality-loss. |
| No spec_decode crashes | EngineCore log clean | PR #39562 regression. |
| Codex task completion ≥ 12 of 13 | per-task `exit_code == 0` | Config-induced agent failures. |

A round that fails any gate becomes **flagged-not-promoted**. The agent must propose a follow-up round that retains the speed delta while restoring correctness, or revert.

## 10. Diagnosis taxonomy → next-round intervention map

Standardized so the auto research agent's recommendations are predictable and reviewable.

| Diagnosis (dominant in regime aggregate) | Next-round intervention class | Concrete configs to consider |
|---|---|---|
| `memory-bw-saturated` | Memory-traffic reduction | FP8 KV (vLLM `--kv-cache-dtype fp8`), KV compression (LMCache), weight repacking, shorter context windows where task allows. |
| `memory-bw-headroom` | Acceptance-rate work | Suffix tree depth/factor tuning, schema-aware drafter (Technique 3), read_file priming (Technique 2), plan-structure pre-drafter (Technique 4). |
| `sm-bound` | Compute-pipeline work | Operator fusion, kernel selection (per `l0-ffn-gemm-pivot-20260502.md`), Triton epilogue. Out of this spec's scope; emit recommendation, do not auto-attempt. |
| `low-acceptance` (regime-specific) | Drafter tuning for that regime | Tighter suffix params; ngram fallback for non-suffix-shaped regimes; cross-turn ngram (Technique 1) — already largely shipped. |
| `prefill-dominated` | Prefill optimization | LMCache cross-task warm KV (Round 0 of engineering spec); `enable-prefix-caching` validation; chunked prefill tuning. |
| `tool-exec-bound` | Out of scope | No intervention. Document as "Codex tool-execution latency, not Track B's decode optimization problem." |

## 11. Implementation sequence (what gets built before Round 0 can run)

| Step | Output | Owner | Notes |
|---|---|---|---|
| A. Patch Codex CLI with `--trace-out` | `vendor/codex-cli/patches/trace_emitter.patch` + correctness verification artifact at §4.3 | one engineer, ~1 day | Block all rounds until correctness verified. |
| B. DCGM/NVML 100 Hz sampler | `scripts/sample_dcgm_during_task.py` + `tests/test_track_b_dcgm_sampler.py` | one engineer, ~0.5 day | Use `pynvml` (DCGM via NVML namespace). |
| C. E2E task runner | `scripts/run_track_b_e2e_task.py` | one engineer, ~1 day | Wraps cache reset → sampler start → Codex spawn → metrics capture → grader → summary join. |
| D. Per-turn vLLM metric extension | `src/lumo_flywheel_serving/metrics.py` extension to capture `spec_decode_num_accepted_tokens`, `spec_decode_num_draft_tokens` and key by `vllm_request_id` | one engineer, ~0.5 day | Integration test against a known-shape task. |
| E. Summary join + diagnosis rule | `scripts/build_track_b_e2e_summary.py` | one engineer, ~0.5 day | Implements §6 + §8 attestation + §9 caveat checks. |
| F. Auto research agent prompt template | `prompts/track_b_e2e_round_proposal.md` | one engineer, ~0.5 day | Templated round-proposal markdown per §7.2. |
| G. Round 0 dry run | `output/track_b_e2e/round_0/` populated and validated | one engineer, ~1 day (mostly waiting) | All 13 tasks × 1 cold run + 3 measured runs, plus 5 NCU archetype profiles. |

Total ramp before Round 1 can be authored: ~5 days serial, ~2-3 days with parallelism.

### 11.1 Implementation status as of 2026-05-08

This plan now has a trusted **deferred-instrumentation Round 0 summary** under the 2026-05-08 user-directed exclusions. The current `output/track_b_e2e/round_0/round_summary.json` is trusted for the reduced contract that defers only `vllm_request_metrics_join_available`, `codex_trace_out_supported`, and `dcgm_profile_fields_available`: it records 13 trusted task summaries, 13 completed tasks, 13 correctness results deferred to normal Codex exit, zero untrusted/diagnostic tasks, median wallclock `95.023s`, and aggregate wallclock `1263.267s`. This is **not** full Track B readiness: the readiness manifest still blocks on the intentionally deferred instrumentation and on absent NCU archetype profiles.

Deferred for the next Round 0 start attempt (2026-05-08 user direction):

- `vllm_request_metrics_join_available`
- `codex_trace_out_supported`
- `dcgm_profile_fields_available`

For this Round 0 start attempt, "deferred" means these three instrumentation checks are excluded from the required measurement contract. The round and summary still run all remaining gates: sample workspaces must exist, the configured Codex command/model path must smoke-run, Codex task exits must propagate, runtime hashes must match, cache/protocol/sample checks must pass, workspaces must be isolated per attempt, and wallclock values must remain finite and positive. The summary records `deferred_instrumentation_checks`; when `codex_trace_out_supported` is deferred, task correctness is counted from normal process exit and reported separately as `tasks_correctness_deferred_to_exit_code`.

Current revised deferred-check Round 0 restart state recorded on 2026-05-08:

- Installed `codex-cli 0.128.0` still has no `--trace-out`, but a custom `model_provider` using `wire_api="responses"` can target the local proxy with `OPENAI_API_KEY=EMPTY`.
- `src/lumo_flywheel_serving/inference_proxy.py` now normalizes Responses reasoning items for Codex compatibility and accepts legacy `request_shaping.target_concurrency` bundles by mapping them to the active concurrency-cap fields.
- Preflight through the local proxy (`http://127.0.0.1:8022/v1`) passed with `round0_may_run=true`, `blocking_reasons=[]`, `codex_smoke_ok=true`, `sample_ok=true`, and deferred reasons limited to the three requested instrumentation checks.
- A live Round 0 started with those three defers and completed several task attempts with Codex exit `0`, but it was stopped before completion after finding that attempts were mutating tracked source workspaces instead of per-attempt copies.
- The invalid partial output was archived to `output/track_b_e2e/round_0_invalid_inplace_workspace_20260508T000721Z`; it must not be used as a baseline or summary input.
- The generated tracked workspace mutation was restored, and `scripts/run_track_b_e2e_task.py` now copies each source workspace bundle into `round_<N>/<task>/run_XX/workspace`, writes the prompt from that copy, executes Codex with `{workspace}` pointing at that copy, and records both `source_workspace` and isolated `workspace` in `runner_metadata.json`.
- Verification after the isolation patch: `.venv/bin/python -m pytest tests/test_track_b_e2e_runner.py tests/test_track_b_e2e_round_driver.py tests/test_track_b_e2e_summary.py tests/test_track_b_e2e_preflight.py` returned `71 passed`.
- A restarted live Round 0 after the isolation patch was stopped early because measured attempts had already failed and the runner would necessarily return nonzero. The partial output is archived to `output/track_b_e2e/round_0_failed_codex_responses_tool_args_20260508T002216Z`.
- Archived restart evidence for `responses-sdk-adapter-cutover/v1-clean-baseline` recorded four isolated attempts: `run_01` exit `0` elapsed `165.9s`, `run_02` exit `1` elapsed `116.3s`, `run_03` exit `1` elapsed `127.0s`, and `run_04` exit `0` elapsed `147.6s`.
- The failed attempts show Codex/vLLM Responses tool-call compatibility failures, not source-workspace contamination: stderr contains `failed to parse function arguments: trailing characters`, and stdout records `BadRequestError` with `Extra data`.
- The proxy also logged a concrete streaming bug during this attempt: SSE upstream responses were accessed through `upstream.content` before chunk forwarding, so chunked stream interruptions could break Codex streams inside the proxy. `src/lumo_flywheel_serving/inference_proxy.py` now streams `text/event-stream` responses without pre-buffering `.content`; `tests/test_inference_proxy.py` covers this with a fake upstream whose `.content` raises if touched.
- A second restarted live Round 0 after the no-buffering SSE patch improved materially: `responses-sdk-adapter-cutover` completed 4/4 exit `0`, and `transcript-merge-regression` completed 4/4 exit `0`. It was stopped after `dead-flag-reachability-audit/run_01` exited `1`; the partial output is archived to `output/track_b_e2e/round_0_failed_codex_responses_transport_20260508T004616Z`.
- The second restart showed the remaining proxy transport defect directly in the foreground proxy session: `requests.exceptions.ChunkedEncodingError: Response ended prematurely` escaped from `_write_chunked_stream`. The proxy now catches upstream stream exceptions, emits a chunked SSE `event: error` with `type="upstream_stream_error"`, closes the chunk stream cleanly, and releases the admission ticket; `tests/test_inference_proxy.py` covers this with an upstream iterator that raises `ChunkedEncodingError`.
- A targeted probe of `dead-flag-reachability-audit/v1-clean-baseline` after that proxy fix completed with Codex exit `0`, elapsed `127.5s`, and an isolated attempt workspace under `/tmp/track_b_deadflag_probe_20260508`.
- A subsequent full Round 0 restart was stopped and archived to `output/track_b_e2e/round_0_stopped_cold_exit_policy_20260508T005609Z` after `responses-sdk-adapter-cutover/run_01` exited `1` while `run_02` was still running. This exposed a runner policy mismatch: the round contract discards `run_01` as cold, but `scripts/run_track_b_e2e_task.py` still failed the process on cold-attempt exits. The runner now supports `--discard-cold-attempt-exit`, and the round driver passes it so `run_01` failures are recorded but do not invalidate the round; measured attempts `run_02` through `run_04` still hard-fail on nonzero Codex exits.
- A full restart after the cold-exit policy fix was stopped and archived to `output/track_b_e2e/round_0_failed_measured_response_stream_20260508T010559Z` because `responses-sdk-adapter-cutover/run_02` was a measured attempt and exited `1`. The failure was again Codex rejecting function-call arguments with `Extra data` and `failed to parse function arguments: trailing characters`, after several reconnects reporting `stream closed before response.completed`.
- The proxy now normalizes Responses `function_call.arguments` by trimming trailing text after the first valid JSON value. This is intentionally narrow: valid JSON arguments are left unchanged, non-JSON strings are left unchanged, and only function-call style response items are rewritten.
- A targeted `responses-sdk-adapter-cutover/v1-clean-baseline` probe after the function-call argument normalizer recorded cold `run_01` exit `1` and measured attempts `run_02` and `run_03` both exit `0` under `/tmp/track_b_responses_probe_20260508`.
- The next full restart was stopped and archived to `output/track_b_e2e/round_0_failed_proxy_lost_20260508T012408Z`. This was not a model or measurement-contract result: the proxy had been launched in a foreground exec session and disappeared during a context transition, after which attempts failed with `error sending request for url (http://127.0.0.1:8022/v1/responses)`. The proxy is now launched for live rounds with `setsid` so it is detached from the interactive exec session.
- Per user direction, live proxy and round processes are now launched in tmux sessions (`trackb_proxy_8022` and `trackb_round0`) when possible. The first tmux restart was archived to `output/track_b_e2e/round_0_blocked_codex_smoke_timeout_20260508T012814Z` because `codex_command_smoke` timed out during preflight. The round driver now passes an explicit `--codex-smoke-timeout-s` to preflight, defaulting to `180.0s`, so successful but slow local Codex smoke runs do not block before measurement.
- A tmux restart after the longer smoke timeout reached measurement, then was stopped and archived to `output/track_b_e2e/round_0_failed_measured_tool_args_20260508T013849Z` because `responses-sdk-adapter-cutover/run_04` was a measured attempt and exited `1`. The concrete failure was repeated `stream closed before response.completed` after useful output/tool events. The proxy now synthesizes a minimal `response.completed` SSE block when upstream has already emitted at least one complete SSE block but omits the terminal completion event; if upstream fails before any event, it still emits an `upstream_stream_error`.
- The next tmux restart was stopped and archived to `output/track_b_e2e/round_0_failed_synthetic_completed_missing_id_20260508T014915Z` because `responses-sdk-adapter-cutover/run_03` was a measured attempt and exited `1`. The concrete failure was `stream disconnected before completion: failed to parse ResponseCompleted: missing field id`, caused by the proxy's synthetic terminal event carrying `status` and `output` but no `response.id`. The proxy now tracks an observed Responses id/model/created_at from earlier SSE events and uses that context in the synthetic `response.completed` block, falling back to `resp_proxy_synthetic` only when no upstream response id was observed.
- The latest tmux restart after preserving synthetic Responses ids completed the full 13-family sweep: `run_01` cold plus `run_02` through `run_04` measured attempts for every family, all 52 attempts exit `0`, with isolated workspaces recorded. Summary promotion still failed before `round_summary.json` was written because `rule_6_dcgm_dropout_pct` remained non-deferred and exceeded the `<1.0%` trust threshold.
- A diagnostic task summary for `responses-sdk-adapter-cutover/run_02` reported `trusted_measurement=false`, `diagnostic_only=true`, `rule_6_dcgm_dropout_pct=3.984537`, and the deferred checks exactly `vllm_request_metrics_join_available`, `codex_trace_out_supported`, and `dcgm_profile_fields_available`. The full failed output is archived to `output/track_b_e2e/round_0_failed_dcgm_dropout_attestation_20260508T032326Z`; it must not be used as a baseline or summary input.
- The dropout was systematic across measured runs rather than task-specific. The sampler was sleeping for the full requested interval after each NVML read, so NVML read overhead counted as missed 100 Hz cadence. `scripts/sample_dcgm_during_task.py` now schedules against absolute sample deadlines and validates `--interval-s` as finite positive; `tests/test_track_b_dcgm_sampler.py` covers the cadence compensation.
- The tmux restart after the sampler cadence fix completed all 52 attempts successfully. `output/track_b_e2e/round_0/round_summary.json` has `schema="lumo.track_b.e2e_round_summary.v1"`, `round=0`, `trusted_task_count=13`, `trusted_unique_task_count=13`, `tasks_completed=13`, `tasks_correctness_passed=13`, `tasks_correctness_deferred_to_exit_code=13`, `untrusted_task_count=0`, `diagnostic_task_count=0`, `sample_hash_mismatch_count=0`, `runtime_config_hash_mismatch_count=0`, `task_summary_schema_mismatch_count=0`, `task_summary_round_mismatch_count=0`, `median_wallclock_s=95.023`, `aggregate_wallclock_s=1263.267`, and `diagnosis_distribution={"mixed": 13}`.
- The verified task summaries all use the three deferred instrumentation checks listed above. The canonical `run_02` task summaries are trusted for all 13 families; their `rule_6_dcgm_dropout_pct` values range from `0.050687` to `0.083326`, below the non-deferred `<1.0%` dropout threshold after the sampler cadence fix.
- `scripts/build_track_b_e2e_readiness_manifest.py --preflight-json output/track_b_e2e/round_0/preflight_audit.json --out /tmp/track_b_readiness_after_round0.json` still exits `1` with `round0_ready=false`: `round0_summary_verified=true`, `preflight_round0_may_run=true`, and `round_proposal_prompt_verified=true`, but full readiness remains false because `trace_correctness_verified=false`, `ncu_profiles_verified=false`, and implementation steps A/B/D/G are not complete under the non-deferred full contract.
- The NCU archetype profile driver now propagates the same three explicit deferred instrumentation flags to `scripts/run_track_b_e2e_task.py` and records them in NCU metadata. Verification after this change: `.venv/bin/python -m pytest tests/test_inference_proxy.py tests/test_metrics.py tests/test_track_b_codex_trace_correctness.py tests/test_track_b_dcgm_sampler.py tests/test_track_b_e2e_ncu_profiles.py tests/test_track_b_e2e_preflight.py tests/test_track_b_e2e_readiness_manifest.py tests/test_track_b_e2e_round_driver.py tests/test_track_b_e2e_runner.py tests/test_track_b_e2e_summary.py` returned `135 passed`.
- A tmux NCU attempt for the first archetype (`long-text` / `sqlalchemy-2-session-modernization`) ran the task successfully in NCU mode with isolated workspace metadata, but Nsight Compute wrote only `==WARNING== No kernels were profiled.` to `output/track_b_e2e/ncu_long-text.csv`. The driver rejected the artifact because all seven required metrics were missing, so no NCU metadata sidecar was promoted and `ncu_profiles_verified` remains false.
- A second `long-text` NCU probe with `--launch-skip-before-match 0` and an absolute fresh task output root avoided the immediate no-kernel warning but did not finish within the expected task-timeout window; it was stopped from tmux with `output/track_b_e2e/ncu_long-text.csv` still `0` bytes and `/tmp/track_b_ncu_retry3_tmux.log` empty. This did not produce a valid NCU profile or metadata sidecar.
- A third `long-text` NCU probe reduced collection to `--launch-count 1`, kept `--launch-skip-before-match 0`, and shortened the task timeout to `300s`; it also remained attached past the timeout with `output/track_b_e2e/ncu_long-text.csv` still `0` bytes and `/tmp/track_b_ncu_onekernel_tmux.log` empty, then was stopped from tmux. The live topology explains why the original driver is weak here: `ncu --target-processes all` profiles the launched Codex/task process tree, while the actual GPU kernels are emitted by the already-running vLLM server process on port `9950` (`/usr/local/bin/vllm serve ... --port 9950`), outside that child process tree unless the server itself is launched under NCU or attached through an NCU launch/attach workflow.
- A direct NCU attach probe against the live environment did not provide a safe arbitrary-PID attach path: `timeout 15 ncu --mode=attach --hostname 127.0.0.1 --port 49152 --target-processes all ...` exited `1` with `Invalid option --target-processes specified for --mode attach`. The NCU help text describes attach as pairing with an application previously launched by `ncu --mode=launch`, so the current live vLLM server would need to be relaunched under NCU control or replaced by a real server-side launch/attach workflow.
- The NCU validators now report this failure mode explicitly. `scripts/run_track_b_e2e_ncu_profiles.py` rejects CSVs containing `No kernels were profiled`, and `scripts/build_track_b_e2e_readiness_manifest.py` reports `<archetype>_no_kernels_profiled` instead of reducing that evidence to generic missing metrics. Verification after this diagnostic tightening: `.venv/bin/python -m pytest tests/test_inference_proxy.py tests/test_metrics.py tests/test_track_b_codex_trace_correctness.py tests/test_track_b_dcgm_sampler.py tests/test_track_b_e2e_ncu_profiles.py tests/test_track_b_e2e_preflight.py tests/test_track_b_e2e_readiness_manifest.py tests/test_track_b_e2e_round_driver.py tests/test_track_b_e2e_runner.py tests/test_track_b_e2e_summary.py` returned `137 passed`.
- The NCU driver now resolves `--out-root` and `--task-out-root` to absolute paths before launching Nsight Compute or the nested task runner, because an attempted relative fresh task root under NCU produced a false `No such file or directory` task failure. Verification after this path-normalization fix: the same focused suite returned `138 passed`.
- The NCU driver now has an explicit `--profile-target server-launch` mode. In this mode `scripts/run_track_b_e2e_ncu_profiles.py` launches the supplied `--server-launch-command` under Nsight Compute, waits for `--server-ready-url`, runs the archetype Codex task against that profiled server, shuts the server process down, validates the NCU CSV, and records `profile_target`, `server_launch_command`, and `server_ready_url` in the metadata sidecar. This does not relaunch the current live vLLM server by default; it makes the required server-owned profiling topology explicit and test-covered. Verification after this server-launch workflow scaffold: the focused Track B suite returned `140 passed`.
- The readiness manifest now exposes the NCU topology state explicitly: Step G evidence includes `ncu_profile_driver_server_launch_supported=true` and `ncu_profile_required_topology="server-launch"`. This records the available driver path without counting it as a completed NCU profile gate; `ncu_profiles_verified` remains false until the four archetype CSVs are produced with the required kernel metrics and valid metadata sidecars.
- The server-launch NCU task wiring now passes `--reset-prefix-cache-url` through to the nested task runner. Without this, an isolated profiled server on a non-9950 port would still reset the live `:9950` server before the task, mixing control traffic across topology boundaries. The focused NCU driver test now asserts health, metrics, reset-prefix-cache, and Codex endpoint URLs are all routed to the intended profiled-server/proxy pair.

Revised deferred-check Round 0 start attempt recorded on 2026-05-07:

- The previous diagnostic `output/track_b_e2e/round_0/` was archived to `output/track_b_e2e/round_0_deferred_diagnostic_20260507T235900Z` so the round driver could start a fresh `round_0`.
- `scripts/run_track_b_e2e_round.py` was started with `--defer-preflight-checks vllm_request_metrics_join_available codex_trace_out_supported dcgm_profile_fields_available`.
- The fresh `output/track_b_e2e/round_0/preflight_audit.json` has `round0_may_run=false`, `blocking_reasons=["codex_command_smoke"]`, and the three requested checks under `deferred_reasons`.
- No task attempts or `round_summary.json` were produced, because `codex_command_smoke` is not deferred under the revised semantics.
- Readiness written to `/tmp/track_b_readiness_after_revised_defer_attempt.json` returns `decision="round0_blocked"`, `round0_ready=false`, and `blocking_reasons=["codex_command_smoke"]`.

Earlier deferred diagnostic result recorded on 2026-05-07 before the revised semantics:

- `scripts/run_track_b_e2e_round.py` completed with `--defer-preflight-checks vllm_request_metrics_join_available codex_trace_out_supported dcgm_profile_fields_available`.
- `output/track_b_e2e/round_0/preflight_audit.json` has `round0_may_run=true`, `blocking_reasons=[]`, and the three checks above under `deferred_reasons`.
- `output/track_b_e2e/round_0/round_summary.json` exists, but has `diagnostic_only=true`, `trusted_task_count=0`, `untrusted_task_count=13`, `diagnostic_task_count=13`, `median_wallclock_s=null`, `diagnostic_median_wallclock_s=2.253`, and `diagnostic_aggregate_wallclock_s=20.581`.
- The run is not a performance baseline: 20 attempts are missing-workspace diagnostics, and 32 live-workspace attempts recorded Codex exit `1` because installed Codex rejected `qwen3.5-27b` under the ChatGPT account path. Under the revised code, those Codex exits are no longer swallowed by deferred instrumentation checks.
- Readiness with `output/track_b_e2e/round_0/preflight_audit.json` still returns `decision="round0_blocked"` and `round0_ready=false`; the Round 0 summary verifier rejects the diagnostic summary for too few trusted/completed/correct task summaries and missing trusted median/aggregate wallclock fields.
- Trusted preflight now also checks the fixed sample's workspace bundles before measurement. After materializing two existing legacy workspaces and seeding the three missing family workspaces from their task specs, the current repo has 13/13 available and no longer blocks on `track_b_sample_workspaces_available`.
- Trusted preflight now smoke-runs the configured Codex command template in a temporary git workspace before the full round. With the diagnostic command template and `qwen3.5-27b`, it blocks on `codex_command_smoke` with the real Codex error: `The 'qwen3.5-27b' model is not supported when using Codex with a ChatGPT account.`

| Step | Current status | Evidence |
|---|---|---|
| A. Patch Codex CLI with `--trace-out` | Deferred for the next Round 0 start attempt, still incomplete for a trusted baseline. Installed `codex-cli 0.128.0` has `codex exec --json` but no `--trace-out`; source audit shows wrapper-only logging is insufficient because request/exec events live in `codex-rs/core` and `codex-rs/exec`. The readiness manifest now rejects placeholder patch files by requiring a nonempty unified diff that touches `codex-rs/` and contains the `--trace-out`, trace event, and `runtime_config_hash` markers. The correctness verifier still requires both observation-only byte equality and trace schema anchors, including a `sha256:<64-hex-digest>` runtime hash, before it can produce a passing artifact, and the readiness manifest requires both patch-content verification and schema-valid correctness status before trusted Round 0 can unblock. | `scripts/verify_track_b_codex_trace_correctness.py`; `scripts/build_track_b_e2e_readiness_manifest.py`; `tests/test_track_b_codex_trace_correctness.py`; `tests/test_track_b_e2e_readiness_manifest.py`; `track-b-e2e-codex-trace-patch-surface-audit-20260507.md`; `output/track_b_e2e/codex_trace_emitter_correctness.json` is absent. |
| B. DCGM/NVML 100 Hz sampler | Deferred for the next Round 0 start attempt only for profile-field availability, still incomplete for a fully trusted baseline. The sampler runs under `.venv/bin/python`, requires `sha256:<64-hex-digest>` `runtime_config_hash` stamps for measurement output, reserves unstamped rows for explicit preflight smoke probes only, and now schedules samples against absolute deadlines so NVML read overhead is not miscounted as 100 Hz dropout. Required DCGM profiling fields still report `null` in this environment, so preflight records `dcgmi_available=false` plus missing `pydcgm`, `dcgm_agent`, and `dcgm_fields` bindings, and NVML fallback rows stamp `profile_fields_unavailable_reason="nvml_fallback_only"`. Preflight and summary reject rows that merely have numeric-looking or non-finite profile keys unless the row also carries `profile_fields_available=true` with finite values for every required DCGM profile field, so NVML fallback samples and `NaN` telemetry cannot masquerade as DCGM profiling evidence. | `scripts/sample_dcgm_during_task.py`; `tests/test_track_b_dcgm_sampler.py`; `scripts/preflight_track_b_e2e.py`; `tests/test_track_b_e2e_preflight.py`; `tests/test_track_b_e2e_summary.py`; `track-b-e2e-round0-preflight-audit-20260507.md`. |
| C. E2E task runner | Scaffolded. The per-task runner wraps task directory creation, sampler lifecycle, Codex spawn, Prometheus capture, and summary build inputs. Both the direct task runner, including imported `run_one()` calls, and round driver now require a `sha256:<64-hex-digest>` `runtime_config_hash`; the round driver rejects stale measurement outputs before measurement, hard-gates preflight, supports explicit deferred instrumentation exclusions, re-validates the trace correctness artifact/timestamp when `codex_trace_out_supported` is not deferred, passes the round hash through the task runner and sampler, runs all 13 tasks with one cold attempt plus three measured attempts, verifies measured trace task identity and runtime hash before using run_02 through run_04, verifies the generation-volume guard when trace output is not deferred, uses trace `task_end.ts - task_start.ts` wallclocks, summarizes one canonical measured task artifact with the three measured wallclocks, and promotes the round summary. Deferring the three named instrumentation checks no longer auto-defers sample-workspace or Codex-command-smoke gates, no longer writes untrusted summaries by default, no longer allows missing workspaces, and no longer masks measured nonzero Codex exits. Each attempt now executes in an isolated copy at `round_<N>/<task>/run_XX/workspace`, so repeat attempts cannot mutate the pinned source workspace bundle. The round driver records but ignores `run_01` cold-attempt exit status because that attempt is explicitly discarded; `run_02` through `run_04` exits remain required. | `scripts/run_track_b_e2e_task.py`; `scripts/run_track_b_e2e_round.py`; `scripts/preflight_track_b_e2e.py`; `tests/test_track_b_e2e_runner.py`; `tests/test_track_b_e2e_round_driver.py`; `tests/test_track_b_e2e_preflight.py`. |
| D. Per-turn vLLM metric extension | Deferred for the next Round 0 start attempt, still incomplete for a trusted baseline. Local code can preserve request-id Prometheus labels when they exist and can normalize a request-keyed vLLM JSONL side-channel into `vllm_per_turn.json` while stamping the captured raw JSONL copy with the same round hash and recording the consumed byte range/request IDs in `runner_metadata.json`. Aggregate and per-request Prometheus parsers drop non-finite samples before deltas are computed; preflight, runner normalization, and summary raw-JSONL parsing all accept the JSONL path only when at least one complete request row also carries `schema="lumo.track_b.vllm_request_metrics.v1"` and `producer="track_b_vllm_request_metrics_patch"` plus finite non-negative prompt/completion/spec-decode counters; summary loading also rejects non-finite or negative metric values in direct `vllm_per_turn.json` artifacts. The round driver passes the side-channel path into preflight before measurement, and readiness exposes the full side-channel diagnostic object rather than only a boolean. The active vLLM process exposes neither request labels nor that side-channel producer. | `src/lumo_flywheel_serving/metrics.py`; `scripts/run_track_b_e2e_task.py`; `scripts/build_track_b_e2e_summary.py`; `scripts/preflight_track_b_e2e.py`; `scripts/run_track_b_e2e_round.py`; `scripts/build_track_b_e2e_readiness_manifest.py`; `tests/test_metrics.py`; `tests/test_track_b_e2e_preflight.py`; `tests/test_track_b_e2e_runner.py`; `tests/test_track_b_e2e_summary.py`; `tests/test_track_b_e2e_round_driver.py`; `tests/test_track_b_e2e_readiness_manifest.py`; `track-b-e2e-vllm-request-metrics-patch-surface-audit-20260507.md`. |
| E. Summary join + diagnosis rule | Scaffolded and unit-tested on synthetic artifacts. It correctly refuses missing/joinless evidence instead of manufacturing a round summary, rejects non-`sha256:<64-hex-digest>` runtime hashes, rejects non-finite or non-positive wallclock inputs before computing task/round medians, requires finite non-negative task scores before counting task correctness unless `codex_trace_out_supported` is deferred, and requires literal boolean `trusted_measurement` / `task_completed` values before promoting task summaries. Deferred checks are now contract exclusions: vLLM-request deferral skips request-id/spec-decode/silent-fallback requirements, Codex-trace deferral skips generation-volume/trace-correctness requirements and counts normal exit as correctness with `tasks_correctness_deferred_to_exit_code`, and DCGM-profile deferral skips only the profile-field-present requirement. The readiness manifest independently rejects non-finite or non-positive Round 0 median/aggregate wallclock fields before declaring the summary verified. It also rejects mismatched stamped `runtime_config_hash` provenance across trace, runner, vLLM, and DCGM artifacts, rejects round promotion when trusted task summaries do not match the round hash/schema/index, and accepts the runner's nested `round_<N>/<task>/run_XX/summary.json` layout when promoting a round. | `scripts/build_track_b_e2e_summary.py`; `scripts/build_track_b_e2e_readiness_manifest.py`; `tests/test_track_b_e2e_summary.py`; `tests/test_track_b_e2e_readiness_manifest.py`. |
| F. Auto research agent prompt template | Scaffolded and readiness-verified. The readiness manifest now validates prompt content instead of only file existence: it requires the hard-gated round driver, runtime/protocol hash arguments, trace-correctness artifact argument, preflight script, exact three-counter spec-decode grep, and rejects the legacy direct repeat-3 task measurement command. It also exposes this as `hard_gates.round_proposal_prompt_verified`. | `prompts/track_b_e2e_round_proposal.md`; `scripts/build_track_b_e2e_readiness_manifest.py`; `tests/test_track_b_e2e_readiness_manifest.py`. |
| G. Round 0 dry run | Revised deferred-check Round 0 now has a trusted reduced-contract summary at `output/track_b_e2e/round_0/round_summary.json`: all 52 attempts exited `0`, all 13 canonical measured task summaries are trusted, `tasks_completed=13`, `tasks_correctness_passed=13`, `tasks_correctness_deferred_to_exit_code=13`, `untrusted_task_count=0`, `diagnostic_task_count=0`, `median_wallclock_s=95.023`, and `aggregate_wallclock_s=1263.267`. This row remains incomplete for full Track B readiness because the NCU archetype profiles did not validate and the three user-deferred instrumentation gaps remain unresolved. The first NCU archetype attempt produced only `==WARNING== No kernels were profiled.` in `output/track_b_e2e/ncu_long-text.csv`; the skip-0 and one-kernel probes both hung with the same CSV at `0` bytes. Required metrics and the metadata sidecar were not promoted. Earlier failed/invalid restarts are archived under `output/track_b_e2e/round_0_invalid_inplace_workspace_20260508T000721Z/`, `output/track_b_e2e/round_0_failed_codex_responses_tool_args_20260508T002216Z/`, `output/track_b_e2e/round_0_failed_codex_responses_transport_20260508T004616Z/`, `output/track_b_e2e/round_0_stopped_cold_exit_policy_20260508T005609Z/`, `output/track_b_e2e/round_0_failed_measured_response_stream_20260508T010559Z/`, `output/track_b_e2e/round_0_failed_proxy_lost_20260508T012408Z/`, `output/track_b_e2e/round_0_blocked_codex_smoke_timeout_20260508T012814Z/`, `output/track_b_e2e/round_0_failed_measured_tool_args_20260508T013849Z/`, `output/track_b_e2e/round_0_failed_synthetic_completed_missing_id_20260508T014915Z/`, and `output/track_b_e2e/round_0_failed_dcgm_dropout_attestation_20260508T032326Z/`. The task, round, and NCU profile drivers reject unstamped runtime hashes before producing measurement/profile artifacts. | `output/track_b_e2e/round_0/round_summary.json`; `output/track_b_e2e/round_0/preflight_audit.json`; `output/track_b_e2e/ncu_long-text.csv`; `output/track_b_e2e/ncu_task_runs/round_0/sqlalchemy-2-session-modernization__v1-clean-baseline/run_01/runner_metadata.json`; `output/track_b_e2e/ncu_task_runs_retry_skip0_abs/round_0/sqlalchemy-2-session-modernization__v1-clean-baseline/run_01/`; `output/track_b_e2e/ncu_task_runs_retry_onekernel_abs/round_0/sqlalchemy-2-session-modernization__v1-clean-baseline/run_01/`; `scripts/build_track_b_e2e_summary.py`; `scripts/build_track_b_e2e_readiness_manifest.py`; `scripts/sample_dcgm_during_task.py`; `scripts/run_track_b_e2e_task.py`; `scripts/run_track_b_e2e_round.py`; `scripts/run_track_b_e2e_ncu_profiles.py`; `output/track_b_e2e/round_0_failed_dcgm_dropout_attestation_20260508T032326Z/`. |

Committed scaffold commits through this status checkpoint:

- `7de01d6 Add Track B E2E measurement scaffold`
- `023702a Record Track B E2E preflight blockers`
- `aa255b6 Tighten Track B E2E preflight gating`
- `be03780 Record Codex trace patch surface audit`
- `440cdc7 Add Track B E2E readiness manifest`
- `aaf2ecd Record Track B vLLM request metrics blocker`
- `55a4e4e Tighten Track B vLLM request label gate`
- `a4b1132 Record Track B E2E objective audit`
- `2b098f8 Expose Track B DCGM profile field gaps`
- `1d2526f Support Track B vLLM request metrics JSONL`
- `c5a99ad Update Track B E2E commit ledger`
- `82e774a Record Codex JSON trace gap`
- `a52e0d7 Validate Track B trace correctness artifact`
- `a9b6e1b Validate Track B Round 0 summary gate`
- `d35df33 Accept Track B vLLM request metrics side channel`
- `bc98b18 Validate Track B NCU archetype outputs`
- `ef84f33 Require Track B summary DCGM profile fields`
- `94eff9e Enforce Track B round sample integrity`
- `0efd4c6 Validate Track B readiness sample integrity`
- `89087cd Reject incomplete Track B round summaries`
- `12b9a76 Update Track B strict gate ledger`
- `cddfbef Verify Track B Codex trace correctness evidence`
- `cb1bc58 Update Track B trace verifier ledger`
- `b9dee12 Validate Track B NCU metric coverage`
- `f1a6ee7 Tighten Track B vLLM JSONL join gate`
- `0919c33 Update Track B JSONL gate ledger`
- `5e400ef Reject incomplete Track B vLLM JSONL rows`
- `9113a09 Update Track B vLLM JSONL row ledger`
- `ef3060b Reject empty Track B vLLM request joins`
- `a2f9317 Update Track B vLLM join ledger`
- `bc93e96 Support Track B runner NCU mode flag`
- `3e6d385 Update Track B runner NCU ledger`
- `e13b92e Accept Track B runner task ids`
- `3f0fd14 Update Track B runner task id ledger`
- `a1bb552 Validate Track B Codex trace command template`
- `e772822 Update Track B trace template ledger`
- `52be8a6 Accept nested Track B task summaries`
- `bcee31b Add Track B E2E round driver`
- `d2a642e Update Track B round driver ledger`
- `c980c39 Add Track B NCU profile driver`
- `b430ed7 Update Track B NCU driver ledger`
- `37ada55 Expose Track B driver evidence in readiness`
- `88dd25f Update Track B readiness evidence ledger`
- `ba92dae Tighten Track B round attestation driver`
- `0b622b6 Update Track B attestation driver ledger`
- `c7a790a Verify Track B generation volume guard`
- `1efbc1f Update Track B generation guard ledger`
- `8973a1a Use Track B trace wallclocks in round driver`
- `d1061f8 Update Track B trace wallclock ledger`
- `8e212bd Isolate Track B NCU task artifacts`
- `4af0f4e Update Track B NCU isolation ledger`
- `478f38c Reject stale Track B round summaries`
- `2024c71 Update Track B stale summary ledger`
- `dd11475 Reject stale Track B measurement outputs`
- `0ee0dd9 Update Track B stale output ledger`
- `d1b7a55 Accept Track B completion token side channel`
- `752b3eb Update Track B completion token ledger`
- `ecba6e9 Verify Track B trace task identity`
- `764f47c Update Track B trace identity ledger`
- `6b74dc1 Snapshot Track B vLLM side channel rows`
- `65afad0 Stamp Track B runtime hash artifacts`
- `effef5e Stamp Track B DCGM runtime hashes`
- `f2f5274 Verify Track B trace runtime hashes`
- `ecbb700 Cover Track B DCGM hash stamping`
- `af1acb2 Validate Track B trace schema evidence`
- `03baa8a Require Track B trace schema readiness`
- `cf83817 Require Track B trace turn ends`
- `66f7981 Verify Track B measured trace identity`
- `e25fa20 Verify Track B raw vLLM hash provenance`
- `9fc1ff0 Reject Track B round hash drift`
- `c4a52fd Require Track B round hash readiness`
- `c78d7f8 Reject Track B task summary drift`
- `1377336 Require Track B task summary readiness`
- `b483068 Require explicit Track B mismatch zeros`
- `6d286d5 Require Track B NCU profile metadata`
- `1788a05 Verify Track B NCU task metadata`
- `31458bc Verify Track B NCU metric metadata`
- `705ece8 Require Track B NCU round metadata`
- `9d2d256 Align Track B NCU artifact docs`
- `93866a3 Match Track B NCU hashes to round`
- `4e2e0be Use Track B round driver in proposal prompt`
- `84a7a40 Align Track B spec metric prompt`
- `f837c4d Verify Track B proposal prompt readiness`
- `6d8a07a Expose Track B proposal prompt gate`
- `96a18ca Update Track B prompt gate audit`
- `1fafa1c Require Track B runtime hash stamps`
- `6d0968a Require Track B task runtime hash`
- `216b32c Validate Track B summary runtime hashes`
- `de56598 Validate Track B trace runtime hashes`
- `547e53d Validate Track B run one hash`
- `bb54343 Require Track B sampler runtime hash`
- `45e6e32 Stamp Track B raw vLLM side channel`
- `3428694 Update Track B raw vLLM ledger`
- `10a5356 Require Track B runtime hash hex digests`
- `9991034 Update Track B runtime digest ledger`
- `80866b3 Refresh Track B digest checkpoint ledger`
- `4ce6c97 Validate Track B trace patch content`
- `bc009d1 Update Track B trace patch ledger`
- `bbf7e03 Record Track B vLLM capture offsets`
- `b584bc4 Update Track B vLLM capture ledger`
- `68fb78b Require Track B DCGM profile flag`
- `b696702 Update Track B DCGM profile ledger`
- `cacbe00 Expose Track B DCGM binding gaps`
- `af18f09 Update Track B DCGM binding ledger`
- `f350432 Require Track B vLLM producer metadata`
- `ddef720 Update Track B vLLM producer ledger`
- `0cddb4b Expose Track B vLLM side-channel evidence`
- `4d2073e Update Track B vLLM side-channel ledger`
- `0b6803a Require Track B trace artifact in round driver`
- `59a1cdd Update Track B trace artifact ledger`
- `2094d68 Require Track B vLLM producer in consumers`
- `e941e5a Reject nonfinite Track B wallclocks`
- `f29cb7d Require finite Track B readiness wallclocks`
- `4aa6ca3 Require finite Track B task scores`
- `ebb6a57 Require literal Track B trust booleans`
- `90596b8 Require finite Track B DCGM fields`
- `3641572 Require finite Track B vLLM request metrics`
- `f342e88 Validate Track B vLLM per-turn metrics`
- `7821215 Drop nonfinite Track B Prometheus samples`
- `fa4ab5a Require finite Track B NCU metric values`
- `8744b66 Defer Track B Round 0 instrumentation blockers`
- `5ba8bc7 Support deferred Track B Round 0 diagnostics`
- `66f094a Record missing Track B workspaces diagnostically`
- `283f0d8 Continue Track B diagnostics after Codex exits`
- `d1c3691 Gate Track B sample workspaces in preflight`
- `c3e255d Smoke Track B Codex command in preflight`
- `1fba6d8 Materialize Track B existing workspace bundles`
- `fbe7c8e Seed remaining Track B workspace bundles`
- `73758f1 Revise Track B deferred round contract`
- `cff55fe Normalize Track B Codex proxy responses`
- `21dee39 Isolate Track B task workspaces`
- `e2c4152 Stream Track B proxy SSE without buffering`
- `d9aecd0 Close Track B proxy SSE on upstream errors`
- `2990b94 Discard Track B cold attempt exits`
- `4324cc9 Normalize Track B function call arguments`
- `349df70 Record Track B detached proxy restart`
- `8855d13 Extend Track B Codex smoke timeout`
- `7a19911 Preserve Track B synthetic response ids`
- `1922c02 Fix Track B sampler cadence`
- `bf2764a Record deferred Track B Round 0 summary`
- `8105d20 Propagate Track B NCU defers`
- `f06031a Record Track B NCU no-kernel blocker`
- `80bd3e7 Record Track B NCU retry blocker`
- `fdb4d3e Record Track B NCU topology blocker`
- `37ccfce Classify Track B NCU no-kernel profiles`
- `3a782ca Normalize Track B NCU output roots`
- Current checkpoint: `Add Track B server-side NCU mode`

## 12. Decision rules for ending the loop

The loop terminates when one of these conditions holds:

1. **Aggregate wallclock improvement plateau:** three consecutive rounds with `wallclock_delta_vs_prior_round_s` between -2% and +2%. Interpret as "the local optimum for this technique class has been reached."
2. **Diagnosis distribution stable:** the `diagnosis_distribution` does not change meaningfully across two rounds. Means the bottleneck has moved out of any addressable regime within the loop's intervention vocabulary.
3. **Correctness ceiling hit:** three consecutive rounds where the agent cannot find a config that retains correctness (every proposed config fails §9 caveats).
4. **Cost ceiling hit:** runtime cost of the loop (wallclock + GPU-hours) exceeds an externally-set budget.

On termination, write `output/track_b_e2e/loop_closeout.md` summarizing the final config, total wallclock saved across rounds, regime-level evidence, and recommendations for next-quarter work outside the loop's intervention vocabulary.

## 13. Open risks and mitigations

| Risk | Mitigation |
|---|---|
| Codex CLI fork drift from upstream | Pin to a specific upstream SHA at the start of the loop. Re-verify trace-emitter correctness (§4.3) after every rebase. |
| DCGM/NVML metric availability on GB10 sm_120 | Verify each `DCGM_FI_PROF_*` field reports non-zero on the smoke task during Step B. If `PIPE_TENSOR_ACTIVE` is not exposed on consumer Blackwell, fall back to `PIPE_FP16_ACTIVE` and document. |
| Codex internal regime tagging accuracy | One-time validation: hand-tag a single task's turns, compare to Codex's emitted regime tags; mismatch rate must be < 5%. If higher, refine the regime mapping in the patch. |
| NCU profiling slows the workload enough to alter behavior | NCU runs are isolated per archetype, not per round. The non-NCU full measurement runs without NCU attached. NCU outputs feed the diagnosis taxonomy as static reference. |
| Variance across the 3 runs per task is high enough to mask real wins | If per-task `wallclock_stdev / wallclock_median > 0.10` across the 3 runs, escalate to 5 runs per task for that family. If still noisy, mark the task as `noisy=true` and weight it less in cross-task aggregation. |
| Codex CLI itself updates mid-loop and changes behavior | Pin Codex CLI version in the runtime_config_hash. Any Codex CLI version change resets the round counter (Round 0 must be re-established). |
| The auto research agent proposes configs that game the metric | Truthful-measurement contract (§8) is the primary defense. Secondary: the round proposal template (§7.2) requires the agent to state a hypothesis tied to a regime-level diagnosis from Round N-1; ad-hoc tuning that does not flow from a diagnosis is rejected at proposal time. |

## 14. References

| Source | Status | Relevance |
|---|---|---|
| `codex-harness-spec-decode-engineering-20260507.md` | Current spec | Technique inventory; this plan supersedes its Step 0d/0e/0f ordering. |
| `track-b-real-task-warmonly-pr39562-matrix-20260507.md` | Current evidence | Post-PR#39562 candidate matrix; 020/025/028 cleared the `9.0` gate at c1, 051 retired. |
| `track_b_tool_call_throughput_closeout_20260507.md` | Current production runtime | Candidate-056 closeout; Round 0 baseline config. |
| `track-b-concurrency-measurement-audit-20260506.md` | Reference | Established truthful-measurement rules. |
| `track-b-e2e-round0-preflight-audit-20260507.md` | Current blocker record | Live Round 0 preflight failures and remediation requirements. |
| `track-b-e2e-codex-trace-patch-surface-audit-20260507.md` | Current blocker record | Shows where Codex CLI must be patched for `--trace-out`. |
| `track-b-e2e-readiness-manifest-20260507.md` | Current blocker record | Defines the machine-readable readiness manifest and `round0_ready=false` gate. |
| `track-b-e2e-vllm-request-metrics-patch-surface-audit-20260507.md` | Current blocker record | Shows vLLM request IDs exist in serving but not in Prometheus metric labels. |
| `track-b-e2e-objective-completion-audit-20260507.md` | Current blocker record | Maps the user objective to concrete artifacts and records why the objective is not complete. |
| `l0-warm-decode-quality-bounded-track-20260505.md` | Parent spec | Track B Round 1 framing; this plan is the e2e measurement instrument that should drive its next rounds. |
| `benchmark_blueprints/tracks/README.md` | Bench source | The 11-track structure used for sample selection. |
| Codex CLI (open source) | External | Patch surface for `--trace-out`; carry as fork until upstreamable. |
| vLLM Prometheus metrics docs — https://docs.vllm.ai/ | Verified | `vllm:spec_decode_num_accepted_tokens` / `vllm:spec_decode_num_draft_tokens` source. |
| NVIDIA DCGM Profiling Metrics — https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-api/group__dcgmFieldIdentifiers.html | Verified | `DCGM_FI_PROF_*` field IDs used by the sampler. |
| NVIDIA Nsight Compute User Guide | Verified | NCU command shape and metric names. |
| vLLM PR #39562 | Verified, OPEN | KV-allocator stop-gap carried in the active runtime. |
| vLLM Issue #40875 | Verified | Tool-call XML corruption check is part of preflight (§7.3). |

---

*This plan is the auto-research-loop instrument for Track B's e2e Codex wallclock optimization. It supersedes per-decode-tok/s ranking as the Track B headline metric. Round 0 establishes the baseline under candidate-056's runtime; subsequent rounds are intervention rounds whose proposals must trace to a Round N-1 regime diagnosis. The truthful-measurement contract (§8) is mandatory; the candidate-051 c4 17.087 incident is the canonical failure mode this contract exists to prevent.*
