# Track B E2E Objective Completion Audit

Generated: 2026-05-07
Repo checkpoint: updated through the Track B deferred Round 0 diagnostic checkpoint in current git history.

## Objective Restated

User objective:

`docs/reports/auto_research/track-b-e2e-agentic-saturation-plan-20260507.md` should truthfully land the Track B E2E auto-research-loop goal, drive measurements without false claims, and commit progress as work lands.

Concrete success criteria:

1. The plan document states the E2E goal and current implementation state truthfully.
2. The auto-research loop has executable scaffolding for preflight, per-task runs, per-request metrics parsing, summary joins, readiness gating, and round proposal authoring.
3. Measurement gates prevent Round 0 from being recorded unless trace, vLLM request correlation, DCGM profiling, correctness, and round-summary artifacts are present.
4. Every made progress checkpoint is committed.
5. No headline E2E measurement is claimed before Round 0 is actually runnable and validated.

## Prompt-to-Artifact Checklist

| Requirement / gate | Required artifact or command | Evidence inspected | Status |
|---|---|---|---|
| Plan states Track B headline metric as E2E Codex task wallclock, not decode tok/s. | `track-b-e2e-agentic-saturation-plan-20260507.md` §1-§2 | Plan says the headline metric is median per-task wallclock across the 13-task sample, with decode tok/s diagnostic-only. | Complete |
| Plan defines fixed 13-task sample. | Plan §3 | Sample covers 13 fixed CNB-55 family/variant slots and says the sample must not change between rounds. | Complete |
| Fixed sample workspace bundles exist. | `scripts/preflight_track_b_e2e.py` | Trusted preflight now checks `track_b_sample_workspaces_available`; after materializing `skill-router-contract-upgrade` and `plugin-scaffold-alignment` from existing legacy `workspace/` directories and seeding the remaining three workspaces from their task specs, current repo reports 13/13 available and no missing task IDs. | Complete |
| Plan defines Codex trace schema and correctness check. | Plan §4 | Trace event schema and `output/track_b_e2e/codex_trace_emitter_correctness.json` verifier schema are present; readiness rejects an existence-only correctness artifact, and the verifier requires `task_start.runtime_config_hash` to be a `sha256:<64-hex-digest>` value. | Spec complete; implementation blocked |
| Codex `--trace-out` implemented and verified. | Patched Codex CLI + correctness artifact | Live preflight reports `codex-cli 0.128.0`, `codex_trace_out_supported=false`; validated correctness artifact is absent. A verifier now exists at `scripts/verify_track_b_codex_trace_correctness.py` to build the artifact from real enabled/disabled evidence once patched Codex is available. The revised deferred-check path excludes this requirement from the selected Round 0 contract instead of forcing diagnostic-only summaries, but Codex task exits and all non-deferred gates still apply. | Deferred for current Round 0 contract; still blocked for full-fidelity baseline |
| Codex command/model path can run. | `scripts/preflight_track_b_e2e.py --codex-command-template ...` | Trusted preflight now smoke-runs the configured Codex command template in a temporary git workspace. With the diagnostic command and `qwen3.5-27b`, current preflight blocks on `codex_command_smoke`; Codex returns `The 'qwen3.5-27b' model is not supported when using Codex with a ChatGPT account.` | Blocked |
| Codex patch surface audited. | `track-b-e2e-codex-trace-patch-surface-audit-20260507.md` | Audit records Rust patch surface, why wrapper-only logging is insufficient, and why installed `codex exec --json` cannot substitute for `--trace-out` because it lacks emitted turn ids, request/response ids, timestamps, and per-tool timing. | Complete blocker record |
| vLLM per-request join requirement specified. | Plan §5.1 | Plan requires `vllm_per_turn.json` keyed by `vllm_request_id`. | Complete |
| vLLM request-label / side-channel consumer implemented. | `src/lumo_flywheel_serving/metrics.py`, `scripts/run_track_b_e2e_task.py`, `scripts/build_track_b_e2e_summary.py` | `parse_prometheus_samples()` preserves labels and drops non-finite samples; `parse_prometheus_text()` also excludes non-finite aggregate samples; `compute_vllm_per_request_metrics()` computes request-keyed deltas when labels exist. Summary and runner normalize request-keyed vLLM JSONL side-channel rows into the same per-turn schema, stamp the captured raw JSONL copy with the round hash, record the consumed side-channel byte range/request IDs in runner metadata, and reject incomplete or empty request-keyed artifacts. Preflight, runner JSONL normalization, and summary raw-JSONL parsing now require side-channel rows to carry `schema="lumo.track_b.vllm_request_metrics.v1"` and `producer="track_b_vllm_request_metrics_patch"` plus finite non-negative prompt/completion/spec-decode counters before accepting JSONL as a join source; summary loading also rejects non-finite or negative values in direct `vllm_per_turn.json` metric rows. The round driver forwards the side-channel path into preflight and readiness exposes the full side-channel diagnostic object. | Scaffold complete |
| vLLM request metrics join available. | `scripts/preflight_track_b_e2e.py --out output/track_b_e2e/preflight_20260507.json` | Preflight now accepts either request-labeled Prometheus metrics or row-complete request-keyed JSONL side-channel metrics; live environment has neither source available. The revised deferred-check path excludes request-id/spec-decode/silent-fallback summary requirements from the selected Round 0 contract when this check is deferred. | Deferred for current Round 0 contract; still blocked for full-fidelity baseline |
| vLLM request-metrics patch surface audited. | `track-b-e2e-vllm-request-metrics-patch-surface-audit-20260507.md` | Audit records that request IDs exist in OpenAI serving but are dropped before Prometheus aggregation. | Complete blocker record |
| DCGM/NVML 100 Hz sampler exists. | `scripts/sample_dcgm_during_task.py` | Sampler script exists, requires `sha256:<64-hex-digest>` runtime-hash stamps for measurement output, keeps an explicit `--allow-unstamped-smoke` path for preflight-only temporary rows, and live preflight reports `dcgm_sampler_runs=true` with `telemetry_sources=["nvml"]`; preflight now records missing `pydcgm`/`dcgm_agent`/`dcgm_fields` bindings and missing `dcgmi`, and preflight plus summary require a row with `profile_fields_available=true` plus finite values for all required profile fields before trusting DCGM profile telemetry. | Scaffold complete |
| Required DCGM profiling fields numeric. | Preflight JSON | Live preflight reports `dcgm_profile_fields_available=false`, no observed numeric profile fields, and missing `dram_active_pct`, `sm_active_pct`, `sm_occupancy_pct`, `pipe_tensor_active_pct`, and `pipe_fp16_active_pct`. The revised deferred-check path excludes only the profile-field-present summary requirement when this check is deferred; sampler/dropout/runtime-hash evidence still applies. | Deferred for current Round 0 contract; still blocked for full-fidelity baseline |
| E2E task runner exists. | `scripts/run_track_b_e2e_task.py` | Readiness manifest reports Step C complete; runner accepts the documented `--ncu-mode` profiling flag and single `family/variant` task-id form, rejects Codex command templates that omit `{trace_out}` unless trace output is explicitly deferred, and the direct task runner, imported `run_one()` entry point, and round driver reject unstamped runtime hashes before measurement. The round driver also re-validates `output/track_b_e2e/codex_trace_emitter_correctness.json` after preflight and before measurement when trace output is not deferred, requiring the supplied `--trace-emitter-correctness-verified-at` timestamp to match the artifact. Deferred instrumentation mode now excludes only the named checks from preflight/summary requirements; it no longer auto-defers sample-workspace or Codex-command-smoke gates, no longer allows missing workspaces, and no longer masks nonzero Codex exits. | Scaffold complete |
| Summary join and deterministic diagnosis exists. | `scripts/build_track_b_e2e_summary.py` | Readiness manifest reports Step E complete; focused tests cover synthetic summary behavior, deferred-check contract exclusions, duplicated/off-sample trusted round rejection, incomplete trusted round rejection, non-finite/non-positive wallclock rejection before task or round median/aggregate summaries, finite non-negative task-score requirements outside trace-deferred mode, literal boolean trust/completion requirements before promotion, and non-`sha256:<64-hex-digest>` runtime hash rejection before writing task/round summaries. Summaries now record `deferred_instrumentation_checks`; trace-deferred correctness counted from process exit is reported separately as `tasks_correctness_deferred_to_exit_code`. | Scaffold complete |
| Auto-research round proposal template exists and drives the hard-gated loop. | `prompts/track_b_e2e_round_proposal.md` | Readiness manifest reports Step F complete and now exposes `hard_gates.round_proposal_prompt_verified=true`; Step F validates the hard-gated round driver, runtime/protocol hash arguments, trace-correctness artifact argument, preflight script, exact three-counter spec-decode grep, and absence of the legacy direct repeat-3 task measurement command. | Complete |
| Machine-readable readiness gate exists. | `scripts/build_track_b_e2e_readiness_manifest.py` | Command exits 1 with `round0_ready=false` and blocking reasons; trace patch content, trace correctness, proposal prompt content, Round 0 summary, sample-integrity, runtime-hash format, finite positive Round 0 wallclock fields, and NCU metric-coverage gates validate artifact content, not only file existence. | Complete |
| Round 0 dry run populated and validated. | `output/track_b_e2e/round_0/round_summary.json` and five named NCU profiles | Earlier deferred diagnostic round completed: 52 runner attempts, 13 task summaries, and one `round_summary.json`. That artifact was archived to `output/track_b_e2e/round_0_deferred_diagnostic_20260507T235900Z/` because it predates the revised deferred-check contract. A fresh Round 0 start attempt wrote `output/track_b_e2e/round_0/preflight_audit.json` and stopped before task execution with `blocking_reasons=["codex_command_smoke"]`; no fresh `round_summary.json` or NCU profiles exist. | Existing artifact diagnostic only; fresh current-contract run blocked before measurement |
| Tests cover new scaffolding. | Focused pytest commands | The full focused Track B suite passed after the deferred-check semantic update: `113 passed` across trace correctness, preflight, readiness, summary, runner, metrics, DCGM sampler, round driver, and NCU profile tests. | Complete for scaffold risk |
| Full repo test suite green. | Full pytest | Earlier `PYTHONPATH=. .venv/bin/pytest -q -x` failed an unrelated existing `tests/test_auto_research.py` expectation. | Not green; unrelated known failure |
| Progress committed. | Git history | Deferred diagnostic checkpoints `5ba8bc7`, `66f094a`, `283f0d8`, sample-workspace preflight gate `d1c3691`, Codex command smoke gate `c3e255d`, workspace materialization checkpoint `1fba6d8`, remaining workspace seed checkpoint `fbe7c8e`, and the current revised deferred-contract checkpoint are on `main`; repo is ahead of origin. | Complete for landed checkpoints |

## Current Readiness Decision

Latest readiness command after the revised deferred-check Round 0 start attempt:

```bash
.venv/bin/python scripts/build_track_b_e2e_readiness_manifest.py \
  --preflight-json output/track_b_e2e/round_0/preflight_audit.json \
  --out /tmp/track_b_readiness_after_revised_defer_attempt.json
```

Result: exit code `1`, `decision="round0_blocked"`, `round0_ready=false`.

Blocking reasons:

- `codex_command_smoke`

The fresh preflight audit has `deferred_reasons=["codex_trace_out_supported", "dcgm_profile_fields_available", "vllm_request_metrics_join_available"]`, `sample_workspaces_available=true`, `available_task_count=13`, and `expected_task_count=13`. No task attempts or summary artifacts were produced.

Latest trusted preflight blocker check:

```bash
.venv/bin/python scripts/preflight_track_b_e2e.py \
  --out /tmp/track_b_preflight_codex_smoke.json \
  --codex-command-template 'codex exec --json -C {workspace} --model {model} "Read the task prompt at {prompt_file} and complete it in this workspace."' \
  --codex-model qwen3.5-27b \
  --codex-endpoint http://127.0.0.1:9950/v1 \
  --codex-api-key local \
  --codex-smoke-timeout-s 20
```

Result without defers: exit code `1`, `round0_may_run=false`, with blockers:

- `vllm_request_metrics_join_available`
- `codex_trace_out_supported`
- `codex_command_smoke`
- `dcgm_profile_fields_available`

The latest sample workspace check after seeding the remaining bundles reports `available_task_count=13`, `expected_task_count=13`, and no missing task IDs. Latest readiness with the fresh deferred-check preflight still returns `decision="round0_blocked"` and `round0_ready=false`.
The Codex smoke check returns exit code `1` with the unsupported `qwen3.5-27b` ChatGPT-account error.

Deferred reasons:

- `vllm_request_metrics_join_available`
- `codex_trace_out_supported`
- `dcgm_profile_fields_available`

Round summary status: no fresh `round_summary.json` exists because the revised deferred-check run stopped at preflight.

## Completion Audit Verdict

The objective is **not complete**. The plan and scaffold now implement the requested deferred-check semantics in code: the three named instrumentation checks can be excluded from the selected Round 0 contract, while all non-deferred gates still apply and Codex exits are preserved. The earlier diagnostic Round 0 artifact remains non-baseline. The remaining work is:

1. Patch/pin Codex CLI with `--trace-out` and produce `output/track_b_e2e/codex_trace_emitter_correctness.json`.
2. Patch vLLM or add a bounded per-request JSONL side-channel, then wire Codex trace request IDs to that side-channel so Codex turns can join to vLLM token/spec-decode metrics.
3. Fix or replace DCGM profile-field collection so required utilization fields are numeric, or explicitly revise the plan with a truthful lower-fidelity diagnosis mode.
4. Route Codex to the intended local `qwen3.5-27b` runtime or choose a supported Codex model path for the diagnostic loop.
5. Start a new Round 0 under the revised deferred-check contract after the configured Codex command/model path smoke-runs successfully.

The archived `round_0_deferred_diagnostic_20260507T235900Z/round_summary.json` predates this semantic change, is diagnostic only, and must not be used as a headline E2E baseline.
