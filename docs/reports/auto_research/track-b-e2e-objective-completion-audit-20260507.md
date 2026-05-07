# Track B E2E Objective Completion Audit

Generated: 2026-05-07
Repo checkpoint: updated through the Track B round-driver trace-artifact gate checkpoint in current git history.

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
| Plan defines Codex trace schema and correctness check. | Plan §4 | Trace event schema and `output/track_b_e2e/codex_trace_emitter_correctness.json` verifier schema are present; readiness rejects an existence-only correctness artifact, and the verifier requires `task_start.runtime_config_hash` to be a `sha256:<64-hex-digest>` value. | Spec complete; implementation blocked |
| Codex `--trace-out` implemented and verified. | Patched Codex CLI + correctness artifact | Live preflight reports `codex-cli 0.128.0`, `codex_trace_out_supported=false`; validated correctness artifact is absent. A verifier now exists at `scripts/verify_track_b_codex_trace_correctness.py` to build the artifact from real enabled/disabled evidence once patched Codex is available. | Blocked |
| Codex patch surface audited. | `track-b-e2e-codex-trace-patch-surface-audit-20260507.md` | Audit records Rust patch surface, why wrapper-only logging is insufficient, and why installed `codex exec --json` cannot substitute for `--trace-out` because it lacks emitted turn ids, request/response ids, timestamps, and per-tool timing. | Complete blocker record |
| vLLM per-request join requirement specified. | Plan §5.1 | Plan requires `vllm_per_turn.json` keyed by `vllm_request_id`. | Complete |
| vLLM request-label / side-channel consumer implemented. | `src/lumo_flywheel_serving/metrics.py`, `scripts/run_track_b_e2e_task.py`, `scripts/build_track_b_e2e_summary.py` | `parse_prometheus_samples()` preserves labels; `compute_vllm_per_request_metrics()` computes request-keyed deltas when labels exist; summary and runner normalize request-keyed vLLM JSONL side-channel rows into the same per-turn schema, stamp the captured raw JSONL copy with the round hash, record the consumed side-channel byte range/request IDs in runner metadata, and reject incomplete or empty request-keyed artifacts. Preflight now requires side-channel rows to carry `schema="lumo.track_b.vllm_request_metrics.v1"` and `producer="track_b_vllm_request_metrics_patch"` before accepting JSONL as a join source; the round driver forwards that side-channel path into preflight and readiness exposes the full side-channel diagnostic object. | Scaffold complete |
| vLLM request metrics join available. | `scripts/preflight_track_b_e2e.py --out output/track_b_e2e/preflight_20260507.json` | Preflight now accepts either request-labeled Prometheus metrics or row-complete request-keyed JSONL side-channel metrics; live environment has neither source available. | Blocked |
| vLLM request-metrics patch surface audited. | `track-b-e2e-vllm-request-metrics-patch-surface-audit-20260507.md` | Audit records that request IDs exist in OpenAI serving but are dropped before Prometheus aggregation. | Complete blocker record |
| DCGM/NVML 100 Hz sampler exists. | `scripts/sample_dcgm_during_task.py` | Sampler script exists, requires `sha256:<64-hex-digest>` runtime-hash stamps for measurement output, keeps an explicit `--allow-unstamped-smoke` path for preflight-only temporary rows, and live preflight reports `dcgm_sampler_runs=true` with `telemetry_sources=["nvml"]`; preflight now records missing `pydcgm`/`dcgm_agent`/`dcgm_fields` bindings and missing `dcgmi`, and summary requires a row with `profile_fields_available=true` plus all numeric profile fields before trusting DCGM profile telemetry. | Scaffold complete |
| Required DCGM profiling fields numeric. | Preflight JSON | Live preflight reports `dcgm_profile_fields_available=false`, no observed numeric profile fields, and missing `dram_active_pct`, `sm_active_pct`, `sm_occupancy_pct`, `pipe_tensor_active_pct`, and `pipe_fp16_active_pct`. | Blocked |
| E2E task runner exists. | `scripts/run_track_b_e2e_task.py` | Readiness manifest reports Step C complete; runner accepts the documented `--ncu-mode` profiling flag and single `family/variant` task-id form, rejects Codex command templates that omit `{trace_out}`, and the direct task runner, imported `run_one()` entry point, and round driver reject unstamped runtime hashes before measurement. The round driver also re-validates `output/track_b_e2e/codex_trace_emitter_correctness.json` after preflight and before measurement, requiring the supplied `--trace-emitter-correctness-verified-at` timestamp to match the artifact. | Scaffold complete |
| Summary join and deterministic diagnosis exists. | `scripts/build_track_b_e2e_summary.py` | Readiness manifest reports Step E complete; focused tests cover synthetic summary behavior, reject missing DCGM profile fields, reject duplicated/off-sample trusted round summaries, reject incomplete trusted round summaries, and reject non-`sha256:<64-hex-digest>` runtime hashes before writing task/round summaries. | Scaffold complete |
| Auto-research round proposal template exists and drives the hard-gated loop. | `prompts/track_b_e2e_round_proposal.md` | Readiness manifest reports Step F complete and now exposes `hard_gates.round_proposal_prompt_verified=true`; Step F validates the hard-gated round driver, runtime/protocol hash arguments, trace-correctness artifact argument, preflight script, exact three-counter spec-decode grep, and absence of the legacy direct repeat-3 task measurement command. | Complete |
| Machine-readable readiness gate exists. | `scripts/build_track_b_e2e_readiness_manifest.py` | Command exits 1 with `round0_ready=false` and blocking reasons; trace patch content, trace correctness, proposal prompt content, Round 0 summary, sample-integrity, runtime-hash format, and NCU metric-coverage gates validate artifact content, not only file existence. | Complete |
| Round 0 dry run populated and validated. | `output/track_b_e2e/round_0/round_summary.json` and five named NCU profiles | Readiness manifest reports `round0_summary_verified=false`, `ncu_profiles_verified=false`, `ncu_profile_count=0`; direct task, sampler, round, NCU profile, summary, and readiness paths now require `sha256:<64-hex-digest>` runtime hash stamps before writing or accepting trusted artifacts. | Blocked |
| Tests cover new scaffolding. | Focused pytest commands | `tests/test_track_b_codex_trace_correctness.py`, `tests/test_track_b_e2e_preflight.py`, `tests/test_track_b_e2e_readiness_manifest.py`, `tests/test_track_b_e2e_summary.py`, `tests/test_track_b_e2e_runner.py`, `tests/test_metrics.py`, `tests/test_track_b_dcgm_sampler.py`, `tests/test_track_b_e2e_round_driver.py`, and `tests/test_track_b_e2e_ncu_profiles.py` passed in focused runs during this work; the latest focused Track B suite reported 87 passed. | Complete for scaffold risk |
| Full repo test suite green. | Full pytest | Earlier `PYTHONPATH=. .venv/bin/pytest -q -x` failed an unrelated existing `tests/test_auto_research.py` expectation. | Not green; unrelated known failure |
| Progress committed. | Git history | The round-driver trace-artifact gate checkpoint (`0b6803a`) is on `main`; repo is ahead of origin. | Complete for landed checkpoints |

## Current Readiness Decision

Latest readiness command:

```bash
.venv/bin/python scripts/build_track_b_e2e_readiness_manifest.py --out /tmp/track_b_readiness_manifest_after_prompt_gate.json
```

Result: exit code `1`, `decision="round0_blocked"`, `round0_ready=false`.

Blocking reasons:

- `vllm_request_metrics_join_available`
- `codex_trace_out_supported`
- `dcgm_profile_fields_available`

## Completion Audit Verdict

The objective is **not complete**. The plan and scaffold now truthfully prevent false Round 0 measurement, and all progress so far has been committed, but the actual E2E auto-research loop cannot yet run truthful measurements. The remaining required implementation is:

1. Patch/pin Codex CLI with `--trace-out` and produce `output/track_b_e2e/codex_trace_emitter_correctness.json`.
2. Patch vLLM or add a bounded per-request JSONL side-channel, then wire Codex trace request IDs to that side-channel so Codex turns can join to vLLM token/spec-decode metrics.
3. Fix or replace DCGM profile-field collection so required utilization fields are numeric, or explicitly revise the plan with a truthful lower-fidelity diagnosis mode.
4. Run Round 0 only after the readiness manifest reports all hard gates passing.

No `round_0/round_summary.json` should be recorded from the current state.
