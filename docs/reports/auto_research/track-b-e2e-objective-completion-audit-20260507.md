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
| Fixed sample workspace bundles exist. | `scripts/preflight_track_b_e2e.py` | Trusted preflight now checks `track_b_sample_workspaces_available`; current repo reports 8/13 available and missing `responsive-checkout-visual-regression/v1-clean-baseline`, `incident-evidence-synthesis/v1-clean-baseline`, `multi-tool-transaction-repair/v1-clean-baseline`, `skill-router-contract-upgrade/v1-clean-baseline`, and `plugin-scaffold-alignment/v1-clean-baseline`. | Blocked |
| Plan defines Codex trace schema and correctness check. | Plan §4 | Trace event schema and `output/track_b_e2e/codex_trace_emitter_correctness.json` verifier schema are present; readiness rejects an existence-only correctness artifact, and the verifier requires `task_start.runtime_config_hash` to be a `sha256:<64-hex-digest>` value. | Spec complete; implementation blocked |
| Codex `--trace-out` implemented and verified. | Patched Codex CLI + correctness artifact | Live preflight reports `codex-cli 0.128.0`, `codex_trace_out_supported=false`; validated correctness artifact is absent. A verifier now exists at `scripts/verify_track_b_codex_trace_correctness.py` to build the artifact from real enabled/disabled evidence once patched Codex is available. The 2026-05-07 deferred diagnostic round synthesized minimal trace anchors and is explicitly untrusted. | Deferred; still blocked for trusted baseline |
| Codex patch surface audited. | `track-b-e2e-codex-trace-patch-surface-audit-20260507.md` | Audit records Rust patch surface, why wrapper-only logging is insufficient, and why installed `codex exec --json` cannot substitute for `--trace-out` because it lacks emitted turn ids, request/response ids, timestamps, and per-tool timing. | Complete blocker record |
| vLLM per-request join requirement specified. | Plan §5.1 | Plan requires `vllm_per_turn.json` keyed by `vllm_request_id`. | Complete |
| vLLM request-label / side-channel consumer implemented. | `src/lumo_flywheel_serving/metrics.py`, `scripts/run_track_b_e2e_task.py`, `scripts/build_track_b_e2e_summary.py` | `parse_prometheus_samples()` preserves labels and drops non-finite samples; `parse_prometheus_text()` also excludes non-finite aggregate samples; `compute_vllm_per_request_metrics()` computes request-keyed deltas when labels exist. Summary and runner normalize request-keyed vLLM JSONL side-channel rows into the same per-turn schema, stamp the captured raw JSONL copy with the round hash, record the consumed side-channel byte range/request IDs in runner metadata, and reject incomplete or empty request-keyed artifacts. Preflight, runner JSONL normalization, and summary raw-JSONL parsing now require side-channel rows to carry `schema="lumo.track_b.vllm_request_metrics.v1"` and `producer="track_b_vllm_request_metrics_patch"` plus finite non-negative prompt/completion/spec-decode counters before accepting JSONL as a join source; summary loading also rejects non-finite or negative values in direct `vllm_per_turn.json` metric rows. The round driver forwards the side-channel path into preflight and readiness exposes the full side-channel diagnostic object. | Scaffold complete |
| vLLM request metrics join available. | `scripts/preflight_track_b_e2e.py --out output/track_b_e2e/preflight_20260507.json` | Preflight now accepts either request-labeled Prometheus metrics or row-complete request-keyed JSONL side-channel metrics; live environment has neither source available. The deferred diagnostic round wrote empty deferred `vllm_per_turn.json` artifacts instead of request-joined metrics. | Deferred; still blocked for trusted baseline |
| vLLM request-metrics patch surface audited. | `track-b-e2e-vllm-request-metrics-patch-surface-audit-20260507.md` | Audit records that request IDs exist in OpenAI serving but are dropped before Prometheus aggregation. | Complete blocker record |
| DCGM/NVML 100 Hz sampler exists. | `scripts/sample_dcgm_during_task.py` | Sampler script exists, requires `sha256:<64-hex-digest>` runtime-hash stamps for measurement output, keeps an explicit `--allow-unstamped-smoke` path for preflight-only temporary rows, and live preflight reports `dcgm_sampler_runs=true` with `telemetry_sources=["nvml"]`; preflight now records missing `pydcgm`/`dcgm_agent`/`dcgm_fields` bindings and missing `dcgmi`, and preflight plus summary require a row with `profile_fields_available=true` plus finite values for all required profile fields before trusting DCGM profile telemetry. | Scaffold complete |
| Required DCGM profiling fields numeric. | Preflight JSON | Live preflight reports `dcgm_profile_fields_available=false`, no observed numeric profile fields, and missing `dram_active_pct`, `sm_active_pct`, `sm_occupancy_pct`, `pipe_tensor_active_pct`, and `pipe_fp16_active_pct`. The deferred diagnostic round recorded this as deferred and produced untrusted summaries. | Deferred; still blocked for trusted baseline |
| E2E task runner exists. | `scripts/run_track_b_e2e_task.py` | Readiness manifest reports Step C complete; runner accepts the documented `--ncu-mode` profiling flag and single `family/variant` task-id form, rejects Codex command templates that omit `{trace_out}` for trusted runs, and the direct task runner, imported `run_one()` entry point, and round driver reject unstamped runtime hashes before measurement. The round driver also re-validates `output/track_b_e2e/codex_trace_emitter_correctness.json` after preflight and before trusted measurement, requiring the supplied `--trace-emitter-correctness-verified-at` timestamp to match the artifact. Deferred diagnostic mode records synthetic trace anchors, empty deferred vLLM metrics, missing-workspace attempts, and nonzero Codex exits as untrusted diagnostics; trusted preflight now catches missing sample workspaces before the round spends measurement time. | Scaffold complete |
| Summary join and deterministic diagnosis exists. | `scripts/build_track_b_e2e_summary.py` | Readiness manifest reports Step E complete; focused tests cover synthetic summary behavior, reject missing DCGM profile fields, reject duplicated/off-sample trusted round summaries, reject incomplete trusted round summaries, reject non-finite/non-positive wallclock inputs before task or round median/aggregate summaries, require finite non-negative task scores before counting task correctness, require literal boolean trust/completion fields before promotion, and reject non-`sha256:<64-hex-digest>` runtime hashes before writing task/round summaries. Deferred diagnostic summaries carry `diagnostic_only=true`, separate diagnostic wallclock fields, and zero trusted tasks. | Scaffold complete |
| Auto-research round proposal template exists and drives the hard-gated loop. | `prompts/track_b_e2e_round_proposal.md` | Readiness manifest reports Step F complete and now exposes `hard_gates.round_proposal_prompt_verified=true`; Step F validates the hard-gated round driver, runtime/protocol hash arguments, trace-correctness artifact argument, preflight script, exact three-counter spec-decode grep, and absence of the legacy direct repeat-3 task measurement command. | Complete |
| Machine-readable readiness gate exists. | `scripts/build_track_b_e2e_readiness_manifest.py` | Command exits 1 with `round0_ready=false` and blocking reasons; trace patch content, trace correctness, proposal prompt content, Round 0 summary, sample-integrity, runtime-hash format, finite positive Round 0 wallclock fields, and NCU metric-coverage gates validate artifact content, not only file existence. | Complete |
| Round 0 dry run populated and validated. | `output/track_b_e2e/round_0/round_summary.json` and five named NCU profiles | Deferred diagnostic round completed: 52 runner attempts, 13 task summaries, and one `round_summary.json`. The summary has `diagnostic_only=true`, `trusted_task_count=0`, `untrusted_task_count=13`, `diagnostic_task_count=13`, `median_wallclock_s=null`, and `diagnostic_median_wallclock_s=2.253`. Readiness still reports `round0_summary_verified=false`, `ncu_profiles_verified=false`, and `round0_ready=false`. | Deferred diagnostic complete; trusted baseline blocked |
| Tests cover new scaffolding. | Focused pytest commands | The full focused Track B suite passed after the sample-workspace gate: `109 passed` across trace correctness, preflight, readiness, summary, runner, metrics, DCGM sampler, round driver, and NCU profile tests. | Complete for scaffold risk |
| Full repo test suite green. | Full pytest | Earlier `PYTHONPATH=. .venv/bin/pytest -q -x` failed an unrelated existing `tests/test_auto_research.py` expectation. | Not green; unrelated known failure |
| Progress committed. | Git history | Deferred diagnostic checkpoints `5ba8bc7`, `66f094a`, `283f0d8`, and sample-workspace preflight gate `d1c3691` are on `main`; repo is ahead of origin. | Complete for landed checkpoints |

## Current Readiness Decision

Latest readiness command:

```bash
.venv/bin/python scripts/build_track_b_e2e_readiness_manifest.py \
  --preflight-json output/track_b_e2e/round_0/preflight_audit.json \
  --out /tmp/track_b_readiness_after_deferred_round0.json
```

Result: exit code `1`, `decision="round0_blocked"`, `round0_ready=false`.

Blocking reasons:

- none in the deferred preflight audit (`blocking_reasons=[]`)

Latest trusted preflight blocker check:

```bash
.venv/bin/python scripts/preflight_track_b_e2e.py --out /tmp/track_b_preflight_workspace_gate.json
```

Result: exit code `1`, `round0_may_run=false`, with blockers:

- `vllm_request_metrics_join_available`
- `codex_trace_out_supported`
- `dcgm_profile_fields_available`
- `track_b_sample_workspaces_available`

The sample workspace check reports `available_task_count=8`, `expected_task_count=13`.

Deferred reasons:

- `vllm_request_metrics_join_available`
- `codex_trace_out_supported`
- `dcgm_profile_fields_available`

Round summary verifier reasons:

- `too_few_trusted_tasks`
- `too_few_unique_trusted_tasks`
- `too_few_completed_tasks`
- `too_few_correct_tasks`
- `median_wallclock_missing`
- `aggregate_wallclock_missing`
- `diagnosis_distribution_missing`

## Completion Audit Verdict

The objective is **not complete**. The plan and scaffold now truthfully prevent false trusted Round 0 measurement, the requested deferred diagnostic Round 0 has been started and summarized, and all progress so far has been committed, but the actual E2E auto-research loop still cannot run trusted measurements. The remaining required implementation is:

1. Patch/pin Codex CLI with `--trace-out` and produce `output/track_b_e2e/codex_trace_emitter_correctness.json`.
2. Patch vLLM or add a bounded per-request JSONL side-channel, then wire Codex trace request IDs to that side-channel so Codex turns can join to vLLM token/spec-decode metrics.
3. Fix or replace DCGM profile-field collection so required utilization fields are numeric, or explicitly revise the plan with a truthful lower-fidelity diagnosis mode.
4. Populate or revise the five missing workspace bundles, or update the fixed sample with a formal sample-hash reset.
5. Route Codex to the intended local `qwen3.5-27b` runtime or choose a supported Codex model path for the diagnostic loop.
6. Run trusted Round 0 only after the readiness manifest reports all hard gates passing.

The current `round_0/round_summary.json` is diagnostic only and must not be used as a headline E2E baseline.
