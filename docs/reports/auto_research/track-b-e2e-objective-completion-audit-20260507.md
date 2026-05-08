# Track B E2E Objective Completion Audit

Generated: 2026-05-08
Repo checkpoint: updated through `cbed669 Refresh Track B NCU status ledger`.

## Objective Restated

User objective:

`docs/reports/auto_research/track-b-e2e-agentic-saturation-plan-20260507.md` should truthfully land the Track B E2E auto-research-loop goal, drive measurements without false claims, and commit progress as work lands.

Concrete success criteria:

1. The plan document states the E2E goal and current implementation state truthfully.
2. The auto-research loop has executable scaffolding for preflight, per-task runs, per-request metrics parsing, summary joins, readiness gating, and round proposal authoring.
3. Measurement gates prevent a full-fidelity Round 0 from being recorded unless trace, vLLM request correlation, DCGM profiling, correctness, round-summary, and NCU artifacts are present; when the user explicitly defers named checks, the reduced-contract summary must record exactly those exclusions.
4. Every made progress checkpoint is committed.
5. No headline full-fidelity E2E measurement is claimed before Round 0 is actually runnable and validated; any reduced-contract measurement is labeled with its deferred checks and remaining blockers.

## Prompt-to-Artifact Checklist

| Requirement / gate | Required artifact or command | Evidence inspected | Status |
|---|---|---|---|
| Plan states Track B headline metric as E2E Codex task wallclock, not decode tok/s. | `track-b-e2e-agentic-saturation-plan-20260507.md` §1-§2 | Plan says the headline metric is median per-task wallclock across the 13-task sample, with decode tok/s diagnostic-only. | Complete |
| Plan defines fixed 13-task sample. | Plan §3 | Sample covers 13 fixed CNB-55 family/variant slots and says the sample must not change between rounds. | Complete |
| Fixed sample workspace bundles exist. | `scripts/preflight_track_b_e2e.py` | Trusted preflight now checks `track_b_sample_workspaces_available`; after materializing `skill-router-contract-upgrade` and `plugin-scaffold-alignment` from existing legacy `workspace/` directories and seeding the remaining three workspaces from their task specs, current repo reports 13/13 available and no missing task IDs. | Complete |
| Plan defines Codex trace schema and correctness check. | Plan §4 | Trace event schema and `output/track_b_e2e/codex_trace_emitter_correctness.json` verifier schema are present; readiness rejects an existence-only correctness artifact, and the verifier requires `task_start.runtime_config_hash` to be a `sha256:<64-hex-digest>` value. | Spec complete; implementation blocked |
| Codex `--trace-out` implemented and verified. | Patched Codex CLI + correctness artifact | Live preflight reports `codex-cli 0.128.0`, `codex_trace_out_supported=false`; validated correctness artifact is absent. A verifier now exists at `scripts/verify_track_b_codex_trace_correctness.py` to build the artifact from real enabled/disabled evidence once patched Codex is available. The revised deferred-check path excludes this requirement from the selected Round 0 contract instead of forcing diagnostic-only summaries, but Codex task exits and all non-deferred gates still apply. | Deferred for current Round 0 contract; still blocked for full-fidelity baseline |
| Codex command/model path can run. | `output/track_b_e2e/round_0/preflight_audit.json` | The current deferred-contract preflight reports `round0_may_run=true`, `blocking_reasons=[]`, and the three user-directed deferrals only. Earlier ChatGPT-account Codex smoke failures remain archived as blocker history, but the current local proxy path was sufficient to run the reduced Round 0 sweep. | Complete for reduced contract |
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
| Machine-readable readiness gate exists. | `scripts/build_track_b_e2e_readiness_manifest.py` | Current command exits `1` with `decision="round0_blocked"` and `round0_ready=false`, while reporting `round0_summary_verified=true`, `preflight_round0_may_run=true`, `round_proposal_prompt_verified=true`, `trace_correctness_verified=false`, and `ncu_profiles_verified=false`. Step statuses are A/B/D `deferred`, C/E/F `complete`, and G `missing`. The manifest validates artifact content, not only file existence. | Complete |
| Round 0 dry run populated and validated. | `output/track_b_e2e/round_0/round_summary.json` and five named NCU profiles | The current reduced-contract `round_summary.json` is trusted for the user-directed deferrals: `trusted_task_count=13`, `trusted_unique_task_count=13`, `tasks_completed=13`, `tasks_correctness_passed=13`, `tasks_correctness_deferred_to_exit_code=13`, `untrusted_task_count=0`, `diagnostic_task_count=0`, `median_wallclock_s=95.023`, and `aggregate_wallclock_s=1263.267`. It is not full Track B readiness because the five NCU profile CSV/metadata pairs are still missing or invalid and the full instrumentation gaps remain deferred. | Reduced-contract Round 0 complete; full readiness blocked |
| NCU archetype profiles produced and verified. | `output/track_b_e2e/ncu_<archetype>.csv` plus `ncu_<archetype>.json` for five archetypes | Current readiness sees `ncu_profile_count=0` of `expected_ncu_profile_count=5` and now carries the verified Round 0 `expected_runtime_config_hash=sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa` into the NCU metadata expectation. The manifest also guards against using malformed Round 0 runtime hashes as NCU expectations. `output/track_b_e2e/ncu_long-text.csv` exists but is `0` bytes and no metadata sidecar exists. Container probe evidence under `/tmp/track_b_container_ncu_probe_20260508T060426Z/profiles/ncu_long-text.csv` records an application error from an isolated second server OOM while live `:9950` remained resident; it was not promoted. | Blocked |
| Tests cover new scaffolding. | Focused pytest commands | The latest focused readiness/NCU check passed in tmux: `45 passed` for `tests/test_track_b_e2e_readiness_manifest.py` and `tests/test_track_b_e2e_ncu_profiles.py`. Earlier full focused Track B suite after the NCU failure-sidecar changes passed with `149 passed`. | Complete for scaffold risk |
| Full repo test suite green. | Full pytest | Earlier `PYTHONPATH=. .venv/bin/pytest -q -x` failed an unrelated existing `tests/test_auto_research.py` expectation. | Not green; unrelated known failure |
| Progress committed. | Git history | Current history includes the deferred Round 0 summary checkpoint `bf2764a`, NCU topology/failure handling through `049985c`, the objective audit refresh `833e991`, and the current NCU runtime-hash guard checkpoint. | Complete for landed checkpoints |

## Current Readiness Decision

Latest readiness command after the reduced-contract Round 0 summary and NCU topology work:

```bash
.venv/bin/python scripts/build_track_b_e2e_readiness_manifest.py \
  --preflight-json output/track_b_e2e/round_0/preflight_audit.json \
  --out /tmp/track_b_objective_audit_readiness.json
```

Result: exit code `1`, `decision="round0_blocked"`, `round0_ready=false`.

Current hard-gate evidence:

- `preflight_round0_may_run=true`
- `round0_summary_verified=true`
- `round_proposal_prompt_verified=true`
- `trace_correctness_verified=false`
- `ncu_profiles_verified=false`
- `all_implementation_steps_complete=false`

Implementation step statuses:

- A. Patch Codex CLI with `--trace-out`: `deferred`
- B. DCGM/NVML profile fields: `deferred`
- C. E2E task runner: `complete`
- D. Per-turn vLLM metric extension: `deferred`
- E. Summary join + diagnosis rule: `complete`
- F. Auto research agent prompt template: `complete`
- G. Round 0 dry run: `missing`

The current preflight audit has `round0_may_run=true`, `blocking_reasons=[]`, and `deferred_reasons=["vllm_request_metrics_join_available", "codex_trace_out_supported", "dcgm_profile_fields_available"]`.

Deferred reasons:

- `vllm_request_metrics_join_available`
- `codex_trace_out_supported`
- `dcgm_profile_fields_available`

Round summary status: the fresh reduced-contract `output/track_b_e2e/round_0/round_summary.json` exists and is trusted for the deferred contract. It must not be represented as full-fidelity Track B readiness because NCU profiles and the three full instrumentation checks remain unresolved.

## Completion Audit Verdict

The objective is **not complete for full Track B readiness**, but the user-directed reduced Round 0 summary has been produced and documented truthfully. The plan and scaffold implement the requested deferred-check semantics in code: the three named instrumentation checks can be excluded from the selected Round 0 contract, while all non-deferred gates still apply and Codex exits are preserved. The earlier diagnostic Round 0 artifact remains non-baseline. The remaining work is:

1. Patch/pin Codex CLI with `--trace-out` and produce `output/track_b_e2e/codex_trace_emitter_correctness.json`.
2. Patch vLLM or add a bounded per-request JSONL side-channel, then wire Codex trace request IDs to that side-channel so Codex turns can join to vLLM token/spec-decode metrics.
3. Fix or replace DCGM profile-field collection so required utilization fields are numeric, or explicitly revise the plan with a truthful lower-fidelity diagnosis mode.
4. Produce five valid NCU archetype profile CSV/metadata pairs. The remaining safe blocker is topology/memory: the valid path requires launching the vLLM server itself under NCU, but the live `:9950` server occupies enough GPU memory that an isolated `:9951` profiled server OOMs and sleep mode frees only `0.03 GiB`.
5. After the full-fidelity gates are resolved, rerun readiness and only then promote `round0_ready=true`.

The archived `round_0_deferred_diagnostic_20260507T235900Z/round_summary.json` predates this semantic change, is diagnostic only, and must not be used as a headline E2E baseline.
