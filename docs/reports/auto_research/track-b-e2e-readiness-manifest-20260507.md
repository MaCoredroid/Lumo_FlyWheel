# Track B E2E Readiness Manifest

Generated: 2026-05-08

Command:

```bash
.venv/bin/python scripts/build_track_b_e2e_readiness_manifest.py \
  --preflight-json output/track_b_e2e/round_0/preflight_audit.json \
  --out /tmp/track_b_readiness_manifest_report_refresh.json
```

Result: exit code `1`, `decision="round0_blocked"`, `round0_ready=false`.

## Current Blockers

- `vllm_request_metrics_join_available`: deferred by user direction for the reduced Round 0 contract; still incomplete for full-fidelity readiness.
- `codex_trace_out_supported`: deferred by user direction for the reduced Round 0 contract; installed `codex-cli 0.128.0` still does not expose `--trace-out`, and no validated `output/track_b_e2e/codex_trace_emitter_correctness.json` exists.
- `dcgm_profile_fields_available`: deferred by user direction for the reduced Round 0 contract; the sampler runs through NVML fallback, but required DCGM profile fields are still unavailable.
- `ncu_profiles_verified`: no valid five-archetype NCU profile set exists. Current readiness reports `ncu_profile_count=0` of `expected_ncu_profile_count=5`; the durable blocker report is `track-b-e2e-ncu-server-profiling-blocker-20260508.md`.

The current preflight artifact has `round0_may_run=true`, `blocking_reasons=[]`, and `deferred_reasons=["vllm_request_metrics_join_available", "codex_trace_out_supported", "dcgm_profile_fields_available"]`. This allowed the reduced-contract Round 0 sweep and summary, but not full Track B readiness.

Current hard-gate values:

- `preflight_round0_may_run=true`
- `round0_summary_verified=true`
- `round_proposal_prompt_verified=true`
- `trace_correctness_verified=false`
- `ncu_profiles_verified=false`
- `all_implementation_steps_complete=false`

## Step Status

| Step | Status | Evidence |
|---|---|---|
| A. Codex `--trace-out` patch + correctness artifact | deferred | user-deferred for reduced Round 0; no content-validated trace patch, no validated `output/track_b_e2e/codex_trace_emitter_correctness.json`, installed Codex lacks `--trace-out` |
| B. DCGM/NVML 100 Hz sampler | deferred | user-deferred only for profile-field availability; sampler script exists and runs through NVML fallback, but `dcgmi` and DCGM Python bindings are unavailable, so profile fields remain unavailable |
| C. E2E task runner | complete | `scripts/run_track_b_e2e_task.py` |
| D. Per-turn vLLM metrics keyed by request id | deferred | user-deferred for reduced Round 0; parser/join code exists, but live metrics do not expose request-id labels and no Track B producer-stamped side-channel is configured |
| E. Summary join + diagnosis rule | complete | `scripts/build_track_b_e2e_summary.py` |
| F. Round proposal prompt | complete | `prompts/track_b_e2e_round_proposal.md` |
| G. Round 0 dry run | missing | reduced-contract `output/track_b_e2e/round_0/round_summary.json` is validated, but full Step G is missing because no five named metric-complete NCU archetype profiles exist |

## Reduced Round 0 Summary

`output/track_b_e2e/round_0/round_summary.json` is trusted only for the user-directed reduced contract:

- `trusted_task_count=13`
- `trusted_unique_task_count=13`
- `tasks_completed=13`
- `tasks_correctness_passed=13`
- `tasks_correctness_deferred_to_exit_code=13`
- `untrusted_task_count=0`
- `diagnostic_task_count=0`
- `median_wallclock_s=95.023`
- `aggregate_wallclock_s=1263.267`

The reduced summary must not be represented as full-fidelity readiness because the deferred full-instrumentation checks and NCU profiles are still unresolved.

## Decision

The readiness manifest intentionally requires both setup gates and actual Round 0 artifacts. Passing preflight and producing a reduced-contract summary are not enough for full readiness. The Codex trace patch must be a non-placeholder unified diff that touches the `codex-rs/` surface and contains the required trace event/runtime-hash markers; a trace correctness artifact must satisfy the `lumo.track_b.codex_trace_correctness.v1` schema with three enabled/disabled byte-equality task checks; full-fidelity vLLM request joins and DCGM profile fields must be available unless explicitly deferred; and NCU must produce the five named archetype CSVs with all required §5.3 metric names and matching metadata.

Do not promote `round0_ready=true` or compare full Track B E2E rounds from this state. The next full-readiness blocker is producing valid server-side NCU profiles, which requires replacing/stopping the live `:9950` server or freeing enough GPU memory for an NCU-launched server.
