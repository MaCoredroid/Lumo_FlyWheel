# Track B E2E Round 0 Preflight Audit

Generated: 2026-05-08

Command:

```bash
.venv/bin/python scripts/build_track_b_e2e_readiness_manifest.py \
  --preflight-json output/track_b_e2e/round_0/preflight_audit.json \
  --out /tmp/track_b_readiness_after_a2db2a8.json
```

Result: exit code `1`, `decision="round0_blocked"`, `round0_ready=false`.

## Current Reduced-Contract Preflight

The current `output/track_b_e2e/round_0/preflight_audit.json` was produced for the user-directed reduced Round 0 contract. It has:

- `round0_may_run=true`
- `blocking_reasons=[]`
- `deferred_reasons=["vllm_request_metrics_join_available", "codex_trace_out_supported", "dcgm_profile_fields_available"]`

Those deferrals mean the reduced Round 0 sweep and summary may run without those three instrumentation checks. They do not mean full Track B readiness is achieved.

## Current Passing Evidence

- vLLM health endpoint `http://127.0.0.1:9950/health` returned 200.
- vLLM exposes aggregate spec_decode counters:
  - `vllm:spec_decode_num_drafts_total`
  - `vllm:spec_decode_num_draft_tokens_total`
  - `vllm:spec_decode_num_accepted_tokens_total`
- Installed Codex is present: `codex-cli 0.128.0`.
- Installed Codex supports `codex exec --json`.
- `nvidia-smi` is present.
- `ncu` is present.
- `pynvml` is now available through the repo dependency `nvidia-ml-py`, and the sampler runs under `.venv/bin/python`.
- The fixed Track B sample workspaces were available for the reduced Round 0 sweep.
- The reduced Round 0 summary at `output/track_b_e2e/round_0/round_summary.json` is verified by readiness for the deferred contract.

## Reduced Round 0 Summary

The reduced-contract summary records:

- `trusted_task_count=13`
- `trusted_unique_task_count=13`
- `tasks_completed=13`
- `tasks_correctness_passed=13`
- `tasks_correctness_deferred_to_exit_code=13`
- `untrusted_task_count=0`
- `diagnostic_task_count=0`
- `median_wallclock_s=95.023`
- `aggregate_wallclock_s=1263.267`

## Full-Readiness Blockers

- `codex exec --help` still does not expose `--trace-out`, and no validated `output/track_b_e2e/codex_trace_emitter_correctness.json` exists.
- Current vLLM `/metrics` output does not expose `request_id=`, `vllm_request_id=`, or `request=` labels, and no Track B producer-stamped per-request JSONL side-channel is configured for full-fidelity per-turn joins.
- The live sampler smoke emitted 100 Hz rows with coarse NVML fields (`gpu_util_pct`, `mem_copy_util_pct`, `power_w`), but the required profiling fields remain unavailable: `dram_active_pct`, `sm_active_pct`, `sm_occupancy_pct`, `pipe_tensor_active_pct`, and `pipe_fp16_active_pct`.
- NCU profiles are not verified. Current readiness reports `ncu_profile_count=0` of `expected_ncu_profile_count=5`; see `track-b-e2e-ncu-server-profiling-blocker-20260508.md`.

## Decision

The reduced-contract `round_summary.json` may be recorded and used only with the three explicit deferrals listed above. Do not represent it as a full-fidelity Track B baseline and do not promote `round0_ready=true`.

Full readiness remains blocked because Rule 7/12/13 cannot be joined per turn, Rule 6 still lacks numeric DCGM profiling fields for bottleneck diagnosis, Rule 14 has no trace-emitter byte-equality artifact, and NCU has not produced the five required server-side archetype profiles.

Next unblockers:

1. Carry or build the Codex CLI `--trace-out` fork/patch and record `output/track_b_e2e/codex_trace_emitter_correctness.json`.
2. Enable or add vLLM per-request metric labels keyed by `vllm_request_id`.
3. Replace the sampler backend with an available DCGM binding or host exporter that emits numeric `dram_active_pct` and `sm_active_pct`; coarse NVML utilization alone is not enough to promote Round 0.
4. Produce the five valid server-side NCU archetype profiles, which currently requires replacing/stopping live `:9950` or freeing enough GPU memory for an NCU-launched server.
