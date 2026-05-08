# Track B E2E NCU Server Profiling Blocker

Generated: 2026-05-08

## Scope

This note records why Track B full readiness still blocks on NCU archetype profiles after the reduced-contract Round 0 summary. It is a blocker record, not a promoted NCU profile and not a performance baseline.

## Current Round 0 State

The reduced-contract Round 0 summary at `output/track_b_e2e/round_0/round_summary.json` is trusted only under the three user-directed deferrals:

- `vllm_request_metrics_join_available`
- `codex_trace_out_supported`
- `dcgm_profile_fields_available`

Current summary values:

- `runtime_config_hash=sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa`
- `trusted_task_count=13`
- `median_wallclock_s=95.023`
- `aggregate_wallclock_s=1263.267`

Readiness remains `round0_ready=false` because full instrumentation is deferred and `ncu_profiles_verified=false`.

## NCU Evidence

The required NCU contract is five valid archetype profiles:

- `long-text`
- `tool-call-frame`
- `pure-investigation`
- `multimodal-prefill`
- `subagent-orchestration`

Current tracked output evidence:

- `output/track_b_e2e/ncu_long-text.csv` exists but is `0` bytes.
- No valid `output/track_b_e2e/ncu_<archetype>.json` metadata sidecars exist.
- Current readiness reports `ncu_profile_count=0` of `expected_ncu_profile_count=5`.

The container-side probe evidence is currently in `/tmp/track_b_container_ncu_probe_20260508T060426Z/profiles/ncu_long-text.csv` and is not a valid profile. Its complete CSV content is:

```text
==PROF== Connected to process 631 (/usr/bin/python3.12)
==PROF== Connected to process 706 (/usr/bin/nvidia-smi)
==PROF== Disconnected from process 706
==PROF== Connected to process 723 (/usr/bin/python3.12)
==PROF== Connected to process 797 (/usr/bin/nvidia-smi)
==PROF== Disconnected from process 797
==PROF== Disconnected from process 723
==PROF== Disconnected from process 631
==ERROR== The application returned an error code (1).
```

This proves Nsight Compute attached to the launched container process tree, but the launched profiled server exited with application error before producing required kernel metrics.

## Live Topology

Current live server check:

- `curl http://127.0.0.1:9950/health` returned HTTP `200`.
- Docker container: `lumo-vllm-l0c-fp8-cutlass-run30`
- Docker image: `lumo-flywheel-vllm:26.01-py3-v0.19.0`
- Container status: `Up 22 hours` at the time of this audit.
- Container binaries: `/usr/local/bin/vllm` and `/usr/local/bin/ncu` both exist.
- Container NCU version: `2025.4.1.0`.
- Host NCU path/version: `/usr/local/cuda/bin/ncu`, `2025.3.1.0`.

The live vLLM server process is already resident on port `9950`. The failed isolated container-side NCU probe attempted to launch a second profiled vLLM server on `9951`; that path reached process attachment but failed before `/health`, consistent with the previously recorded CUDA OOM during `cudaMemGetInfo` while the live `:9950` server remained resident.

## Conclusion

The task-wrapper NCU topology is insufficient because it profiles the Codex/task child process tree, while the GPU kernels are emitted by the already-running vLLM server. Valid NCU profiles require launching the vLLM server itself under NCU.

The currently safe path cannot produce valid profiles while the live `:9950` server remains resident. The next full-readiness step requires explicit operator approval to either:

1. Stop or replace the live `:9950` server with an NCU-launched server, then run the five archetype profiles.
2. Free enough GPU memory to launch an isolated profiled server alongside the live server.

Until one of those is done, `ncu_profiles_verified=false` and Step G remains `missing`.
