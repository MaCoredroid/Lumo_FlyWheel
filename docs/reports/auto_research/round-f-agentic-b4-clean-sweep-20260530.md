# Round-F Agentic B=4 Clean Sweep

Date: 2026-05-30

## Methodology

Workload is the fixed four-instance SWE-bench Verified subset:
`astropy__astropy-12907`, `astropy__astropy-13033`,
`astropy__astropy-13236`, `astropy__astropy-13398`.

Every arm uses the real B=4 agentic harness:
`scripts/run_codex_experiment.py --suite swe --concurrency 4 --limit 4 --no-commit`.
Inference runs on DGX vLLM `:9950`; the Codex/SWE harness runs on alienware/x86.

Clean-slate policy per arm:

- Recreate the vLLM container and verify no prior vLLM/EngineCore process remains.
- Recover host memory before relaunch and verify free memory is restored.
- Relaunch with `LUMO_BATCH_INVARIANT_VLLM=1` and `--attention-backend FLASH_ATTN`.
- Reset vLLM prefix cache via `POST /reset_prefix_cache` before the harness starts.
- Verify cold counters before the harness: prefix-cache queries/hits, generation tokens,
  spec-draft events, accepted tokens, and KV-cache usage are all zero.
- Use a unique `exp_tag` per arm so `--skip-existing` can only resume within that arm.
- Slice `/tmp/swe_dgx_steptrace.jsonl` by the arm's own driver start/end timestamps.
- Keep proxy sampling constant: temperature as-set, model default `top_p=0.95`.
- The SWE harness creates fresh per-task worktrees at each task base commit for each
  unique arm output directory.

`--nsight first-task` is retained for bounded GPU metrics capture. The exported sqlite
for E3 contains GPU metrics tables but no CUDA kernel activity tables, so it is not a
kernel-name breakdown. Kernel-level attribution is limited to vLLM telemetry and
available Round-F instrumentation unless a later arm is captured with CUDA tracing.

## Sweep Matrix

| Arm | Depth | Width | Nodes | Accept/event | Accept/draft | Decode TPS | Event ms | Resolved | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| E3 chain | 3 | 1 | 3 | 2.323 | 0.774 | 37.15 | 89.50 | 1/4 | clean rerun; prefix cache cold at start |

## Per-Arm Details

### E3 chain depth 3

- Experiment: `output/roundf_clean_agentic_b4_E3_d3_20260530T1846Z`
- Clean-slate proof: `clean_slate.json`
- Summary: `summary.json`
- Sliced traces: `dgx_steptrace_window.jsonl`, `per_req_spec_trace_window.jsonl`
- Acceptance distribution: `{0: 2278, 1: 2482, 2: 2276, 3: 13747}`
- Tasks: `astropy__astropy-12907` resolved; the other three failed.
- Nsight export: available, but no CUDA kernel tables in sqlite.
