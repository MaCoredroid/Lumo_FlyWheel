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

For F width-2 arms launched after commit `692d3be3`, the verifier also writes
`tree_accept_path.jsonl` with the accepted tree node path per event. The alt-branch
columns count events whose accepted path includes any non-top-1 child and the
fraction of accepted tokens at or below the first alternate child. The depth-3
F-width2 arm was already running before that logger was added, so its alt-branch
path statistic is unavailable.

## Sweep Matrix

| Arm | Depth | Width | Nodes | Accept/event | Accept/draft | Decode TPS | Event ms | Alt-branch events | Alt accepted tokens | Resolved | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| E3 chain | 3 | 1 | 3 | 2.323 | 0.774 | 37.15 | 89.50 | n/a | n/a | 1/4 | clean rerun; prefix cache cold at start |
| F tree-delta spine | 3 | 1 | 3 | 2.363 | 0.788 | 34.21 | 98.42 | n/a | n/a | 1/4 | clean rerun; FULL requested, GDN forced FULL_AND_PIECEWISE |
| F tree-delta width2 | 3 | 2 | 14 | 1.981 | 0.142 | 14.85 | 199.47 | unavailable | unavailable | 1/4 | clean rerun; launched before accepted-path logger |
| F tree-delta width2 remeasure | 3 | 2 | 14 | 2.048 | 0.146 | 15.89 | 190.80 | 3635/18607 (19.5%) | 7158/36965 (19.4%) | 2/4 | clean rerun; accepted-path logger live |
| F tree-delta width2 | 5 | 2 | 62 | 2.421 | 0.039 | 7.44 | 455.81 | 1646/6422 (25.6%) | 4400/14962 (29.4%) | 1/4 | clean rerun; accepted-path logger live; GPU memory target lowered to 0.85 after launch-threshold miss at 0.86 |
| E5 chain | 5 | 1 | 5 | 3.150 | 0.630 | 26.86 | 154.54 | n/a | n/a | 2/4 | clean rerun; FULL requested; GPU memory target lowered to 0.85 after launch-threshold miss at 0.86 |

Depth-3 result: tree-delta spine remains lossless on the real B=4 agentic
workload but is slower than E3. The 14-node real k=2 tree does not improve
acceptance over the spine/E3 on this workload and roughly doubles event time, so
the width-2 depth-3 branch is not a speed win.

Final narrowed result: alternate branches are not zero. The depth-3 k=2 tree
accepted alternate-branch tokens on `19.4%` of accepted path tokens, and the
depth-5 k=2 tree accepted alternate-branch tokens on `29.4%` of accepted path
tokens. That extra branching did not translate into throughput: E5 chain is the
depth-paired baseline for F-w2-d5 and is much faster (`26.86` vs `7.44` decode
TPS) while also accepting more per event (`3.150` vs `2.421`). E3 remains the
fastest arm in this five-arm closeout matrix. E6 and F-w2-d6 were explicitly
dropped from the narrowed scope and were not run.

## Final Closeout Matrix

| Arm | Depth | Width | Nodes | Accept/event | Decode TPS | Event ms | Alt branch token % | Resolved | Nsight tables |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| E3 chain | 3 | 1 | 3 | 2.323 | 37.15 | 89.50 | n/a | 1/4 | 29; no CUDA kernel tables |
| F tree-delta spine | 3 | 1 | 3 | 2.363 | 34.21 | 98.42 | n/a | 1/4 | 29; no CUDA kernel tables |
| F tree-delta width2 remeasure | 3 | 2 | 14 | 2.048 | 15.89 | 190.80 | 19.4% | 2/4 | 29; no CUDA kernel tables |
| F tree-delta width2 | 5 | 2 | 62 | 2.421 | 7.44 | 455.81 | 29.4% | 1/4 | 29; no CUDA kernel tables |
| E5 chain | 5 | 1 | 5 | 3.150 | 26.86 | 154.54 | n/a | 2/4 | 29; no CUDA kernel tables |

## Per-Arm Details

### E3 chain depth 3

- Experiment: `output/roundf_clean_agentic_b4_E3_d3_20260530T1846Z`
- Clean-slate proof: `clean_slate.json`
- Summary: `summary.json`
- Sliced traces: `dgx_steptrace_window.jsonl`, `per_req_spec_trace_window.jsonl`
- Acceptance distribution: `{0: 2278, 1: 2482, 2: 2276, 3: 13747}`
- Tasks: `astropy__astropy-12907` resolved; the other three failed.
- Nsight export: available, but no CUDA kernel tables in sqlite.

### F Tree-Delta Spine Depth 3

- Experiment: `output/roundf_clean_agentic_b4_Fspine_d3_20260530T1938Z`
- Clean-slate proof: `clean_slate.json`
- Summary: `summary.json`
- Sliced traces: `dgx_steptrace_window.jsonl`, `per_req_spec_trace_window.jsonl`
- Acceptance distribution: `{0: 3501, 1: 2696, 2: 2429, 3: 20156}`
- Tasks: `astropy__astropy-12907` resolved; the other three failed.
- Nsight export: available, but no CUDA kernel tables in sqlite.
- Launch note: `CUDAGraphMode.FULL` was requested, but vLLM 0.19.0 reported
  `GDNAttentionBackend` support as `UNIFORM_BATCH` and forced
  `FULL_AND_PIECEWISE`.
- Repair note: the first clean F-spine launch exposed a CUDA graph capture
  failure from allocating depth-row tensors inside capture; commit `fb6c99fa`
  caches/reuses those static tensors before capture.

### F Tree-Delta Width2 Depth 3

- Experiment: `output/roundf_clean_agentic_b4_Fw2_d3_20260530T2037Z`
- Clean-slate proof: `clean_slate.json`
- Summary: `summary.json`
- Sliced traces: `dgx_steptrace_window.jsonl`, `per_req_spec_trace_window.jsonl`
- Acceptance distribution: `{0: 3464, 1: 1743, 2: 1251, 3: 8390}`
- Tasks: `astropy__astropy-12907` resolved; the other three failed.
- Nsight export: available, but no CUDA kernel tables in sqlite.
- Alt-branch path log: unavailable for this arm because it was launched before
  commit `692d3be3` added `tree_accept_path.jsonl`.

### F Tree-Delta Width2 Depth 3 Remeasure

- Experiment: `output/roundf_clean_agentic_b4_Fw2_d3_remeasure_20260530T2254Z`
- Clean-slate proof: `clean_slate.json`
- Summary: `summary.json`
- Sliced traces: `dgx_steptrace_window.jsonl`, `per_req_spec_trace_window.jsonl`,
  `tree_accept_path_window.jsonl`
- Acceptance distribution: `{0: 4037, 1: 1798, 2: 1475, 3: 10738}`
- Alt-branch path events: `3635/18607` events (`19.5%`) included at least one
  non-top-1 child; `7158/36965` accepted path tokens (`19.4%`) were at or below
  an alternate branch.
- Tasks: `astropy__astropy-12907` and `astropy__astropy-13236` resolved; the
  other two failed.
- Nsight export: available, but no CUDA kernel tables in sqlite.

### F Tree-Delta Width2 Depth 5

- Experiment: `output/roundf_clean_agentic_b4_Fw2_d5_20260530T2155Z`
- Clean-slate proof: `clean_slate.json`
- Summary: `summary.json`
- Sliced traces: `dgx_steptrace_window.jsonl`, `per_req_spec_trace_window.jsonl`,
  `tree_accept_path_window.jsonl`
- Acceptance distribution: `{0: 2008, 1: 835, 2: 434, 3: 473, 4: 312, 5: 2118}`
- Alt-branch path events: `1646/6422` events (`25.6%`) included at least one
  non-top-1 child; `4400/14962` accepted path tokens (`29.4%`) were at or below
  an alternate branch.
- Tasks: `astropy__astropy-12907` resolved; the other three failed.
- Nsight export: available, but no CUDA kernel tables in sqlite.
- Launch note: accepted-path logging was verified with a short authenticated
  probe before the clean relaunch. The first clean depth-5 launch at GPU memory
  target `0.86` missed the vLLM startup free-memory threshold by about 0.13 GiB,
  so this arm used `LUMO_GPU_MEMORY_UTILIZATION=0.85`.

### E5 Chain Depth 5

- Experiment: `output/roundf_clean_agentic_b4_E5_d5_20260531T0008Z`
- Clean-slate proof: `clean_slate.json`
- Summary: `summary.json`
- Sliced traces: `dgx_steptrace_window.jsonl`, `per_req_spec_trace_window.jsonl`
- Acceptance distribution: `{0: 3151, 1: 3335, 2: 2834, 3: 2249, 4: 1830, 5: 10347}`
- Tasks: `astropy__astropy-12907` and `astropy__astropy-13236` resolved; the
  other two failed.
- Nsight export: available, but no CUDA kernel tables in sqlite.
- Launch note: the first FULL E5 launch hit a transient `lmcache` download
  failure before serving; the retry at GPU memory target `0.86` then missed the
  vLLM startup free-memory threshold by about 0.23 GiB. The measured arm used a
  fresh clean relaunch with `LUMO_GPU_MEMORY_UTILIZATION=0.85`.
