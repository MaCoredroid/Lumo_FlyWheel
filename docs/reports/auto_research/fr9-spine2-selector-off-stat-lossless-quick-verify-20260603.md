# FR9 selector-off spine-2 statistical-lossless quick verify

Date: 2026-06-03 UTC

Branch: `fr9-spine2-lossless-winner`

Objective: test whether selector-off `spines=2` preserves the served public output distribution relative to `spines=1` on the exact quick-verify SWE setup before any speed work.

## Setup

- Config: `Fb`, `mtp=5`, row mode `independent`, `LUMO_GPU_MEMORY_UTILIZATION=0.88`
- Workload: fixed 4-task SWE-Verified subset, `B=4`, `temperature=0.6`, `agent-wall-s=1800`, `eval-timeout-s=1800`
- Arms:
  - `s1a`: `fr9_lossless4_s1a_20260602T2216Z`, `spines=1`
  - `s1b`: `fr9_lossless4_s1b_wave_20260602T2351Z`, `spines=1`
  - `s2`: `fr9_lossless4_s2_wave_20260603T0023Z`, `spines=2`, selector off, `policy=lossless`, launched only with diagnostic override `LUMO_IR_DIAGNOSTIC_UNISOLATED=1`

Production remains fail-closed for unisolated `spines>1`; the diagnostic override is not a production default.

## Profiler status

No CUDA kernel timeline is claimed.

The built-in `--nsight` artifact for `s1a` exported a sqlite, but it had no CUDA kernel tables (`has_cuda_kernel_tables=false`). Host-side system-wide nsys attempts produced zero CUPTI kernel rows. The single allowed in-container vLLM-wrap attempt failed because the expected nsys binary path was missing inside the container:

```text
/opt/nvidia/nsight-systems-cli/2026.2.1/bin/nsys: No such file or directory
```

The fallback trace comparison is therefore limited to the served decode/speculation artifacts available from the runs. `s1b` and `s2` did not produce `dgx_steptrace.jsonl`; their `agentic_summary.steptrace` entries are empty. This quick verify cannot identify a CUDA-kernel ordering difference in the GDN/Mamba recurrent path.

## Per-arm task results

Driver elapsed below includes task execution plus eval handling from the experiment driver logs.

| Task | s1a verdict / elapsed | s1b verdict / elapsed | s2 verdict / elapsed | Wall-gate note |
| --- | --- | --- | --- | --- |
| `astropy__astropy-12907` | resolved / 1854.7s | resolved / 1855.9s | resolved / 1853.3s | matched wall hit |
| `astropy__astropy-13033` | failed / 1861.5s | failed / 1853.5s | failed / 1856.8s | matched wall hit |
| `astropy__astropy-13236` | resolved / 1860.7s | failed / 1854.9s | failed / 3653.7s | s2 first attempt hit wall with empty patch and retried; confounded |
| `astropy__astropy-13398` | failed / 1530.7s | failed / 1667.9s | failed / 1868.4s | wall behavior differs, but verdict and patch are identical |

Aggregate quality:

| Comparison | Result |
| --- | --- |
| s1 self-noise (`s1a` vs `s1b`) | 1 task verdict flip: `13236` resolved in `s1a`, failed in `s1b` |
| `s1b` vs `s2` | 0 task verdict flips on the 4 completed tasks |
| Cleanest wall-matched subset | `12907` and `13033`; both verdicts match |

## Distribution checks

Acceptance-length distributions from `agentic_summary.acceptance.acc_dist`:

| Arm | Events | `acc_dist` | Accept/event | Accept/draft |
| --- | ---: | --- | ---: | ---: |
| s1a | 16233 | `{0:1936,1:2041,2:1693,3:1457,4:1183,5:7923}` | 3.33549 | 0.66710 |
| s1b | 18369 | `{0:2260,1:2494,2:1991,3:1659,4:1344,5:8621}` | 3.26278 | 0.65256 |
| s2 | 35804 | `{0:4486,1:4922,2:3876,3:3340,4:2718,5:16462}` | 3.23640 | 0.64728 |

Total-variation distances:

| Comparison | TV |
| --- | ---: |
| s1 self-noise: `s1a` vs `s1b` | 0.0187565 |
| selector-off spine-2: `s1b` vs `s2` | 0.00967536 |
| reference only: `s1a` vs `s2` | 0.0282988 |

The measured `s1b` vs `s2` acceptance-length TV is within the `spines=1` relaunch self-distance floor.

Patch-output checks:

| Task | s1a patch | s1b patch | s2 patch | Note |
| --- | --- | --- | --- | --- |
| `12907` | `0172ca0d430e` / 504B | `0172ca0d430e` / 504B | `0172ca0d430e` / 504B | exact match |
| `13033` | `4d17018b852a` / 1399B | `650a857edca9` / 1086B | `903c7248552b` / 1025B | all fail; s1b-s2 text similarity is closer than s1a-s1b |
| `13236` | `09542980a217` / 1119B | `b29037e47577` / 716B | `cfbd1f98249d` / 774B | s2 retry confound; s1 self already flips verdict |
| `13398` | `6290cab28963` / 580B | `6290cab28963` / 580B | `6290cab28963` / 580B | exact match |

## Selector-off policy trace

The `s2` independent-winner trace confirms selector-off/public-stream behavior:

- `rows=18351`
- `policy=lossless` for all rows
- `selector_enabled_events=0`
- `winner_nonzero_spine_events=0`
- `non_lossless_public_stream_events=0`
- `hidden_winner_suppressed_events=831`
- `lossless_suppressed_superset_events=831`
- `recovered_token_total=0`

This means the hidden spine was not published into the public stream in the diagnostic run.

## Verdict

For this 4-task quick verify, selector-off `spines=2` is statistically within the `spines=1` self-noise floor on served quality and acceptance-length distribution:

- Quality: `s1b` vs `s2` has no verdict flips, while `s1a` vs `s1b` has one verdict flip.
- Distribution: `TV(s1b,s2)=0.00967536`, below the measured self-distance `TV(s1a,s1b)=0.0187565`.
- Public stream policy: hidden winners are suppressed and no non-lossless public-stream events were recorded.

This is not a bitwise-losslessness proof and not a CUDA-kernel isolation proof. The wall-gate caveats are real: `13236` is confounded by the s2 empty-patch retry, and `13398` has different wall-hit timing despite identical patch and verdict. Under the requested statistical criterion, the diagnostic run lands in the "already statistically lossless" case for this quick verify, so no recurrent-state isolation implementation is triggered by this gate yet. Speed and cost are intentionally not claimed.
