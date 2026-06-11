# FR13 B=1 native-vs-chain5 profiler bind

Date: 2026-06-11 UTC.

## Verdict

The canonical FR10/FR13 launchers now have an off-by-default Nsight Systems wrapper around the actual in-container `vllm serve` command. With the wrapper enabled, a short B=1 native MTP-5 vs chain5 replay-on profile was captured under the canonical launchers.

The profile binds CUDA graph-node, runtime API, and GPU memcpy/memset evidence for the remaining B=1 speed tax. It does not bind per-kernel attribution: both exported kernel summary tables are empty, so `_tree_gdn_kernel`, `_tree_gdn_replay_kernel`, native GDN/update kernels, attention kernels, and individual graph kernel nodes are not visible in this export.

## Launcher change

Updated:

- `scripts/fr10_launch_speed_server.sh`
- `scripts/fr13_launch_forked_fa2_tree_server.sh`

The wrapper is disabled by default:

- `LUMO_NSYS_WRAP_VLLM=${LUMO_NSYS_WRAP_VLLM:-0}`
- default output: `/logs/nsys_vllm_${CONTAINER}`
- no `/opt/nvidia` or `/usr/local/cuda-13.0` host mounts are added unless `LUMO_NSYS_WRAP_VLLM` is truthy

When enabled, the launchers mount `/opt/nvidia` and `/usr/local/cuda-13.0` read-only, pass the `LUMO_NSYS_*` env vars into the container, build `NSYS_PREFIX=()` inside the container shell, and execute:

```bash
exec "${NSYS_PREFIX[@]}" vllm serve /models/qwen3.6-27b-fp8 ...
```

When disabled, `NSYS_PREFIX` remains empty and the expansion is the normal `exec vllm serve ...` path.

## B=1 live profile

Run root: `output/fr13_b1_profile_bind/`

Native arm:

- launcher: `scripts/fr10_launch_speed_server.sh`
- container: `fr13-b1-prof-native`
- `MAX_NUM_SEQS=1`, `BATCH_INVARIANT=0`, `FR10_METRICS=0`
- `FR10_ENABLE_TREE_GDN=0`
- `FR10_DECODE_MODE_DEFAULT=naive_mtp`
- `SPEC_CONFIG='{"method":"qwen3_5_mtp","num_speculative_tokens":5}'`
- `LUMO_NSYS_WRAP_VLLM=1`, `LUMO_NSYS_DELAY_S=360`, `LUMO_NSYS_DURATION_S=240`
- `LUMO_NSYS_OUTPUT=/logs/native_nsys`
- artifacts: `output/fr13_b1_profile_bind/native/logs/native_nsys.nsys-rep`, `output/fr13_b1_profile_bind/native/logs/native_nsys.sqlite`

Chain5 replay-on arm:

- launcher: `scripts/fr13_launch_forked_fa2_tree_server.sh`
- container: `fr13-b1-prof-chain`
- `MAX_NUM_SEQS=1`, `BATCH_INVARIANT=0`, `FR10_METRICS=0`
- `FR13_REPLAY_ROUTE=1`
- `TREE='[(0,), (0, 0), (0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0, 0)]'`
- `LUMO_NSYS_WRAP_VLLM=1`, `LUMO_NSYS_DELAY_S=360`, `LUMO_NSYS_DURATION_S=240`
- `LUMO_NSYS_OUTPUT=/logs/chain_nsys`
- artifacts: `output/fr13_b1_profile_bind/chain_replay_on/logs/chain_nsys.nsys-rep`, `output/fr13_b1_profile_bind/chain_replay_on/logs/chain_nsys.sqlite`

Probe command shape for both arms:

```bash
python3 scripts/fr12_deliverable_swe4_probe.py \
  --endpoint http://127.0.0.1:9950 \
  --model qwen3.6-27b \
  --mode naive_mtp|tree_mtp \
  --samples-per-prompt 1 \
  --batch-size 1 \
  --max-tokens 64 \
  --temperature 0.0 \
  --top-p 1.0 \
  --seed 1313 \
  --wait-health 0 \
  --request-timeout 600
```

Speed numbers below are from `/metrics` deltas in the probe JSON, not TPS/accept division.

| arm | `/metrics` decode seconds | spec drafts | s/forward | accept/event |
|---|---:|---:|---:|---:|
| native MTP-5 | 15.626623 | 72 | 0.217036 | 2.597222 |
| chain5 replay-on | 21.003351 | 69 | 0.304396 | 2.797101 |
| chain/native | - | - | 1.402513x | - |

This reproduces the previously bound residual tax class under a profiled launch.

## Profile table

Nsight export/reduction used `/opt/nvidia/nsight-systems-cli/2026.2.1/bin/nsys`. The host `nsys` on `PATH` was older and could not read these reports.

| signal | native MTP-5 | chain5 replay-on | chain/native |
|---|---:|---:|---:|
| CUDA graph node events, name grouped as `Graph Node Creation` | 14,368 | 43,270 | 3.011553x |
| graph node events per draft | 199.555556 | 627.101449 | 3.142491x |
| `cudaGraphLaunch` calls | 868 | 824 | 0.949309x |
| `cudaGraphLaunch` total runtime API time | 22.016800 ms | 58.725568 ms | 2.667306x |
| `cudaGraphLaunch` avg runtime API time | 25.364977 us | 71.268893 us | 2.809724x |
| `cudaMemcpyAsync` runtime API calls | 7,865 | 18,512 | 2.353719x |
| `cudaMemcpyAsync` runtime API time | 447.655232 ms | 22,809.608800 ms | 50.952404x |
| GPU memcpy/memset total time | 2.763616 ms | 38.549056 ms | 13.947004x |
| GPU Device-to-Host memcpy count | 326 | 7,752 | 23.779141x |
| GPU Device-to-Host memcpy bytes | 5,704 | 4,017,072 | 704.253857x |
| GPU Device-to-Host memcpy time | 0.454176 ms | 36.621952 ms | 80.633825x |
| `cudaLaunchKernel` calls | 37,807 | 61,836 | 1.635570x |
| `cuLaunchKernel` calls | 14,112 | 16,030 | 1.135913x |

Top runtime API rows by cumulative time:

| arm | top rows |
|---|---|
| native MTP-5 | `cudaMalloc` 55.933 s / 74 calls; `cudaEventSynchronize` 26.145 s / 477 calls; `cudaStreamSynchronize` 6.553 s / 519 calls; `cudaDeviceSynchronize` 3.021 s / 472 calls; `cudaLaunchKernel` 1.552 s / 37,807 calls |
| chain5 replay-on | `cudaMalloc` 23.567 s / 118 calls; `cudaMemcpyAsync` 22.810 s / 18,512 calls; `cudaDeviceSynchronize` 4.941 s / 636 calls; `cudaLaunchKernel` 1.821 s / 61,836 calls; `cuLaunchKernel` 1.136 s / 16,030 calls |

GPU memcpy/memset rows:

| arm | rows |
|---|---|
| native MTP-5 | HtoD memcpy 3,363 ops / 2.028512 ms / 3,050,430 bytes; DtoH memcpy 326 ops / 0.454176 ms / 5,704 bytes; memset 574 ops / 0.280928 ms |
| chain5 replay-on | DtoH memcpy 7,752 ops / 36.621952 ms / 4,017,072 bytes; HtoD memcpy 4,302 ops / 1.229216 ms / 357,898 bytes; memset 1,378 ops / 0.697888 ms |

## Kernel visibility limit

Attempted Nsight reductions:

- `cuda_gpu_kern_sum`
- `cuda_kern_exec_sum`

Both produced empty CSVs for both arms:

- `output/fr13_b1_profile_bind/native/native_cuda_gpu_kern_sum_cuda_gpu_kern_sum.csv`
- `output/fr13_b1_profile_bind/native/native_cuda_extra_cuda_kern_exec_sum.csv`
- `output/fr13_b1_profile_bind/chain_replay_on/chain_cuda_gpu_kern_sum_cuda_gpu_kern_sum.csv`
- `output/fr13_b1_profile_bind/chain_replay_on/chain_cuda_extra_cuda_kern_exec_sum.csv`

The SQLite exports do contain `CUDA_GRAPH_NODE_EVENTS`, `CUPTI_ACTIVITY_KIND_MEMCPY`, `CUPTI_ACTIVITY_KIND_MEMSET`, and `CUPTI_ACTIVITY_KIND_RUNTIME`. Diagnostics also report CUDA/CUPTI event collection (`215,760` CUDA events in native, `326,777` CUDA events in chain), but the kernel event summaries remain absent. Therefore this bind is graph/runtime/mem evidence, not per-kernel attribution.

## Tests

```bash
bash -n scripts/fr10_launch_speed_server.sh
bash -n scripts/fr13_launch_forked_fa2_tree_server.sh
pytest -q \
  tests/test_fr13_nsys_launcher_wiring.py \
  tests/test_fr13_replay_route_wiring.py::test_launcher_passthrough_defaults_replay_route_on \
  tests/test_fr13_s1_bonus_row.py::test_launcher_forwards_bonus_self_flag \
  tests/test_fr13_nondet_chase_fixes.py::test_launcher_forwards_fr13_chase_flags
```

Result: `5 passed in 0.77s`.

## Next speed target

The remaining B=1 speed target is still the broader `tree_mtp` graph/row-shape/scheduler path. This profile makes the next concrete target the chain5 pure-spine path's extra graph-node creation/runtime and host/device copy surface, especially the DtoH memcpy activity. Do not move to B=4 from this bind; the next B=1 speed work should collapse or bypass the pure-spine `tree_mtp` row/metadata/scheduler path, or route chain5 through the native MTP-5 path while preserving the tree verifier contract.
