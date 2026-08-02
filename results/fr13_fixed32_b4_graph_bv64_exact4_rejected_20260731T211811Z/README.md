# Rejected B4 graph-byte gate: 2026-07-31T21:18:11Z

This is a zero-task boot-capacity rejection from the real SWE-Verified exact4
B4 graph-byte campaign. It is not kernel correctness, timing, throughput, or
floor-acceptance evidence.

## Result

- Source: `6bbafd5caee2d95081ec049faeaa2a2d1b4743a5`
- Candidate: fixed32 batched GDN `BV=64`, shadowed behind served `BV=8`
- Workload: canonical exact4, B4, concurrency 4
- Engine result: rejected before health, graph capture, and task launch
- Available KV cache after CUDA-graph memory profiling: `6.45 GiB`
- Minimum KV cache required for `MAX_MODEL_LEN=131072`: `15.07 GiB`
- Tasks started: 0
- Candidate graph records: 0

The source and external manifests are byte-identical at launch and end. The
stopped container was retained long enough to preserve its complete Docker log,
then removed; no compute process remained.

## Correction

Commit `d46b40dac` pins fixed32 B4 to a manual 20 GiB KV cache for both the graph
gate and the paired stock/candidate timing runner. This preserves the canonical
task set, B4 concurrency, model context length, dtype, and kernel selectors
while bypassing only the unstable CUDA-graph memory estimate. The correction
passed `747` tests with `7` skips plus shell syntax checks.
