# Rejected B4 graph-byte gate: 2026-07-31T21:38:57Z

This is a zero-task process-attestation rejection from the real SWE-Verified
exact4 B4 graph-byte campaign. It proves that the 20 GiB KV-cache correction
boots and captures all graphs, but it is not task-level kernel correctness,
timing, throughput, or floor-acceptance evidence.

## Result

- Source: `04cd4d39b4741f0637db5613d6b5fcb86f364605`
- Workload: canonical exact4, B4, concurrency 4
- Manual KV cache: `21,474,836,480` bytes
- KV cache created: `76,800` tokens
- PIECEWISE graphs completed: 8 of 8
- FULL graphs completed: 4 of 4
- FULL graph memory: `6.93 GiB`
- Health: passed after 443 seconds
- Tasks started: 0

The run was rejected because the exact PID1 argv allowlist still described the
pre-correction B4 command and therefore rejected the new pinned
`--kv-cache-memory-bytes` pair. The guard failed closed before task launch.
The terminal flush consequently returned `error:RuntimeError` with an empty
work census; this is rejection lifecycle evidence, not task or timing evidence.

The preserved `free_after_teardown.txt` snapshot was taken before the rejected
container was manually removed, so it is not final cleanup proof. The separately
timestamped Docker and GPU recaptures record the final zero-container,
zero-compute-process state.

## Correction

Commit `6c44fa780` adds the B4-only argument pair to the exact process identity,
keeps B1 free of the manual KV argument, and adds a value-tamper rejection test.
The focused suite passed 112 tests and the broad fixed32/kernel suite passed 748
tests with 7 skips.
