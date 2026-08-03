# Fixed32 ordered GDN real-task gate runners

Verdict: **SOURCE_READY_REAL_TASK_RUNNERS_UNQUALIFIED**.

This supersedes the earlier route-only source record. The default-off
`fixed32_gdn_single_launch_tree_v2` diagnostic now has three fixed-scope real
SWE-Verified entrypoints:

- Hydra27 B1 on the pinned one-task diagnostic.
- Tail23 B4 on the canonical exact4 task set.
- Hydra27 B4 on the canonical exact4 task set.

Each process bakes exactly one expected batch. Captures are keyed by batch,
FULL-graph identity, and graph signature; a B1 replay cannot consume or inherit
a B4 capture or PASS. The live result binds mode, logical topology, batch,
physical32, BV8, K64/root1, authenticated trigger task, graph signature, source,
reference service, byte equality, and restored state.

The reducer does not trust the live result alone. It requires clean task
completion and terminal SWE verdicts, reconstructs the finalized authenticated
traffic audit from raw per-task evidence, replays the exact Qwen compaction
algebra and per-task/campaign proof bindings, validates finalized proxy and
engine ingress, validates the complete graph/work census, and requires byte-for-
byte stable runtime and external manifests from launch through end. Credentials
are distinct for `hydra27:b1`, `tail23:b4`, and `hydra27:b4`.

## Production boundary

Every live result and credential records `performance_measurement=false` and
`acceptance_valid=false`. The production resolver still rejects
`single_launch`. Production and timing remain unavailable until all three real
GPU gates pass and a later production credential binds those disjoint results.

## Validation

- Combined GDN, runner/reducer, ingress, draft-head, timing-contract, and final
  FULL-preseed compatibility suite: `275 passed, 1 skipped` (the skip requires
  CUDA/Triton).
- `bash -n`, `py_compile`, and `git diff --check`: pass.

No GPU kernel, SWE-Verified task, probe, timing arm, TPS measurement,
hardware-floor measurement, or production authorization ran for this source
artifact.
