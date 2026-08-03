# Fixed32 ordered GDN real-task byte-gate route

Verdict: **SOURCE_READY_REAL_TASK_UNQUALIFIED**.

This artifact records the default-off, fail-closed qualification route for
`fixed32_gdn_single_launch_tree_v2`. The route is selected only with
`FR13_FIXED32_GDN_PATH_BV_CANDIDATE=single_launch` and requires physical32,
BV8, K64/root1, FULL CUDA graphs, metrics/ring/flags, and either the exact B1
or exact B4 serving shape. Both Tail23 (`tail6_fixed32`) and Hydra27
(`hydra27_fixed32`) are accepted as distinct mode-bound qualifications.

The route reuses authenticated fixed32 ingress. Canonical B1 admits only
`astropy__astropy-12907`; B4 admits only the canonical exact4 task set. The
candidate comparison runs after the first measured FULL-graph replay. It runs
the incumbent two-launch GDN and the ordered single-launch candidate from the
same persistent state, compares output plus K/V/A/B rings, flags, and counter
bytes, restores output and all persistent state, and serves the incumbent.
B2/B3 graph shapes are deliberately skipped and cannot satisfy the gate.

## Production boundary

The emitted live result is source-, mode-, batch-, topology-, physical32-,
BV8-, K64/root1-, candidate-, and authenticated-task-bound. It records
`production_eligible=false`. The production resolver accepts only prior
numeric BV credentials and rejects `single_launch`; the launcher also removes
the legacy direct-arm sidecar and does not forward its environment selector.
Production and timing remain unavailable until real B1 and both Tail23 and
Hydra27 exact4 credentials exist and are independently bound by a later
production credential.

## Validation

- GDN, ingress, graph-gate, schedule, exact-I/O, and BV8 compatibility suite:
  `152 passed, 1 skipped` (the skip requires CUDA/Triton).
- Full-preseed and SFWD compatibility suite: `83 passed`.
- `bash -n`, `py_compile`, and `git diff --check`: pass.

No GPU kernel, SWE-Verified task, synthetic probe, timing arm, TPS measurement,
hardware-floor measurement, or production authorization ran for this artifact.

