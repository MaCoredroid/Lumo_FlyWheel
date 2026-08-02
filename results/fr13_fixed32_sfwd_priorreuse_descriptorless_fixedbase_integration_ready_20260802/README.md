# Descriptorless fixed-base SFWD B1 gate readiness

Status: **READY_NOT_EXECUTED**.

Source commit `8ec4c5126cbb323f27143a06064629b76e142550` integrates the
descriptorless fixed-base row32/C64 prior-reuse kernel into the default-off,
reference-served real SWE-Verified K64/root1 B1 byte gate.

The launch path enforces the source-bound live layout before the candidate
kernel starts: `x` is `[32,10240]` with strides `[16384,1]`; output is dense
with strides `[10240,1]`; commit-source stage is `[36,10240]` with strides
`[10240,1]`; and convolution weights are `[10240,4]` with strides `[4,1]`.
The kernel signature has no source descriptor or runtime x/weight strides.
The host still validates the fixed32 source descriptor before launch.

The shadow output is allocated explicitly dense. The candidate runs once per
layer, while the incumbent remains the only served output and commit source.
The validator requires all 48 layers and exact bytes on both `conv_out` and
`commit_source_stage`. It cannot emit PASS unless the one authenticated B1
task uses K64/root1 with the pinned block map and unchanged launch/end source,
runtime, and external manifests.

This port reuses the gate protocol that qualified the older descriptor-based
C64 prior-reuse candidate on one real B1 task at result commit
`580784d8c7efa2b40ec5d6edd850bd84a45d3647`. That predecessor compared
34,893,987,840 bytes with zero differences while always serving the reference.
Those bytes do not qualify this new kernel source; a fresh run is required.

Prior offline SM121a evidence for the exact unchanged kernel function reports
40 registers/thread, 718 static and 736 encoded SASS instructions, 37 LDG,
20 STG, and no stack, local memory, spills, shared memory, or calls at B1/B4.
Those are codegen facts, not runtime timing or floor acceptance.

Launch only when a real task slot is assigned:

```bash
RUNROOT=output/<new-root> TAG=<unique-tag> FORKED_FA2_SO=<pinned-so> \
  bash scripts/fr13_run_b1_sfwd_prior_reuse_gate.sh
```

No GPU kernel, Docker service, SWE task, timing run, or acceptance run was
launched while creating this READY package.
