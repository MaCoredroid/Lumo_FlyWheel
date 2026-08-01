# FR13 fixed32 GDN parent-group source candidate

Status: **source candidate only, default OFF, no GPU measurement**.

## Change

The exact fixed32 logical GDN schedule remains `[1, 11]`: one level-0 root
path and eleven level-1 child paths. The candidate changes only level 1. Its
five programs correspond to the five distinct exported parents and execute
each parent's original child paths in descriptor order:

```text
parent 14: paths 0, 9, 10
parent  0: paths 1, 2
parent  1: paths 3, 4
parent  4: paths 5, 6
parent  9: paths 7, 8
```

Each program loads its fp32 parent-state tile once, resets the carried state to
those bytes for each member path, and calls the unchanged `_gdn_node_step` in
the original per-path node order. Level 0 stays on the established kernel and
remains the only state-export, invocation-counter, and freshness-flag writer.
All 32 output and K/V/A/B ring rows retain exactly one logical writer.

The same kernel supports B1 with physical parent-node export rows and B2-B4
with the established compact five-slot export layout. The route requires an
exact fixed32 mode and `FR13_TREE_GDN_GEOM_OVERRIDE=BV=8`; it is armed only by
`FR13_FIXED32_GDN_PARENT_GROUP=1` or a worker-visible `.arm` file containing
exactly `1`. Missing, malformed, non-fixed32, non-BV8, wider-BV, and descriptor
drift cases fail closed. No production authorization is included.

## Static work model

At the deployed 48 value heads, `DIM_V=128`, `DIM_K=128`, and `BV=8`, the
candidate retains two level launches and all 32 node updates per request/layer.
It changes the per-request level grids from `[1, 11]` to `[1, 5]`:

- Physical program units: 12 to 6 per request/layer.
- Logical level-1 parent-state reads: 11 to 5.
- One complete fp32 parent state: 3,145,728 bytes.
- B1 avoided CTAs across 48 layers: 221,184.
- B4 avoided CTAs across 48 layers: 884,736.
- B1 modeled parent-read bytes avoided: 905,969,664.
- B4 modeled parent-read bytes avoided: 3,623,878,656.
- B1 modeled export-plus-read handoff: 2,415,919,104 to 1,509,949,440 bytes.

These are descriptor-derived work counts, not measured bandwidth or latency.

## Verification boundary

CPU/static tests validate exact path/parent coverage, original member order,
all-node output/ring single writers, compact slots, B1/B4 launch wiring,
default-off/fail-closed selection, unchanged generic-kernel AST, and observer
separation of the logical `[1, 11]` contract from physical `[1, 5]` work.

No GPU or container command was run because the existing B4 campaign owned the
device. Triton compilation, SM121 resource usage, CUDA graph capture, raw-byte
equivalence, real SWE-Verified B1/B4 timing, and full-wall acceptance remain
unresolved. The primary compile/performance risk is retaining the parent tile
live across up to three child paths, which can increase registers or spill and
erase the modeled read reduction.

## Required next gate

Compile and inspect resources on SM121, then compare the candidate and
incumbent on the same authenticated real SWE-Verified work at B1 and canonical
exact4 B4. Require raw-byte equality for output, compact/actual export state,
K/V/A/B rings, flags, and counter before measuring full-step TPS and the GDN
phase breakdown. Synthetic or probe timing is not acceptance evidence.
