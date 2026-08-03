# Fixed32 committer gate ring SM121a codegen

This artifact qualifies the default-off
`FR13_FIXED32_COMMITTER_GATE_RING` kernel layer at source revision
`5700ddaf3ff51e0b8dba0d571069ba0d8c158ce6`.

The physical32 B1/B4 SFWD kernel already evaluates the exact FP32 recurrence
gates for every node. The candidate stores those existing `g/beta` scalars
once per `(layer, request, node, value-head)`. The one-launch native committer
reloads them for each live accepted-path step instead of reloading raw `a/b`
and repeating softplus, exponentials, and sigmoid. It adds no producer
nonlinear evaluation and no launch.

## Static result

| Kernel | Batch | Registers incumbent -> candidate | Static SASS | Key delta |
| --- | ---: | ---: | ---: | --- |
| SFWD producer | B1 | 80 -> 80 | 1152 -> 1198 | gate stores; arithmetic/MUFU counts unchanged |
| SFWD producer | B4 | 80 -> 80 | 1152 -> 1198 | gate stores; arithmetic/MUFU counts unchanged |
| Native committer | B1 | 193 -> 167 | 917 -> 817 | EX2 3 -> 1, RCP 1 -> 0, LDG 42 -> 40 |
| Native committer | B4 | 189 -> 167 | 938 -> 872 | EX2 3 -> 1, RCP 1 -> 0, LDG 42 -> 40 |

All 24 primary/rebuild builds are stack-, local-, LDL-, STL-, and CALL-free.
The producer candidate is explicitly capped at the incumbent 80-register
schedule. Parent and current K-norm-only producer and committer SASS are
byte-identical at B1 and B4.

At four accepted drafts, the committer removes 11,520 B1 or 46,080 B4 gate
nonlinear sets and replaces them with 23,040 or 92,160 FP32 scalar loads. The
producer writes 147,456 B1 or 589,824 B4 FP32 values, 589,824 bytes or
2,359,296 bytes per event across 48 layers.

## Qualification boundary

This is offline CUDA `sm_121a` codegen evidence. It did not execute a GPU,
SWE-Verified task, acceptance measurement, or timing campaign. Serving remains
behind the authenticated real-event exact-state-byte depth gate; unseen
accepted depths serve the native reference after shadow comparison.

Raw cubin/PTX/SASS build trees are intentionally omitted. The reduced summary,
reproduction scripts, source bindings, focused test output, and checksums are
committed here.
