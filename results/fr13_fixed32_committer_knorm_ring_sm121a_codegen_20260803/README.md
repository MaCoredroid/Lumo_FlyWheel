# Fixed32 committer K-norm ring SM121a codegen

This artifact qualifies the default-off `FR13_FIXED32_COMMITTER_KNORM_RING`
kernel layer at source revision
`b2b4ab6f5ec4ec1f7ac6b5606b711ef2c1f68d37`.

The physical32 B1/B4 SFWD kernel already computes the FP32 inverse norm of
each raw BF16 K vector. The candidate stores that existing scalar once per
`(layer, request, node, key-head)`. The one-launch native committer reloads it
for each live value-head recurrence instead of repeating a 128-wide reduction
and `rsqrt`. It adds no producer reduction and no launch.

## Static result

| Kernel | Batch | Registers incumbent -> candidate | Static SASS | Key delta |
| --- | ---: | ---: | ---: | --- |
| SFWD producer | B1 | 80 -> 80 | 1124 -> 1152 | scalar stores; RSQ/shuffle/barrier counts unchanged |
| SFWD producer | B4 | 80 -> 80 | 1124 -> 1152 | scalar stores; RSQ/shuffle/barrier counts unchanged |
| Native committer | B1 | 172 -> 193 | 913 -> 917 | RSQ 1 -> 0, shuffle 87 -> 80, barriers 3 -> 0 |
| Native committer | B4 | 202 -> 189 | 941 -> 938 | RSQ 1 -> 0, shuffle 87 -> 80, barriers 3 -> 0 |

All 24 primary/rebuild builds are stack-, local-, LDL-, STL-, and CALL-free.
The committer candidate also removes its reduction shared traffic and launch
scratch (`LDS/STS 2 -> 0`, launch shared bytes `16 -> 0`). Parent and current
default-off producer and committer SASS are byte-identical at both B1 and B4.

At four accepted drafts, the committer replaces 11,520 B1 or 46,080 B4 K-norm
reductions with the same count of FP32 scalar loads. The producer writes 24,576
B1 or 98,304 B4 scalar values while adding zero norm reductions.

## Qualification boundary

This is offline CUDA `sm_121a` codegen evidence. It did not execute a GPU,
SWE-Verified task, acceptance measurement, or timing campaign. Serving remains
behind the authenticated real-event, exact-state-byte depth gate; unseen
accepted depths serve the native reference after shadow comparison.

Raw cubin/PTX/SASS build trees are intentionally omitted. The reduced summary,
reproduction scripts, source bindings, focused test output, and checksums are
committed here.
