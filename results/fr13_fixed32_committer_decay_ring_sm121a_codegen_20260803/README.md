# Fixed32 committer decay ring SM121a codegen

This artifact qualifies the default-off
`FR13_FIXED32_COMMITTER_DECAY_RING` kernel layer at source revision
`ecfd1bd30` relative to the gate-ring parent `9dbad6245`.

The physical32 B1/B4 SFWD producer already evaluates the exact FP32
`decay = exp(g)` used by its live recurrence. The candidate writes that same
decay value into the existing packed FP32 gate ring and reuses it in the
committer. It adds no buffer and no launch. Raw BF16 `a/b` stores remain intact
because the authenticated native shadow graph consumes them when qualifying
previously unseen accepted depths.

## Static result

| Kernel | Batch | Registers incumbent -> candidate | Static SASS | Key delta |
| --- | ---: | ---: | ---: | --- |
| SFWD producer | B1 | 80 -> 80 | 1198 -> 1199 | STG 34 -> 34; nonlinear counts unchanged |
| SFWD producer | B4 | 80 -> 80 | 1198 -> 1199 | STG 34 -> 34; nonlinear counts unchanged |
| Native committer | B1 | 167 -> 169 | 817 -> 828 | EX2 1 -> 0; no spill/local/shared traffic |
| Native committer | B4 | 167 -> 161 | 872 -> 887 | EX2 1 -> 0; no spill/local/shared traffic |

All 24 primary/rebuild builds are stack-, local-, LDL-, STL-, and CALL-free.
Parent and current gate-only producer and committer SASS are byte-identical at
B1 and B4. The B2-B4 committer candidate uses a 167-register ceiling after an
uncapped B4 build exposed an unacceptable 217-register schedule; the accepted
B4 cubin uses 161 registers.

At four accepted drafts, the committer removes 11,520 B1 or 46,080 B4
state-decay exponentials. The two existing FP32 gate-ring scalar loads per live
step and all incumbent-reference ring stores are unchanged.

## Qualification boundary

This is offline CUDA `sm_121a` codegen evidence. It did not execute a GPU,
SWE-Verified task, acceptance measurement, or timing campaign. Serving remains
behind the authenticated real-event exact-state-byte depth gate; unseen
accepted depths serve the native reference after shadow comparison.

Raw cubin/PTX/SASS build trees are intentionally omitted. The reduced summary,
reproduction scripts, source bindings, focused test output, and checksums are
committed here.
