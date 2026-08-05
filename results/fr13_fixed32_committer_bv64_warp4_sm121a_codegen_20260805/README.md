# Fixed32 committer BV64/4-warp SM121a codegen

This artifact statically qualifies the default-off
`FR13_FIXED32_COMMITTER_BV64_WARP4` candidate at source revision
`5d15020c99aa58365096ee1c27a2c1afc4825644`.

The candidate keeps the existing one-launch, 48-layer ordered committer kernel
and changes only its independent value-row geometry. Each physical value head
uses two BV64/4-warp CTAs instead of one BV128/8-warp CTA. The K reduction,
accepted-path order, live-step recurrence, and final disjoint state write are
the same kernel instructions.

## Static result

| Batch | Registers/thread incumbent -> candidate | Registers/CTA | Threads/CTA | Static SASS | cuobjdump shared |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1 | 169 -> 128 | 43,264 -> 16,384 | 256 -> 128 | 828 -> 895 | 0 -> 1,024 B |
| B4 | 161 -> 164 | 41,216 -> 20,992 | 256 -> 128 | 887 -> 894 | 0 -> 1,024 B |

All eight primary/rebuild cubins are byte-identical across fresh caches. All
variants are stack- and local-memory-free. Triton reports 256 launch shared
bytes for the candidate and zero for the incumbent.

The lower CTA register allocation is not free. Programs per event double from
2,304 to 4,608 at B1 and from 9,216 to 18,432 at B4. H-state and V-vector
payload bytes are split across the two disjoint row tiles and remain constant
in aggregate. The K vector, accepted-path metadata, and scalar gate inputs are
loaded once by each tile, so those inputs are duplicated once.

## Qualification boundary

The source guard accepts only exact Hydra27 physical32 K64/root1 layer batching
at B1 or B4. The arm defaults off. This is offline CUDA `sm_121a` codegen
evidence: it did not execute a GPU, SWE-Verified task, acceptance measurement,
or timing campaign. The resource reduction makes the variant eligible for a
real B1/B4 comparison; it is not performance acceptance.

Raw cubin/PTX/SASS trees are intentionally omitted. The reduced summary,
reproduction commands, source bindings, focused test output, and checksums are
committed here.
