# Fixed32 GDN GQA3 BV16 source-bound candidate

Status: **offline SM121a codegen/resource gate PASS; default off; no
performance promotion**.

Exact scope: Tail23 B1 and Hydra27 B4, physical32, K64/root1. The served
reference remains GQA3 BV8. The explicit `gqa_group3_bv16` selector halves
the value-tile grid from 16 to 8 CTAs per key head.

Across 48 GDN layers this removes 6,144 B1 or 24,576 B4 CTAs, 96 MiB or
384 MiB of repeated q+k reads, and 393,216 or 1,572,864 q/k norm reductions.
Both independent fresh-cache builds were byte-identical and spill-free.

The base profile rises from 108 to 128 registers/thread, which can reduce
occupancy and must be resolved by real B1/B4 timing. The committer profile
drops from 118 to 112 registers/thread. This artifact is source/resource
evidence only: no GPU kernel, serving task, or timing measurement ran.

Raw cubin, PTX, SASS, compiler IR, and caches are intentionally excluded.
