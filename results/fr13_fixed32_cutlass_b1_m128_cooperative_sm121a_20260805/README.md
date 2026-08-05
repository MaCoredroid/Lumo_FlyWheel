# Fixed32 B1 M128 cooperative target

Status: **default off; pinned SM121a compile, link, and codegen audit passed;
real SWE-Verified byte gate and timing pending**.

## Kernel change

This candidate retains the B1 target path's `128x32x128` tile, StageCount2,
full-K operation, scale contract, cluster `(1,1,1)`, output mapping, and
source-bound 48-CTA direct scheduler. It changes only the CUTLASS kernel
schedule from blockwise ping-pong to blockwise cooperative SM120.

That schedule change alters the generated machine execution plan, so raw-byte
equivalence is not assumed. The candidate remains unavailable to production
until its exact linked binary passes the authenticated real SWE-Verified B1
physical32/K64/root1 target comparator over all five projection shapes.

## Offline evidence

The complete target translation unit compiled twice against vLLM
`fe9c3d6c5f66c873d196800384ed6880687b9e52` and CUTLASS
`da5e086dab31d63815acafdac9a9c5893b1c69e2` for SM121a. ELF metadata differs
between the two objects, while the extracted target-kernel SASS is
byte-identical.

- FP16 and BF16 target kernels each fall from 760 to 560 encoded SASS
  instructions, a reduction of 200 or 26.316% from the live-gated incumbent.
- Branches fall from 38 to 29 per dtype kernel.
- Registers remain 168 and static shared storage remains 1,024 bytes per CTA.
  Stack storage falls from 8 to zero bytes, eliminating three `LDL` and one
  `STL` instruction per dtype kernel.
- The exact shared object linked successfully and loaded through
  `torch.ops.load_library` with `CUDA_VISIBLE_DEVICES` empty; no GPU runtime was
  used.
- The focused patch suite passes all 43 tests.

This is compiler and CPU-load evidence only. It contains no synthetic timing,
TPS measurement, or hardware-floor claim.

## Required live gate

Run the authenticated real SWE-Verified B1 physical32/K64/root1 target
comparator against shared-object SHA-256
`d8c6502e7a166e6d2124576a9e36814401d6dbc215516adfffa7ac436f93ba0f`.
It must exercise all five projection shapes, complete 320 comparisons with
zero differing bytes, and resolve the real task cleanly. Only that exact binary
may enter the standing exact-four full-step timing protocol.
