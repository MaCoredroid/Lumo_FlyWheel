# Fixed32 B1 wide256 direct full-grid scheduler

Status: **pinned SM121a translation-unit compile and codegen audit passed;
default off; real SWE-Verified raw-byte and timing gates pending**.

## Kernel change

The existing B1 `identity_wide256_fullgrid_b1` target uses an exact
`256x32x128` StageCount2 cooperative GEMM with a 48-CTA full grid. Its generic
CUTLASS static scheduler decodes every linear work index through batch,
cluster, swizzle, and raster div/mod state even though every admitted wide
shape has exactly one scheduler-N tile, one batch plane, and cluster
`(1,1,1)`.

This candidate gives only that M256 specialization a distinct scheduler tag
and maps its bounded persistent cursor directly to `{M_idx, 0, 0}`. It keeps
the initialized CUTLASS base, cursor advance, last-tile test, fetch contract,
48-CTA grid, full-K ordered reduction, tile geometry, launch count, and output
coordinate contract. N5120 remains on the existing M128 specialization. No
GEMM arithmetic, epilogue arithmetic, weight bytes, activation bytes, or
reduction order changed.

## Offline evidence

The complete stable-libtorch translation unit compiled for SM121a against
vLLM `fe9c3d6c5f66c873d196800384ed6880687b9e52` and CUTLASS
`da5e086dab31d63815acafdac9a9c5893b1c69e2`.

- Both FP16 and BF16 M256 kernels fell from 1,080 to 784 encoded SASS
  instructions, a reduction of 296 or 27.407%.
- Branches fell from 35 to 31.
- Resources stayed at 168 registers, zero stack, zero local memory, 1,024
  bytes static shared memory, and zero `LDL`, `STL`, or `CALL` instructions.
- The neighboring FP16 and BF16 M128 full-grid SASS dumps are byte-identical
  to the incumbent, at 968 instructions and 39 branches.
- The direct cursor covers each admitted wide shape exactly once: 56, 64, and
  136 tiles distributed over 48 CTAs. Across the fixed target-step histogram,
  this removes generic coordinate decoding from 12,672 wide tile assignments.

This is compile/codegen evidence, not a GPU correctness or speed result. The
candidate has not been linked as a replacement shared object and makes no
hardware-floor acceptance claim.

## Required real gate

Build and attest a shared object from the pinned closure, then run the
authenticated real SWE-Verified B1 physical32/K64/root1 byte diagnostic. The
gate must resolve a real task cleanly, exercise all five admitted projection
shapes, produce all 320 candidate/stock comparisons, and report zero
mismatching comparisons and zero differing bytes while continuing to serve
stock output.

Only after that gate passes may the direct selector be timed on the standing
real SWE-Verified task protocol. Compare the same source and binary closure
against the incumbent full-grid arm, report both target-SFWD component time
and full-step wall TPS, and retain the exact-four then exact-16 campaign rules
for acceptance. Synthetic kernels, probes, and modeled ceilings are not
measurement evidence.
