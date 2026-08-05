# Fixed32 B1 constant-grid target scheduler

Status: **default off; pinned SM121a translation-unit compile and codegen
audit passed; real SWE-Verified byte gate and timing pending**.

## Kernel change

The admitted B1 target path uses an exact `128x32x128`, StageCount2,
ping-pong CUTLASS collective on a source-bound 48-SM GB10 launch. Its direct
scheduler still initialized the generic base device cursor and retained a
runtime grid stride even though the production contract fixes the grid at
`(1,48,1)`.

This candidate constructs the unused base state empty, starts directly from
`blockIdx.y`, and advances by the compile-time stride 48. It retains the
runtime logical-tile bound because the admitted projection widths contain
112, 128, or 272 complete tiles. The output mapping remains `{M_idx,0,0}` and
covers each logical tile exactly once.

The GEMM tile, mainloop, epilogue, StageCount, full-K ordered reduction,
48-CTA launch, requested bytes, and projection selector are unchanged.

## Offline evidence

The complete target translation unit compiled twice, including a
cache-disabled rebuild, against vLLM
`fe9c3d6c5f66c873d196800384ed6880687b9e52` and CUTLASS
`da5e086dab31d63815acafdac9a9c5893b1c69e2` for SM121a. The two ELF objects
differ, but the extracted target-kernel SASS is byte-identical.

- FP16 and BF16 target kernels each fall from 760 to 744 encoded SASS
  instructions, a reduction of 16 or 2.105%.
- Across both dtype kernels, the device scheduler removes 14 `UIMAD`, 14
  `LDCU/LDCU.64`, 12 `UMOV`, and two `S2UR` sites. Compiler rescheduling adds
  smaller integer/predicate setup; the net reduction is 32 instructions.
- Resources remain 168 registers, 8 stack bytes, zero local bytes, and 1,024
  static shared bytes per CTA. `LDL`, `STL`, and branch counts are unchanged.
- The focused patch suite passes all 43 tests, including exact coverage for
  all three admitted logical tile counts.

This is compiler evidence only. No GPU workload, synthetic timing, Docker
launch, TPS measurement, or hardware-floor claim was used for this artifact.

## Required live gate

Link a shared object from this exact source, then run the authenticated real
SWE-Verified B1 physical32/K64/root1 target comparator. It must exercise all
five projection shapes, complete 320 comparisons with zero differing bytes,
and return the real task cleanly. Only that passed binary may enter the
standing exact-four full-step timing protocol.
