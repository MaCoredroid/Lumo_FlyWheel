# Fixed32 B1 M128 exact-shape cooperative scheduler

Status: **default off; pinned SM121a double-compile and codegen audit passed;
real SWE-Verified byte gate and timing pending**.

## Kernel change

The three wide B1 projections use exactly 112, 128, or 272 complete M128
tiles on the source-bound 48-CTA GB10 grid. This candidate moves that logical
tile bound from runtime scheduler state to a `ProblemTiles` template argument.
It retains the M128 StageCount2 cooperative collective, cluster `(1,1,1)`,
full-K reduction, output mapping, tile ownership, and tile order of the
currently gated target candidate. The two `N=5120` projections retain their
existing 40-CTA one-tile scheduler.

The selector remains opt-in. Its production credential is still pinned to the
earlier cooperative shared object and cannot install this source follow-on.

## Offline result

The complete vLLM translation unit compiled twice for SM121a with GPU access
disabled. The two objects differ in ELF metadata, while their full SASS dumps,
resource reports, and the extracted six candidate kernels are byte-identical.

- All six exact kernels use 168 registers, 1,024 bytes of static shared
  storage, zero stack, zero local memory, and no `LDL`, `STL`, or `CALL`.
- Each exact kernel retains 560 encoded SASS slots, 29 branches, 32 QMMA, 32
  FFMA, 24 LDSM, and four STSM instructions from the in-object generic M128
  cooperative comparator.
- Constant-memory loads fall from five `LDC` plus 15 `LDCU` to four `LDC` plus
  12 `LDCU`, removing four load opcodes per kernel. Alignment padding replaces
  the saved slots, so this is not an encoded-size reduction.
- The exact route covers eight of the 16 calls in the real projection-shape
  histogram. The other eight calls are the retained `N=5120` route.

This is compiler/codegen evidence only. It is not a timing result, TPS result,
or hardware-floor claim. No GPU kernel, Docker service, synthetic timing probe,
or SWE-Verified task was run in this lane.

## Required live validation

Build and pin a linked shared object from this exact source, then run the
authenticated real SWE-Verified B1 physical32/K64/root1 raw-byte comparator
over all five projection shapes. Only an exact 320/320 zero-byte result may
advance to the standing exact-four full-step timing protocol.
