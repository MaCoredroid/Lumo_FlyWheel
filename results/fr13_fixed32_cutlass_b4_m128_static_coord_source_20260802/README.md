# Fixed32 B4 M128 static-coordinate source checkpoint

Status: source-only, default off, and not acceptance-valid.

## Existing kernel census

The latest locally built static-persistent M128 BF16 kernel contains 1,440
SASS instructions in 23,040 text bytes. It uses 168 registers, zero stack,
zero local memory, zero detected spills, 1,024 bytes of static shared memory,
and 384 threads per CTA. It contains zero device `CALL` instructions.

The mandatory projection instruction counts are 128 QMMA, 128 FFMA, 72 FMUL,
48 LDSM, 16 STSM, and 32 BF16 output packs. The remaining instruction stream
includes 152 predicated instructions, 46 branches, and seven reads of
`SR_CgaCtaId`. Several cluster-ID reads belong to shared pipeline, barrier, and
TMEM lifecycle code and are outside this checkpoint. No exact SASS reduction
is claimed before compilation.

The census is from the existing pre-direct-linear static kernel. The
direct-linear scheduler and this coordinate follow-up are source-only layers
on top of it.

## Change

The fixed candidate has exactly one M tile, one L tile, and a 1x1x1 cluster.
Its derived scheduler now converts work to the CTA coordinate
`(M=0, N=work.N_idx, K=_, L=0)` directly. A compile-time cluster-shape check
fails construction if the candidate is instantiated outside 1x1x1.

This replaces five source-level calls to CUTLASS's generic cluster-coordinate
conversion: the initial coordinate plus the main-load, MMA, epilogue-load, and
epilogue next-work updates. Those five generic conversions contain fifteen
coordinate additions in total. It also exposes M and L as compile-time zero to
downstream mainloop and epilogue address generation.

All five allowlisted B4 projections have M=128, L=1, and N and K dimensions
that are exact multiples of the 128x128x128 tile. The dynamic N-tile index,
complete-output-tile ownership, physical-grid stride, ordered K iteration,
mainloop, epilogue math, output tile count, and stock-default dispatch remain
unchanged.

## Verification

- 25 focused source tests passed.
- Python bytecode compilation and `git diff --check` passed.
- The patch applied to pinned vLLM source and pinned CUTLASS v4.4.2, and a
  second application was idempotent.
- Generated dispatch SHA256:
  `21ba590c5fc8fb998f72787f7601bbddaf632065676310ce85ba33a07d7e1982`.

No NVCC/C++ compile, link, new SASS/resource audit, GPU execution, container
run, synthetic performance probe, or real-task run was performed while the B4
pair was active. Compile admission, real SWE-Verified exact4 raw-byte gates,
paired timing, floor ratio, and U95 remain pending.
