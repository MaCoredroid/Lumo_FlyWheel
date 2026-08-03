# Fixed32 B1 full-grid scheduler contract repair

Status: **host compile, link, and CPU import passed for the fixed binary;
default off; GPU requalification required**.

## Invalidated gate attempt

The prior `identity_onen_n5120_fullgrid_b1` candidate reached its candidate
call during a real-task gate, but produced no completed comparator record.
The gate was terminated. It yielded no valid gate credential, byte comparison,
timing result, or acceptance evidence. No raw logs or runtime identities are
included in this artifact.

This artifact supersedes the candidate identity and device-code claims in
`fr13_fixed32_cutlass_b1_n5120_fullgrid_live_ready_20260803`. The earlier
744-instruction full-grid kernels are rejected: their reduction removed device
scheduler state required by the SM120 ping-pong kernel, so the smaller SASS was
not a valid optimization.

## Root cause and repair

The host grid was already correct. The three wide shapes use CUTLASS's native
full static grid, which is `(1,48,1)` on the 48-SM target. The fault was the
custom device scheduler. Its parent skipped initialization of the native
static scheduler cursor and grid state, while its local `get_current_work()`
mapping did not implement the `advance_to_next_work()` contract used directly
by the ping-pong kernel. Warp groups could therefore replay or disagree on
work instead of reaching the native termination state.

The scoped repair makes the wide B1 scheduler inherit
`StaticPersistentTileScheduler100` directly and preserves `using Base::Base`.
This restores CUTLASS's initialized cursor, advancement, and termination
semantics while retaining the native full-grid host policy. The two physical
`N=5120` projections keep their existing exact 40-CTA cooperative scheduler.
Unset, unknown, and out-of-contract configurations remain stock.

## Static evidence

The complete pinned vLLM translation unit compiled for SM121a and linked into
an AArch64 stable-libtorch extension. CPU import with no visible CUDA device
passed. The repaired binary is pinned by SHA-256 and size in `manifest.json`.
Its dynamic dependency set and 183-entry undefined-symbol set match the
rejected candidate library.

All 44 kernel resource records match the prior object. Only the FP16 and BF16
wide full-grid instruction streams changed; the other 42 streams are
byte-identical. Each repaired wide stream is byte-identical to the existing
same-dtype, swap-AB, StageCount2 initialized static-persistent ping-pong device
implementation in the same object. The two N5120 single-tile streams are
unchanged.

The repaired wide kernels each contain 968 encoded instructions, including 31
NOPs and 39 `BRA`/`BRX` operations. They retain 168 registers, zero stack,
zero local memory, 1,024 bytes static shared memory, and no `LDL`, `STL`, or
`CALL` instructions. These are code-generation facts, not performance claims.

## Qualification boundary

- 195 CUTLASS-focused, artifact, and B1 diagnostic host tests passed.
- Python byte compilation, shell syntax, whitespace, source identity, SM121a
  compile, full link, CPU import, ELF, ABI, resource, and SASS checks passed.
- This audit used no GPU and no Docker, synthetic timing, probe timing, task
  output, or performance sampling.
- `acceptance_valid` and `performance_claim` remain false.

The smallest valid next step is a fresh authenticated one-task real
SWE-Verified B1 K64/root1 byte gate against the repaired binary. No additional
scheduler reduction is defensible before that gate passes. Exact-four
real-task full-step timing remains gated on a byte pass.
