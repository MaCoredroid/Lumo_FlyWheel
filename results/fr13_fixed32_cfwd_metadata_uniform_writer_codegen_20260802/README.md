# FR13 fixed32 CFWD metadata uniform-writer codegen

Status: offline SM121 code generation passed for the B1 and B4 deployed
specializations. This is static evidence only: no GPU kernel launch, real task,
timing, acceptance, or hardware-floor measurement was run.

The metadata-fused direct conv committer now guards its accepted-path and
accepted-length copies with the CTA-uniform predicate
`pid_l == 0 && pid_c == 0`. Triton 3.6 lowers that predicate to a real branch
on `SR_CTAID.X/Z`, not a metadata load/store sequence predicated in every CTA.
For every request, exactly the `(layer=0, channel_tile=0)` CTA enters the
metadata block; the remaining `48 * ceil(10240 / 1024) - 1 = 479` CTAs skip
that block. B4 retains one writer CTA per request, so metadata traffic is four
copies per event rather than scaling with the 1,920 conv CTAs.

## Codegen result

Both B1 (`B=1`) and B4 (`B=4`) deployed specializations compile for `sm_121a`
with:

- 56 registers per thread, zero stack, local memory, shared memory, or device
  calls;
- 2,088 SASS instructions;
- a CTA-uniform branch around the vectorized 16-column metadata path;
- no `LDL`, `STL`, or `CALL` instructions.

The otherwise-equivalent direct conv kernel without metadata fusion compiles
at 48 registers and 2,064 SASS instructions. The fusion therefore adds 24
static instructions and raises the whole-kernel allocation by eight registers.
That cost is recorded explicitly; this artifact claims elimination of
non-writer metadata memory work, not zero occupancy cost.

## Compile contract

- implementation commit: `c94943b4887eca8bd1ef1857e69a8d11ce21ac9a`
- codegen input commit: `2946061c8fe5e01bb10ac70d0f98987d7fdfb359`
- target: CUDA `sm_121a`, warp size 32
- Triton: 3.6.0
- Torch: 2.10.0+cu130
- deployed constants: `CONV_C=10240`, `CONV_L=34`, `SOURCE_ROWS=36`,
  `ELEM_BYTES=2`, `SPEC_COLS=32`, `PATH_COLS=16`, `BLOCK_C=1024`
- launch options: four warps, three stages

The generated cubins and PTX were inspected from isolated temporary Triton
caches and are identified by digest in `manifest.json`; the binary blobs are
not committed. No task prompt, model output, token sequence, or raw
SWE-Verified material is present.
