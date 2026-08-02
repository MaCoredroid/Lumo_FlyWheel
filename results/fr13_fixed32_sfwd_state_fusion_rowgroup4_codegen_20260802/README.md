# FR13 fixed32 SFWD row-group-4 offline codegen

Status: **offline SM121a codegen passed; default off, not byte-qualified, and
not timing eligible**.

This artifact compares the row-group-4 fused conv/state kernel with its
immediate row-per-program parent at the deployed Qwen fixed32 specialization.
It contains no task text, model outputs, patches, traces, raw logs, process or
container identifiers, environment dumps, or secrets.

## Exact compile contract

- Model config SHA256:
  `f78c412bfdec65a88c8aa2a031d39c2fda32e3377ae48a77f971bc40a4f095df`
- `C=10240`: `2 * (16 key heads * 128) + (48 value heads * 128)`
- `N=32`, conv width `4`, state length `12`, source rows `36`
- BF16 activations, state, weights, output, and source stage
- int32 state indices; int64 source descriptor; bias disabled
- `BLOCK_C=256`, Triton 3.6.0, Torch 2.10.0+cu130, CUDA 13.0
- target `sm_121a`, `num_stages=3`
- baseline: one row/program, four warps
- candidate: four rows/program, eight warps

Both B1 and B4 were compiled independently. Their cubins are byte-identical
within each variant, as expected because `B` is a compile-time contract check
but does not alter this kernel body.

## Result

The candidate compiles without stack, local memory, spill instructions, or
calls. Per CTA, it changes from 40 to 64 registers and introduces 2048 bytes
of launch shared memory (the cubin resource report exposes 1024 bytes). The
static SASS body grows from 293 to 633 instructions because a CTA owns four
rows and Triton introduces shared-memory exchange.

The grid reduction is exact:

| Batch | Baseline CTAs/request | Candidate CTAs/request | Ratio |
|---:|---:|---:|---:|
| 1 | 1280 | 320 | 0.25 |
| 4 | 5120 | 1280 | 0.25 |

This is not a latency claim. The lower CTA count and weight reuse trade against
more registers, twice the warps, shared-memory traffic, and a larger body.
Only a live byte gate followed by real-task timing can decide whether it wins.

## Qualification boundary

The next allowed GPU action is the existing authenticated real SWE-Verified B1
reference-returning byte gate across all 48 GDN layers. It must require exact
bytes for both conv output and the full commit-source stage while continuing to
serve incumbent bytes. No production selector or timing is allowed before that
passes. If it passes, measure on the standing exact4 real task set, then exact16
and one-sided U95 only after a material exact4 improvement.

Source commit: `f7456b7fc83bdc292cf25b4f2d15e22a2f224363`.

