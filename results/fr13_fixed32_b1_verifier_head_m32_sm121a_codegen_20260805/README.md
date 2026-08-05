# Fixed32 B1 BF16 verifier-head M32 candidate

Status: **source and SM121a code generation pass; live qualification pending**.

This branch adds a default-off CUDA candidate for the separate verifier
full-vocabulary projection. It does not change the verifier vocabulary, input
or weight dtype, output dtype, or logical matrix product.

## Measured basis

The only available per-kernel evidence remains the canonical real
SWE-Verified Nsight attribution captured at older source commit
`1a7a765447c8ce6068e0dd5d3a344d58ace85f2b`:

- incumbent operation: BF16 `M=32, N=248320, K=5120` projection
- measured kernel symbol:
  `nvjet_sm121_tst_mma_128x208x64_2_32x104x64_tmaAB_bz_TNNN`
- measured time: `12.152306946651533 ms/event`
- mandatory weight bytes: `2542796800`
- weight-only lower bound at 273 GB/s: `9.314273992673993 ms`
- observed excess over that lower bound: `2.83803295397754 ms`

These numbers motivate the candidate but do not predict its performance.
Compute-versus-memory NCU evidence remains unmeasured.

## Candidate

The CUDA op expresses the existing buffers as a GEMM of
`[248320,5120] row-major` by `[5120,32] column-major`, writing
`[248320,32] column-major`. Those column-major views are byte-for-byte the
existing contiguous PyTorch `hidden[32,5120]` and `output[32,248320]`
storage.

The fixed kernel geometry is:

- threadblock: `128x32x64`
- warp: `64x32x64`
- tensor-core instruction: `16x8x16`
- stages: `3`
- split-K slices: `1`
- workspace: `0 bytes`
- dynamic shared storage: `61440 bytes`

The physical row dimension is exactly one 32-column tile, rather than the
incumbent symbol's 208-column CTA tile. K remains 64 and there is no split-K,
which makes this the narrowest defensible scheduler change found from the
available evidence. A scheduler change can still alter BF16 rounding, so the
candidate is not distribution-qualified by construction.

## Code generation

CPU-only CUDA 13.0 code generation for `sm_121a` passed against pinned
CUTLASS commit `da5e086dab31d63815acafdac9a9c5893b1c69e2`.

- registers: `158`
- stack frame: `0 bytes`
- spill stores: `0 bytes`
- spill loads: `0 bytes`
- barriers: `1`
- static shared reported by cuobjdump: `1024 bytes`
- dynamic shared storage locked by source assertion: `61440 bytes`
- SASS census: `64` BF16 HMMA, `60` asynchronous 128-bit
  global-to-shared loads, `8` ordinary 128-bit global loads, and `16` 128-bit
  stores

No GPU, Docker container, synthetic timing, live workload, or performance
measurement was used.

## Required qualification

1. Build the full shared object in the pinned CUDA 13 / Torch 2.11 runtime.
2. Run one real SWE-Verified B1 shadow task at Hydra27 physical32 and compare
   every BF16 output element against the incumbent while always serving the
   incumbent output. Any mismatch rejects the candidate.
3. If and only if raw mismatches are zero, run frozen-source real exact4 B1
   stock-versus-candidate full-step timing and require clean engagement.
4. Only after B1 passes, repeat the byte gate and real exact4 timing on B4.

There is no byte-equality, verifier-distribution, latency, TPS, or
hardware-floor claim in this artifact.
