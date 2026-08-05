# Fixed32 verifier-head M32 N256/K32/stage3 candidate

Status: **SM121a code generation PASS; real-task qualification pending**.

This exact-f81 branch stages a default-off successor to the existing
verifier-head M32 N128/K64/stage3 candidate. It keeps the full BF16 verifier
projection, output layout, vocabulary, warp output tile, K16 tensor-core
instruction, and no-split-K contract.

## Change

The CTA tile changes from `32x128x64` to `32x256x32`. Four `32x64x32` warps
share a CTA instead of two `32x64x64` warps. Three asynchronous stages remain.

Static consequences for the full `M=32, N=248320, K=5120` projection:

- logical grid: `1940 -> 970` CTAs
- registers per thread: `158 -> 128`
- dynamic shared memory per CTA: `61440 -> 55296` bytes
- shared-limited resident CTAs per 100 KiB maximum carveout: unchanged at `1`
- shared-limited resident warps per SM: `2 -> 4`
- logical repeated hidden-tile bytes before cache: halved from `635699200`
  to `317849600`
- mandatory weight bytes and output bytes: unchanged

These are source and code-generation deltas, not measured speedups.

## Audit

CPU-only CUDA 13.0 code generation for `sm_121a` passed against pinned CUTLASS
commit `da5e086dab31d63815acafdac9a9c5893b1c69e2`.

- registers: `128`
- stack frame: `0 bytes`
- spill stores and loads: `0 bytes`
- barriers: `1`
- static shared memory: `1024 bytes`
- dynamic shared memory: `55296 bytes`
- SASS: `32` BF16 HMMA, `27` asynchronous 128-bit global-to-shared loads,
  `8` ordinary 128-bit global loads, and `16` 128-bit global stores in the
  emitted loop body

The K32 body contains half the static HMMA instructions and executes twice as
many K iterations. This artifact does not claim fewer dynamic tensor-core
operations.

N256/K64/stage2 was rejected because it generated `214` registers and lost the
asynchronous mainloop. N256/K64/stage3 restored the asynchronous mainloop but
required `110592` bytes of dynamic shared memory, exceeding the 99 KiB
per-block limit for compute capability 12.x. The selected 55,296-byte dynamic
allocation remains in the 64 KiB carveout class. Exact exploratory hashes are
in `explored_variants.tsv`.

No GPU device, Docker container, synthetic workload, timing run, or probe was
used. Repeated compiles produced byte-identical resource reports and SASS.

## Required qualification

1. Build the full shared object in the pinned CUDA 13 / Torch 2.11 runtime.
2. Run a real SWE-Verified B1 shadow task and compare every BF16 output element
   against the incumbent while always serving incumbent logits.
3. Only after a zero-mismatch PASS, measure the frozen candidate on the
   standing real task set.
4. Repeat correctness and timing qualification on B4 only after B1 passes.

There is no byte-equality, latency, TPS, acceptance, or hardware-floor claim.
