# FR13 fixed32 DFWD K64 M1 shuffle source checkpoint

Status: **source-only, default off, not compiled, not byte-qualified, and not
timing eligible**.

This reduced artifact records a bounded B1 drafter-head kernel checkpoint made
while the canonical real B4 timing campaign was using the host. It contains no
prompts, responses, task patches, traces, logs, process or container IDs,
environment dumps, or secrets.

## Why this kernel

The real SWE-Verified B1 Nsight attribution measured DFWD at 47.435717 ms per
event. BF16 draft-vocabulary heads were the largest DFWD group at 26.227316 ms
per event, ahead of MTP FP8 GEMMs at 8.514285 ms and unified attention at
6.967564 ms. The head group executed exactly five M1 K64 projections per event
and achieved only 127.94 GB/s over 3,355,443,200 mandatory weight bytes.

The later valid K64 Hydra27 exact4 result measured aggregate DFWD at
36.813368 ms per event but did not carry a fresh kernel-group split. Therefore
the older attribution identifies the narrow target, while the newer result is
only the current aggregate anchor.

## Candidate

`csrc/fr13_bf16_gemvx_k64_m1_shuffle.cu` implements only contiguous BF16
`[1,5120] x [65536,5120]^T -> [1,65536]`. It retains the incumbent-aligned
16-lane K partition, 320 dependent scalar `__fmaf_rn` operations per lane,
the exact `8+4+2+1` FP32 `__fadd_rn` tree, the alpha-one/beta-positive-zero
FFMA epilogue, and round-to-nearest BF16 output conversion.

Relative to the direct K64 form of the prior exact-order M1 source candidate,
it makes two structural changes without changing any output row's arithmetic:

- 16 output rows per CTA instead of 8, reducing the K64 grid from 8,192 to
  4,096 CTAs.
- Four width-16 shuffle exchanges instead of shared partials plus four
  CTA-wide barriers. A 256-thread block contains two independent 16-lane rows
  per warp, and the shuffle width prevents cross-row values from entering the
  reduction.

The op is not referenced by the runtime, so production behavior remains stock.
The build script emits `BUILT_UNQUALIFIED`, false byte/resource/performance
claims, and `production_default_enabled=false`.

## Required closure

1. After the live campaign releases the host, compile for the pinned
   PyTorch 2.10/CUDA 13.0 `sm_121a` toolchain.
2. Require zero stack, zero local memory, zero calls, zero shared memory, no
   barrier instructions, and the intended shuffle/FADD reduction sequence in
   SASS. A compiler result that changes the dependent FMA or reduction tree is
   rejected.
3. Wire a default-off real B1 shadow gate at exact K64/root1/physical32. For
   root, MTP1, MTP2, MTP3, and MTP4, compute stock first, candidate second,
   compare all 65,536 BF16 values bitwise, and return stock. Require one exact
   comparison per authenticated measured event and zero mismatches.
4. Only after that credential passes may the candidate serve in matched real
   SWE-Verified exact4 Tail23 and Hydra27 timing. Exact16 and one-sided U95 are
   still required for formal floor acceptance.

No GPU, Docker, CUDA build, synthetic traffic, performance probe, or real task
was run for this checkpoint. It makes no speed, byte-parity, acceptance,
quality, B4, or hardware-floor claim.
