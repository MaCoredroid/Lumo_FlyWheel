# FR13 fixed32 DFWD MTP M1 static scheduler source checkpoint

Status: **source-only, default off, not compiled, not byte-qualified, and not
timing eligible**.

This reduced artifact records one bounded kernel source change made while the
canonical real B4 timing campaign owned the host. It contains no prompts,
responses, task patches, traces, raw logs, process or container IDs,
environment dumps, or secrets.

## Target choice

The provenance-bound real SWE-Verified B1 attribution measured the remaining
DFWD groups as follows:

- MTP FP8 CUTLASS: 8.514285 ms/event, 20 launches/event.
- Unified attention: 6.967564 ms/event, 4 launches/event.

The MTP group is larger. Unified attention already has a separate default-off
BM8 source candidate, whereas exact `M=1` MTP projections were still using the
stock Blackwell dynamic CLC tile scheduler. This checkpoint therefore targets
the MTP projection scheduler and does not duplicate or alter the attention
candidate.

The MTP forward ledger carries 1,908,798,976 mandatory weight bytes across four
passes. At the 273 GB/s planning bandwidth that is a 6.991938 ms floor, leaving
only 1.522347 ms between the observed CUTLASS group and the weight floor. That
is the optimistic scheduler headroom, not a measured saving.

## Candidate

`scripts/fr13_patch_cutlass_fixed32_wave.py` now exposes two additional
default-off selectors:

- `static_persistent_mtp_m1`
- `static_persistent_mtp_m1_byte_ab`

Both selectors are restricted to exact `M=1` and the five observed real Qwen
projection `(N,K)` pairs. Physical projection rows `32/64/96/128`, all other
shapes, unset selectors, and unknown selectors remain stock.

The candidate inherits the stock swapped-AB FP8 GEMM and retains:

- tile `128x32x128` and cluster `1x1x1`;
- FP8 block-scale granularity `128x1x128`;
- cooperative SM120 mainloop and automatic epilogue;
- full-K traversal, accumulator type, output conversion, and output tile
  boundaries.

Only `TileSchedulerSelector` changes from the stock dynamic scheduler to
`StaticPersistentTileScheduler100`. Since output tiles retain full-K ownership,
the scheduler changes allocation order without adding cross-CTA reductions or
changing any output element's arithmetic order.

The byte-A/B path is bounded to 320 authenticated real-task calls. It computes
stock first, candidate second, compares the complete output bytewise, and
returns stock. A formal phase-bound reducer and live credential are still
required; this source hook alone is not qualification.

## Required closure

1. After host teardown, emit the pinned vLLM/CUTLASS source and compile for
   `sm_121a`.
2. Require the candidate to preserve the stock mainloop/epilogue symbols and
   report zero stack and zero local memory. Compare instruction/resource
   records with the earlier static-scheduler specialization; any spill, call,
   split-K, or reduction-order drift rejects the build.
3. Run the allowed real SWE-Verified B1 K64/root1 shadow gate. Require all five
   shapes, the exact phase-bound 20-launch/event DFWD census, complete output
   byte equality, zero differing bytes, and stock served throughout.
4. Only after that credential passes may matched real exact4 Tail23 and Hydra27
   timing enable the production selector. Exact16 and one-sided U95 remain the
   formal floor gate.

No nvcc, Triton compilation, C++ build, Docker, GPU, synthetic timing, or real
task ran for this checkpoint. It makes no speed, exactness, quality,
acceptance, B4, or hardware-floor claim.
