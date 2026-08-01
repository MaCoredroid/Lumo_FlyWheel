# FR13 fixed32 B4 persistent-M128 live-gate preflight

Status: static preflight PASS; canonical GPU byte gate not launched.

This artifact prepares the persistent-M128 CUTLASS candidate for the real
SWE-Verified exact4 B4 byte gate. No GPU command was launched during this
repair, so the preflight does not contain a byte-equality result, a throughput
result, or a hardware-floor acceptance claim.

## Correct acceptance contract

- Full-vocabulary mandatory-weight floor: `153.9383846446886 ms/step`
- One-sided U95 cap at 1.15x: `177.0291423413919 ms/step`
- Mandatory weight bytes: `42025179008`
- B4 shape: batch size 4, concurrency 4, physical rows 128
- Target topology: `hydra27_fixed32` (27 active nodes)
- Task set: the four pinned real SWE-Verified instances in
  `config/fr13_fixed32/subset_b4_four.json`
- Draft vocabulary: full (`FR13_DRAFT_VOCAB_ROOT=0`, `FR13_DRAFT_VOCAB_K=0`)

The stale `98.6 ms/step` floor was removed from the three B4 workflow prompts.

## Kernel audit

The candidate configuration
`sm120_blockwise_fp8_config_b4_persistent_m128` inherits
`cutlass_3x_gemm_fp8_blockwise`, not the Stream-K specialization. Its tile is
`128x128x128`, its schedule is
`KernelTmaWarpSpecializedBlockwiseCooperativeSm120`, and it does not declare
`use_stream_k`, `force_stream_k`, or `StreamKScheduler`. Therefore it does not
introduce a K split or a cross-CTA K reduction.

The base kernel keeps `ElementAccumulator = float` and
`ElementCompute = float`. The candidate changes the M tile from stock 64 to
128 while retaining K tile 128; it does not override the accumulator type,
compute type, or K tile. This is the static no-reassociation prerequisite. The
live byte gate remains required because source inspection is not a substitute
for comparing every produced BF16 byte.

## Diagnostic behavior

For every eligible real B4 projection, the diagnostic computes stock into the
served output, computes persistent-M128 into a temporary output, compares every
BF16 byte, records mismatch details, and still serves stock. Eager diagnostic
bracketing now records authenticated task boundaries without invoking graph-only
census or flush requirements. Any comparison mismatch fails the gate.

The B4 comparator and both credential reducers are bounded at 320 calls. This
leaves the B1 Stream-K bound at 256 while covering the required MTP projection
that follows 256 target-projection calls in a real B4 event. The reducer also
requires `FR13_FIXED32_MODE=hydra27_fixed32`; legacy Tail6 execution cannot
qualify this candidate.

The exact command is in `prepared_command.txt`. It was prepared only and was not
executed. After an authenticated byte PASS, run the paired stock/candidate B4
timing gate; only that timing result may be evaluated against the 177.029142 ms
one-sided U95 cap.
