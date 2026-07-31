# FR13 fixed32 CUTLASS floor audit and PDL candidate

Status: source-only, exact-math, default-off candidate. No GPU build, byte
gate, performance result, B4 result, or acceptance result is claimed.

## Bottom line

The `112.312954 ms/event` real SWE-Verified B1 CUTLASS number and the
`98.627093 ms` target-plus-verifier mandatory-weight floor are not the same
operator slice. The former is the target model's 256 block-FP8 projections;
the latter also includes the separate BF16 verifier head and other target
weights. Subtracting them produces a misleading `13.686 ms` "CUTLASS gap."

For the actual CUTLASS slice, the Qwen3.6 layer dimensions imply:

- FP8 projection weights: `23,823,646,720 B`.
- FP32 block scales: `5,816,320 B`.
- Fixed32 FP8 activation reads: `69,206,016 B`.
- Fixed32 activation-scale reads: `2,162,688 B`.
- BF16 output writes: `243,269,632 B`.
- Minimum explicit tensor traffic: `24,144,101,376 B`, or
  `88.439932 ms` at the optimistic `273 GB/s` hardware ceiling.
- Profiled CUTLASS self-time: `112.312954 ms/event`, equivalent to
  `214.972 GB/s` against that traffic ledger.
- Residual above the CUTLASS traffic floor: `23.873022 ms`.

This is strongly memory-bound. The 256 M=32 GEMMs perform about
`1.5247 TFLOP/event`, only `63.15 FLOP/B` against the explicit traffic ledger.
At `273 GB/s`, the memory roofline needs only `17.24 TFLOP/s`; the local GB10
hardware ledger places dense FP8 tensor throughput near `214 TFLOP/s`.

## Exact live launch topology

Pinned vLLM source `fe9c3d6c5f66c873d196800384ed6880687b9e52` routes block-FP8
`M <= 64` through `sm120_blockwise_fp8_config_swapab`: A/B are swapped and a
cooperative `128x32x128`, cluster-1 kernel is launched. That source exactly
matches the Nsight kernel name. At B1, the architecture issues four packed FP8
projections in each of 64 layers, nominally 256 launches/event.

| Projection | Calls | N x K | FP8 weights/event | CTAs/call | 48-SM waves | Last-wave use |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gate_up | 64 | 34816 x 5120 | 11,408,506,880 B | 272 | 6 | 66.7% |
| down | 64 | 5120 x 17408 | 5,704,253,440 B | 40 | 1 | 83.3% |
| attention out | 64 | 5120 x 6144 | 2,013,265,920 B | 40 | 1 | 83.3% |
| GDN qkvz | 48 | 16384 x 5120 | 4,026,531,840 B | 128 | 3 | 66.7% |
| full-attention qkv | 16 | 8192 x 5120 | 671,088,640 B | 64 | 2 | 33.3% |

`CTAs/call` is `ceil(N/128)` after the source's A/B swap. The last-wave column
describes only the final wave; the whole-launch slot utilizations are 94.4%,
83.3%, 83.3%, 88.9%, and 66.7%, respectively. A simple full-wave-equivalent
weight model adds `3.053 GB` or `11.185 ms` at 273 GB/s. It is a geometry bound,
not a measured attribution: cache behavior, CTA residency, memory-controller
utilization, and cooperative scheduling require NCU to separate.

The remaining evidence does constrain the gap:

- Only `1.441087 ms/event` lies outside the published top-20 SFWD kernel
  self-times and the SFWD first-to-last envelope. Host launch/idle removal alone
  therefore cannot recover the `23.873 ms` in-kernel traffic gap.
- The old 40-attempt CUTLASS loop found no winner across small-M tiles,
  pingpong/cooperative/persistent schedules, stage counts, padding, swap-A/B,
  and epilogue variants. It targeted an older M=1/M=16 serving regime and does
  not validate the current fixed32 kernel, but it is strong negative evidence
  against repeating an ungated schedule sweep.
- No NCU counter run exists for this fixed32 real-task shape, so an exact split
  among sustained-DRAM shortfall, wave tails, scale visitor work, and internal
  pipeline stalls is not defensible yet.

## Candidate: corrected fixed32 PDL

`scripts/fr13_patch_cutlass_fixed32_pdl.py` patches the live
`csrc/libtorch_stable/.../cutlass_gemm_caller.cuh`. The earlier source-only PDL
candidate targeted the obsolete `csrc/quantization/...` path and gated only
`problem M == 32`; neither condition reaches the pinned B1 block-FP8 path,
because that path swaps A/B and carries the row count in problem N.

Set `FR13_FIXED32_CUTLASS_PDL=1` at process start to pass
`launch_with_pdl=true` for row counts `32`, `64`, `96`, or `128` and
`K >= 5120`. Unset or any other value is stock. These rows cover B1 and partial
or full B4 co-batches after either source dispatch orientation. PDL changes only
dependency admission; it does not change pointers, shapes, layouts, scales,
tile selection, accumulation, epilogue, dtype, or bytes.

The credible ceiling is small: the `1.441087 ms/event` SFWD top-20 remainder is
an upper bound on ordinary launch/idle bubbles, though PDL may additionally
overlap legal consumer preamble with predecessor tail. This candidate cannot
close the hardware-floor gap by itself.

## Ranked next real-shape experiments

1. **Corrected PDL, B1 then B4.** Lowest semantic risk. Build the pinned source,
   byte-compare full logits/state under captured fixed32 graphs, then use the
   standing exact4 real SWE-Verified gate. Expected ceiling is about 1.4 ms/B1.
2. **B4 M=64/96/128 route to existing 128x128x128 cooperative config.** The
   current M64 pingpong path creates two row tiles at M=96/128, while the
   already-instantiated M128 config creates one. This may reduce repeated
   weight-tile traffic, but B4 is unmeasured and byte equality must be proved.
3. **NCU on the five exact projection classes during bounded real SWE.** Record
   DRAM bytes/throughput, L2 hit rate, active warps, tensor utilization, and
   eligible-warps stalls. This decides whether experiment 2 attacks real DRAM
   rereads or only L2-resident reuse.

Do not retry K-tile/stage changes first: K regrouping has a higher exact-output
risk, and adjacent old candidates did not improve the real objective.

## Required gates

1. Apply only to vLLM `fe9c3d6c5f66c873d196800384ed6880687b9e52`; the script
   fails closed on the unpatched caller SHA256.
2. Recompile both SM120 scaled-mm translation units and link the exact
   `_C_stable_libtorch` replacement after the active campaign releases the GPU;
   this CUTLASS path moved out of `_C` in the pinned source.
3. Capture/replay B1 and B4 fixed32 graphs with flag off/on.
4. Require byte-equal full logits and all persistent state before serving the
   candidate output.
5. Profile only bounded real SWE-Verified work; no synthetic timing counts.
6. Acceptance remains exact4/exact16 full wall TPS under the standing rule.
