# FR13 fixed32 CUTLASS tail-balancing candidates

Status: source-only, default-off kernel candidates. No GPU was used. There is
no build, byte-equality, B1/B4 timing, or acceptance claim in this artifact.

## Kernel decision

Pinned vLLM uses CUTLASS v4.4.2. Its default SM120 tile scheduler is already
`PersistentTileSchedulerSm100`: Blackwell cluster launch control lets a running
CTA cancel an unlaunched CTA and consume its full output tile. The earlier
11.185 ms model is therefore not removable launch-wave overhead. It models the
remaining full-tile granularity at the end of the persistent grid. For example,
272 equal output tiles over 48 SMs still leave 32 SMs doing a sixth tile while
16 finish after five.

The obvious swapped `64x32x128` B1 tile is also illegal. The live small-M path
swaps A/B and has scale granularity M=128; CUTLASS statically requires
`TileM % ScaleGranularityM == 0`. Changing the scale granularity to 64 would
reinterpret the live scale buffer and change FP8 math.

The only pinned CUTLASS scheduler that attacks the remaining geometry is
Stream-K. It divides selected tail tiles along K and performs deterministic
FP32 fixup. This preserves the live FP8 operands, FP32 scales, K tile 128,
FP32 accumulator, output epilogue, and cluster `(1,1,1)`. It can reassociate
FP32 additions, so a candidate is rejected unless the real-workload output is
byte-identical to stock.

| Selector | B1 M=32/64 | B4 M=96/128 | Purpose |
| --- | --- | --- | --- |
| unset/unknown | stock | stock | control |
| `streamk_coop64` | `128x32x128` cooperative Stream-K | `64x128x128` cooperative Stream-K | retain B4 tile geometry |
| `streamk_coop128` | `128x32x128` cooperative Stream-K | `128x128x128` cooperative Stream-K | halve B4 logical row tiles |

Ping-pong is not combined with Stream-K: CUTLASS v4.4.2 rejects that pairing at
compile time. Both selectors are admitted only for rows `32/64/96/128` and the
five projection `(N,K)` pairs observed in the real B1 profile.

## Headroom model

The real B1 profile attributes 112.313 ms/event to CUTLASS, versus an 88.440 ms
explicit-traffic floor at 273 GB/s. The full-output-tile model pads 23.824 GB of
compulsory weight traffic to 26.877 GB. An ideal K-granular tail balance reduces
that equivalent work to 23.895 GB, a modeled 2.982 GB or 10.924 ms recovery
before workspace, fixup, and scheduling costs. Even the ideal model leaves
about 12.949 ms of the measured CUTLASS residual; this candidate alone cannot
close the full end-to-end 1.15x goal.

These are selection calculations, not synthetic timing and not performance
evidence. Only the real SWE-Verified gate below can retain a candidate.

At B4, the `coop64` candidate keeps 59,392 logical output tiles/event. The
`coop128` candidate reduces that to 29,696 by covering all 128 physical rows in
one CTA. It may reduce repeated weight-tile work, but no DRAM-byte reduction is
claimed until a real B4 profile measures it.

## Exact source and build

Apply only to vLLM `fe9c3d6c5f66c873d196800384ed6880687b9e52`:

```bash
python3 scripts/fr13_patch_cutlass_fixed32_wave.py \
  --cutlass-root "$CUTLASS_SRC" "$VLLM_SRC"
```

The patcher fails closed on the exact dispatch and CMake digests. That CMake
file pins CUTLASS `v4.4.2`, resolved here to
`da5e086dab31d63815acafdac9a9c5893b1c69e2`.

```bash
test "$(git -C "$VLLM_SRC" rev-parse HEAD)" = \
  fe9c3d6c5f66c873d196800384ed6880687b9e52
test "$(git -C "$CUTLASS_SRC" rev-parse HEAD)" = \
  da5e086dab31d63815acafdac9a9c5893b1c69e2

VLLM_CUTLASS_SRC_DIR="$CUTLASS_SRC" cmake \
  -S "$VLLM_SRC" -B "$VLLM_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DVLLM_TARGET_DEVICE=cuda \
  -DVLLM_PYTHON_EXECUTABLE="$(command -v python3)" \
  -DVLLM_CUTLASS_SRC_DIR="$CUTLASS_SRC"
cmake --build "$VLLM_BUILD" --target _C_stable_libtorch -j1
```

Only
`csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm120_fp8.cu`
directly includes the patched header. In an existing Ninja build, this rebuilds
that CUDA translation unit and relinks `_C_stable_libtorch`; `_C` and FA2 are
not build targets. Preserve the `ptxas -v` output and
`cuobjdump --dump-resource-usage` for the selected kernel identities.

## Required real gate

First qualify each selector on real SWE-Verified work. Use one allowed B1 task
only to capture stock and candidate full logits plus every persistent state
tensor at identical graph boundaries, then byte-compare them. Repeat at B4 on
the canonical exact4 tasks. A single differing byte, missing Stream-K kernel,
zero workspace, absent selector in `container_env.txt`, or missing patched
extension digest rejects the candidate. Diagnostic runs are not acceptance.

Then time stock and each byte-clean candidate with the canonical real exact4
set. The launcher already forwards `FR13_*` into the vLLM container and worker.

```bash
# B1 exact4
SEQUENCE_FILE=scripts/fr13_fixed32_floor_timers_seq.sh \
BSIZE=1 CONC=1 TAG=cutlass_wave_b1 \
SUBSET=config/fr13_fixed32/subset_b4_four.json \
FR13_FIXED32_CUTLASS_WAVE=streamk_coop64 \
bash scripts/fr13_b4_campaign_driver.sh

# B4 exact4
SEQUENCE_FILE=scripts/fr13_fixed32_floor_timers_seq.sh \
BSIZE=4 CONC=4 TAG=cutlass_wave_b4 \
SUBSET=config/fr13_fixed32/subset_b4_four.json \
FR13_FIXED32_CUTLASS_WAVE=streamk_coop64 \
bash scripts/fr13_b4_campaign_driver.sh
```

Run the same commands with the selector unset for the control and with
`streamk_coop128` for the second candidate. Promote only a byte-clean exact4
winner to `config/fr13_fixed32/subset_b4_sixteen.json`. Report full wall TPS,
accepted drafts/event, kernel attribution, and one-sided U95. A synthetic GEMM
time or one-task result is never acceptance evidence.
