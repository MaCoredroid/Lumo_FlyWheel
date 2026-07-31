# FR13 fixed32 CUTLASS PDL source candidate

Status: source-only, default-off candidate. No GPU build, CUDA-graph capture,
correctness run, or performance result is claimed here.

## Candidate

`scripts/fr13_patch_cutlass_fixed32_pdl.py` patches only
`csrc/quantization/w8a8/cutlass/c3x/cutlass_gemm_caller.cuh` in an explicitly
supplied vLLM source tree. Set `FR13_FIXED32_CUTLASS_PDL=1` at process start to
pass `launch_with_pdl=true` to CUTLASS only when GEMM `M == 32`. Unset, empty,
or any value other than `1` uses the stock launch path.

The candidate does not change GEMM arguments, A/B pointers, block scales,
layouts, tile or cluster shapes, scheduler arguments, accumulation, or the
epilogue. The mechanism is limited to CUDA Programmatic Dependent Launch (PDL):
dependent CUTLASS grids may be admitted before their predecessor completes,
allowing opportunistic overlap of launch latency and legal kernel preamble/tail
work. CUDA does not guarantee overlap.

Pinned source facts:

- vLLM image: `vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776`
- vLLM version: `0.19.2rc1.dev134+gfe9c3d6c5`
- CUTLASS revision in that source: `v4.2.1`
- Unpatched caller SHA256: `9c53d17d8590786c995bf3d01dabccf323d7882daa708b0dab4fa0471e665ec5`
- Model config SHA256: `f78c412bfdec65a88c8aa2a031d39c2fda32e3377ae48a77f971bc40a4f095df`

CUTLASS v4.2.1 exposes the five-argument `run(args, workspace, stream,
cuda_adapter, launch_with_pdl)` overload. CUDA documents PDL stream-capture
support, but that does not replace a capture/replay gate for this exact binary.

## Real-task evidence and ceiling

Source: `results/fr13_fixed32_b1_nsys_20260731T013952Z_curated/nsys_attribution.json`.
It is bounded real SWE-Verified B1 attribution (`882` SFWD events), not an
acceptance campaign. It was captured on the pre-final kernel stack, so it sizes
the mechanism only and cannot close the current goal.

| SFWD component | ms/event | launches/event |
| --- | ---: | ---: |
| First-to-last GPU envelope | 174.813673 | n/a |
| Blockwise FP8 CUTLASS GEMMs | 112.312954 | 255.79 |
| Standalone group-128 activation quant | 0.351219 | 191.84 |
| Fused SiLU + group-128 quant | 0.819935 | 63.95 |
| Fused producer + group-128 quant | 0.152387 | 47.96 |
| Named producer/quant kernels total | 1.323540 | 303.75 |
| Sum of published SFWD top 20 kernels | 173.372587 | n/a |
| Envelope minus published top 20 | 1.441087 | n/a |

The `1.441087 ms/event` remainder is an upper bound on all unpublished kernel
self-time plus inter-kernel idle time in this trace. It is therefore the
mechanism-targeted ceiling if PDL only hides launch admission/idle bubbles.
It is not a strict physical PDL ceiling: PDL can also overlap a consumer's
independent preamble with a predecessor tail, and aggregate kernel totals do not
expose those internal synchronization points. Only a post-build timeline can
measure that overlap. PDL cannot remove any of the 112.312954 ms/event of FP8
GEMM arithmetic/weight work or any mandatory model-weight bytes.

Keep the measurements separate: the current target+verifier mandatory-weight
floor is `98.627093 ms/step` for `26,925,196,288` bytes at `273 GB/s`. It is not
the SFWD FP8 GEMM time and is not changed by this candidate. The corrected full
hardware floor is `119.658 ms/step`; the acceptance cap is `137.607 ms/step`.
There is no defensible floor-closing claim for PDL.

## Activation-byte accounting

At fixed32, the nominal 192 standalone group-128 quant calls read:

- 64 qkv/qkvz inputs: `64 * 32 * 5120 * 2 = 20,971,520` bytes.
- 64 attention/GDN output-projection inputs:
  `64 * 32 * 6144 * 2 = 25,165,824` bytes.
- 64 MLP gate/up inputs: `64 * 32 * 5120 * 2 = 20,971,520` bytes.

Together they read `67,108,864` BF16 bytes, write `33,554,432` FP8 bytes,
and write `1,048,576` FP32 scale bytes: `101,711,872` bytes/event total. The
traffic-only floor at `273 GB/s` is `0.372571 ms/event`, consistent with the
profiled `0.351219 ms/event` self-time within attribution and overlap limits.

The fused SiLU+quant kernels move `179,372,032` bytes/event and already avoid
`142,606,336` bytes/event of intermediate BF16 write plus reread. Splitting
them is a regression, not a fusion opportunity.

## Why projection stacking stops here

The model already packs full-attention `qkv_proj`, GDN `in_proj_qkvz`, and MLP
`gate_up_proj`. Every next FP8 projection is separated by attention/GDN, SiLU,
or residual dependencies. GDN `in_proj_ba` shares the hidden input but is BF16
and explicitly excluded from FP8 conversion, so stacking it into a block-FP8
projection would change the quantization and arithmetic contract.

## Required gates before real SWE

1. Build the patched pinned source and verify the exact CUTLASS caller object is
   present in the replacement `_C` binary.
2. Capture and replay the exact fixed32 CUDA graph with the flag off and on.
3. Require byte-equal state tensors and full logits before timing.
4. Profile a bounded real SWE-Verified B1 run to establish whether PDL produces
   any overlap and whether GEMM self-time regresses.
5. Only a passing standing-rule 4-task or 16-task campaign can count toward
   acceptance. B4 remains separately required; no one-task probe can qualify.

Official source references:

- https://github.com/NVIDIA/cutlass/blob/v4.2.1/include/cutlass/gemm/device/gemm_universal_adapter.h
- https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html
