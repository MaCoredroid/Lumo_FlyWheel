# Fixed32 DFWD BF16 padded-row selector

Status: **source-only, default off, and not deployable**. No GPU was used to
build or test this branch, and it carries no performance or exactness claim.

## Decision

M64 and M128 are not statically dominated, but neither has earned a campaign.
The pinned dispatch and roofline leave a small, real possibility that either
shape improves memory latency hiding over M32. This branch therefore exposes a
strict selector instead of running an unbounded padding sweep.

`FR13_DRAFT_HEAD_PAD_ROWS` accepts exactly `0`, `32`, `64`, or `128`; `0` is the
default. A nonzero value replicates the real B1 BF16 hidden row into a
preallocated matrix, runs one `torch.mm(..., out=...)` against the unchanged
contiguous BF16 `65536 x 5120` draft-head weight, and returns output row zero.
It is fail-loud outside exact fixed32 B1 root64 single-logits serving.

`FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB=1` is a mutually exclusive diagnostic. It runs
M32, M64, and M128 against every real draft-head input, compares the complete
65,536-element BF16 row from each shape bitwise with the existing `gemvx`,
accumulates per-shape mismatch and comparison counters on device, and always
returns the reference logits. Thus one real SWE-Verified B1 task can reject any
numerically unsafe row count. Candidate-only timing still requires three arms
using the same real task; diagnostic timing is invalid.

## Pinned dispatch

The production image is
`vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776`.
It contains PyTorch `2.11.0+cu130`, source commit `70d99e998b4955e0049d13a98d77ae1b14db1f45`,
CUDA 13.0, and reports the preferred BLAS backend as cuBLAS.

At that exact PyTorch commit, `mm_out_cuda` enters the shared `addmm` path with
`beta=0`. The cuBLASLt eligibility predicate requires `beta=1`, so this call
uses the ordinary cuBLAS GEMM route. Its BF16 specialization invokes
`cublasGemmEx` with BF16 A/B/C, FP32 compute, and
`CUBLAS_GEMM_DEFAULT_TENSOR_OP`. cuBLAS still chooses the internal kernel by
the complete matrix shape, so source inspection cannot identify the M64 or
M128 kernel. Real-task Nsight confirmation remains mandatory.

Primary source:

- `aten/src/ATen/native/cuda/Blas.cpp` at PyTorch commit `70d99e9`: `mm_out_cuda`,
  `addmm_out_cuda_impl`, and `isInputCompliesAddmmCudaLt`.
- `aten/src/ATen/cuda/CUDABlas.cpp` at the same commit:
  `gemm_internal_cublas_bfloat16_helper`.

## Real Nsight evidence

The real SWE-Verified B1 trace contains 4,405 draft-head BF16 `gemvx` kernels
over 881 complete events, exactly five per event. They cost 26.227316 ms/event
for 3,355,443,200 mandatory weight bytes, or 127.94 GB/s.

The same trace contains one M32 verifier-head
`nvjet_sm121_tst_mma_128x208x64_2_32x104x64_tmaAB_bz_TNNN` per event. Its
postprocess phase costs 12.153933 ms/event for a `248320 x 5120` BF16 weight
(2,542,796,800 bytes), or 209.22 GB/s. Applying that measured efficiency to
the smaller draft weight projects M32 at 16.038180 ms/event, saving about
10.189136 ms/event versus `gemvx`. This is only a projection: the draft
`65536 x 5120` shape has fewer vocabulary tiles and may select another opaque
cuBLAS kernel.

## Roofline

Every row count reads the same 671,088,640 weight bytes/pass and
3,355,443,200 bytes across five passes/event. At 273 GB/s the mandatory-weight
floor is 2.458200 ms/pass or 12.291001 ms/event.

| Rows | FLOPs/pass | Arithmetic intensity | Persistent input+output | TF/s needed at weight floor | Min event floor incl. input/output |
|---:|---:|---:|---:|---:|---:|
| 32 | 21.475 GF | 31.786 flop/B | 4,521,984 B | 8.736 | 12.373821 ms |
| 64 | 42.950 GF | 63.149 flop/B | 9,043,968 B | 17.472 | 12.456641 ms |
| 128 | 85.899 GF | 124.641 flop/B | 18,087,936 B | 34.944 | 12.622282 ms |

Against the 125 TF/s dense-BF16 planning ceiling, even M128 remains on the
memory side of this roofline. M32 already supplies at least 512 vocabulary-side
tiles if the observed 128-wide `nvjet` tile transfers, so larger rows are not a
CTA-starvation fix. Their only plausible benefit is better selected-kernel
efficiency or latency hiding. The absolute weight-floor headroom below the M32
projection is only 3.747179 ms/event, before extra input/output traffic.

All-row byte-A/B mode owns 31,653,888 persistent input/output bytes plus two
three-element int64 counters. Candidate-only mode allocates only its selected
row geometry. Steady state performs no allocation.

## Real-task gate

1. With candidate rows off, run `FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB=1` on real
   SWE-Verified B1 `astropy__astropy-12907`. Require zero mismatches for all
   three counters across root and captured-loop head positions; returned logits
   remain reference logits.
2. On that same real task, run candidate-only rows 32, 64, and 128 separately.
   Record full wall TPS, DFWD GPU component time, head kernel identity/count,
   and acceptance. No synthetic timing traffic is valid.
3. Retain only a byte-safe row count that beats M32 on the real task. Then run
   Tail/Hydra exact4 and exact16 before deployment. Any logit, token,
   acceptance, task-resolution, or head-count drift rejects it.

