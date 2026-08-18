// FR14 fused full-vocabulary draft top-k (K0 drafter head).
//
// WHAT IT REPLACES.  Under the served K0 (full-vocabulary) drafter profile each
// of the five fixed32 MTP head reads runs, on the SAME [rows, 248320] bf16
// logits row that `compute_logits` just wrote:
//
//     draft_token_ids       = logits.argmax(dim=-1)          # ATen reduce
//     _fr10_wide_topk[pos]  = torch.topk(logits, 3, -1).indices  # ATen mbtopk
//
// `torch.topk` on a 248 320-element slice takes ATen's MULTI-BLOCK radix-select
// path (`at::native::mbtopk`), which is a chain of kernels -- fill, radix
// histogram passes, blockwise within-k counts, blockwise kth counts, a cub
// scan, then `gatherTopK`, then a bitonic sort of the k survivors.  Together
// with `argmax` that is one full-vocabulary write followed by several
// full-vocabulary re-reads, five times per decode step, all of them
// latency-bound at this geometry.  This kernel does the whole selection in ONE
// launch and ONE read of the row.
//
// EXACTNESS.  Selection is defined by a TOTAL ORDER on (value, index) encoded
// as a single unsigned 64-bit key:
//
//     key = (order_preserving_u32(value) << 32) | (0xFFFFFFFF - index)
//
// `order_preserving_u32` is the standard monotone float->uint bijection, with
// NaN forced to 0xFFFFFFFF so a NaN outranks every finite value (this is ATen's
// `at::_isnan(a) || a > b` max semantics).  Because indices are unique, every
// key is unique, so "the three largest keys" is a SET-VALUED function of the
// row alone: it does not depend on how many CTAs ran, in what order they
// finished, or in what order two candidate lists were merged.  Determinism is
// therefore structural, not empirical -- there is no reduction-order freedom
// left to be non-deterministic about.
//
// Decoding the low half recovers the index; the high half orders by value
// descending and the low half orders by index ASCENDING on equal values.
//
// THE TIE-BREAK IS MEASURED, NOT ASSUMED.  ATen's `argmax` and ATen's `topk`
// do NOT agree with each other on ties at this geometry, and neither one is
// "index ascending" end to end.  The offline probe swept 411 adversarial rows
// at V = 248320 and found exactly one rule that reproduces `torch.topk` on
// 411/411, with zero run-to-run drift:
//
//     SET   = the k largest by (value desc, index ASCENDING)
//     ORDER = (value desc, index DESCENDING)
//
// while `argmax` is (value desc, index ASCENDING) -- i.e. `argmax` is rank 0 of
// the SET, which is NOT in general `topk(...).indices[:, 0]`.  This kernel
// therefore emits BOTH: `spine_output` from the pre-reorder ladder (argmax
// semantics) and `topk_output` after `fr14_emit_order` reverses each
// equal-value run (topk semantics).  Reproducing that disagreement is the
// point: byte-exactness means matching what ships, including where what ships
// is internally inconsistent.
//
// Parity -- INCLUDING the tie-break -- is proven, not asserted, by
// `results/fr14_nvfp4_port_20260816/fr14_fused_draft_topk_probe.py`, which
// plants exact ties at the real geometry and requires zero raw-byte mismatches
// with a powered negative control in every configuration.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

// NAMED namespace, deliberately. An anonymous namespace makes nvcc mangle the
// kernel as `_GLOBAL__N__<tu-hash>_<file>_<build-hash>_...`, and that build-hash
// lands inside the cubin's symbol table -- so two builds of byte-identical
// source produce different device code digests, and the build has no
// reproducibility credential at all. This is the device-code-level twin of the
// `.so` sha nondeterminism banked in pass 37. Naming the namespace fixes the
// symbol, and `build_attestation.json` records the cubin sha that results.
namespace fr14_fused_draft_topk_impl {

constexpr int kVocab = 248320;      // Qwen3.x text_config.vocab_size, pinned
constexpr int kTopK = 3;            // fixed32 SAMPLER_MAX_FANOUT
constexpr int kThreads = 256;
constexpr int kWarp = 32;
constexpr int kWarps = kThreads / kWarp;
constexpr int kMaxRows = 4;         // fixed32 serves B in {1,2,3,4}
constexpr int kMaxBlocksPerRow = 512;
constexpr int kVecWidth = 8;        // uint4 == 8 bf16
constexpr int kVecPerRow = kVocab / kVecWidth;
constexpr unsigned kFullWarpMask = 0xffffffffu;
constexpr unsigned long long kEmptyKey = 0ull;

static_assert(kVocab % kVecWidth == 0, "row must be uint4-loadable");
static_assert(kWarps == 8, "block reduction assumes 8 warps");
static_assert(kTopK == 3, "insert ladder is unrolled for k=3");
static_assert(sizeof(at::BFloat16) == sizeof(__nv_bfloat16));

// Monotone float -> uint32 map.  Finite values keep their float ordering; any
// NaN is hoisted above +inf so it wins a max exactly as ATen's max does.
__device__ __forceinline__ uint32_t fr14_order_bits(const float value) {
  if (isnan(value)) {
    return 0xffffffffu;
  }
  const uint32_t bits = __float_as_uint(value);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ unsigned long long fr14_key(const float value,
                                             const int index) {
  return (static_cast<unsigned long long>(fr14_order_bits(value)) << 32) |
         static_cast<unsigned long long>(0xffffffffu - static_cast<uint32_t>(index));
}

__device__ __forceinline__ int fr14_key_index(const unsigned long long key) {
  return static_cast<int>(0xffffffffu - static_cast<uint32_t>(key & 0xffffffffull));
}

__device__ __forceinline__ uint32_t fr14_key_value_bits(
    const unsigned long long key) {
  return static_cast<uint32_t>(key >> 32);
}

// ATen's emission order (measured, not assumed -- see the probe's H1 sweep):
// the SET is the k largest by (value desc, index ASC), but the ORDER they are
// emitted in is (value desc, index DESC).  The selection ladder above already
// produces the set in (value desc, index ASC) order, so equal-valued entries
// are contiguous and ascending; recovering ATen's order is exactly reversing
// each equal-value run of the three survivors.
__device__ __forceinline__ void fr14_emit_order(unsigned long long& first,
                                                unsigned long long& second,
                                                unsigned long long& third) {
  const uint32_t va = fr14_key_value_bits(first);
  const uint32_t vb = fr14_key_value_bits(second);
  const uint32_t vc = fr14_key_value_bits(third);
  unsigned long long a = first;
  unsigned long long b = second;
  unsigned long long c = third;
  if (va == vb && vb == vc) {
    first = c;
    second = b;
    third = a;
  } else if (va == vb) {
    first = b;
    second = a;
  } else if (vb == vc) {
    second = c;
    third = b;
  }
}

__device__ __forceinline__ void fr14_insert(const unsigned long long candidate,
                                            unsigned long long& first,
                                            unsigned long long& second,
                                            unsigned long long& third) {
  if (candidate > first) {
    third = second;
    second = first;
    first = candidate;
  } else if (candidate > second) {
    third = second;
    second = candidate;
  } else if (candidate > third) {
    third = candidate;
  }
}

__device__ __forceinline__ void fr14_merge(const unsigned long long other_first,
                                           const unsigned long long other_second,
                                           const unsigned long long other_third,
                                           unsigned long long& first,
                                           unsigned long long& second,
                                           unsigned long long& third) {
  fr14_insert(other_first, first, second, third);
  fr14_insert(other_second, first, second, third);
  fr14_insert(other_third, first, second, third);
}

__device__ __forceinline__ void fr14_warp_reduce(unsigned long long& first,
                                                 unsigned long long& second,
                                                 unsigned long long& third,
                                                 const int lane) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    const unsigned long long other_first =
        __shfl_down_sync(kFullWarpMask, first, offset, kWarp);
    const unsigned long long other_second =
        __shfl_down_sync(kFullWarpMask, second, offset, kWarp);
    const unsigned long long other_third =
        __shfl_down_sync(kFullWarpMask, third, offset, kWarp);
    if (lane + offset < kWarp) {
      fr14_merge(other_first, other_second, other_third, first, second, third);
    }
  }
}

// Reduce the block's per-thread ladders into warp 0 lane 0.
__device__ __forceinline__ void fr14_block_reduce(unsigned long long& first,
                                                  unsigned long long& second,
                                                  unsigned long long& third,
                                                  unsigned long long* __restrict__ shared,
                                                  const int lane,
                                                  const int warp) {
  fr14_warp_reduce(first, second, third, lane);
  if (lane == 0) {
    shared[warp * kTopK + 0] = first;
    shared[warp * kTopK + 1] = second;
    shared[warp * kTopK + 2] = third;
  }
  __syncthreads();
  if (warp == 0) {
    unsigned long long block_first = kEmptyKey;
    unsigned long long block_second = kEmptyKey;
    unsigned long long block_third = kEmptyKey;
    if (lane < kWarps) {
      block_first = shared[lane * kTopK + 0];
      block_second = shared[lane * kTopK + 1];
      block_third = shared[lane * kTopK + 2];
    }
    fr14_warp_reduce(block_first, block_second, block_third, lane);
    first = block_first;
    second = block_second;
    third = block_third;
  }
}

// grid = (blocks_per_row, rows).  scratch layout: [rows][blocks_per_row][kTopK]
// keys, followed by `rows` int32 tickets (the tail is aliased as int32 through
// the same int64 buffer so the op takes ONE scratch tensor).
__global__ __launch_bounds__(kThreads, 1) void fr14_fused_draft_topk_kernel(
    int64_t* __restrict__ spine_output,
    int64_t* __restrict__ topk_output,
    const __nv_bfloat16* __restrict__ logits,
    unsigned long long* __restrict__ scratch,
    const int blocks_per_row,
    const int rows) {
  const int row = static_cast<int>(blockIdx.y);
  const int block = static_cast<int>(blockIdx.x);
  const int thread = static_cast<int>(threadIdx.x);
  const int lane = thread & (kWarp - 1);
  const int warp = thread / kWarp;

  unsigned long long first = kEmptyKey;
  unsigned long long second = kEmptyKey;
  unsigned long long third = kEmptyKey;

  const uint4* __restrict__ row_vec =
      reinterpret_cast<const uint4*>(logits + static_cast<size_t>(row) * kVocab);
  const int stride = blocks_per_row * kThreads;
#pragma unroll 1
  for (int vec = block * kThreads + thread; vec < kVecPerRow; vec += stride) {
    const uint4 packed = row_vec[vec];
    const int base = vec * kVecWidth;
    const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      const __nv_bfloat16 lo = __ushort_as_bfloat16(
          static_cast<unsigned short>(words[w] & 0xffffu));
      const __nv_bfloat16 hi = __ushort_as_bfloat16(
          static_cast<unsigned short>(words[w] >> 16));
      fr14_insert(fr14_key(__bfloat162float(lo), base + 2 * w), first, second,
                  third);
      fr14_insert(fr14_key(__bfloat162float(hi), base + 2 * w + 1), first,
                  second, third);
    }
  }

  __shared__ unsigned long long shared[kWarps * kTopK];
  fr14_block_reduce(first, second, third, shared, lane, warp);

  if (blocks_per_row == 1) {
    if (thread == 0) {
      // argmax semantics: lowest index among the maximum value == the
      // selection ladder's rank 0 BEFORE the emission reorder.
      spine_output[row] = static_cast<int64_t>(fr14_key_index(first));
      fr14_emit_order(first, second, third);
      topk_output[row * kTopK + 0] = static_cast<int64_t>(fr14_key_index(first));
      topk_output[row * kTopK + 1] = static_cast<int64_t>(fr14_key_index(second));
      topk_output[row * kTopK + 2] = static_cast<int64_t>(fr14_key_index(third));
    }
    return;
  }

  unsigned long long* const row_scratch =
      scratch + static_cast<size_t>(row) * blocks_per_row * kTopK;
  int* const tickets = reinterpret_cast<int*>(
      scratch + static_cast<size_t>(rows) * blocks_per_row * kTopK);

  __shared__ bool is_last;
  if (thread == 0) {
    row_scratch[block * kTopK + 0] = first;
    row_scratch[block * kTopK + 1] = second;
    row_scratch[block * kTopK + 2] = third;
    // Publish the three keys before the ticket that advertises them.
    __threadfence();
    is_last = (atomicAdd(tickets + row, 1) == blocks_per_row - 1);
  }
  __syncthreads();
  if (!is_last) {
    return;
  }

  // Self-cleaning: the last block resets its own ticket so the buffer is
  // reusable by the next launch, including a CUDA-graph replay.
  if (thread == 0) {
    tickets[row] = 0;
  }

  unsigned long long merged_first = kEmptyKey;
  unsigned long long merged_second = kEmptyKey;
  unsigned long long merged_third = kEmptyKey;
#pragma unroll 1
  for (int slot = thread; slot < blocks_per_row; slot += kThreads) {
    fr14_merge(row_scratch[slot * kTopK + 0], row_scratch[slot * kTopK + 1],
               row_scratch[slot * kTopK + 2], merged_first, merged_second,
               merged_third);
  }
  __syncthreads();
  fr14_block_reduce(merged_first, merged_second, merged_third, shared, lane,
                    warp);
  if (thread == 0) {
    spine_output[row] = static_cast<int64_t>(fr14_key_index(merged_first));
    fr14_emit_order(merged_first, merged_second, merged_third);
    topk_output[row * kTopK + 0] =
        static_cast<int64_t>(fr14_key_index(merged_first));
    topk_output[row * kTopK + 1] =
        static_cast<int64_t>(fr14_key_index(merged_second));
    topk_output[row * kTopK + 2] =
        static_cast<int64_t>(fr14_key_index(merged_third));
  }
}

int64_t fr14_fused_draft_topk_scratch_elements(int64_t rows,
                                               int64_t blocks_per_row) {
  // keys + one int32 ticket per row, rounded up to int64 slots.
  return rows * blocks_per_row * kTopK + (rows + 1) / 2 + 1;
}

void fr14_fused_draft_topk_out(at::Tensor spine_output,
                               at::Tensor topk_output,
                               const at::Tensor& logits,
                               at::Tensor scratch,
                               int64_t blocks_per_row) {
  TORCH_CHECK(spine_output.is_cuda() && topk_output.is_cuda() &&
                  logits.is_cuda() && scratch.is_cuda(),
              "FR14 fused draft top-k requires CUDA tensors");
  TORCH_CHECK(spine_output.device() == logits.device() &&
                  topk_output.device() == logits.device() &&
                  scratch.device() == logits.device(),
              "FR14 fused draft top-k tensors must share one CUDA device");
  TORCH_CHECK(logits.scalar_type() == at::kBFloat16,
              "FR14 fused draft top-k logits must be BF16");
  TORCH_CHECK(spine_output.scalar_type() == at::kLong &&
                  topk_output.scalar_type() == at::kLong &&
                  scratch.scalar_type() == at::kLong,
              "FR14 fused draft top-k outputs and scratch must be int64");
  TORCH_CHECK(logits.dim() == 2 && logits.size(1) == kVocab,
              "FR14 fused draft top-k logits must be [rows, 248320]");
  const int64_t rows = logits.size(0);
  TORCH_CHECK(rows >= 1 && rows <= kMaxRows,
              "FR14 fused draft top-k supports 1..4 rows");
  TORCH_CHECK(logits.stride(0) == kVocab && logits.stride(1) == 1,
              "FR14 fused draft top-k logits must be contiguous");
  TORCH_CHECK(spine_output.dim() == 1 && spine_output.size(0) == rows &&
                  spine_output.stride(0) == 1,
              "FR14 fused draft top-k spine output must be contiguous [rows]");
  TORCH_CHECK(topk_output.dim() == 2 && topk_output.size(0) == rows &&
                  topk_output.size(1) == kTopK &&
                  topk_output.stride(0) == kTopK && topk_output.stride(1) == 1,
              "FR14 fused draft top-k output must be contiguous [rows,3]");
  TORCH_CHECK(spine_output.data_ptr() != topk_output.data_ptr(),
              "FR14 fused draft top-k outputs must not alias");
  TORCH_CHECK(blocks_per_row >= 1 && blocks_per_row <= kMaxBlocksPerRow,
              "FR14 fused draft top-k blocks_per_row out of range");
  TORCH_CHECK(blocks_per_row * kThreads <= kVecPerRow + kThreads - 1,
              "FR14 fused draft top-k: every CTA must own >=1 vector lane");
  TORCH_CHECK(scratch.is_contiguous() &&
                  scratch.numel() >= fr14_fused_draft_topk_scratch_elements(
                                          rows, blocks_per_row),
              "FR14 fused draft top-k scratch is too small");

  const c10::cuda::CUDAGuard device_guard(logits.device());
  const cudaDeviceProp* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties != nullptr && properties->major == 12 &&
                  properties->minor == 1,
              "FR14 fused draft top-k is qualified only for SM121");

  const dim3 grid(static_cast<unsigned>(blocks_per_row),
                  static_cast<unsigned>(rows));
  fr14_fused_draft_topk_kernel<<<grid, kThreads, 0,
                                 at::cuda::getCurrentCUDAStream()>>>(
      spine_output.data_ptr<int64_t>(), topk_output.data_ptr<int64_t>(),
      reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr<at::BFloat16>()),
      reinterpret_cast<unsigned long long*>(scratch.data_ptr<int64_t>()),
      static_cast<int>(blocks_per_row), static_cast<int>(rows));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int64_t fr14_fused_draft_topk_scratch_numel(int64_t rows,
                                            int64_t blocks_per_row) {
  TORCH_CHECK(rows >= 1 && rows <= kMaxRows, "rows out of range");
  TORCH_CHECK(blocks_per_row >= 1 && blocks_per_row <= kMaxBlocksPerRow,
              "blocks_per_row out of range");
  return fr14_fused_draft_topk_scratch_elements(rows, blocks_per_row);
}

}  // namespace fr14_fused_draft_topk_impl

using fr14_fused_draft_topk_impl::fr14_fused_draft_topk_out;
using fr14_fused_draft_topk_impl::fr14_fused_draft_topk_scratch_numel;

TORCH_LIBRARY_FRAGMENT(fr14_fused_draft_topk, library) {
  library.def(
      "select_out(Tensor(a!) spine_output, Tensor(b!) topk_output, Tensor "
      "logits, Tensor(c!) scratch, int blocks_per_row) -> ()");
  library.def("scratch_numel(int rows, int blocks_per_row) -> int",
              &fr14_fused_draft_topk_scratch_numel);
}

TORCH_LIBRARY_IMPL(fr14_fused_draft_topk, CUDA, library) {
  library.impl("select_out", &fr14_fused_draft_topk_out);
}
