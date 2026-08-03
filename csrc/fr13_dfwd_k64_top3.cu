#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

namespace {

constexpr int kVocab = 65536;
constexpr int kTopK = 3;
constexpr int kThreads = 256;
constexpr int kWarp = 32;
constexpr int kWarps = kThreads / kWarp;
constexpr unsigned kFullWarpMask = 0xffffffffu;

static_assert(kVocab % kThreads == 0);
static_assert(kTopK == 3);
static_assert(kWarps == 8);
static_assert(sizeof(at::BFloat16) == sizeof(__nv_bfloat16));

struct Candidate {
  float value;
  int index;
};

__device__ __forceinline__ bool fr13_better(const Candidate lhs,
                                             const Candidate rhs) {
  const bool lhs_nan = isnan(lhs.value);
  const bool rhs_nan = isnan(rhs.value);
  if (lhs_nan != rhs_nan) {
    return lhs_nan;
  }
  if (lhs.value > rhs.value) {
    return true;
  }
  if (lhs.value < rhs.value) {
    return false;
  }
  return lhs.index < rhs.index;
}

__device__ __forceinline__ void fr13_insert(Candidate value,
                                             Candidate& first,
                                             Candidate& second,
                                             Candidate& third) {
  if (fr13_better(value, first)) {
    third = second;
    second = first;
    first = value;
  } else if (fr13_better(value, second)) {
    third = second;
    second = value;
  } else if (fr13_better(value, third)) {
    third = value;
  }
}

__device__ __forceinline__ Candidate fr13_shuffle_down(const Candidate value,
                                                        const int offset) {
  return Candidate{
      __shfl_down_sync(kFullWarpMask, value.value, offset, kWarp),
      __shfl_down_sync(kFullWarpMask, value.index, offset, kWarp),
  };
}

__device__ __forceinline__ void fr13_warp_top3(Candidate& first,
                                                Candidate& second,
                                                Candidate& third,
                                                const int lane) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    const Candidate other_first = fr13_shuffle_down(first, offset);
    const Candidate other_second = fr13_shuffle_down(second, offset);
    const Candidate other_third = fr13_shuffle_down(third, offset);
    if (lane + offset < kWarp) {
      fr13_insert(other_first, first, second, third);
      fr13_insert(other_second, first, second, third);
      fr13_insert(other_third, first, second, third);
    }
  }
}

// One exact-geometry launch replaces the separate argmax, multi-kernel top-k,
// two ID-map gathers, and graph-buffer copies for one K64 drafter head.
__global__ __launch_bounds__(kThreads, 1) void
fr13_dfwd_k64_mapped_top3_kernel(
    int64_t* __restrict__ spine_output,
    int64_t* __restrict__ top3_output,
    const __nv_bfloat16* __restrict__ logits,
    const int64_t* __restrict__ id_map) {
  const int thread = static_cast<int>(threadIdx.x);
  const int lane = thread & (kWarp - 1);
  const int warp = thread / kWarp;
  const float negative_infinity = -__int_as_float(0x7f800000);
  Candidate first{negative_infinity, std::numeric_limits<int>::max()};
  Candidate second = first;
  Candidate third = first;

#pragma unroll 1
  for (int index = thread; index < kVocab; index += kThreads) {
    fr13_insert(
        Candidate{__bfloat162float(logits[index]), index},
        first,
        second,
        third);
  }
  fr13_warp_top3(first, second, third, lane);

  __shared__ Candidate warp_candidates[kWarps][kTopK];
  if (lane == 0) {
    warp_candidates[warp][0] = first;
    warp_candidates[warp][1] = second;
    warp_candidates[warp][2] = third;
  }
  __syncthreads();

  if (warp == 0) {
    Candidate block_first{negative_infinity,
                          std::numeric_limits<int>::max()};
    Candidate block_second = block_first;
    Candidate block_third = block_first;
    if (lane < kWarps) {
      block_first = warp_candidates[lane][0];
      block_second = warp_candidates[lane][1];
      block_third = warp_candidates[lane][2];
    }
    fr13_warp_top3(block_first, block_second, block_third, lane);
    if (lane == 0) {
      const int64_t mapped_first = id_map[block_first.index];
      spine_output[0] = mapped_first;
      top3_output[0] = mapped_first;
      top3_output[1] = id_map[block_second.index];
      top3_output[2] = id_map[block_third.index];
    }
  }
}

void fr13_dfwd_k64_mapped_top3_out(at::Tensor spine_output,
                                    at::Tensor top3_output,
                                    const at::Tensor& logits,
                                    const at::Tensor& id_map) {
  TORCH_CHECK(spine_output.is_cuda() && top3_output.is_cuda() &&
                  logits.is_cuda() && id_map.is_cuda(),
              "FR13 DFWD K64 top3 requires CUDA tensors");
  TORCH_CHECK(spine_output.device() == logits.device() &&
                  top3_output.device() == logits.device() &&
                  id_map.device() == logits.device(),
              "FR13 DFWD K64 top3 tensors must share one CUDA device");
  TORCH_CHECK(logits.scalar_type() == at::kBFloat16,
              "FR13 DFWD K64 top3 logits must be BF16");
  TORCH_CHECK(spine_output.scalar_type() == at::kLong &&
                  top3_output.scalar_type() == at::kLong &&
                  id_map.scalar_type() == at::kLong,
              "FR13 DFWD K64 top3 IDs must be int64");
  TORCH_CHECK(logits.sizes() == at::IntArrayRef({1, kVocab}) &&
                  logits.strides() == at::IntArrayRef({kVocab, 1}),
              "FR13 DFWD K64 top3 logits must be contiguous [1,65536]");
  TORCH_CHECK(id_map.sizes() == at::IntArrayRef({kVocab}) &&
                  id_map.strides() == at::IntArrayRef({1}),
              "FR13 DFWD K64 top3 ID map must be contiguous [65536]");
  TORCH_CHECK(spine_output.sizes() == at::IntArrayRef({1}) &&
                  spine_output.strides() == at::IntArrayRef({1}),
              "FR13 DFWD K64 top3 spine output must be contiguous [1]");
  TORCH_CHECK(top3_output.sizes() == at::IntArrayRef({1, kTopK}) &&
                  top3_output.strides() == at::IntArrayRef({kTopK, 1}),
              "FR13 DFWD K64 top3 output must be contiguous [1,3]");
  TORCH_CHECK(spine_output.data_ptr() != top3_output.data_ptr(),
              "FR13 DFWD K64 top3 outputs must not alias");

  const c10::cuda::CUDAGuard device_guard(logits.device());
  const cudaDeviceProp* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties != nullptr && properties->major == 12 &&
                  properties->minor == 1,
              "FR13 DFWD K64 top3 is qualified only for SM121");
  fr13_dfwd_k64_mapped_top3_kernel<<<1, kThreads, 0,
                                      at::cuda::getCurrentCUDAStream()>>>(
      spine_output.data_ptr<int64_t>(),
      top3_output.data_ptr<int64_t>(),
      reinterpret_cast<const __nv_bfloat16*>(
          logits.data_ptr<at::BFloat16>()),
      id_map.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(fr13_dfwd_top3, library) {
  library.def(
      "mapped_top3_out(Tensor(a!) spine_output, Tensor(b!) top3_output, Tensor logits, Tensor id_map) -> ()");
}

TORCH_LIBRARY_IMPL(fr13_dfwd_top3, CUDA, library) {
  library.impl("mapped_top3_out", &fr13_dfwd_k64_mapped_top3_out);
}
