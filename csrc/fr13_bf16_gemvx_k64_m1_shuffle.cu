#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

struct alignas(16) PackedBf16x16 {
  uint4 lo;
  uint4 hi;
};

constexpr int kHidden = 5120;
constexpr int kVocab = 65536;
constexpr int kLanes = 32;
constexpr int kRowsPerCta = 32;
constexpr int kCtas = kVocab / kRowsPerCta;
constexpr int kElementsPerLoad = 16;
constexpr int kGroups = kHidden / kElementsPerLoad;
constexpr unsigned kFullWarpMask = 0xffffffffu;

static_assert(kHidden % (kLanes * kElementsPerLoad) == 0);
static_assert(kVocab % kRowsPerCta == 0);
static_assert(kLanes * kRowsPerCta == 1024);
static_assert(kCtas == 2048);
static_assert(sizeof(at::BFloat16) == sizeof(__nv_bfloat16));
static_assert(sizeof(PackedBf16x16) == 16 * sizeof(__nv_bfloat16));
static_assert(alignof(PackedBf16x16) == 16);

// One full warp owns each output row. Each lane reads sixteen aligned packed
// BF16 values and expands their bit patterns directly into exact FP32 values.
__global__ __launch_bounds__(kLanes * kRowsPerCta) void
fr13_bf16_gemvx_k64_m1_warp32_r32_pair16bits_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight, const float alpha,
    const float beta) {
  const int lane = static_cast<int>(threadIdx.x);
  const int row_in_cta = static_cast<int>(threadIdx.y);
  const int row = static_cast<int>(blockIdx.x) * kRowsPerCta + row_in_cta;
  const auto* input_groups = reinterpret_cast<const PackedBf16x16*>(input);
  const auto* weight_groups =
      reinterpret_cast<const PackedBf16x16*>(weight) + row * kGroups;

  float accumulator = 0.0f;
#pragma unroll 1
  for (int group = lane; group < kGroups; group += kLanes) {
    const PackedBf16x16 x = input_groups[group];
    const PackedBf16x16 w = weight_groups[group];
#define FR13_ACCUMULATE_PACKED_WORD(X, W)                                  \
  do {                                                                    \
    const float x0 = __uint_as_float((X) << 16);                          \
    const float x1 = __uint_as_float((X) & 0xffff0000u);                  \
    const float w0 = __uint_as_float((W) << 16);                          \
    const float w1 = __uint_as_float((W) & 0xffff0000u);                  \
    accumulator = __fmaf_rn(x0, w0, accumulator);                        \
    accumulator = __fmaf_rn(x1, w1, accumulator);                        \
  } while (false)
    FR13_ACCUMULATE_PACKED_WORD(x.lo.x, w.lo.x);
    FR13_ACCUMULATE_PACKED_WORD(x.lo.y, w.lo.y);
    FR13_ACCUMULATE_PACKED_WORD(x.lo.z, w.lo.z);
    FR13_ACCUMULATE_PACKED_WORD(x.lo.w, w.lo.w);
    FR13_ACCUMULATE_PACKED_WORD(x.hi.x, w.hi.x);
    FR13_ACCUMULATE_PACKED_WORD(x.hi.y, w.hi.y);
    FR13_ACCUMULATE_PACKED_WORD(x.hi.z, w.hi.z);
    FR13_ACCUMULATE_PACKED_WORD(x.hi.w, w.hi.w);
#undef FR13_ACCUMULATE_PACKED_WORD
  }

  float peer =
      __shfl_down_sync(kFullWarpMask, accumulator, 16, kLanes);
  if (lane < 16) {
    accumulator = __fadd_rn(accumulator, peer);
  }
  peer = __shfl_down_sync(kFullWarpMask, accumulator, 8, kLanes);
  if (lane < 8) {
    accumulator = __fadd_rn(accumulator, peer);
  }
  peer = __shfl_down_sync(kFullWarpMask, accumulator, 4, kLanes);
  if (lane < 4) {
    accumulator = __fadd_rn(accumulator, peer);
  }
  peer = __shfl_down_sync(kFullWarpMask, accumulator, 2, kLanes);
  if (lane < 2) {
    accumulator = __fadd_rn(accumulator, peer);
  }
  peer = __shfl_down_sync(kFullWarpMask, accumulator, 1, kLanes);
  if (lane == 0) {
    const float reduced_sum = __fadd_rn(accumulator, peer);
    const float sum = __fmaf_rn(alpha, reduced_sum, beta);
    output[row] = __float2bfloat16_rn(sum);
  }
}

void fr13_bf16_gemvx_k64_m1_warp32_r32_pair16bits_out(
    at::Tensor output, const at::Tensor& input, const at::Tensor& weight) {
  TORCH_CHECK(input.is_cuda() && weight.is_cuda() && output.is_cuda(),
              "FR13 BF16 K64 M1 shuffle requires CUDA tensors");
  TORCH_CHECK(input.device() == weight.device() &&
                  output.device() == input.device(),
              "FR13 BF16 K64 M1 shuffle tensors must share one CUDA device");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16 &&
                  weight.scalar_type() == at::kBFloat16 &&
                  output.scalar_type() == at::kBFloat16,
              "FR13 BF16 K64 M1 shuffle requires BF16 tensors");
  TORCH_CHECK(input.sizes() == at::IntArrayRef({1, kHidden}) &&
                  input.strides() == at::IntArrayRef({kHidden, 1}),
              "FR13 BF16 K64 M1 input must be contiguous [1,5120]");
  TORCH_CHECK(weight.sizes() == at::IntArrayRef({kVocab, kHidden}) &&
                  weight.strides() == at::IntArrayRef({kHidden, 1}),
              "FR13 BF16 K64 M1 weight must be contiguous [65536,5120]");
  TORCH_CHECK(output.sizes() == at::IntArrayRef({1, kVocab}) &&
                  output.strides() == at::IntArrayRef({kVocab, 1}),
              "FR13 BF16 K64 M1 output must be contiguous [1,65536]");
  TORCH_CHECK(
      reinterpret_cast<std::uintptr_t>(input.data_ptr()) %
                  alignof(PackedBf16x16) ==
              0 &&
          reinterpret_cast<std::uintptr_t>(weight.data_ptr()) %
                  alignof(PackedBf16x16) ==
              0,
      "FR13 BF16 K64 M1 pair16bits inputs must be 16-byte aligned");

  const c10::cuda::CUDAGuard device_guard(input.device());
  const dim3 block(kLanes, kRowsPerCta, 1);
  fr13_bf16_gemvx_k64_m1_warp32_r32_pair16bits_kernel
      <<<kCtas, block, 0, at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<__nv_bfloat16*>(
              output.data_ptr<at::BFloat16>()),
          reinterpret_cast<const __nv_bfloat16*>(
              input.data_ptr<at::BFloat16>()),
          reinterpret_cast<const __nv_bfloat16*>(
              weight.data_ptr<at::BFloat16>()),
          1.0f, 0.0f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(fr13_bf16_k64_head, library) {
  library.def(
      "gemvx_m1_warp32_r32_pair16bits_out(Tensor(a!) output, Tensor input, Tensor weight) -> ()");
}

TORCH_LIBRARY_IMPL(fr13_bf16_k64_head, CUDA, library) {
  library.impl("gemvx_m1_warp32_r32_pair16bits_out",
               &fr13_bf16_gemvx_k64_m1_warp32_r32_pair16bits_out);
}
