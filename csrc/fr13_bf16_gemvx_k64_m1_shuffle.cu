#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kHidden = 5120;
constexpr int kVocab = 65536;
constexpr int kLanes = 32;
constexpr int kRowsPerCta = 32;
constexpr int kCtas = kVocab / kRowsPerCta;
constexpr int kElementsPerLoad = 8;
constexpr int kOctets = kHidden / kElementsPerLoad;
constexpr unsigned kFullWarpMask = 0xffffffffu;

static_assert(kHidden % (kLanes * kElementsPerLoad) == 0);
static_assert(kVocab % kRowsPerCta == 0);
static_assert(kLanes * kRowsPerCta == 1024);
static_assert(kCtas == 2048);
static_assert(sizeof(at::BFloat16) == sizeof(__nv_bfloat16));
static_assert(sizeof(uint4) == 8 * sizeof(__nv_bfloat16));

// One full warp owns each output row. Each lane reads eight aligned packed BF16
// values and expands their bit patterns directly into exact FP32 values.
__global__ __launch_bounds__(kLanes * kRowsPerCta) void
fr13_bf16_gemvx_k64_m1_warp32_r32_pair8bits_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight, const float alpha,
    const float beta) {
  const int lane = static_cast<int>(threadIdx.x);
  const int row_in_cta = static_cast<int>(threadIdx.y);
  const int row = static_cast<int>(blockIdx.x) * kRowsPerCta + row_in_cta;
  const auto* input_octets = reinterpret_cast<const uint4*>(input);
  const auto* weight_octets =
      reinterpret_cast<const uint4*>(weight) + row * kOctets;

  float accumulator = 0.0f;
#pragma unroll 1
  for (int octet = lane; octet < kOctets; octet += kLanes) {
    const uint4 x = input_octets[octet];
    const uint4 w = weight_octets[octet];
    const float x0 = __uint_as_float(x.x << 16);
    const float x1 = __uint_as_float(x.x & 0xffff0000u);
    const float x2 = __uint_as_float(x.y << 16);
    const float x3 = __uint_as_float(x.y & 0xffff0000u);
    const float x4 = __uint_as_float(x.z << 16);
    const float x5 = __uint_as_float(x.z & 0xffff0000u);
    const float x6 = __uint_as_float(x.w << 16);
    const float x7 = __uint_as_float(x.w & 0xffff0000u);
    const float w0 = __uint_as_float(w.x << 16);
    const float w1 = __uint_as_float(w.x & 0xffff0000u);
    const float w2 = __uint_as_float(w.y << 16);
    const float w3 = __uint_as_float(w.y & 0xffff0000u);
    const float w4 = __uint_as_float(w.z << 16);
    const float w5 = __uint_as_float(w.z & 0xffff0000u);
    const float w6 = __uint_as_float(w.w << 16);
    const float w7 = __uint_as_float(w.w & 0xffff0000u);
    accumulator = __fmaf_rn(x0, w0, accumulator);
    accumulator = __fmaf_rn(x1, w1, accumulator);
    accumulator = __fmaf_rn(x2, w2, accumulator);
    accumulator = __fmaf_rn(x3, w3, accumulator);
    accumulator = __fmaf_rn(x4, w4, accumulator);
    accumulator = __fmaf_rn(x5, w5, accumulator);
    accumulator = __fmaf_rn(x6, w6, accumulator);
    accumulator = __fmaf_rn(x7, w7, accumulator);
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

void fr13_bf16_gemvx_k64_m1_warp32_r32_pair8bits_out(
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
                  alignof(uint4) ==
              0 &&
          reinterpret_cast<std::uintptr_t>(weight.data_ptr()) %
                  alignof(uint4) ==
              0,
      "FR13 BF16 K64 M1 pair8bits inputs must be 16-byte aligned");

  const c10::cuda::CUDAGuard device_guard(input.device());
  const dim3 block(kLanes, kRowsPerCta, 1);
  fr13_bf16_gemvx_k64_m1_warp32_r32_pair8bits_kernel
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
      "gemvx_m1_warp32_r32_pair8bits_out(Tensor(a!) output, Tensor input, Tensor weight) -> ()");
}

TORCH_LIBRARY_IMPL(fr13_bf16_k64_head, CUDA, library) {
  library.impl("gemvx_m1_warp32_r32_pair8bits_out",
               &fr13_bf16_gemvx_k64_m1_warp32_r32_pair8bits_out);
}
