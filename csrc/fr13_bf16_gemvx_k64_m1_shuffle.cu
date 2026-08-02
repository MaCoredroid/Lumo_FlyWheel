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
constexpr int kWarpsPerCta = 8;
constexpr int kRowsPerWarp = 4;
constexpr int kRowsPerCta = kWarpsPerCta * kRowsPerWarp;
constexpr int kThreadsPerCta = kLanes * kWarpsPerCta;
constexpr int kCtas = kVocab / kRowsPerCta;
constexpr int kElementsPerLoad = 8;
constexpr int kOctets = kHidden / kElementsPerLoad;
constexpr unsigned kFullWarpMask = 0xffffffffu;

static_assert(kHidden % (kLanes * kElementsPerLoad) == 0);
static_assert(kVocab % kRowsPerCta == 0);
static_assert(kRowsPerCta == 32);
static_assert(kThreadsPerCta == 256);
static_assert(kCtas == 2048);
static_assert(kOctets == 640);
static_assert(alignof(uint4) == 16);
static_assert((kHidden * sizeof(__nv_bfloat16)) % alignof(uint4) == 0);
static_assert(sizeof(at::BFloat16) == sizeof(__nv_bfloat16));
static_assert(sizeof(uint4) == kElementsPerLoad * sizeof(__nv_bfloat16));

__device__ __forceinline__ float fr13_reduce_full_warp(float accumulator,
                                                       const int lane) {
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
  return lane == 0 ? __fadd_rn(accumulator, peer) : accumulator;
}

// Eight warps own 32 output rows. Each warp loads one hidden octet and reuses
// it across four independent rows in the same octet, FMA, and shuffle order
// as the one-row pair8bits kernel.
__global__ __launch_bounds__(kThreadsPerCta) void
fr13_bf16_gemvx_k64_m1_warp4_globalx_pair8bits_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight, const float alpha,
    const float beta) {
  const int lane = static_cast<int>(threadIdx.x);
  const int warp = static_cast<int>(threadIdx.y);
  const auto* input_octets = reinterpret_cast<const uint4*>(input);

  const int first_row =
      static_cast<int>(blockIdx.x) * kRowsPerCta + warp * kRowsPerWarp;
  const auto* weight0 =
      reinterpret_cast<const uint4*>(weight) + first_row * kOctets;
  const auto* weight1 = weight0 + kOctets;
  const auto* weight2 = weight1 + kOctets;
  const auto* weight3 = weight2 + kOctets;
  float accumulator0 = 0.0f;
  float accumulator1 = 0.0f;
  float accumulator2 = 0.0f;
  float accumulator3 = 0.0f;

#pragma unroll 1
  for (int octet = lane; octet < kOctets; octet += kLanes) {
    const uint4 x = input_octets[octet];
    const float x0 = __uint_as_float(x.x << 16);
    const float x1 = __uint_as_float(x.x & 0xffff0000u);
    const float x2 = __uint_as_float(x.y << 16);
    const float x3 = __uint_as_float(x.y & 0xffff0000u);
    const float x4 = __uint_as_float(x.z << 16);
    const float x5 = __uint_as_float(x.z & 0xffff0000u);
    const float x6 = __uint_as_float(x.w << 16);
    const float x7 = __uint_as_float(x.w & 0xffff0000u);

    {
      const uint4 w0 = weight0[octet];
      accumulator0 = __fmaf_rn(x0, __uint_as_float(w0.x << 16), accumulator0);
      accumulator0 =
          __fmaf_rn(x1, __uint_as_float(w0.x & 0xffff0000u), accumulator0);
      accumulator0 = __fmaf_rn(x2, __uint_as_float(w0.y << 16), accumulator0);
      accumulator0 =
          __fmaf_rn(x3, __uint_as_float(w0.y & 0xffff0000u), accumulator0);
      accumulator0 = __fmaf_rn(x4, __uint_as_float(w0.z << 16), accumulator0);
      accumulator0 =
          __fmaf_rn(x5, __uint_as_float(w0.z & 0xffff0000u), accumulator0);
      accumulator0 = __fmaf_rn(x6, __uint_as_float(w0.w << 16), accumulator0);
      accumulator0 =
          __fmaf_rn(x7, __uint_as_float(w0.w & 0xffff0000u), accumulator0);
    }
    {
      const uint4 w1 = weight1[octet];
      accumulator1 = __fmaf_rn(x0, __uint_as_float(w1.x << 16), accumulator1);
      accumulator1 =
          __fmaf_rn(x1, __uint_as_float(w1.x & 0xffff0000u), accumulator1);
      accumulator1 = __fmaf_rn(x2, __uint_as_float(w1.y << 16), accumulator1);
      accumulator1 =
          __fmaf_rn(x3, __uint_as_float(w1.y & 0xffff0000u), accumulator1);
      accumulator1 = __fmaf_rn(x4, __uint_as_float(w1.z << 16), accumulator1);
      accumulator1 =
          __fmaf_rn(x5, __uint_as_float(w1.z & 0xffff0000u), accumulator1);
      accumulator1 = __fmaf_rn(x6, __uint_as_float(w1.w << 16), accumulator1);
      accumulator1 =
          __fmaf_rn(x7, __uint_as_float(w1.w & 0xffff0000u), accumulator1);
    }
    {
      const uint4 w2 = weight2[octet];
      accumulator2 = __fmaf_rn(x0, __uint_as_float(w2.x << 16), accumulator2);
      accumulator2 =
          __fmaf_rn(x1, __uint_as_float(w2.x & 0xffff0000u), accumulator2);
      accumulator2 = __fmaf_rn(x2, __uint_as_float(w2.y << 16), accumulator2);
      accumulator2 =
          __fmaf_rn(x3, __uint_as_float(w2.y & 0xffff0000u), accumulator2);
      accumulator2 = __fmaf_rn(x4, __uint_as_float(w2.z << 16), accumulator2);
      accumulator2 =
          __fmaf_rn(x5, __uint_as_float(w2.z & 0xffff0000u), accumulator2);
      accumulator2 = __fmaf_rn(x6, __uint_as_float(w2.w << 16), accumulator2);
      accumulator2 =
          __fmaf_rn(x7, __uint_as_float(w2.w & 0xffff0000u), accumulator2);
    }
    {
      const uint4 w3 = weight3[octet];
      accumulator3 = __fmaf_rn(x0, __uint_as_float(w3.x << 16), accumulator3);
      accumulator3 =
          __fmaf_rn(x1, __uint_as_float(w3.x & 0xffff0000u), accumulator3);
      accumulator3 = __fmaf_rn(x2, __uint_as_float(w3.y << 16), accumulator3);
      accumulator3 =
          __fmaf_rn(x3, __uint_as_float(w3.y & 0xffff0000u), accumulator3);
      accumulator3 = __fmaf_rn(x4, __uint_as_float(w3.z << 16), accumulator3);
      accumulator3 =
          __fmaf_rn(x5, __uint_as_float(w3.z & 0xffff0000u), accumulator3);
      accumulator3 = __fmaf_rn(x6, __uint_as_float(w3.w << 16), accumulator3);
      accumulator3 =
          __fmaf_rn(x7, __uint_as_float(w3.w & 0xffff0000u), accumulator3);
    }
  }

  const float reduced0 = fr13_reduce_full_warp(accumulator0, lane);
  const float reduced1 = fr13_reduce_full_warp(accumulator1, lane);
  const float reduced2 = fr13_reduce_full_warp(accumulator2, lane);
  const float reduced3 = fr13_reduce_full_warp(accumulator3, lane);
  if (lane == 0) {
    output[first_row] =
        __float2bfloat16_rn(__fmaf_rn(alpha, reduced0, beta));
    output[first_row + 1] =
        __float2bfloat16_rn(__fmaf_rn(alpha, reduced1, beta));
    output[first_row + 2] =
        __float2bfloat16_rn(__fmaf_rn(alpha, reduced2, beta));
    output[first_row + 3] =
        __float2bfloat16_rn(__fmaf_rn(alpha, reduced3, beta));
  }
}

void fr13_bf16_gemvx_k64_m1_warp4_globalx_pair8bits_out(
    at::Tensor output, const at::Tensor& input, const at::Tensor& weight) {
  TORCH_CHECK(input.is_cuda() && weight.is_cuda() && output.is_cuda(),
              "FR13 BF16 K64 M1 warp4 global-x requires CUDA tensors");
  TORCH_CHECK(input.device() == weight.device() &&
                  output.device() == input.device(),
              "FR13 BF16 K64 M1 warp4 global-x tensors must share one CUDA device");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16 &&
                  weight.scalar_type() == at::kBFloat16 &&
                  output.scalar_type() == at::kBFloat16,
              "FR13 BF16 K64 M1 warp4 global-x requires BF16 tensors");
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
      reinterpret_cast<std::uintptr_t>(input.data_ptr()) % alignof(uint4) ==
              0 &&
          reinterpret_cast<std::uintptr_t>(weight.data_ptr()) %
                  alignof(uint4) ==
              0,
      "FR13 BF16 K64 M1 warp4 global-x inputs must be 16-byte aligned");

  const c10::cuda::CUDAGuard device_guard(input.device());
  const dim3 block(kLanes, kWarpsPerCta, 1);
  fr13_bf16_gemvx_k64_m1_warp4_globalx_pair8bits_kernel
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
      "gemvx_m1_warp4_globalx_pair8bits_out(Tensor(a!) output, Tensor input, Tensor weight) -> ()");
}

TORCH_LIBRARY_IMPL(fr13_bf16_k64_head, CUDA, library) {
  library.impl("gemvx_m1_warp4_globalx_pair8bits_out",
               &fr13_bf16_gemvx_k64_m1_warp4_globalx_pair8bits_out);
}
