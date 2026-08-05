#if !defined(FR13_DEVICE_CODEGEN_ONLY)
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>
#endif

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kHidden = 5120;
constexpr int kVocab = 65536;
constexpr int kBatch4 = 4;
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
static_assert(sizeof(uint4) == kElementsPerLoad * sizeof(__nv_bfloat16));
static_assert((kHidden * sizeof(__nv_bfloat16)) % alignof(uint4) == 0);
#if !defined(FR13_DEVICE_CODEGEN_ONLY)
static_assert(sizeof(at::BFloat16) == sizeof(__nv_bfloat16));
#endif

struct FloatOctet {
  float v0;
  float v1;
  float v2;
  float v3;
  float v4;
  float v5;
  float v6;
  float v7;
};

__device__ __forceinline__ FloatOctet fr13_unpack_bf16_octet(
    const uint4 bits) {
  return FloatOctet{
      __uint_as_float(bits.x << 16),
      __uint_as_float(bits.x & 0xffff0000u),
      __uint_as_float(bits.y << 16),
      __uint_as_float(bits.y & 0xffff0000u),
      __uint_as_float(bits.z << 16),
      __uint_as_float(bits.z & 0xffff0000u),
      __uint_as_float(bits.w << 16),
      __uint_as_float(bits.w & 0xffff0000u),
  };
}

__device__ __forceinline__ float fr13_fma_float_octets(
    float accumulator, const FloatOctet input, const FloatOctet weight) {
  accumulator = __fmaf_rn(input.v0, weight.v0, accumulator);
  accumulator = __fmaf_rn(input.v1, weight.v1, accumulator);
  accumulator = __fmaf_rn(input.v2, weight.v2, accumulator);
  accumulator = __fmaf_rn(input.v3, weight.v3, accumulator);
  accumulator = __fmaf_rn(input.v4, weight.v4, accumulator);
  accumulator = __fmaf_rn(input.v5, weight.v5, accumulator);
  accumulator = __fmaf_rn(input.v6, weight.v6, accumulator);
  accumulator = __fmaf_rn(input.v7, weight.v7, accumulator);
  return accumulator;
}

__device__ __forceinline__ float fr13_fma_octet(
    float accumulator, const FloatOctet input, const uint4 weight_bits) {
  return fr13_fma_float_octets(
      accumulator, input, fr13_unpack_bf16_octet(weight_bits));
}

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

// Each warp owns four adjacent vocabulary rows. A single packed hidden load is
// reused by all four rows, while 16-byte weight loads replace scalar BF16 load
// issue. The width-32 reduction can change draft logits slightly; proposals are
// still consumed only by target-authoritative rejection sampling.
__global__ __launch_bounds__(kThreadsPerCta, 2) void
fr13_bf16_gemvx_k64_m1_warp4_pair8_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight) {
  const int lane = static_cast<int>(threadIdx.x);
  const int warp = static_cast<int>(threadIdx.y);
  const int first_row =
      static_cast<int>(blockIdx.x) * kRowsPerCta + warp * kRowsPerWarp;
  const auto* input_octets = reinterpret_cast<const uint4*>(input);
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
    const FloatOctet x = fr13_unpack_bf16_octet(input_octets[octet]);
    accumulator0 = fr13_fma_octet(accumulator0, x, weight0[octet]);
    accumulator1 = fr13_fma_octet(accumulator1, x, weight1[octet]);
    accumulator2 = fr13_fma_octet(accumulator2, x, weight2[octet]);
    accumulator3 = fr13_fma_octet(accumulator3, x, weight3[octet]);
  }

  const float reduced0 = fr13_reduce_full_warp(accumulator0, lane);
  const float reduced1 = fr13_reduce_full_warp(accumulator1, lane);
  const float reduced2 = fr13_reduce_full_warp(accumulator2, lane);
  const float reduced3 = fr13_reduce_full_warp(accumulator3, lane);
  if (lane == 0) {
    output[first_row] = __float2bfloat16_rn(reduced0);
    output[first_row + 1] = __float2bfloat16_rn(reduced1);
    output[first_row + 2] = __float2bfloat16_rn(reduced2);
    output[first_row + 3] = __float2bfloat16_rn(reduced3);
  }
}

// B4 reuses each packed weight octet across all four requests as well as one
// packed hidden octet across four vocabulary rows. Sixteen independent FP32
// accumulators retain request and output-row isolation.
__global__ __launch_bounds__(kThreadsPerCta, 2) void
fr13_bf16_gemvx_k64_m4_warp4_pair8_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight) {
  const int lane = static_cast<int>(threadIdx.x);
  const int warp = static_cast<int>(threadIdx.y);
  const int first_row =
      static_cast<int>(blockIdx.x) * kRowsPerCta + warp * kRowsPerWarp;
  const auto* input_octets = reinterpret_cast<const uint4*>(input);
  const auto* weight0 =
      reinterpret_cast<const uint4*>(weight) + first_row * kOctets;
  const auto* weight1 = weight0 + kOctets;
  const auto* weight2 = weight1 + kOctets;
  const auto* weight3 = weight2 + kOctets;

  float accumulator00 = 0.0f;
  float accumulator01 = 0.0f;
  float accumulator02 = 0.0f;
  float accumulator03 = 0.0f;
  float accumulator10 = 0.0f;
  float accumulator11 = 0.0f;
  float accumulator12 = 0.0f;
  float accumulator13 = 0.0f;
  float accumulator20 = 0.0f;
  float accumulator21 = 0.0f;
  float accumulator22 = 0.0f;
  float accumulator23 = 0.0f;
  float accumulator30 = 0.0f;
  float accumulator31 = 0.0f;
  float accumulator32 = 0.0f;
  float accumulator33 = 0.0f;

#pragma unroll 1
  for (int octet = lane; octet < kOctets; octet += kLanes) {
    const FloatOctet weight_values0 =
        fr13_unpack_bf16_octet(weight0[octet]);
    const FloatOctet weight_values1 =
        fr13_unpack_bf16_octet(weight1[octet]);
    const FloatOctet weight_values2 =
        fr13_unpack_bf16_octet(weight2[octet]);
    const FloatOctet weight_values3 =
        fr13_unpack_bf16_octet(weight3[octet]);

#define FR13_ACCUMULATE_REQUEST(request, accumulator0, accumulator1,          \
                                accumulator2, accumulator3)                   \
  do {                                                                        \
    const FloatOctet x = fr13_unpack_bf16_octet(                              \
        input_octets[(request) * kOctets + octet]);                           \
    accumulator0 =                                                            \
        fr13_fma_float_octets(accumulator0, x, weight_values0);               \
    accumulator1 =                                                            \
        fr13_fma_float_octets(accumulator1, x, weight_values1);               \
    accumulator2 =                                                            \
        fr13_fma_float_octets(accumulator2, x, weight_values2);               \
    accumulator3 =                                                            \
        fr13_fma_float_octets(accumulator3, x, weight_values3);               \
  } while (false)

    FR13_ACCUMULATE_REQUEST(0, accumulator00, accumulator01, accumulator02,
                            accumulator03);
    FR13_ACCUMULATE_REQUEST(1, accumulator10, accumulator11, accumulator12,
                            accumulator13);
    FR13_ACCUMULATE_REQUEST(2, accumulator20, accumulator21, accumulator22,
                            accumulator23);
    FR13_ACCUMULATE_REQUEST(3, accumulator30, accumulator31, accumulator32,
                            accumulator33);
#undef FR13_ACCUMULATE_REQUEST
  }

#define FR13_STORE_REQUEST(request, accumulator0, accumulator1, accumulator2, \
                           accumulator3)                                      \
  do {                                                                        \
    const float reduced0 = fr13_reduce_full_warp(accumulator0, lane);          \
    const float reduced1 = fr13_reduce_full_warp(accumulator1, lane);          \
    const float reduced2 = fr13_reduce_full_warp(accumulator2, lane);          \
    const float reduced3 = fr13_reduce_full_warp(accumulator3, lane);          \
    if (lane == 0) {                                                          \
      output[(request) * kVocab + first_row] =                                 \
          __float2bfloat16_rn(reduced0);                                       \
      output[(request) * kVocab + first_row + 1] =                             \
          __float2bfloat16_rn(reduced1);                                       \
      output[(request) * kVocab + first_row + 2] =                             \
          __float2bfloat16_rn(reduced2);                                       \
      output[(request) * kVocab + first_row + 3] =                             \
          __float2bfloat16_rn(reduced3);                                       \
    }                                                                         \
  } while (false)

  FR13_STORE_REQUEST(0, accumulator00, accumulator01, accumulator02,
                     accumulator03);
  FR13_STORE_REQUEST(1, accumulator10, accumulator11, accumulator12,
                     accumulator13);
  FR13_STORE_REQUEST(2, accumulator20, accumulator21, accumulator22,
                     accumulator23);
  FR13_STORE_REQUEST(3, accumulator30, accumulator31, accumulator32,
                     accumulator33);
#undef FR13_STORE_REQUEST
}

#if !defined(FR13_DEVICE_CODEGEN_ONLY)
void fr13_require_common(const at::Tensor& output, const at::Tensor& input,
                         const at::Tensor& weight, const int batch) {
  TORCH_CHECK(input.is_cuda() && weight.is_cuda() && output.is_cuda(),
              "FR13 BF16 K64 warp4-pair8 requires CUDA tensors");
  TORCH_CHECK(input.device() == weight.device() &&
                  output.device() == input.device(),
              "FR13 BF16 K64 warp4-pair8 tensors must share one CUDA device");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16 &&
                  weight.scalar_type() == at::kBFloat16 &&
                  output.scalar_type() == at::kBFloat16,
              "FR13 BF16 K64 warp4-pair8 requires BF16 tensors");
  TORCH_CHECK(input.sizes() == at::IntArrayRef({batch, kHidden}) &&
                  input.strides() == at::IntArrayRef({kHidden, 1}),
              "FR13 BF16 K64 warp4-pair8 input geometry drifted");
  TORCH_CHECK(weight.sizes() == at::IntArrayRef({kVocab, kHidden}) &&
                  weight.strides() == at::IntArrayRef({kHidden, 1}),
              "FR13 BF16 K64 warp4-pair8 weight geometry drifted");
  TORCH_CHECK(output.sizes() == at::IntArrayRef({batch, kVocab}) &&
                  output.strides() == at::IntArrayRef({kVocab, 1}),
              "FR13 BF16 K64 warp4-pair8 output geometry drifted");
  TORCH_CHECK(
      reinterpret_cast<std::uintptr_t>(input.data_ptr()) % alignof(uint4) == 0 &&
          reinterpret_cast<std::uintptr_t>(weight.data_ptr()) % alignof(uint4) ==
              0,
      "FR13 BF16 K64 warp4-pair8 operands must be 16-byte aligned");
}

void fr13_require_sm121() {
  const cudaDeviceProp* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties != nullptr && properties->major == 12 &&
                  properties->minor == 1,
              "FR13 BF16 K64 warp4-pair8 is qualified only for SM121");
}

void fr13_bf16_gemvx_k64_m1_warp4_pair8_out(
    at::Tensor output, const at::Tensor& input, const at::Tensor& weight) {
  fr13_require_common(output, input, weight, 1);
  const c10::cuda::CUDAGuard device_guard(input.device());
  fr13_require_sm121();
  const dim3 block(kLanes, kWarpsPerCta, 1);
  fr13_bf16_gemvx_k64_m1_warp4_pair8_kernel
      <<<kCtas, block, 0, at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
          reinterpret_cast<const __nv_bfloat16*>(
              input.data_ptr<at::BFloat16>()),
          reinterpret_cast<const __nv_bfloat16*>(
              weight.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fr13_bf16_gemvx_k64_m4_warp4_pair8_out(
    at::Tensor output, const at::Tensor& input, const at::Tensor& weight) {
  fr13_require_common(output, input, weight, kBatch4);
  const c10::cuda::CUDAGuard device_guard(input.device());
  fr13_require_sm121();
  const dim3 block(kLanes, kWarpsPerCta, 1);
  fr13_bf16_gemvx_k64_m4_warp4_pair8_kernel
      <<<kCtas, block, 0, at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
          reinterpret_cast<const __nv_bfloat16*>(
              input.data_ptr<at::BFloat16>()),
          reinterpret_cast<const __nv_bfloat16*>(
              weight.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
#endif

}  // namespace

#if !defined(FR13_DEVICE_CODEGEN_ONLY)
TORCH_LIBRARY_FRAGMENT(fr13_bf16_k64_head, library) {
  library.def(
      "gemvx_m1_warp4_pair8_out(Tensor(a!) output, Tensor input, Tensor weight) -> ()");
  library.def(
      "gemvx_m4_warp4_pair8_out(Tensor(a!) output, Tensor input, Tensor weight) -> ()");
}

TORCH_LIBRARY_IMPL(fr13_bf16_k64_head, CUDA, library) {
  library.impl("gemvx_m1_warp4_pair8_out",
               &fr13_bf16_gemvx_k64_m1_warp4_pair8_out);
  library.impl("gemvx_m4_warp4_pair8_out",
               &fr13_bf16_gemvx_k64_m4_warp4_pair8_out);
}
#endif
