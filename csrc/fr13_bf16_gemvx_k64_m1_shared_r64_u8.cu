#if !defined(FR13_DEVICE_CODEGEN_ONLY)
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>
#endif

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kHidden = 5120;
constexpr int kVocab = 65536;
constexpr int kLanes = 16;
constexpr int kRowsPerCta = 64;
constexpr int kSharedRowStride = 17;
constexpr int kCtas = kVocab / kRowsPerCta;

static_assert(kHidden % (kLanes * 8) == 0);
static_assert(kVocab % kRowsPerCta == 0);
static_assert(kLanes * kRowsPerCta == 1024);
static_assert(kCtas == 1024);
#if !defined(FR13_DEVICE_CODEGEN_ONLY)
static_assert(sizeof(at::BFloat16) == sizeof(__nv_bfloat16));
#endif

// Preserve the U8 load schedule while matching the incumbent shared-memory
// reduction association. The 17-float row stride avoids shared bank aliasing
// between the 64 independent output rows in a CTA.
__global__ __launch_bounds__(kLanes * kRowsPerCta) void
fr13_bf16_gemvx_k64_m1_shared_r64_u8_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight) {
  const int lane = static_cast<int>(threadIdx.x);
  const int row_in_cta = static_cast<int>(threadIdx.y);
  const int row = static_cast<int>(blockIdx.x) * kRowsPerCta + row_in_cta;

  float accumulator = 0.0f;
  const int row_base = row * kHidden;
#pragma unroll 1
  for (int k = lane; k < kHidden; k += kLanes * 8) {
#pragma unroll
    for (int step = 0; step < 8; ++step) {
      const int offset = k + step * kLanes;
      const float x = __bfloat162float(input[offset]);
      const float w = __bfloat162float(weight[row_base + offset]);
      accumulator = __fmaf_rn(x, w, accumulator);
    }
  }

  extern __shared__ float partials[];
  float* row_partials = partials + row_in_cta * kSharedRowStride;
  row_partials[lane] = accumulator;
  __syncthreads();
  if (lane < 8) {
    row_partials[lane] =
        __fadd_rn(row_partials[lane], row_partials[lane + 8]);
  }
  __syncthreads();
  if (lane < 4) {
    row_partials[lane] =
        __fadd_rn(row_partials[lane], row_partials[lane + 4]);
  }
  __syncthreads();
  if (lane < 2) {
    row_partials[lane] =
        __fadd_rn(row_partials[lane], row_partials[lane + 2]);
  }
  __syncthreads();
  if (lane == 0) {
    const float sum = __fadd_rn(row_partials[0], row_partials[1]);
    output[row] = __float2bfloat16_rn(sum);
  }
}

#if !defined(FR13_DEVICE_CODEGEN_ONLY)
void fr13_bf16_gemvx_k64_m1_shared_r64_u8_out(
    at::Tensor output, const at::Tensor& input, const at::Tensor& weight) {
  TORCH_CHECK(input.is_cuda() && weight.is_cuda() && output.is_cuda(),
              "FR13 BF16 K64 M1 shared R64 U8 requires CUDA tensors");
  TORCH_CHECK(input.device() == weight.device() &&
                  output.device() == input.device(),
              "FR13 BF16 K64 M1 shared R64 U8 tensors must share one device");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16 &&
                  weight.scalar_type() == at::kBFloat16 &&
                  output.scalar_type() == at::kBFloat16,
              "FR13 BF16 K64 M1 shared R64 U8 requires BF16 tensors");
  TORCH_CHECK(input.sizes() == at::IntArrayRef({1, kHidden}) &&
                  input.strides() == at::IntArrayRef({kHidden, 1}),
              "FR13 BF16 K64 M1 shared R64 U8 input must be [1,5120]");
  TORCH_CHECK(weight.sizes() == at::IntArrayRef({kVocab, kHidden}) &&
                  weight.strides() == at::IntArrayRef({kHidden, 1}),
              "FR13 BF16 K64 M1 shared R64 U8 weight must be [65536,5120]");
  TORCH_CHECK(output.sizes() == at::IntArrayRef({1, kVocab}) &&
                  output.strides() == at::IntArrayRef({kVocab, 1}),
              "FR13 BF16 K64 M1 shared R64 U8 output must be [1,65536]");

  const c10::cuda::CUDAGuard device_guard(input.device());
  const cudaDeviceProp* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties != nullptr && properties->major == 12 &&
                  properties->minor == 1,
              "FR13 BF16 K64 M1 shared R64 U8 is qualified only for SM121");
  const dim3 block(kLanes, kRowsPerCta, 1);
  constexpr size_t shared_bytes =
      kRowsPerCta * kSharedRowStride * sizeof(float);
  fr13_bf16_gemvx_k64_m1_shared_r64_u8_kernel
      <<<kCtas, block, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<__nv_bfloat16*>(
              output.data_ptr<at::BFloat16>()),
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
      "gemvx_m1_shared_r64_u8_out(Tensor(a!) output, Tensor input, Tensor weight) -> ()");
}

TORCH_LIBRARY_IMPL(fr13_bf16_k64_head, CUDA, library) {
  library.impl("gemvx_m1_shared_r64_u8_out",
               &fr13_bf16_gemvx_k64_m1_shared_r64_u8_out);
}
#endif
