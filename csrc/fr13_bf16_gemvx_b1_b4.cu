#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kHidden = 5120;
constexpr int kVocab = 248320;
constexpr int kLanes = 16;
constexpr int kRowsPerCta = 8;
constexpr int kSharedRowStride = 17;
constexpr int kMaxBatch = 4;
constexpr int kCtas = kVocab / kRowsPerCta;

static_assert(kHidden % kLanes == 0);
static_assert(kVocab % kRowsPerCta == 0);
static_assert(kCtas == 31040);
static_assert(sizeof(at::BFloat16) == sizeof(__nv_bfloat16));

// B1-B4 uses the same per-logit arithmetic as the exact-M1 kernel, while a
// CTA reuses each weight value across every request row. Each batch size has a
// distinct specialization so CUDA graphs retain a fixed kernel geometry.
template <int kBatch>
__global__ __launch_bounds__(kLanes * kRowsPerCta) void
fr13_bf16_gemvx_b1_b4_kernel(__nv_bfloat16* __restrict__ output,
                             const __nv_bfloat16* __restrict__ input,
                             const __nv_bfloat16* __restrict__ weight,
                             const float alpha, const float beta) {
  static_assert(kBatch >= 1 && kBatch <= kMaxBatch);
  const int lane = static_cast<int>(threadIdx.x);
  const int row_in_cta = static_cast<int>(threadIdx.y);
  const int row = static_cast<int>(blockIdx.x) * kRowsPerCta + row_in_cta;

  float accumulators[kBatch];
#pragma unroll
  for (int batch = 0; batch < kBatch; ++batch) {
    accumulators[batch] = 0.0f;
  }
#pragma unroll 1
  for (int k = lane; k < kHidden; k += kLanes) {
    const float w = __bfloat162float(weight[row * kHidden + k]);
#pragma unroll
    for (int batch = 0; batch < kBatch; ++batch) {
      const float x = __bfloat162float(input[batch * kHidden + k]);
      accumulators[batch] = __fmaf_rn(x, w, accumulators[batch]);
    }
  }

  extern __shared__ float partials[];
#pragma unroll
  for (int batch = 0; batch < kBatch; ++batch) {
    float* row_partials =
        partials + (batch * kRowsPerCta + row_in_cta) * kSharedRowStride;
    row_partials[lane] = accumulators[batch];
  }
  __syncthreads();

#pragma unroll
  for (int batch = 0; batch < kBatch; ++batch) {
    float* row_partials =
        partials + (batch * kRowsPerCta + row_in_cta) * kSharedRowStride;
    if (lane < 8) {
      row_partials[lane] =
          __fadd_rn(row_partials[lane], row_partials[lane + 8]);
    }
  }
  __syncthreads();
#pragma unroll
  for (int batch = 0; batch < kBatch; ++batch) {
    float* row_partials =
        partials + (batch * kRowsPerCta + row_in_cta) * kSharedRowStride;
    if (lane < 4) {
      row_partials[lane] =
          __fadd_rn(row_partials[lane], row_partials[lane + 4]);
    }
  }
  __syncthreads();
#pragma unroll
  for (int batch = 0; batch < kBatch; ++batch) {
    float* row_partials =
        partials + (batch * kRowsPerCta + row_in_cta) * kSharedRowStride;
    if (lane < 2) {
      row_partials[lane] =
          __fadd_rn(row_partials[lane], row_partials[lane + 2]);
    }
  }
  __syncthreads();
  if (lane == 0) {
#pragma unroll
    for (int batch = 0; batch < kBatch; ++batch) {
      float* row_partials =
          partials + (batch * kRowsPerCta + row_in_cta) * kSharedRowStride;
      const float reduced_sum =
          __fadd_rn(row_partials[0], row_partials[1]);
      const float sum = __fmaf_rn(alpha, reduced_sum, beta);
      output[batch * kVocab + row] = __float2bfloat16_rn(sum);
    }
  }
}

template <int kBatch>
void launch_b1_b4(at::Tensor& output, const at::Tensor& input,
                  const at::Tensor& weight) {
  const dim3 block(kLanes, kRowsPerCta, 1);
  constexpr size_t shared_bytes =
      kBatch * kRowsPerCta * kSharedRowStride * sizeof(float);
  fr13_bf16_gemvx_b1_b4_kernel<kBatch>
      <<<kCtas, block, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
          reinterpret_cast<const __nv_bfloat16*>(
              input.data_ptr<at::BFloat16>()),
          reinterpret_cast<const __nv_bfloat16*>(
              weight.data_ptr<at::BFloat16>()),
          1.0f, 0.0f);
}

void fr13_bf16_gemvx_b1_b4_out(at::Tensor output, const at::Tensor& input,
                               const at::Tensor& weight) {
  TORCH_CHECK(input.is_cuda() && weight.is_cuda() && output.is_cuda(),
              "FR13 BF16 B1-B4 GEMV requires CUDA tensors");
  TORCH_CHECK(input.device() == weight.device() &&
                  output.device() == input.device(),
              "FR13 BF16 B1-B4 GEMV tensors must share one CUDA device");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16 &&
                  weight.scalar_type() == at::kBFloat16 &&
                  output.scalar_type() == at::kBFloat16,
              "FR13 BF16 B1-B4 GEMV requires BF16 tensors");
  TORCH_CHECK(input.dim() == 2 && input.size(0) >= 1 &&
                  input.size(0) <= kMaxBatch && input.size(1) == kHidden &&
                  input.stride(0) == kHidden && input.stride(1) == 1,
              "FR13 BF16 B1-B4 GEMV input must be contiguous [B,5120], B=1..4");
  const int batch = static_cast<int>(input.size(0));
  TORCH_CHECK(weight.sizes() == at::IntArrayRef({kVocab, kHidden}) &&
                  weight.strides() == at::IntArrayRef({kHidden, 1}),
              "FR13 BF16 B1-B4 GEMV weight must be contiguous [248320,5120]");
  TORCH_CHECK(output.dim() == 2 && output.size(0) == batch &&
                  output.size(1) == kVocab && output.stride(0) == kVocab &&
                  output.stride(1) == 1,
              "FR13 BF16 B1-B4 GEMV output must be contiguous [B,248320]");

  const c10::cuda::CUDAGuard device_guard(input.device());
  switch (batch) {
    case 1:
      launch_b1_b4<1>(output, input, weight);
      break;
    case 2:
      launch_b1_b4<2>(output, input, weight);
      break;
    case 3:
      launch_b1_b4<3>(output, input, weight);
      break;
    case 4:
      launch_b1_b4<4>(output, input, weight);
      break;
    default:
      TORCH_CHECK(false, "FR13 BF16 B1-B4 GEMV batch dispatch drifted");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

TORCH_LIBRARY(fr13_bf16_head, library) {
  library.def(
      "gemvx_b1_b4_out(Tensor(a!) output, Tensor input, Tensor weight) -> ()");
}

TORCH_LIBRARY_IMPL(fr13_bf16_head, CUDA, library) {
  library.impl("gemvx_b1_b4_out", &fr13_bf16_gemvx_b1_b4_out);
}
