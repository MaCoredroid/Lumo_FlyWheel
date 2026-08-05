#if !defined(FR13_CODEGEN_ONLY)
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>
#endif

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/matrix.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace {

constexpr int kRows = 32;
constexpr int kVocab = 248320;
constexpr int kHidden = 5120;
constexpr int kThreadblockM = 32;
constexpr int kThreadblockN = 256;
constexpr int kThreadblockK = 32;
constexpr int kWarpM = 32;
constexpr int kWarpN = 64;
constexpr int kWarpK = 32;
constexpr int kStages = 3;

static_assert(kRows == kThreadblockM);
static_assert(kVocab % kThreadblockN == 0);
static_assert(kHidden % kThreadblockK == 0);
#if !defined(FR13_CODEGEN_ONLY)
static_assert(sizeof(at::BFloat16) == sizeof(cutlass::bfloat16_t));
#endif

using Element = cutlass::bfloat16_t;
using Accumulator = float;
using HiddenLayout = cutlass::layout::RowMajor;
using WeightLayout = cutlass::layout::ColumnMajor;
using OutputLayout = cutlass::layout::RowMajor;
using Epilogue = cutlass::epilogue::thread::LinearCombination<
    Element,
    128 / cutlass::sizeof_bits<Element>::value,
    Accumulator,
    Accumulator>;

// Four 32x64 warps share each 32x256 CTA. Relative to the existing
// N128xK64/stage3 M32 candidate, K32 keeps the same ordered K16 tensor-core
// updates while halving CTA count and shared storage. No split-K is used.
using VerifierHeadGemm = cutlass::gemm::device::Gemm<
    Element,
    HiddenLayout,
    Element,
    WeightLayout,
    Element,
    OutputLayout,
    Accumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<kThreadblockM, kThreadblockN, kThreadblockK>,
    cutlass::gemm::GemmShape<kWarpM, kWarpN, kWarpK>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    Epilogue,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    kStages>;

constexpr std::size_t kSharedStorageBytes =
    sizeof(typename VerifierHeadGemm::GemmKernel::SharedStorage);
static_assert(kSharedStorageBytes == 55296);

typename VerifierHeadGemm::Arguments make_arguments(
    Element* output_ptr,
    const Element* hidden_ptr,
    const Element* weight_ptr) {
  return typename VerifierHeadGemm::Arguments(
      {kRows, kVocab, kHidden},
      {hidden_ptr, kHidden},
      {weight_ptr, kHidden},
      {output_ptr, kVocab},
      {output_ptr, kVocab},
      {1.0f, 0.0f},
      1);
}

#if defined(FR13_CODEGEN_ONLY)

extern "C" cutlass::Status fr13_bf16_verifier_head_m32_n256k32s3_codegen(
    Element* output_ptr,
    const Element* hidden_ptr,
    const Element* weight_ptr,
    cudaStream_t stream) {
  auto arguments = make_arguments(output_ptr, hidden_ptr, weight_ptr);
  VerifierHeadGemm gemm;
  return gemm(arguments, nullptr, stream);
}

#else

void fr13_bf16_verifier_head_m32_n256k32s3_out(
    at::Tensor output,
    const at::Tensor& hidden,
    const at::Tensor& weight) {
  TORCH_CHECK(output.is_cuda() && hidden.is_cuda() && weight.is_cuda(),
              "FR13 verifier head N256/K32/stage3 requires CUDA tensors");
  TORCH_CHECK(output.device() == hidden.device() &&
                  weight.device() == hidden.device(),
              "FR13 verifier head N256/K32/stage3 tensors must share one CUDA device");
  TORCH_CHECK(output.scalar_type() == at::kBFloat16 &&
                  hidden.scalar_type() == at::kBFloat16 &&
                  weight.scalar_type() == at::kBFloat16,
              "FR13 verifier head N256/K32/stage3 tensors must be BF16");
  TORCH_CHECK(hidden.sizes() == at::IntArrayRef({kRows, kHidden}) &&
                  hidden.strides() == at::IntArrayRef({kHidden, 1}),
              "FR13 verifier hidden must be contiguous BF16[32,5120]");
  TORCH_CHECK(weight.sizes() == at::IntArrayRef({kVocab, kHidden}) &&
                  weight.strides() == at::IntArrayRef({kHidden, 1}),
              "FR13 verifier weight must be contiguous BF16[248320,5120]");
  TORCH_CHECK(output.sizes() == at::IntArrayRef({kRows, kVocab}) &&
                  output.strides() == at::IntArrayRef({kVocab, 1}),
              "FR13 verifier output must be contiguous BF16[32,248320]");
  TORCH_CHECK(!output.is_alias_of(hidden) && !output.is_alias_of(weight),
              "FR13 verifier output must not alias either input");

  const auto hidden_address =
      reinterpret_cast<std::uintptr_t>(hidden.data_ptr());
  const auto weight_address =
      reinterpret_cast<std::uintptr_t>(weight.data_ptr());
  const auto output_address =
      reinterpret_cast<std::uintptr_t>(output.data_ptr());
  TORCH_CHECK((hidden_address | weight_address | output_address) % 16 == 0,
              "FR13 verifier head tensors must be 16-byte aligned");

  const c10::cuda::CUDAGuard device_guard(hidden.device());
  const cudaDeviceProp* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties != nullptr && properties->major == 12 &&
                  properties->minor == 1,
              "FR13 verifier head N256/K32/stage3 is qualified only for SM121");

  auto* output_ptr =
      reinterpret_cast<Element*>(output.data_ptr<at::BFloat16>());
  const auto* hidden_ptr =
      reinterpret_cast<const Element*>(hidden.data_ptr<at::BFloat16>());
  const auto* weight_ptr =
      reinterpret_cast<const Element*>(weight.data_ptr<at::BFloat16>());
  auto arguments = make_arguments(output_ptr, hidden_ptr, weight_ptr);

  TORCH_CHECK(VerifierHeadGemm::get_workspace_size(arguments) == 0,
              "FR13 verifier head unexpectedly requires GEMM workspace");
  TORCH_CHECK(VerifierHeadGemm::can_implement(arguments) ==
                  cutlass::Status::kSuccess,
              "FR13 verifier head N256/K32/stage3 CUTLASS geometry is unsupported");

  VerifierHeadGemm gemm;
  const cutlass::Status status = gemm(
      arguments, nullptr, at::cuda::getCurrentCUDAStream(hidden.device().index()));
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "FR13 verifier head N256/K32/stage3 CUTLASS launch failed");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#endif

}  // namespace

#if !defined(FR13_CODEGEN_ONLY)
TORCH_LIBRARY_FRAGMENT(fr13_verifier_head, library) {
  library.def(
      "bf16_m32_n256k32s3_out(Tensor(a!) output, Tensor hidden, Tensor weight) -> ()");
}

TORCH_LIBRARY_IMPL(fr13_verifier_head, CUDA, library) {
  library.impl("bf16_m32_n256k32s3_out",
               &fr13_bf16_verifier_head_m32_n256k32s3_out);
}
#endif
