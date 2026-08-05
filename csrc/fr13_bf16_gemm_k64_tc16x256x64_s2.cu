#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/matrix.h"

#include <cstddef>
#include <cstdint>

namespace {

constexpr int kHidden = 5120;
constexpr int kVocab = 65536;
constexpr int kBatch1 = 1;
constexpr int kBatch4 = 4;
constexpr int kThreadblockM = 16;
constexpr int kThreadblockN = 256;
constexpr int kThreadblockK = 64;
constexpr int kWarpM = 16;
constexpr int kWarpN = 64;
constexpr int kWarpK = 64;
constexpr int kInstructionM = 16;
constexpr int kInstructionN = 8;
constexpr int kInstructionK = 16;
constexpr int kStages = 2;
constexpr int kLogicalCtas = kVocab / kThreadblockN;
constexpr int kWarpsPerCta = kThreadblockN / kWarpN;
constexpr int kThreadsPerCta = kWarpsPerCta * 32;

static_assert(kVocab % kThreadblockN == 0);
static_assert(kHidden % kThreadblockK == 0);
static_assert(kLogicalCtas == 256);
static_assert(kWarpsPerCta == 4);
static_assert(kThreadsPerCta == 128);
static_assert(sizeof(at::BFloat16) == sizeof(cutlass::bfloat16_t));

using Element = cutlass::bfloat16_t;
using Accumulator = float;
using HiddenLayout = cutlass::layout::RowMajor;
using WeightLayout = cutlass::layout::ColumnMajor;
using OutputLayout = cutlass::layout::RowMajor;
using Epilogue = cutlass::epilogue::thread::LinearCombination<
    Element,
    128 / cutlass::sizeof_bits<Element>::value,
    Accumulator,
    Accumulator,
    cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>;

// Distinct swizzle types force separate fixed-B1 and fixed-B4 kernel symbols
// while retaining the same one-dimensional 256-column tile order.
struct M1IdentitySwizzle
    : cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1> {};
struct M4IdentitySwizzle
    : cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1> {};

template <class Swizzle>
using FixedK64TensorHead = cutlass::gemm::device::Gemm<
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
    cutlass::gemm::GemmShape<kInstructionM, kInstructionN, kInstructionK>,
    Epilogue,
    Swizzle,
    kStages,
    8,
    8>;

using M1TensorHead = FixedK64TensorHead<M1IdentitySwizzle>;
using M4TensorHead = FixedK64TensorHead<M4IdentitySwizzle>;

constexpr std::size_t kM1SharedStorageBytes =
    sizeof(typename M1TensorHead::GemmKernel::SharedStorage);
constexpr std::size_t kM4SharedStorageBytes =
    sizeof(typename M4TensorHead::GemmKernel::SharedStorage);
static_assert(kM1SharedStorageBytes == kM4SharedStorageBytes);
static_assert(kM1SharedStorageBytes == 69632);

void fr13_require_common(const at::Tensor& output, const at::Tensor& input,
                         const at::Tensor& weight, const int batch) {
  TORCH_CHECK(input.is_cuda() && weight.is_cuda() && output.is_cuda(),
              "FR13 K64 tensor head requires CUDA tensors");
  TORCH_CHECK(input.device() == weight.device() &&
                  output.device() == input.device(),
              "FR13 K64 tensor head tensors must share one CUDA device");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16 &&
                  weight.scalar_type() == at::kBFloat16 &&
                  output.scalar_type() == at::kBFloat16,
              "FR13 K64 tensor head requires BF16 tensors");
  TORCH_CHECK(input.sizes() == at::IntArrayRef({batch, kHidden}) &&
                  input.strides() == at::IntArrayRef({kHidden, 1}),
              "FR13 K64 tensor head input geometry drifted");
  TORCH_CHECK(weight.sizes() == at::IntArrayRef({kVocab, kHidden}) &&
                  weight.strides() == at::IntArrayRef({kHidden, 1}),
              "FR13 K64 tensor head weight geometry drifted");
  TORCH_CHECK(output.sizes() == at::IntArrayRef({batch, kVocab}) &&
                  output.strides() == at::IntArrayRef({kVocab, 1}),
              "FR13 K64 tensor head output geometry drifted");
  TORCH_CHECK(!output.is_alias_of(input) && !output.is_alias_of(weight),
              "FR13 K64 tensor head output must not alias an input");
  const auto input_address =
      reinterpret_cast<std::uintptr_t>(input.data_ptr());
  const auto weight_address =
      reinterpret_cast<std::uintptr_t>(weight.data_ptr());
  const auto output_address =
      reinterpret_cast<std::uintptr_t>(output.data_ptr());
  TORCH_CHECK((input_address | weight_address | output_address) % 16 == 0,
              "FR13 K64 tensor head tensors must be 16-byte aligned");
}

void fr13_require_sm121() {
  const cudaDeviceProp* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties != nullptr && properties->major == 12 &&
                  properties->minor == 1,
              "FR13 K64 tensor head is qualified only for SM121");
}

template <class Gemm, int Batch>
void fr13_launch_tensor_head(at::Tensor output, const at::Tensor& input,
                             const at::Tensor& weight) {
  fr13_require_common(output, input, weight, Batch);
  const c10::cuda::CUDAGuard device_guard(input.device());
  fr13_require_sm121();

  auto* output_ptr =
      reinterpret_cast<Element*>(output.data_ptr<at::BFloat16>());
  const auto* input_ptr =
      reinterpret_cast<const Element*>(input.data_ptr<at::BFloat16>());
  const auto* weight_ptr =
      reinterpret_cast<const Element*>(weight.data_ptr<at::BFloat16>());
  typename Gemm::Arguments arguments(
      {Batch, kVocab, kHidden},
      {input_ptr, kHidden},
      {weight_ptr, kHidden},
      {output_ptr, kVocab},
      {output_ptr, kVocab},
      {1.0f, 0.0f},
      1);
  TORCH_CHECK(Gemm::get_workspace_size(arguments) == 0,
              "FR13 K64 tensor head unexpectedly requires workspace");
  TORCH_CHECK(Gemm::can_implement(arguments) == cutlass::Status::kSuccess,
              "FR13 K64 tensor-head geometry is unsupported");
  Gemm gemm;
  const cutlass::Status status = gemm(
      arguments, nullptr,
      at::cuda::getCurrentCUDAStream(input.device().index()));
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "FR13 K64 tensor-head launch failed");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fr13_bf16_gemm_k64_m1_tc16x256x64_s2_out(
    at::Tensor output, const at::Tensor& input, const at::Tensor& weight) {
  fr13_launch_tensor_head<M1TensorHead, kBatch1>(output, input, weight);
}

void fr13_bf16_gemm_k64_m4_tc16x256x64_s2_out(
    at::Tensor output, const at::Tensor& input, const at::Tensor& weight) {
  fr13_launch_tensor_head<M4TensorHead, kBatch4>(output, input, weight);
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(fr13_bf16_k64_tc_head, library) {
  library.def(
      "gemm_m1_tc16x256x64_s2_out(Tensor(a!) output, Tensor input, Tensor weight) -> ()");
  library.def(
      "gemm_m4_tc16x256x64_s2_out(Tensor(a!) output, Tensor input, Tensor weight) -> ()");
}

TORCH_LIBRARY_IMPL(fr13_bf16_k64_tc_head, CUDA, library) {
  library.impl("gemm_m1_tc16x256x64_s2_out",
               &fr13_bf16_gemm_k64_m1_tc16x256x64_s2_out);
  library.impl("gemm_m4_tc16x256x64_s2_out",
               &fr13_bf16_gemm_k64_m4_tc16x256x64_s2_out);
}
