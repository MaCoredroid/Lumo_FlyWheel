#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

namespace {

constexpr int kVocab = 65536;
constexpr int kHidden = 5120;
constexpr int kScaleGroup = 128;
constexpr int kGroups = kHidden / kScaleGroup;
constexpr int kWeightScaleRows = kVocab / kScaleGroup;
constexpr int kTopK = 3;
constexpr int kMaxBatch = 4;
constexpr int kRowsPerPartial = 128;
constexpr int kPartials = kVocab / kRowsPerPartial;
constexpr int kThreads = 256;
constexpr int kWarp = 32;
constexpr int kWarps = kThreads / kWarp;
constexpr int kRowsPerWarp = kRowsPerPartial / kWarps;
constexpr unsigned kFullWarpMask = 0xffffffffu;

static_assert(kGroups == 40);
static_assert(kWeightScaleRows == 512);
static_assert(kPartials == 512);
static_assert(kWarps == 8);
static_assert(kRowsPerWarp == 16);
static_assert(sizeof(at::Float8_e4m3fn) == sizeof(__nv_fp8_e4m3));
static_assert(sizeof(at::BFloat16) == sizeof(__nv_bfloat16));

struct Candidate {
  float value;
  int index;
};

__device__ __forceinline__ Candidate fr13_sentinel() {
  return Candidate{-__int_as_float(0x7f800000),
                   std::numeric_limits<int>::max()};
}

// This is the same explicit total order as the already-qualified K64 mapped
// top3 reducer: NaNs rank first, then descending BF16 score, then lower subset
// index. Mapping never participates in the order.
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

__device__ __forceinline__ void fr13_insert(const Candidate value,
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

__device__ __forceinline__ float fr13_warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(kFullWarpMask, value, offset, kWarp);
  }
  return value;
}

// One block owns exactly one 128-row FP8 weight-scale tile. It loads every
// qweight byte once and reuses it across B1 or all four B4 activation rows.
// Scores are rounded to BF16 before selection, matching the removed materialized
// logits' observable precision. Only three BF16 scores and subset IDs per tile
// are written to the persistent workspace.
__global__ __launch_bounds__(kThreads, 2) void
fr13_dfwd_k64_fp8_partial_top3_kernel(
    __nv_bfloat16* __restrict__ partial_values,
    int32_t* __restrict__ partial_indices,
    const __nv_fp8_e4m3* __restrict__ activation_q,
    const __nv_fp8_e4m3* __restrict__ qweight,
    const float* __restrict__ activation_scale,
    const float* __restrict__ weight_scale,
    const int batch) {
  const int partial = static_cast<int>(blockIdx.x);
  const int thread = static_cast<int>(threadIdx.x);
  const int lane = thread & (kWarp - 1);
  const int warp = thread / kWarp;

  __shared__ __nv_fp8_e4m3 shared_activation[kMaxBatch * kHidden];
  __shared__ float shared_scale[kMaxBatch * kGroups];
  __shared__ Candidate shared_candidates[kMaxBatch][kRowsPerPartial];

  for (int index = thread; index < batch * kHidden; index += kThreads) {
    shared_activation[index] = activation_q[index];
  }
  for (int index = thread; index < batch * kGroups; index += kThreads) {
    const int batch_index = index / kGroups;
    const int group = index - batch_index * kGroups;
    // Activation scales are [B,40] column-major with stride (1,B).
    shared_scale[index] =
        activation_scale[group * batch + batch_index] *
        weight_scale[partial * kGroups + group];
  }
  __syncthreads();

#pragma unroll 1
  for (int row_iteration = 0; row_iteration < kRowsPerWarp;
       ++row_iteration) {
    const int row_in_partial = row_iteration * kWarps + warp;
    const int vocab_index = partial * kRowsPerPartial + row_in_partial;
    float accumulators[kMaxBatch] = {0.0f, 0.0f, 0.0f, 0.0f};

#pragma unroll 1
    for (int group = 0; group < kGroups; ++group) {
      const int group_base = group * kScaleGroup;
#pragma unroll
      for (int element = lane; element < kScaleGroup; element += kWarp) {
        const int hidden_index = group_base + element;
        const float weight_value = static_cast<float>(
            qweight[vocab_index * kHidden + hidden_index]);
#pragma unroll
        for (int batch_index = 0; batch_index < kMaxBatch; ++batch_index) {
          if (batch_index < batch) {
            const float activation_value = static_cast<float>(
                shared_activation[batch_index * kHidden + hidden_index]);
            accumulators[batch_index] = fmaf(
                activation_value,
                weight_value * shared_scale[batch_index * kGroups + group],
                accumulators[batch_index]);
          }
        }
      }
    }

#pragma unroll
    for (int batch_index = 0; batch_index < kMaxBatch; ++batch_index) {
      if (batch_index < batch) {
        const float score = fr13_warp_sum(accumulators[batch_index]);
        if (lane == 0) {
          const __nv_bfloat16 rounded = __float2bfloat16_rn(score);
          shared_candidates[batch_index][row_in_partial] = Candidate{
              __bfloat162float(rounded), vocab_index};
        }
      }
    }
  }
  __syncthreads();

  if (warp == 0) {
#pragma unroll
    for (int batch_index = 0; batch_index < kMaxBatch; ++batch_index) {
      if (batch_index < batch) {
        Candidate first = fr13_sentinel();
        Candidate second = first;
        Candidate third = first;
#pragma unroll
        for (int index = lane; index < kRowsPerPartial; index += kWarp) {
          fr13_insert(shared_candidates[batch_index][index], first, second,
                      third);
        }
        fr13_warp_top3(first, second, third, lane);
        if (lane == 0) {
          const int output_base =
              (batch_index * kPartials + partial) * kTopK;
          partial_values[output_base] = __float2bfloat16_rn(first.value);
          partial_values[output_base + 1] =
              __float2bfloat16_rn(second.value);
          partial_values[output_base + 2] =
              __float2bfloat16_rn(third.value);
          partial_indices[output_base] = first.index;
          partial_indices[output_base + 1] = second.index;
          partial_indices[output_base + 2] = third.index;
        }
      }
    }
  }
}

__global__ __launch_bounds__(kThreads, 1) void
fr13_dfwd_k64_fp8_finish_top3_kernel(
    int64_t* __restrict__ spine_output,
    int64_t* __restrict__ top3_ids,
    __nv_bfloat16* __restrict__ top3_scores,
    const __nv_bfloat16* __restrict__ partial_values,
    const int32_t* __restrict__ partial_indices,
    const int64_t* __restrict__ id_map,
    const int batch) {
  const int batch_index = static_cast<int>(blockIdx.x);
  const int thread = static_cast<int>(threadIdx.x);
  const int lane = thread & (kWarp - 1);
  const int warp = thread / kWarp;
  Candidate first = fr13_sentinel();
  Candidate second = first;
  Candidate third = first;

  const int input_base = batch_index * kPartials * kTopK;
  for (int index = thread; index < kPartials * kTopK; index += kThreads) {
    fr13_insert(
        Candidate{__bfloat162float(partial_values[input_base + index]),
                  partial_indices[input_base + index]},
        first, second, third);
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
    Candidate block_first = fr13_sentinel();
    Candidate block_second = block_first;
    Candidate block_third = block_first;
    if (lane < kWarps) {
      block_first = warp_candidates[lane][0];
      block_second = warp_candidates[lane][1];
      block_third = warp_candidates[lane][2];
    }
    fr13_warp_top3(block_first, block_second, block_third, lane);
    if (lane == 0) {
      const int output_base = batch_index * kTopK;
      const int64_t mapped_first = id_map[block_first.index];
      spine_output[batch_index] = mapped_first;
      top3_ids[output_base] = mapped_first;
      top3_ids[output_base + 1] = id_map[block_second.index];
      top3_ids[output_base + 2] = id_map[block_third.index];
      top3_scores[output_base] = __float2bfloat16_rn(block_first.value);
      top3_scores[output_base + 1] =
          __float2bfloat16_rn(block_second.value);
      top3_scores[output_base + 2] =
          __float2bfloat16_rn(block_third.value);
    }
  }
}

void fr13_dfwd_k64_fp8_mapped_top3_out(
    at::Tensor spine_output,
    at::Tensor top3_ids,
    at::Tensor top3_scores,
    at::Tensor partial_values,
    at::Tensor partial_indices,
    const at::Tensor& activation_q,
    const at::Tensor& qweight,
    const at::Tensor& activation_scale,
    const at::Tensor& weight_scale,
    const at::Tensor& id_map) {
  TORCH_CHECK(spine_output.is_cuda() && top3_ids.is_cuda() &&
                  top3_scores.is_cuda() && partial_values.is_cuda() &&
                  partial_indices.is_cuda() && activation_q.is_cuda() &&
                  qweight.is_cuda() && activation_scale.is_cuda() &&
                  weight_scale.is_cuda() && id_map.is_cuda(),
              "FR13 DFWD FP8 mapped top3 requires CUDA tensors");
  TORCH_CHECK(spine_output.device() == activation_q.device() &&
                  top3_ids.device() == activation_q.device() &&
                  top3_scores.device() == activation_q.device() &&
                  partial_values.device() == activation_q.device() &&
                  partial_indices.device() == activation_q.device() &&
                  qweight.device() == activation_q.device() &&
                  activation_scale.device() == activation_q.device() &&
                  weight_scale.device() == activation_q.device() &&
                  id_map.device() == activation_q.device(),
              "FR13 DFWD FP8 mapped top3 tensors must share one CUDA device");
  TORCH_CHECK(activation_q.scalar_type() == at::kFloat8_e4m3fn &&
                  qweight.scalar_type() == at::kFloat8_e4m3fn,
              "FR13 DFWD FP8 activations and weights must be float8_e4m3fn");
  TORCH_CHECK(activation_scale.scalar_type() == at::kFloat &&
                  weight_scale.scalar_type() == at::kFloat,
              "FR13 DFWD FP8 scales must be FP32");
  TORCH_CHECK(spine_output.scalar_type() == at::kLong &&
                  top3_ids.scalar_type() == at::kLong &&
                  id_map.scalar_type() == at::kLong,
              "FR13 DFWD FP8 IDs must be int64");
  TORCH_CHECK(top3_scores.scalar_type() == at::kBFloat16 &&
                  partial_values.scalar_type() == at::kBFloat16,
              "FR13 DFWD FP8 scores must be BF16");
  TORCH_CHECK(partial_indices.scalar_type() == at::kInt,
              "FR13 DFWD FP8 partial indices must be int32");

  const int batch = static_cast<int>(activation_q.size(0));
  TORCH_CHECK(batch == 1 || batch == 4,
              "FR13 DFWD FP8 mapped top3 serves only exact B1 or B4");
  TORCH_CHECK(activation_q.sizes() == at::IntArrayRef({batch, kHidden}) &&
                  activation_q.strides() ==
                      at::IntArrayRef({kHidden, 1}),
              "FR13 DFWD FP8 activation must be contiguous [B,5120]");
  TORCH_CHECK(qweight.sizes() == at::IntArrayRef({kVocab, kHidden}) &&
                  qweight.strides() == at::IntArrayRef({kHidden, 1}),
              "FR13 DFWD FP8 qweight must be contiguous [65536,5120]");
  TORCH_CHECK(activation_scale.sizes() ==
                      at::IntArrayRef({batch, kGroups}) &&
                  activation_scale.strides() ==
                      at::IntArrayRef({1, batch}),
              "FR13 DFWD FP8 activation scales must be [B,40] stride (1,B)");
  TORCH_CHECK(weight_scale.sizes() ==
                      at::IntArrayRef({kWeightScaleRows, kGroups}) &&
                  weight_scale.strides() == at::IntArrayRef({kGroups, 1}),
              "FR13 DFWD FP8 weight scales must be contiguous [512,40]");
  TORCH_CHECK(id_map.sizes() == at::IntArrayRef({kVocab}) &&
                  id_map.strides() == at::IntArrayRef({1}),
              "FR13 DFWD FP8 ID map must be contiguous [65536]");
  TORCH_CHECK(spine_output.sizes() == at::IntArrayRef({batch}) &&
                  spine_output.strides() == at::IntArrayRef({1}),
              "FR13 DFWD FP8 spine output must be contiguous [B]");
  TORCH_CHECK(top3_ids.sizes() == at::IntArrayRef({batch, kTopK}) &&
                  top3_ids.strides() == at::IntArrayRef({kTopK, 1}) &&
                  top3_scores.sizes() ==
                      at::IntArrayRef({batch, kTopK}) &&
                  top3_scores.strides() == at::IntArrayRef({kTopK, 1}),
              "FR13 DFWD FP8 top3 outputs must be contiguous [B,3]");
  TORCH_CHECK(partial_values.sizes() ==
                      at::IntArrayRef({batch, kPartials, kTopK}) &&
                  partial_values.strides() ==
                      at::IntArrayRef({kPartials * kTopK, kTopK, 1}) &&
                  partial_indices.sizes() ==
                      at::IntArrayRef({batch, kPartials, kTopK}) &&
                  partial_indices.strides() ==
                      at::IntArrayRef({kPartials * kTopK, kTopK, 1}),
              "FR13 DFWD FP8 workspace must be contiguous [B,512,3]");

  const c10::cuda::CUDAGuard device_guard(activation_q.device());
  const cudaDeviceProp* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties != nullptr && properties->major == 12 &&
                  properties->minor == 1,
              "FR13 DFWD FP8 mapped top3 is qualified only for SM121");
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  fr13_dfwd_k64_fp8_partial_top3_kernel<<<kPartials, kThreads, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(
          partial_values.data_ptr<at::BFloat16>()),
      partial_indices.data_ptr<int32_t>(),
      reinterpret_cast<const __nv_fp8_e4m3*>(
          activation_q.data_ptr<at::Float8_e4m3fn>()),
      reinterpret_cast<const __nv_fp8_e4m3*>(
          qweight.data_ptr<at::Float8_e4m3fn>()),
      activation_scale.data_ptr<float>(), weight_scale.data_ptr<float>(), batch);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  fr13_dfwd_k64_fp8_finish_top3_kernel<<<batch, kThreads, 0, stream>>>(
      spine_output.data_ptr<int64_t>(), top3_ids.data_ptr<int64_t>(),
      reinterpret_cast<__nv_bfloat16*>(
          top3_scores.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(
          partial_values.data_ptr<at::BFloat16>()),
      partial_indices.data_ptr<int32_t>(), id_map.data_ptr<int64_t>(), batch);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(fr13_dfwd_fp8_top3, library) {
  library.def(
      "mapped_top3_out(Tensor(a!) spine_output, Tensor(b!) top3_ids, Tensor(c!) top3_scores, Tensor(d!) partial_values, Tensor(e!) partial_indices, Tensor activation_q, Tensor qweight, Tensor activation_scale, Tensor weight_scale, Tensor id_map) -> ()");
}

TORCH_LIBRARY_IMPL(fr13_dfwd_fp8_top3, CUDA, library) {
  library.impl("mapped_top3_out", &fr13_dfwd_k64_fp8_mapped_top3_out);
}
