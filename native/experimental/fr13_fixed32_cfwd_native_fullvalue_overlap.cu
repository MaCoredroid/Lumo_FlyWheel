#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include <cstdint>
#include <utility>

namespace {

constexpr int kLayers = 48;
constexpr int kRingNodes = 32;
constexpr int kKeyHeads = 16;
constexpr int kValueHeads = 48;
constexpr int kDimK = 128;
constexpr int kDimV = 128;
constexpr int kPathCap = 16;
constexpr int kMaxAcceptedLength = 11;
constexpr int kPrecomputedSteps = kMaxAcceptedLength + 1;
constexpr int kHeadGroup = kValueHeads / kKeyHeads;
constexpr int kWarpsPerBlock = 16;
constexpr int kThreadsPerBlock = kWarpsPerBlock * 32;
constexpr int kValuesPerWarp = kDimV / kWarpsPerBlock;
constexpr int kKeyQuads = kDimK / 32;
constexpr int kStateElementsPerThread = kValuesPerWarp * kKeyQuads;
constexpr int kNormPartialWarps = kDimK / 32;
constexpr int kStepsPerWave = kWarpsPerBlock / kNormPartialWarps;
constexpr int kPrecomputeWaves =
    (kPrecomputedSteps + kStepsPerWave - 1) / kStepsPerWave;
constexpr int kSharedBytes =
    kPrecomputedSteps * kDimK * sizeof(float) +
    kStepsPerWave * kNormPartialWarps * sizeof(float) +
    kStepsPerWave * sizeof(float) +
    kPrecomputedSteps * kHeadGroup * 2 * sizeof(float) +
    kPrecomputedSteps * sizeof(int32_t) + 2 * sizeof(int32_t);
constexpr unsigned kFullWarpMask = 0xffffffffu;
constexpr float kLog2E = 0x1.715476p+0f;

static_assert(kHeadGroup == 3);
static_assert(kPrecomputedSteps == 12);
static_assert(kThreadsPerBlock == 512);
static_assert(kValuesPerWarp == 8);
static_assert(kKeyQuads == 4);
static_assert(kStateElementsPerThread == 32);
static_assert(kNormPartialWarps == 4);
static_assert(kStepsPerWave == 4);
static_assert(kPrecomputeWaves == 3);
static_assert(kSharedBytes == 6568);

__device__ __forceinline__ float load_bf16(const __nv_bfloat16* pointer) {
  return __bfloat162float(*pointer);
}

__device__ __forceinline__ float triton_exp(float value) {
  const float exponent = __fmul_rn(value, kLog2E);
  float result;
  asm volatile("ex2.approx.f32 %0, %1;" : "=f"(result) : "f"(exponent));
  return result;
}

__device__ __forceinline__ float triton_rsqrt(float value) {
  float result;
  asm volatile("rsqrt.approx.ftz.f32 %0, %1;"
               : "=f"(result)
               : "f"(value));
  return result;
}

__device__ __forceinline__ float triton_divide(float numerator,
                                                float denominator) {
  float result;
  asm volatile("div.full.f32 %0, %1, %2;"
               : "=f"(result)
               : "f"(numerator), "f"(denominator));
  return result;
}

__device__ __forceinline__ float triton_butterfly_product_sum(float lhs,
                                                               float rhs) {
  const float product = __fmul_rn(lhs, rhs);
  const float partner =
      __shfl_xor_sync(kFullWarpMask, product, 16);
  float value = __fmaf_rn(lhs, rhs, partner);
#pragma unroll
  for (int mask = 8; mask > 0; mask >>= 1) {
    value = __fadd_rn(
        value, __shfl_xor_sync(kFullWarpMask, value, mask));
  }
  return value;
}

__device__ __forceinline__ float triton_butterfly_four_sum(float value) {
  value = __fadd_rn(value,
                    __shfl_xor_sync(kFullWarpMask, value, 2));
  return __fadd_rn(value,
                   __shfl_xor_sync(kFullWarpMask, value, 1));
}

__device__ __forceinline__ float softplus(float value) {
  return value <= 20.0f
             ? logf(__fadd_rn(1.0f, triton_exp(value)))
             : value;
}

__device__ __forceinline__ float sigmoid(float value) {
  return triton_divide(
      1.0f, __fadd_rn(1.0f, triton_exp(__fsub_rn(0.0f, value))));
}

__global__ __launch_bounds__(kThreadsPerBlock, 2)
void fixed32_cfwd_native_fullvalue_kernel(
    float* bank_anchor, const int64_t* bank_off16,
    const int32_t* accepted_paths, const int32_t* accepted_lens,
    const int32_t* spec_state_indices, const __nv_bfloat16* k_rings,
    const __nv_bfloat16* v_rings, const __nv_bfloat16* a_rings,
    const __nv_bfloat16* b_rings, const float* gate_coeffs,
    int64_t bank_stride, int64_t gate_layer_stride,
    int64_t ring_k_layer_stride, int64_t ring_k_batch_stride,
    int64_t ring_k_node_stride, int64_t ring_v_layer_stride,
    int64_t ring_v_batch_stride, int64_t ring_v_node_stride,
    int64_t ring_ab_layer_stride, int64_t ring_ab_batch_stride,
    int64_t ring_ab_node_stride, int64_t spec_layer_stride,
    int64_t spec_batch_stride, int batch_size) {
  __shared__ float normalized_ks[kPrecomputedSteps][kDimK];
  __shared__ float norm_partials[kStepsPerWave][kNormPartialWarps];
  __shared__ float inverse_norms[kStepsPerWave];
  __shared__ float recurrence_scalars[kPrecomputedSteps][kHeadGroup][2];
  __shared__ int32_t shared_nodes[kPrecomputedSteps];
  __shared__ int32_t shared_steps;
  __shared__ int32_t shared_state_index;

  const int thread_id = threadIdx.x;
  const int warp = thread_id >> 5;
  const int lane = thread_id & 31;
  const int key_head = blockIdx.x % kKeyHeads;
  const int layer_request = blockIdx.x / kKeyHeads;
  const int request = layer_request % batch_size;
  const int layer = layer_request / batch_size;

  if (thread_id == 0) {
    const int accepted = accepted_lens[request];
    shared_steps = min(max(accepted, 0) + 1, kMaxAcceptedLength + 1);
    shared_state_index = spec_state_indices[
        static_cast<int64_t>(layer) * spec_layer_stride +
        static_cast<int64_t>(request) * spec_batch_stride];
  }
  __syncthreads();
  if (shared_state_index <= 0) {
    return;
  }

  const int steps = shared_steps;
  if (thread_id < steps) {
    const int step = thread_id;
    int node = 0;
    if (step > 0) {
      node = accepted_paths[
          static_cast<int64_t>(request) * kPathCap + step - 1];
      node = min(max(node, 0), kRingNodes - 1);
    }
    shared_nodes[step] = node;
  }
  // Both gate-scalar producers and normalization consumers need the node table.
  __syncthreads();

  // Do not join the CTA after this branch. Warps without gate tasks can begin K
  // normalization while the first two warps execute the gate transcendentals.
  const int gate_task_count = steps * kHeadGroup;
  if (thread_id < gate_task_count) {
    const int step = thread_id / kHeadGroup;
    const int local_value_head = thread_id % kHeadGroup;
    const int node = shared_nodes[step];
    const int value_head = key_head * kHeadGroup + local_value_head;
    const int64_t gate_offset =
        static_cast<int64_t>(layer) * gate_layer_stride + value_head * 2;
    const int ab_offset =
        layer * static_cast<int>(ring_ab_layer_stride) +
        request * static_cast<int>(ring_ab_batch_stride) +
        node * static_cast<int>(ring_ab_node_stride) + value_head;
    const float x = __fadd_rn(load_bf16(a_rings + ab_offset),
                              gate_coeffs[gate_offset + 1]);
    const float decay =
        __fmul_rn(gate_coeffs[gate_offset], softplus(x));
    recurrence_scalars[step][local_value_head][0] = triton_exp(decay);
    recurrence_scalars[step][local_value_head][1] =
        sigmoid(load_bf16(b_rings + ab_offset));
  }

  const int step_slot = warp / kNormPartialWarps;
  const int norm_warp = warp % kNormPartialWarps;
#pragma unroll
  for (int wave = 0; wave < kPrecomputeWaves; ++wave) {
    const int step = wave * kStepsPerWave + step_slot;
    const bool active_step = step < steps;
    float warp_partial = 0.0f;
    if (active_step) {
      const int key_index = norm_warp * 32 + lane;
      const int64_t key_offset =
          static_cast<int64_t>(layer) * ring_k_layer_stride +
          static_cast<int64_t>(request) * ring_k_batch_stride +
          static_cast<int64_t>(shared_nodes[step]) * ring_k_node_stride +
          static_cast<int64_t>(key_head) * kDimK + key_index;
      const float key_value = load_bf16(k_rings + key_offset);
      normalized_ks[step][key_index] = key_value;
      warp_partial = triton_butterfly_product_sum(key_value, key_value);
    }
    if (active_step && lane == 0) {
      norm_partials[step_slot][norm_warp] = warp_partial;
    }
    __syncthreads();

    if (active_step && norm_warp == 0) {
      float norm = lane < kNormPartialWarps
                       ? norm_partials[step_slot][lane]
                       : 0.0f;
      norm = triton_butterfly_four_sum(norm);
      if (lane == 0) {
        inverse_norms[step_slot] =
            triton_rsqrt(__fadd_rn(norm, 1.0e-6f));
      }
    }
    __syncthreads();
    if (active_step) {
      const int key_index = norm_warp * 32 + lane;
      normalized_ks[step][key_index] = __fmul_rn(
          normalized_ks[step][key_index], inverse_norms[step_slot]);
    }
  }
  // Publish every normalized K row and gate scalar before recurrence consumes it.
  __syncthreads();

  // Process one value head at a time so the same 32-element register tile is
  // reused three times without persistent shared-state storage.
#pragma unroll 1
  for (int local_value_head = 0; local_value_head < kHeadGroup;
       ++local_value_head) {
    const int value_head = key_head * kHeadGroup + local_value_head;
    const int value_row_base = warp * kValuesPerWarp;
    float state[kValuesPerWarp][kKeyQuads];
    float* load_state_bank = bank_anchor + bank_off16[layer] * 4;
    const int64_t load_state_row_offset =
        static_cast<int64_t>(shared_state_index) * bank_stride;
#pragma unroll
    for (int value_lane = 0; value_lane < kValuesPerWarp; ++value_lane) {
      const int value_index = value_row_base + value_lane;
#pragma unroll
      for (int key_quad = 0; key_quad < kKeyQuads; ++key_quad) {
        const int key_index = lane + key_quad * 32;
        const int state_inner_offset =
            (value_head * kDimV + value_index) * kDimK + key_index;
        const int64_t state_offset = load_state_row_offset + state_inner_offset;
        state[value_lane][key_quad] =
            load_state_bank[state_offset] + 0.0f;
      }
    }

    for (int step = 0; step < steps; ++step) {
      const float decay_scale =
          recurrence_scalars[step][local_value_head][0];
      const float beta = recurrence_scalars[step][local_value_head][1];
      float step_k[kKeyQuads];
#pragma unroll
      for (int key_quad = 0; key_quad < kKeyQuads; ++key_quad) {
        step_k[key_quad] = normalized_ks[step][lane + key_quad * 32];
      }
      const int64_t value_base =
          static_cast<int64_t>(layer) * ring_v_layer_stride +
          static_cast<int64_t>(request) * ring_v_batch_stride +
          static_cast<int64_t>(shared_nodes[step]) * ring_v_node_stride +
          static_cast<int64_t>(value_head) * kDimV;
#pragma unroll
      for (int value_lane = 0; value_lane < kValuesPerWarp; ++value_lane) {
        const int value_index = value_row_base + value_lane;
#pragma unroll
        for (int key_quad = 0; key_quad < kKeyQuads; ++key_quad) {
          state[value_lane][key_quad] =
              __fmul_rn(state[value_lane][key_quad], decay_scale);
        }
        float partial02 = triton_butterfly_product_sum(
            state[value_lane][0], step_k[0]);
        partial02 = __fadd_rn(
            partial02,
            triton_butterfly_product_sum(state[value_lane][2], step_k[2]));
        float partial13 = triton_butterfly_product_sum(
            state[value_lane][1], step_k[1]);
        partial13 = __fadd_rn(
            partial13,
            triton_butterfly_product_sum(state[value_lane][3], step_k[3]));
        float state_k = __fadd_rn(partial02, partial13);
        state_k = __shfl_sync(kFullWarpMask, state_k, 0);
        float value = lane == 0
                          ? load_bf16(v_rings + value_base + value_index)
                          : 0.0f;
        value = __shfl_sync(kFullWarpMask, value, 0);
        const float residual =
            __fmul_rn(__fsub_rn(value, state_k), beta);
#pragma unroll
        for (int key_quad = 0; key_quad < kKeyQuads; ++key_quad) {
          state[value_lane][key_quad] = __fmaf_rn(
              residual, step_k[key_quad], state[value_lane][key_quad]);
        }
      }
    }

    float* store_state_bank = bank_anchor + bank_off16[layer] * 4;
    int store_thread_id;
    asm volatile("mov.u32 %0, %%tid.x;" : "=r"(store_thread_id));
    const int store_warp = store_thread_id >> 5;
    const int store_lane = store_thread_id & 31;
    const int store_value_row_base = store_warp * kValuesPerWarp;
    const int store_value_head = key_head * kHeadGroup + local_value_head;
    const int64_t store_state_row_offset =
        static_cast<int64_t>(shared_state_index) * bank_stride;
#pragma unroll
    for (int value_lane = 0; value_lane < kValuesPerWarp; ++value_lane) {
      const int value_index = store_value_row_base + value_lane;
#pragma unroll
      for (int key_quad = 0; key_quad < kKeyQuads; ++key_quad) {
        const int key_index = store_lane + key_quad * 32;
        const int state_inner_offset =
            (store_value_head * kDimV + value_index) * kDimK + key_index;
        const int64_t state_offset = store_state_row_offset + state_inner_offset;
        store_state_bank[state_offset] = state[value_lane][key_quad];
      }
    }
  }
}

void check_cuda_contiguous(const torch::Tensor& tensor, const char* name,
                           int device) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.get_device() == device, name, " device mismatch");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

void fr13_fixed32_cfwd_native_fullvalue(
    torch::Tensor& bank_anchor, const torch::Tensor& bank_off16,
    const torch::Tensor& accepted_paths, const torch::Tensor& accepted_lens,
    const torch::Tensor& spec_state_indices, const torch::Tensor& k_rings,
    const torch::Tensor& v_rings, const torch::Tensor& a_rings,
    const torch::Tensor& b_rings, const torch::Tensor& gate_coeffs,
    int64_t batch_size, bool bank_offset_table_prevalidated,
    bool accepted_values_device_guarded) {
  TORCH_CHECK(batch_size >= 1 && batch_size <= 4,
              "FR13 native full-value CFWD requires B1-B4");
  TORCH_CHECK(bank_offset_table_prevalidated,
              "FR13 native full-value CFWD requires a prevalidated bank table");
  TORCH_CHECK(accepted_values_device_guarded,
              "FR13 native full-value CFWD requires guarded accepted values");
  const int device = bank_anchor.get_device();
  TORCH_CHECK(bank_anchor.is_cuda(), "bank_anchor must be CUDA");
  TORCH_CHECK(bank_anchor.scalar_type() == torch::kFloat32,
              "FR13 native full-value CFWD requires FP32 state banks");
  TORCH_CHECK(bank_anchor.dim() == 4 &&
                  bank_anchor.size(1) == kValueHeads &&
                  bank_anchor.size(2) == kDimV &&
                  bank_anchor.size(3) == kDimK &&
                  bank_anchor.stride(3) == 1 &&
                  bank_anchor.stride(2) == kDimK &&
                  bank_anchor.stride(1) == kDimV * kDimK,
              "FR13 native full-value CFWD bank geometry/stride drift");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(bank_anchor.data_ptr()) % 16 == 0,
              "FR13 native full-value CFWD bank anchor must be 16B aligned");

  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>(&bank_off16, "bank_off16"),
           {&accepted_paths, "accepted_paths"},
           {&accepted_lens, "accepted_lens"},
           {&spec_state_indices, "spec_state_indices"},
           {&k_rings, "k_rings"},
           {&v_rings, "v_rings"},
           {&a_rings, "a_rings"},
           {&b_rings, "b_rings"},
           {&gate_coeffs, "gate_coeffs"},
       }) {
    check_cuda_contiguous(*item.first, item.second, device);
  }
  TORCH_CHECK(bank_off16.scalar_type() == torch::kInt64 &&
                  bank_off16.numel() == kLayers,
              "FR13 native full-value CFWD bank offset table drift");
  TORCH_CHECK(accepted_paths.scalar_type() == torch::kInt32 &&
                  accepted_lens.scalar_type() == torch::kInt32 &&
                  spec_state_indices.scalar_type() == torch::kInt32,
              "FR13 native full-value CFWD path/state indices must be int32");
  TORCH_CHECK(accepted_paths.dim() == 2 &&
                  accepted_paths.size(0) == batch_size &&
                  accepted_paths.size(1) == kPathCap &&
                  accepted_lens.dim() == 1 &&
                  accepted_lens.size(0) == batch_size,
              "FR13 native full-value CFWD accepted-input geometry drift");
  TORCH_CHECK(spec_state_indices.dim() == 3 &&
                  spec_state_indices.size(0) == kLayers &&
                  spec_state_indices.size(1) >= batch_size &&
                  spec_state_indices.size(2) >= 1,
              "FR13 native full-value CFWD state-index geometry drift");

  TORCH_CHECK(k_rings.scalar_type() == torch::kBFloat16 &&
                  v_rings.scalar_type() == torch::kBFloat16 &&
                  a_rings.scalar_type() == torch::kBFloat16 &&
                  b_rings.scalar_type() == torch::kBFloat16,
              "FR13 native full-value CFWD rings must be BF16");
  TORCH_CHECK(k_rings.dim() == 5 &&
                  k_rings.size(0) == kLayers &&
                  k_rings.size(1) >= batch_size &&
                  k_rings.size(2) == kRingNodes &&
                  k_rings.size(3) == kKeyHeads &&
                  k_rings.size(4) == kDimK,
              "FR13 native full-value CFWD K-ring geometry drift");
  TORCH_CHECK(v_rings.dim() == 5 &&
                  v_rings.size(0) == kLayers &&
                  v_rings.size(1) == k_rings.size(1) &&
                  v_rings.size(2) == kRingNodes &&
                  v_rings.size(3) == kValueHeads &&
                  v_rings.size(4) == kDimV,
              "FR13 native full-value CFWD V-ring geometry drift");
  TORCH_CHECK(a_rings.dim() == 4 && b_rings.sizes() == a_rings.sizes() &&
                  a_rings.size(0) == kLayers &&
                  a_rings.size(1) == k_rings.size(1) &&
                  a_rings.size(2) == kRingNodes &&
                  a_rings.size(3) == kValueHeads,
              "FR13 native full-value CFWD gate-ring geometry drift");
  TORCH_CHECK(a_rings.numel() <= INT32_MAX,
              "FR13 native full-value CFWD gate-ring offset exceeds int32");
  TORCH_CHECK(gate_coeffs.scalar_type() == torch::kFloat32 &&
                  gate_coeffs.dim() == 3 &&
                  gate_coeffs.size(0) == kLayers &&
                  gate_coeffs.size(1) == kValueHeads &&
                  gate_coeffs.size(2) == 2,
              "FR13 native full-value CFWD gate coefficient drift");

  const auto* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->major == 12 && properties->minor == 1,
              "FR13 native full-value CFWD is source-qualified only for SM121");
  TORCH_CHECK(properties->warpSize == 32 &&
                  properties->maxThreadsPerBlock >= kThreadsPerBlock &&
                  properties->sharedMemPerBlock >= kSharedBytes,
              "FR13 native full-value CFWD launch resource unsupported");

  const dim3 grid(static_cast<unsigned int>(
      kLayers * batch_size * kKeyHeads));
  const dim3 block(kThreadsPerBlock);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(device);
  fixed32_cfwd_native_fullvalue_kernel<<<grid, block, 0, stream>>>(
      bank_anchor.mutable_data_ptr<float>(), bank_off16.data_ptr<int64_t>(),
      accepted_paths.data_ptr<int32_t>(), accepted_lens.data_ptr<int32_t>(),
      spec_state_indices.data_ptr<int32_t>(),
      reinterpret_cast<const __nv_bfloat16*>(
          k_rings.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(
          v_rings.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(
          a_rings.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(
          b_rings.data_ptr<at::BFloat16>()),
      gate_coeffs.data_ptr<float>(), bank_anchor.stride(0),
      gate_coeffs.stride(0), k_rings.stride(0), k_rings.stride(1),
      k_rings.stride(2), v_rings.stride(0), v_rings.stride(1),
      v_rings.stride(2), a_rings.stride(0), a_rings.stride(1),
      a_rings.stride(2), spec_state_indices.stride(0),
      spec_state_indices.stride(1), static_cast<int>(batch_size));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
