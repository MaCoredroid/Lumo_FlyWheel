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
constexpr int kHeadGroup = kValueHeads / kKeyHeads;
constexpr int kWarpsPerBlock = 16;
constexpr int kThreadsPerBlock = kWarpsPerBlock * 32;
constexpr int kValuesPerWarp = kDimV / kWarpsPerBlock;
constexpr int kKeyQuads = kDimK / 32;
constexpr int kStateElementsPerThread = kValuesPerWarp * kKeyQuads;
constexpr int kNormPartialWarps = kDimK / 32;
constexpr int kSharedBytes =
    kDimK * sizeof(float) + kNormPartialWarps * sizeof(float) +
    2 * sizeof(float) + 3 * sizeof(int32_t);

static_assert(kThreadsPerBlock == 512);
static_assert(kValuesPerWarp == 8);
static_assert(kKeyQuads == 4);
static_assert(kStateElementsPerThread == 32);
static_assert(kNormPartialWarps == 4);
static_assert(kSharedBytes == 548);

__device__ __forceinline__ float load_bf16(const __nv_bfloat16* pointer) {
  return __bfloat162float(*pointer);
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int delta = 16; delta > 0; delta >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, delta);
  }
  return __shfl_sync(0xffffffffu, value, 0);
}

__device__ __forceinline__ float softplus(float value) {
  return value <= 20.0f ? logf(1.0f + expf(value)) : value;
}

__device__ __forceinline__ float sigmoid(float value) {
  return 1.0f / (1.0f + expf(-value));
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
  __shared__ float normalized_k[kDimK];
  __shared__ float norm_partials[kNormPartialWarps];
  __shared__ float recurrence_scalars[2];
  __shared__ int32_t shared_node;
  __shared__ int32_t shared_steps;
  __shared__ int32_t shared_state_index;

  const int thread_id = threadIdx.x;
  const int warp = thread_id >> 5;
  const int lane = thread_id & 31;
  const int value_head = blockIdx.x % kValueHeads;
  const int layer_request = blockIdx.x / kValueHeads;
  const int request = layer_request % batch_size;
  const int layer = layer_request / batch_size;
  const int key_head = value_head / kHeadGroup;

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

  float state[kValuesPerWarp][kKeyQuads];
  // Keep the bank base out of the recurrence live set. Recompute it for the
  // final store instead of carrying a 64-bit address across every step.
  float* load_state_bank = bank_anchor + bank_off16[layer] * 4;
  const int64_t load_state_row_offset =
      static_cast<int64_t>(shared_state_index) * bank_stride;
#pragma unroll
  for (int value_lane = 0; value_lane < kValuesPerWarp; ++value_lane) {
    const int value_index = warp * kValuesPerWarp + value_lane;
#pragma unroll
    for (int key_quad = 0; key_quad < kKeyQuads; ++key_quad) {
      const int key_index = lane + key_quad * 32;
      const int state_inner_offset =
          (value_head * kDimV + value_index) * kDimK + key_index;
      const int64_t state_offset = load_state_row_offset + state_inner_offset;
      // Match the incumbent Triton b_h=zeros; b_h+=load initialization,
      // including its -0.0 to +0.0 normalization.
      state[value_lane][key_quad] =
          load_state_bank[state_offset] + 0.0f;
    }
  }

  for (int step = 0; step < shared_steps; ++step) {
    if (thread_id == 0) {
      int node = 0;
      if (step > 0) {
        node = accepted_paths[
            static_cast<int64_t>(request) * kPathCap + step - 1];
        node = min(max(node, 0), kRingNodes - 1);
      }
      shared_node = node;
      const int64_t gate_offset =
          static_cast<int64_t>(layer) * gate_layer_stride + value_head * 2;
      const int64_t ab_offset =
          static_cast<int64_t>(layer) * ring_ab_layer_stride +
          static_cast<int64_t>(request) * ring_ab_batch_stride +
          static_cast<int64_t>(node) * ring_ab_node_stride + value_head;
      const float x = load_bf16(a_rings + ab_offset) + gate_coeffs[gate_offset + 1];
      const float decay = gate_coeffs[gate_offset] * softplus(x);
      recurrence_scalars[0] = expf(decay);
      recurrence_scalars[1] = sigmoid(load_bf16(b_rings + ab_offset));
    }
    // Publish the node/scalars after every warp finished the preceding step.
    __syncthreads();

    float norm_term = 0.0f;
    if (thread_id < kDimK) {
      const int64_t key_offset =
          static_cast<int64_t>(layer) * ring_k_layer_stride +
          static_cast<int64_t>(request) * ring_k_batch_stride +
          static_cast<int64_t>(shared_node) * ring_k_node_stride +
          static_cast<int64_t>(key_head) * kDimK + thread_id;
      const float key_value = load_bf16(k_rings + key_offset);
      normalized_k[thread_id] = key_value;
      norm_term = key_value * key_value;
    }
    const float warp_partial = warp_sum(norm_term);
    if (lane == 0 && warp < kNormPartialWarps) {
      norm_partials[warp] = warp_partial;
    }
    __syncthreads();

    if (warp == 0) {
      float norm = lane < kNormPartialWarps ? norm_partials[lane] : 0.0f;
      norm = warp_sum(norm);
      if (lane == 0) {
        norm_partials[0] = rsqrtf(norm + 1.0e-6f);
      }
    }
    __syncthreads();
    if (thread_id < kDimK) {
      normalized_k[thread_id] *= norm_partials[0];
    }
    // All 16 warps consume the same normalized K vector below.
    __syncthreads();

    const float decay_scale = recurrence_scalars[0];
    const float beta = recurrence_scalars[1];
#pragma unroll
    for (int value_lane = 0; value_lane < kValuesPerWarp; ++value_lane) {
      const int value_index = warp * kValuesPerWarp + value_lane;
#pragma unroll
      for (int key_quad = 0; key_quad < kKeyQuads; ++key_quad) {
        state[value_lane][key_quad] *= decay_scale;
      }
      float state_k = 0.0f;
#pragma unroll
      for (int key_quad = 0; key_quad < kKeyQuads; ++key_quad) {
        const int key_index = lane + key_quad * 32;
        state_k += state[value_lane][key_quad] * normalized_k[key_index];
      }
      state_k = warp_sum(state_k);
      const int64_t value_offset =
          static_cast<int64_t>(layer) * ring_v_layer_stride +
          static_cast<int64_t>(request) * ring_v_batch_stride +
          static_cast<int64_t>(shared_node) * ring_v_node_stride +
          static_cast<int64_t>(value_head) * kDimV + value_index;
      float value = lane == 0 ? load_bf16(v_rings + value_offset) : 0.0f;
      value = __shfl_sync(0xffffffffu, value, 0);
      const float residual = (value - state_k) * beta;
#pragma unroll
      for (int key_quad = 0; key_quad < kKeyQuads; ++key_quad) {
        const int key_index = lane + key_quad * 32;
        state[value_lane][key_quad] += residual * normalized_k[key_index];
      }
    }
    // Prevent the cooperative loader from overwriting K while a peer warp
    // still consumes the current recurrence step.
    __syncthreads();
  }

  float* store_state_bank = bank_anchor + bank_off16[layer] * 4;
  const int store_thread_id = threadIdx.x;
  const int store_warp = store_thread_id >> 5;
  const int store_lane = store_thread_id & 31;
  const int64_t store_state_row_offset =
      static_cast<int64_t>(shared_state_index) * bank_stride;
#pragma unroll
  for (int value_lane = 0; value_lane < kValuesPerWarp; ++value_lane) {
    const int value_index = store_warp * kValuesPerWarp + value_lane;
#pragma unroll
    for (int key_quad = 0; key_quad < kKeyQuads; ++key_quad) {
      const int key_index = store_lane + key_quad * 32;
      const int state_inner_offset =
          (value_head * kDimV + value_index) * kDimK + key_index;
      const int64_t state_offset = store_state_row_offset + state_inner_offset;
      store_state_bank[state_offset] = state[value_lane][key_quad];
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
      kLayers * batch_size * kValueHeads));
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
