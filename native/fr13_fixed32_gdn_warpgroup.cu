#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include <cstdint>

namespace {

constexpr int kNodes = 32;
constexpr int kKeyHeads = 16;
constexpr int kValueHeads = 48;
constexpr int kDimK = 128;
constexpr int kDimV = 128;
constexpr int kBlockV = 8;
constexpr int kValueTiles = kDimV / kBlockV;
constexpr int kHeadGroup = kValueHeads / kKeyHeads;
constexpr int kGroups = 5;
constexpr int kWarpsPerGroup = 4;
constexpr int kWarpsPerBlock = kGroups * kWarpsPerGroup;
constexpr int kThreadsPerBlock = kWarpsPerBlock * 32;
constexpr int kStateElements = kBlockV * kDimK;
constexpr int kParentSharedBytes = kGroups * kStateElements * sizeof(float);

static_assert(kThreadsPerBlock == 640);
static_assert(kParentSharedBytes == 20480);

// The root path is 0 -> 1 -> 4 -> 9 -> 14. Its states are placed in the
// shared-memory slot consumed by each same-parent group below.
__device__ __constant__ int8_t kRootNodes[5] = {0, 1, 4, 9, 14};
__device__ __constant__ int8_t kRootSharedSlots[5] = {1, 2, 3, 4, 0};

// group 0: parent 14, paths 0/9/10
// group 1: parent 0, paths 1/2
// group 2: parent 1, paths 3/4
// group 3: parent 4, paths 5/6
// group 4: parent 9, paths 7/8
__device__ __constant__ int8_t kBranchLengths[kGroups][kWarpsPerGroup] = {
    {7, 1, 1, 0},
    {5, 7, 0, 0},
    {1, 1, 0, 0},
    {1, 1, 0, 0},
    {1, 1, 0, 0},
};

__device__ __constant__ int8_t
    kBranchNodes[kGroups][kWarpsPerGroup][7] = {
        {
            {19, 24, 26, 28, 29, 30, 31},
            {20, -1, -1, -1, -1, -1, -1},
            {21, -1, -1, -1, -1, -1, -1},
            {-1, -1, -1, -1, -1, -1, -1},
        },
        {
            {2, 7, 12, 17, 22, -1, -1},
            {3, 8, 13, 18, 23, 25, 27},
            {-1, -1, -1, -1, -1, -1, -1},
            {-1, -1, -1, -1, -1, -1, -1},
        },
        {
            {5, -1, -1, -1, -1, -1, -1},
            {6, -1, -1, -1, -1, -1, -1},
            {-1, -1, -1, -1, -1, -1, -1},
            {-1, -1, -1, -1, -1, -1, -1},
        },
        {
            {10, -1, -1, -1, -1, -1, -1},
            {11, -1, -1, -1, -1, -1, -1},
            {-1, -1, -1, -1, -1, -1, -1},
            {-1, -1, -1, -1, -1, -1, -1},
        },
        {
            {15, -1, -1, -1, -1, -1, -1},
            {16, -1, -1, -1, -1, -1, -1},
            {-1, -1, -1, -1, -1, -1, -1},
            {-1, -1, -1, -1, -1, -1, -1},
        },
};

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int delta = 16; delta > 0; delta >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, delta);
  }
  return __shfl_sync(0xffffffffu, value, 0);
}

__device__ __forceinline__ float load_bf16(const __nv_bfloat16* ptr) {
  return __bfloat162float(*ptr);
}

__device__ __forceinline__ float bf16_round(float value) {
  return __bfloat162float(__float2bfloat16_rn(value));
}

__device__ __forceinline__ float sigmoid(float value) {
  return 1.0f / (1.0f + __expf(-value));
}

__device__ __forceinline__ float softplus(float value) {
  return value <= 20.0f ? __logf(1.0f + __expf(value)) : value;
}

__device__ __forceinline__ void load_initial_state(
    float (&state)[kBlockV][4], const float* h0, int64_t bank_row,
    int64_t bank_stride, int value_head, int value_tile, int lane) {
#pragma unroll
  for (int value_lane = 0; value_lane < kBlockV; ++value_lane) {
    const int value_index = value_tile * kBlockV + value_lane;
#pragma unroll
    for (int key_quad = 0; key_quad < 4; ++key_quad) {
      const int key_index = lane + key_quad * 32;
      const int64_t offset =
          bank_row * bank_stride +
          (static_cast<int64_t>(value_head) * kDimV + value_index) * kDimK +
          key_index;
      state[value_lane][key_quad] = h0[offset];
    }
  }
}

__device__ __forceinline__ void store_parent_state(
    float* parent_states, int slot, const float (&state)[kBlockV][4],
    int lane) {
#pragma unroll
  for (int value_lane = 0; value_lane < kBlockV; ++value_lane) {
#pragma unroll
    for (int key_quad = 0; key_quad < 4; ++key_quad) {
      const int key_index = lane + key_quad * 32;
      parent_states[slot * kStateElements + value_lane * kDimK + key_index] =
          state[value_lane][key_quad];
    }
  }
}

__device__ __forceinline__ void load_parent_state(
    float (&state)[kBlockV][4], const float* parent_states, int slot,
    int lane) {
#pragma unroll
  for (int value_lane = 0; value_lane < kBlockV; ++value_lane) {
#pragma unroll
    for (int key_quad = 0; key_quad < 4; ++key_quad) {
      const int key_index = lane + key_quad * 32;
      state[value_lane][key_quad] =
          parent_states[slot * kStateElements + value_lane * kDimK + key_index];
    }
  }
}

__device__ __forceinline__ void node_step(
    float (&state)[kBlockV][4], int batch, int node, int key_head,
    int value_head, int value_tile, int lane, const __nv_bfloat16* q,
    const __nv_bfloat16* k, const __nv_bfloat16* v,
    const __nv_bfloat16* raw_a, const __nv_bfloat16* raw_b,
    const float* a_log, const float* dt_bias, __nv_bfloat16* out,
    __nv_bfloat16* ring_k, __nv_bfloat16* ring_v,
    __nv_bfloat16* ring_a, __nv_bfloat16* ring_b, float output_scale,
    bool use_qk_l2norm, bool scan_align, bool ring_export) {
  const int global_node = batch * kNodes + node;
  const int64_t qk_base =
      (static_cast<int64_t>(global_node) * kKeyHeads + key_head) * kDimK;
  float q_values[4];
  float k_values[4];
  float q_norm = 0.0f;
  float k_norm = 0.0f;
#pragma unroll
  for (int key_quad = 0; key_quad < 4; ++key_quad) {
    const int key_index = lane + key_quad * 32;
    const __nv_bfloat16 q_raw = q[qk_base + key_index];
    const __nv_bfloat16 k_raw = k[qk_base + key_index];
    q_values[key_quad] = __bfloat162float(q_raw);
    k_values[key_quad] = __bfloat162float(k_raw);
    q_norm = fmaf(q_values[key_quad], q_values[key_quad], q_norm);
    k_norm = fmaf(k_values[key_quad], k_values[key_quad], k_norm);
    if (ring_export && value_tile == 0 && value_head % kHeadGroup == 0) {
      ring_k[qk_base + key_index] = k_raw;
    }
  }
  if (use_qk_l2norm) {
    q_norm = sqrtf(warp_sum(q_norm) + 1.0e-6f);
    k_norm = sqrtf(warp_sum(k_norm) + 1.0e-6f);
#pragma unroll
    for (int key_quad = 0; key_quad < 4; ++key_quad) {
      q_values[key_quad] /= q_norm;
      k_values[key_quad] /= k_norm;
    }
  }
#pragma unroll
  for (int key_quad = 0; key_quad < 4; ++key_quad) {
    q_values[key_quad] *= output_scale;
  }

  const int64_t gate_index =
      static_cast<int64_t>(global_node) * kValueHeads + value_head;
  float raw_a_value = lane == 0 ? load_bf16(raw_a + gate_index) : 0.0f;
  float raw_b_value = lane == 0 ? load_bf16(raw_b + gate_index) : 0.0f;
  float a_log_value = lane == 0 ? a_log[value_head] : 0.0f;
  float dt_bias_value = lane == 0 ? dt_bias[value_head] : 0.0f;
  raw_a_value = __shfl_sync(0xffffffffu, raw_a_value, 0);
  raw_b_value = __shfl_sync(0xffffffffu, raw_b_value, 0);
  a_log_value = __shfl_sync(0xffffffffu, a_log_value, 0);
  dt_bias_value = __shfl_sync(0xffffffffu, dt_bias_value, 0);

  const float decay =
      -__expf(a_log_value) * softplus(raw_a_value + dt_bias_value);
  const float decay_scale = __expf(decay);
  float beta = sigmoid(raw_b_value);
  if (scan_align) {
    beta = bf16_round(beta);
  }

#pragma unroll
  for (int value_lane = 0; value_lane < kBlockV; ++value_lane) {
#pragma unroll
    for (int key_quad = 0; key_quad < 4; ++key_quad) {
      state[value_lane][key_quad] *= decay_scale;
    }
    float state_k = 0.0f;
#pragma unroll
    for (int key_quad = 0; key_quad < 4; ++key_quad) {
      state_k = fmaf(
          state[value_lane][key_quad], k_values[key_quad], state_k);
    }
    state_k = warp_sum(state_k);
    const int value_index = value_tile * kBlockV + value_lane;
    const int64_t value_offset =
        (static_cast<int64_t>(global_node) * kValueHeads + value_head) * kDimV +
        value_index;
    float value_input = lane == 0 ? load_bf16(v + value_offset) : 0.0f;
    value_input = __shfl_sync(0xffffffffu, value_input, 0);
    if (ring_export && lane == 0) {
      ring_v[value_offset] = v[value_offset];
    }
    const float residual = (value_input - state_k) * beta;
#pragma unroll
    for (int key_quad = 0; key_quad < 4; ++key_quad) {
      state[value_lane][key_quad] = fmaf(
          residual, k_values[key_quad], state[value_lane][key_quad]);
    }
    float output_value = 0.0f;
#pragma unroll
    for (int key_quad = 0; key_quad < 4; ++key_quad) {
      output_value = fmaf(
          state[value_lane][key_quad], q_values[key_quad], output_value);
    }
    output_value = warp_sum(output_value);
    if (lane == 0) {
      out[value_offset] = __float2bfloat16_rn(output_value);
    }
  }
  if (ring_export && value_tile == 0 && lane == 0) {
    ring_a[gate_index] = raw_a[gate_index];
    ring_b[gate_index] = raw_b[gate_index];
  }
  if (scan_align) {
#pragma unroll
    for (int value_lane = 0; value_lane < kBlockV; ++value_lane) {
#pragma unroll
      for (int key_quad = 0; key_quad < 4; ++key_quad) {
        state[value_lane][key_quad] = bf16_round(state[value_lane][key_quad]);
      }
    }
  }
}

__global__ __launch_bounds__(kThreadsPerBlock, 1)
void fixed32_gdn_warpgroup_kernel(
    __nv_bfloat16* out, const __nv_bfloat16* q, const __nv_bfloat16* k,
    const __nv_bfloat16* v, const __nv_bfloat16* raw_a,
    const __nv_bfloat16* raw_b, const float* a_log, const float* dt_bias,
    const float* h0, const int32_t* h0_indices, int64_t h0_bank_stride,
    __nv_bfloat16* ring_k, __nv_bfloat16* ring_v,
    __nv_bfloat16* ring_a, __nv_bfloat16* ring_b, int32_t* flags,
    int32_t* invocation_counter, int batch_size, int h0_index_row,
    int h0_index_batch_stride, float output_scale, bool use_qk_l2norm,
    bool scan_align, bool ring_export, bool flags_export,
    bool count_invocation) {
  __shared__ float parent_states[kGroups * kStateElements];

  const int value_tile = blockIdx.x % kValueTiles;
  const int head_batch = blockIdx.x / kValueTiles;
  const int value_head = head_batch % kValueHeads;
  const int batch = head_batch / kValueHeads;
  const int key_head = value_head / kHeadGroup;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;

  if (count_invocation && value_head == 0 && value_tile == 0 &&
      threadIdx.x == 0) {
    atomicAdd(invocation_counter, 1);
  }

  float state[kBlockV][4];
  if (warp == 0) {
    const int h0_flat_index =
        h0_index_row + batch * h0_index_batch_stride;
    const int64_t h0_bank_row = h0_indices[h0_flat_index];
    load_initial_state(state, h0, h0_bank_row, h0_bank_stride, value_head,
                       value_tile, lane);
#pragma unroll
    for (int step = 0; step < 5; ++step) {
      const int node = kRootNodes[step];
      node_step(state, batch, node, key_head, value_head, value_tile, lane, q,
                k, v, raw_a, raw_b, a_log, dt_bias, out, ring_k, ring_v,
                ring_a, ring_b, output_scale, use_qk_l2norm, scan_align,
                ring_export);
      store_parent_state(parent_states, kRootSharedSlots[step], state, lane);
    }
  }
  __syncthreads();

  const int group = warp / kWarpsPerGroup;
  const int member = warp % kWarpsPerGroup;
  const int path_length = kBranchLengths[group][member];
  if (path_length > 0) {
    load_parent_state(state, parent_states, group, lane);
#pragma unroll
    for (int step = 0; step < 7; ++step) {
      if (step < path_length) {
        const int node = kBranchNodes[group][member][step];
        node_step(state, batch, node, key_head, value_head, value_tile, lane,
                  q, k, v, raw_a, raw_b, a_log, dt_bias, out, ring_k, ring_v,
                  ring_a, ring_b, output_scale, use_qk_l2norm, scan_align,
                  ring_export);
      }
    }
  }
  __syncthreads();

  if (flags_export && batch == 0 && value_head == 0 && value_tile == 0 &&
      threadIdx.x == 0) {
    flags[0] = 1;
    flags[1] = batch_size;
  }
}

void check_cuda_contiguous(const torch::Tensor& tensor, const char* name,
                           int device) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.get_device() == device, name, " device mismatch");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

void fr13_fixed32_gdn_warpgroup(
    torch::Tensor& out, const torch::Tensor& q, const torch::Tensor& k,
    const torch::Tensor& v, const torch::Tensor& raw_a,
    const torch::Tensor& raw_b, const torch::Tensor& a_log,
    const torch::Tensor& dt_bias, const torch::Tensor& h0,
    const torch::Tensor& h0_indices, torch::Tensor& ring_k,
    torch::Tensor& ring_v, torch::Tensor& ring_a, torch::Tensor& ring_b,
    torch::Tensor& flags, torch::Tensor& invocation_counter,
    int64_t batch_size, int64_t h0_index_row,
    int64_t h0_index_batch_stride, int64_t h0_bank_stride,
    double output_scale, bool use_qk_l2norm, bool scan_align,
    bool ring_export, bool flags_export, bool count_invocation) {
  TORCH_CHECK(batch_size >= 1 && batch_size <= 4,
              "FR13 warp-group GDN requires B1-B4");
  const int device = q.get_device();
  check_cuda_contiguous(q, "q", device);
  check_cuda_contiguous(k, "k", device);
  check_cuda_contiguous(v, "v", device);
  check_cuda_contiguous(raw_a, "raw_a", device);
  check_cuda_contiguous(raw_b, "raw_b", device);
  check_cuda_contiguous(a_log, "A_log", device);
  check_cuda_contiguous(dt_bias, "dt_bias", device);
  check_cuda_contiguous(h0, "h0", device);
  check_cuda_contiguous(h0_indices, "h0_indices", device);
  check_cuda_contiguous(out, "out", device);

  const int64_t rows = batch_size * kNodes;
  TORCH_CHECK(q.scalar_type() == torch::kBFloat16 &&
                  k.scalar_type() == torch::kBFloat16 &&
                  v.scalar_type() == torch::kBFloat16 &&
                  raw_a.scalar_type() == torch::kBFloat16 &&
                  raw_b.scalar_type() == torch::kBFloat16 &&
                  out.scalar_type() == torch::kBFloat16,
              "FR13 warp-group GDN requires BF16 activations/output");
  TORCH_CHECK(a_log.scalar_type() == torch::kFloat32 &&
                  dt_bias.scalar_type() == torch::kFloat32 &&
                  h0.scalar_type() == torch::kFloat32,
              "FR13 warp-group GDN requires FP32 state/gate parameters");
  TORCH_CHECK(h0_indices.scalar_type() == torch::kInt32,
              "FR13 warp-group GDN requires int32 h0 indices");
  TORCH_CHECK(q.sizes() == torch::IntArrayRef({rows, kKeyHeads, kDimK}) &&
                  k.sizes() == q.sizes(),
              "FR13 warp-group GDN q/k geometry drift");
  TORCH_CHECK(v.sizes() == torch::IntArrayRef({rows, kValueHeads, kDimV}) &&
                  out.sizes() == v.sizes(),
              "FR13 warp-group GDN v/out geometry drift");
  TORCH_CHECK(raw_a.sizes() == torch::IntArrayRef({rows, kValueHeads}) &&
                  raw_b.sizes() == raw_a.sizes(),
              "FR13 warp-group GDN raw gate geometry drift");
  TORCH_CHECK(a_log.numel() == kValueHeads &&
                  dt_bias.numel() == kValueHeads,
              "FR13 warp-group GDN A_log/dt_bias geometry drift");
  TORCH_CHECK(h0.dim() == 4 && h0.size(1) == kValueHeads &&
                  h0.size(2) == kDimV && h0.size(3) == kDimK,
              "FR13 warp-group GDN h0 bank geometry drift");
  TORCH_CHECK(h0_bank_stride == h0.stride(0),
              "FR13 warp-group GDN h0 bank stride drift");
  TORCH_CHECK(h0_indices.dim() == 2 && h0_indices.size(0) >= batch_size &&
                  h0_indices.size(1) >= 1,
              "FR13 warp-group GDN h0 index geometry drift");
  TORCH_CHECK(h0_index_row >= 0 && h0_index_batch_stride > 0 &&
                  h0_index_row + (batch_size - 1) * h0_index_batch_stride <
                      h0_indices.numel(),
              "FR13 warp-group GDN h0 index addressing drift");
  TORCH_CHECK(use_qk_l2norm && scan_align,
              "FR13 warp-group GDN exact candidate requires qk norm and "
              "SCAN_ALIGN");

  if (ring_export) {
    check_cuda_contiguous(ring_k, "ring_k", device);
    check_cuda_contiguous(ring_v, "ring_v", device);
    check_cuda_contiguous(ring_a, "ring_a", device);
    check_cuda_contiguous(ring_b, "ring_b", device);
    TORCH_CHECK(ring_k.scalar_type() == torch::kBFloat16 &&
                    ring_v.scalar_type() == torch::kBFloat16 &&
                    ring_a.scalar_type() == torch::kBFloat16 &&
                    ring_b.scalar_type() == torch::kBFloat16,
                "FR13 warp-group GDN ring dtype drift");
    TORCH_CHECK(
        ring_k.sizes() == torch::IntArrayRef({rows, kKeyHeads, kDimK}) &&
            ring_v.sizes() == torch::IntArrayRef({rows, kValueHeads, kDimV}) &&
            ring_a.sizes() == torch::IntArrayRef({rows, kValueHeads}) &&
            ring_b.sizes() == ring_a.sizes(),
        "FR13 warp-group GDN ring geometry drift");
  }
  if (flags_export) {
    check_cuda_contiguous(flags, "flags", device);
    TORCH_CHECK(flags.scalar_type() == torch::kInt32 && flags.numel() >= 2,
                "FR13 warp-group GDN flags drift");
  }
  if (count_invocation) {
    check_cuda_contiguous(invocation_counter, "invocation_counter", device);
    TORCH_CHECK(invocation_counter.scalar_type() == torch::kInt32 &&
                    invocation_counter.numel() >= 1,
                "FR13 warp-group GDN invocation counter drift");
  }

  const auto* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->major == 12 && properties->minor == 1,
              "FR13 warp-group GDN is source-qualified only for SM121");
  TORCH_CHECK(properties->warpSize == 32 &&
                  properties->maxThreadsPerBlock >= kThreadsPerBlock &&
                  properties->sharedMemPerBlock >= kParentSharedBytes,
              "FR13 warp-group GDN launch resource contract unsupported");

  const dim3 grid(static_cast<unsigned int>(batch_size * kValueHeads *
                                            kValueTiles));
  const dim3 block(kThreadsPerBlock);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(device);
  fixed32_gdn_warpgroup_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(out.mutable_data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(raw_a.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(raw_b.data_ptr<at::BFloat16>()),
      a_log.data_ptr<float>(), dt_bias.data_ptr<float>(), h0.data_ptr<float>(),
      h0_indices.data_ptr<int32_t>(), h0_bank_stride,
      reinterpret_cast<__nv_bfloat16*>(ring_k.mutable_data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(ring_v.mutable_data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(ring_a.mutable_data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(ring_b.mutable_data_ptr()),
      reinterpret_cast<int32_t*>(flags.mutable_data_ptr()),
      reinterpret_cast<int32_t*>(invocation_counter.mutable_data_ptr()),
      static_cast<int>(batch_size), static_cast<int>(h0_index_row),
      static_cast<int>(h0_index_batch_stride), static_cast<float>(output_scale),
      use_qk_l2norm, scan_align, ring_export, flags_export, count_invocation);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
