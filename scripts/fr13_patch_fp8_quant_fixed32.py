#!/usr/bin/env python3
"""Add a default-off fixed32 B1 FP8 activation-quantization kernel.

The candidate is restricted to the deployed row32/K5120/group128 BF16 input
and column-major FP32 scale layout.  It preserves the stock half-warp load,
maximum-reduction, divide, clamp, and FP8-conversion order while retaining the
eight BF16 values owned by each lane in registers.  Stock stores those values
to shared memory, synchronizes all 256 threads, and loads them back before
conversion.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


TARGET_RELATIVE_PATH = Path(
    "csrc/libtorch_stable/quantization/w8a8/fp8/per_token_group_quant.cu"
)
EXPECTED_UNPATCHED_SHA256 = (
    "f3516d6813c6d231b745558c488e98e80f0f61f07e4cc3509b8e2472dac9694e"
)
MARKER = "// FR13_FIXED32_B1_FP8_QUANT_REGCACHE:"

INCLUDE_ANCHOR = "#include <cmath>\n"
INCLUDE_REPLACEMENT = """#include <cmath>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
"""

KERNEL_ANCHOR = """template <typename T, typename DST_DTYPE, bool IS_COLUMN_MAJOR = false,
          bool SCALE_UE8M0 = false, typename scale_packed_t = float>
__global__ void per_token_group_quant_8bit_kernel(
"""
KERNEL_REPLACEMENT = r"""// FR13_FIXED32_B1_FP8_QUANT_REGCACHE: candidate kernel.
// One half warp owns one 128-element group, exactly as in the stock kernel.
// Each lane owns one aligned eight-BF16 pack.  Keeping that pack live across
// GroupReduceMax removes the stock shared store, CTA barrier, and shared load
// without changing the per-value arithmetic or the half-warp reduction tree.
template <typename T, typename DST_DTYPE>
__global__ __launch_bounds__(256)
void fr13_fixed32_b1_fp8_quant_regcache_r32k5120_kernel(
    const T* __restrict__ input, void* __restrict__ output_q,
    float* __restrict__ output_s, const float eps, const float min_8bit,
    const float max_8bit) {
  constexpr int kGroupSize = 128;
  constexpr int kThreadsPerGroup = 16;
  constexpr int kGroupsPerBlock = 16;
  constexpr int kValuesPerLane = 8;
  constexpr int kScaleColumns = 40;
  constexpr int kScaleStride = 32;

  const int local_group_id = threadIdx.x / kThreadsPerGroup;
  const int lane_id = threadIdx.x % kThreadsPerGroup;
  const int global_group_id =
      static_cast<int>(blockIdx.x) * kGroupsPerBlock + local_group_id;
  const int64_t group_offset =
      static_cast<int64_t>(global_group_id) * kGroupSize;

  using InputPack = vllm::vec_n_t<T, kValuesPerLane>;
  using OutputPack = vllm::vec_n_t<DST_DTYPE, kValuesPerLane>;
  const InputPack values =
      reinterpret_cast<const InputPack*>(input + group_offset)[lane_id];

  float local_absmax = eps;
#pragma unroll
  for (int index = 0; index < kValuesPerLane; ++index) {
    const float abs_value = fabsf(static_cast<float>(values.val[index]));
    local_absmax = fmaxf(local_absmax, abs_value);
  }
  const float y_s = GroupReduceMax(local_absmax) / max_8bit;

  OutputPack quantized;
#pragma unroll
  for (int index = 0; index < kValuesPerLane; ++index) {
    const float value = static_cast<float>(values.val[index]);
    const float q = fminf(fmaxf(value / y_s, min_8bit), max_8bit);
    quantized.val[index] = DST_DTYPE(q);
  }
  reinterpret_cast<OutputPack*>(
      static_cast<DST_DTYPE*>(output_q) + group_offset)[lane_id] = quantized;

  if (lane_id == 0) {
    const int scale_row = global_group_id / kScaleColumns;
    const int scale_col = global_group_id % kScaleColumns;
    output_s[scale_col * kScaleStride + scale_row] = y_s;
  }
}

enum class fr13_fixed32_b1_fp8_quant_regcache_mode {
  stock,
  byte_ab,
  production,
};

inline fr13_fixed32_b1_fp8_quant_regcache_mode
fr13_fixed32_b1_fp8_quant_regcache_selection() {
  const char* value =
      std::getenv("FR13_FIXED32_B1_FP8_QUANT_REGCACHE");
  if (value != nullptr && std::strcmp(value, "byte_ab") == 0) {
    return fr13_fixed32_b1_fp8_quant_regcache_mode::byte_ab;
  }
  if (value != nullptr && std::strcmp(value, "1") == 0) {
    return fr13_fixed32_b1_fp8_quant_regcache_mode::production;
  }
  return fr13_fixed32_b1_fp8_quant_regcache_mode::stock;
}

inline std::string fr13_fixed32_b1_fp8_quant_regcache_task_marker() {
  constexpr const char* path =
      "/logs/fr13_fixed32_cutlass_streamk.real_event.arm";
  std::ifstream input(path);
  if (!input.good()) {
    return "";
  }
  std::string marker;
  std::string trailing;
  std::getline(input, marker);
  STD_TORCH_CHECK(!marker.empty() && !std::getline(input, trailing),
                  "FR13 FP8 quant real-task marker is malformed");
  constexpr const char* prefix = "swe_verified:";
  constexpr const char* allowed =
      "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-/";
  STD_TORCH_CHECK(
      marker.size() > std::strlen(prefix) && marker.size() <= 256 &&
          marker.rfind(prefix, 0) == 0 &&
          marker.substr(std::strlen(prefix)).find_first_not_of(allowed) ==
              std::string::npos,
      "FR13 FP8 quant real-task marker is noncanonical");
  return marker;
}

template <typename T, typename DST_DTYPE, bool IS_COLUMN_MAJOR = false,
          bool SCALE_UE8M0 = false, typename scale_packed_t = float>
__global__ void per_token_group_quant_8bit_kernel(
"""

DISPATCH_ANCHOR = """  const bool is_column_major = output_s.stride(0) < output_s.stride(1);
  const int scale_num_rows = output_s.size(1);
  const int scale_stride = output_s.stride(1);

#define LAUNCH_KERNEL(T, DST_DTYPE)                                        \\
"""
DISPATCH_REPLACEMENT = r"""  const bool is_column_major = output_s.stride(0) < output_s.stride(1);
  const int scale_num_rows = output_s.size(1);
  const int scale_stride = output_s.stride(1);

  // FR13_FIXED32_B1_FP8_QUANT_REGCACHE: exact deployed-shape admission.
  // Any dtype, layout, alignment, scale mode, or shape drift stays stock.
  const auto fr13_regcache_mode =
      fr13_fixed32_b1_fp8_quant_regcache_selection();
  const bool fr13_regcache_shape =
      input.scalar_type() == torch::headeronly::ScalarType::BFloat16 &&
      dst_type == torch::headeronly::ScalarType::Float8_e4m3fn &&
      output_s.scalar_type() == torch::headeronly::ScalarType::Float &&
      input.dim() == 2 && input.size(0) == 32 && input.size(1) == 5120 &&
      output_q.dim() == 2 && output_q.size(0) == 32 &&
      output_q.size(1) == 5120 && group_size == 128 &&
      num_groups == 1280 && groups_per_block == 16 && num_blocks == 80 &&
      num_threads == 256 && !scale_ue8m0 && is_column_major &&
      output_s.size(0) == 32 && output_s.size(1) == 40 &&
      output_s.stride(0) == 1 && scale_stride == 32 &&
      (reinterpret_cast<uintptr_t>(input.data_ptr()) & 15u) == 0u &&
      (reinterpret_cast<uintptr_t>(output_q.data_ptr()) & 7u) == 0u;
  auto fr13_run_regcache = [&](torch::stable::Tensor& destination_q,
                               torch::stable::Tensor& destination_s) {
    fr13_fixed32_b1_fp8_quant_regcache_r32k5120_kernel<
        c10::BFloat16, __nv_fp8_e4m3><<<80, 256, 0, stream>>>(
        static_cast<c10::BFloat16*>(input.data_ptr()),
        destination_q.data_ptr(),
        static_cast<float*>(destination_s.data_ptr()), static_cast<float>(eps),
        static_cast<float>(min_8bit), static_cast<float>(max_8bit));
  };
  if (fr13_regcache_shape &&
      fr13_regcache_mode ==
          fr13_fixed32_b1_fp8_quant_regcache_mode::production) {
    fr13_run_regcache(output_q, output_s);
    return;
  }

#define LAUNCH_KERNEL(T, DST_DTYPE)                                        \
"""

POST_DISPATCH_ANCHOR = """#undef LAUNCH_KERNEL
}
"""
POST_DISPATCH_REPLACEMENT = r"""  if (fr13_regcache_shape &&
      fr13_regcache_mode ==
          fr13_fixed32_b1_fp8_quant_regcache_mode::byte_ab) {
    const std::string task_marker =
        fr13_fixed32_b1_fp8_quant_regcache_task_marker();
    if (!task_marker.empty()) {
      // Diagnostic only.  Candidate output is shadow-only and every served byte
      // remains the stock result produced above.
      torch::stable::Tensor candidate_q = torch::stable::empty_like(output_q);
      torch::stable::Tensor candidate_s = torch::stable::empty_like(output_s);
      STD_TORCH_CHECK(candidate_s.stride(0) == 1 &&
                          candidate_s.stride(1) == 32,
                      "FR13 FP8 quant shadow scale layout drifted");
      fr13_run_regcache(candidate_q, candidate_s);

      const size_t output_bytes =
          static_cast<size_t>(output_q.numel()) * output_q.element_size();
      const size_t scale_bytes =
          static_cast<size_t>(output_s.numel()) * output_s.element_size();
      std::vector<unsigned char> stock_output(output_bytes);
      std::vector<unsigned char> candidate_output(output_bytes);
      std::vector<unsigned char> stock_scale(scale_bytes);
      std::vector<unsigned char> candidate_scale(scale_bytes);
      cudaError_t status = cudaMemcpyAsync(
          stock_output.data(), output_q.const_data_ptr(), output_bytes,
          cudaMemcpyDeviceToHost, stream);
      STD_TORCH_CHECK(status == cudaSuccess,
                      "FR13 FP8 quant stock-output D2H failed: ",
                      cudaGetErrorString(status));
      status = cudaMemcpyAsync(candidate_output.data(),
                               candidate_q.const_data_ptr(), output_bytes,
                               cudaMemcpyDeviceToHost, stream);
      STD_TORCH_CHECK(status == cudaSuccess,
                      "FR13 FP8 quant candidate-output D2H failed: ",
                      cudaGetErrorString(status));
      status = cudaMemcpyAsync(stock_scale.data(), output_s.const_data_ptr(),
                               scale_bytes, cudaMemcpyDeviceToHost, stream);
      STD_TORCH_CHECK(status == cudaSuccess,
                      "FR13 FP8 quant stock-scale D2H failed: ",
                      cudaGetErrorString(status));
      status = cudaMemcpyAsync(candidate_scale.data(),
                               candidate_s.const_data_ptr(), scale_bytes,
                               cudaMemcpyDeviceToHost, stream);
      STD_TORCH_CHECK(status == cudaSuccess,
                      "FR13 FP8 quant candidate-scale D2H failed: ",
                      cudaGetErrorString(status));
      status = cudaStreamSynchronize(stream);
      STD_TORCH_CHECK(status == cudaSuccess,
                      "FR13 FP8 quant byte A/B synchronize failed: ",
                      cudaGetErrorString(status));

      auto compare = [](const std::vector<unsigned char>& stock,
                        const std::vector<unsigned char>& candidate) {
        size_t mismatches = 0;
        size_t first = stock.size();
        for (size_t index = 0; index < stock.size(); ++index) {
          if (stock[index] != candidate[index]) {
            if (first == stock.size()) {
              first = index;
            }
            ++mismatches;
          }
        }
        return std::pair<size_t, size_t>{mismatches, first};
      };
      const auto output_comparison = compare(stock_output, candidate_output);
      const auto scale_comparison = compare(stock_scale, candidate_scale);

      static std::atomic<int64_t> next_invocation{0};
      const int64_t invocation = next_invocation.fetch_add(1);
      static std::mutex log_mutex;
      std::lock_guard<std::mutex> lock(log_mutex);
      std::ofstream log(
          "/logs/fr13_fixed32_b1_fp8_quant_regcache.byte_ab.jsonl",
          std::ios::app);
      STD_TORCH_CHECK(log.good(),
                      "FR13 FP8 quant byte A/B could not open JSONL");
      log << "{\"schema\":\"fr13.fixed32.b1_fp8_quant_regcache.byte_ab.v1\",";
      log << "\"invocation\":" << invocation << ",";
      log << "\"target_forward_ordinal\":" << invocation / 128 << ",";
      log << "\"call_in_target_forward\":" << invocation % 128 << ",";
      log << "\"task_marker\":\"" << task_marker << "\",";
      log << "\"rows\":32,\"k\":5120,\"group_size\":128,";
      log << "\"groups\":1280,\"groups_per_cta\":16,\"ctas\":80,";
      log << "\"threads_per_cta\":256,";
      log << "\"output_bytes\":" << output_bytes << ",";
      log << "\"output_byte_equal\":"
          << (output_comparison.first == 0 ? "true" : "false") << ",";
      log << "\"output_mismatch_count\":" << output_comparison.first << ",";
      log << "\"output_first_mismatch\":";
      if (output_comparison.second == output_bytes) {
        log << "null";
      } else {
        log << output_comparison.second;
      }
      log << ",\"scale_bytes\":" << scale_bytes << ",";
      log << "\"scale_byte_equal\":"
          << (scale_comparison.first == 0 ? "true" : "false") << ",";
      log << "\"scale_mismatch_count\":" << scale_comparison.first << ",";
      log << "\"scale_first_mismatch\":";
      if (scale_comparison.second == scale_bytes) {
        log << "null";
      } else {
        log << scale_comparison.second;
      }
      log << ",\"scale_layout\":\"column_major_fp32_32x40_stride_1_32\",";
      log << "\"stock_served\":true,\"comparison_sampled\":false}\n";
      log.flush();
      STD_TORCH_CHECK(log.good(),
                      "FR13 FP8 quant byte A/B JSONL write failed");
    }
  }

#undef LAUNCH_KERNEL
}
"""


class PatchError(RuntimeError):
    """Raised when pinned upstream source no longer matches the patch."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def patch_text(source: str) -> tuple[str, bool]:
    """Return patched source and whether this invocation changed it."""
    if MARKER in source:
        required = (
            "fr13_fixed32_b1_fp8_quant_regcache_r32k5120_kernel",
            'std::getenv("FR13_FIXED32_B1_FP8_QUANT_REGCACHE")',
            "fr13.fixed32.b1_fp8_quant_regcache.byte_ab.v1",
            "comparison_sampled\\\":false",
            "input.size(0) == 32 && input.size(1) == 5120",
            "group_size == 128",
            "output_s.stride(0) == 1 && scale_stride == 32",
        )
        missing = tuple(token for token in required if token not in source)
        if missing:
            raise PatchError(
                "existing FR13 FP8 quant marker is incomplete: " + ",".join(missing)
            )
        return source, False

    replacements = (
        (INCLUDE_ANCHOR, INCLUDE_REPLACEMENT, "include anchor"),
        (KERNEL_ANCHOR, KERNEL_REPLACEMENT, "kernel anchor"),
        (DISPATCH_ANCHOR, DISPATCH_REPLACEMENT, "dispatch anchor"),
        (POST_DISPATCH_ANCHOR, POST_DISPATCH_REPLACEMENT, "post-dispatch anchor"),
    )
    patched = source
    for anchor, replacement, label in replacements:
        if patched.count(anchor) != 1:
            raise PatchError(
                f"expected exactly one {label}, found {patched.count(anchor)}"
            )
        patched = patched.replace(anchor, replacement, 1)
    return patched, True


def patch_vllm_root(vllm_root: Path) -> tuple[Path, bool, str]:
    target = vllm_root / TARGET_RELATIVE_PATH
    try:
        raw = target.read_bytes()
    except OSError as error:
        raise PatchError(f"cannot read pinned vLLM source {target}: {error}") from error
    try:
        source = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise PatchError(f"pinned vLLM source is not ASCII: {target}") from error

    if MARKER not in source and _sha256(raw) != EXPECTED_UNPATCHED_SHA256:
        raise PatchError(
            "pinned vLLM FP8 quant source SHA-256 drift: "
            f"expected {EXPECTED_UNPATCHED_SHA256}, observed {_sha256(raw)}"
        )
    patched, changed = patch_text(source)
    if changed:
        target.write_text(patched, encoding="ascii")
    return target, changed, _sha256(patched.encode("ascii"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-root", type=Path, required=True)
    args = parser.parse_args()
    target, changed, digest = patch_vllm_root(args.vllm_root.resolve())
    state = "patched" if changed else "already-patched"
    print(f"{state}\t{digest}\t{target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
