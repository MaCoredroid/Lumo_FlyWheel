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
#include <cstdint>
#include <cstdlib>
#include <cstring>
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

inline bool fr13_fixed32_b1_fp8_quant_regcache_enabled() {
  const char* value =
      std::getenv("FR13_FIXED32_B1_FP8_QUANT_REGCACHE");
  return value != nullptr && std::strcmp(value, "1") == 0;
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
  const bool fr13_regcache_shape =
      fr13_fixed32_b1_fp8_quant_regcache_enabled() &&
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
  if (fr13_regcache_shape) {
    fr13_fixed32_b1_fp8_quant_regcache_r32k5120_kernel<
        c10::BFloat16, __nv_fp8_e4m3><<<80, 256, 0, stream>>>(
        static_cast<c10::BFloat16*>(input.data_ptr()), output_q.data_ptr(),
        static_cast<float*>(output_s.data_ptr()), static_cast<float>(eps),
        static_cast<float>(min_8bit), static_cast<float>(max_8bit));
    return;
  }

#define LAUNCH_KERNEL(T, DST_DTYPE)                                        \
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
