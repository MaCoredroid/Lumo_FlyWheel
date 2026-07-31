#!/usr/bin/env python3
"""Add default-off fixed32 CUTLASS Stream-K candidates to pinned vLLM.

The candidate selector is restricted to the real Qwen3.6 projection histogram.
It does not alter quantization granularity, tile K, accumulation type, or the
BF16/FP16 epilogue. Unset and unknown selector values retain stock dispatch.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path


TARGET_RELATIVE_PATH = Path(
    "csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/"
    "scaled_mm_blockwise_sm120_fp8_dispatch.cuh"
)
EXPECTED_UNPATCHED_SHA256 = (
    "6e1df3f4701f58f233b3831b848c7bbf7936e6cb34b3bc28ded208fd66c48a7f"
)
CMAKE_RELATIVE_PATH = Path("CMakeLists.txt")
EXPECTED_CMAKE_SHA256 = (
    "b12cd47f5761442551d6e1966e8a37ad94175382c1b014d2b65f67b74fbb6e3b"
)
EXPECTED_CUTLASS_REVISION_LINE = 'set(CUTLASS_REVISION "v4.4.2")'
CUTLASS_TAG_COMMIT = "da5e086dab31d63815acafdac9a9c5893b1c69e2"
CUTLASS_REQUIRED_SHA256 = {
    Path("include/cutlass/gemm/collective/builders/sm120_blockwise_mma_builder.inl"):
        "40409c39fbbc5f023e8030472efab2a7b94baf41109eaa59fb009e52ce0d6509",
    Path("include/cutlass/gemm/kernel/tile_scheduler.hpp"):
        "acc90548b9e2b19f944764ced57e1459d5c2ed7e118d6a1af476add26c3d5e73",
    Path("include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp"):
        "54ebcaee08d4fc0663169f97c7fa665cec90d29ce5d01336c2c714f2a911b010",
    Path("include/cutlass/gemm/kernel/sm100_tile_scheduler_stream_k.hpp"):
        "f9baf471896f03c530344a489942758f48c63acd6f31f5d6a41b3e0da0f8eee4",
    Path("include/cutlass/gemm/kernel/tile_scheduler_params.h"):
        "ef48a12e8920183e88259d0b685279c2232fc2fb12c4fb4db7e8d0fbfdc019e9",
    Path("include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp"):
        "41f0cc048be7a70c6dbd72afb5509b71991b54bd4e01dfa40702a1e2dffc3781",
}
MARKER = "// FR13_FIXED32_CUTLASS_WAVE:"

INCLUDE_ANCHOR = "#pragma once\n\n#include <torch/headeronly/util/shim_utils.h>\n"
INCLUDE_REPLACEMENT = """#pragma once

#include <cstdlib>
#include <cstring>

#include <torch/headeronly/util/shim_utils.h>
"""

TEMPLATE_ANCHOR = """          class EpilogueScheduler, class MainloopScheduler,
          bool swap_ab_ = false>
struct cutlass_3x_gemm_fp8_blockwise {
"""
TEMPLATE_REPLACEMENT = """          class EpilogueScheduler, class MainloopScheduler,
          bool swap_ab_ = false, class TileScheduler = void>
struct cutlass_3x_gemm_fp8_blockwise {
  static constexpr bool use_stream_k = !cute::is_void_v<TileScheduler>;
"""

KERNEL_ANCHOR = """  using KernelType = enable_sm120_family<cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>>;
"""
KERNEL_REPLACEMENT = """  using KernelType = enable_sm120_family<cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue,
      TileScheduler>>;
"""

CONFIG_ANCHOR = """template <typename Gemm>
void cutlass_gemm_caller_blockwise(torch::stable::Tensor& out, torch::stable::Tensor const& a,
"""
CONFIG_REPLACEMENT = r"""// FR13_FIXED32_CUTLASS_WAVE: source-only candidates; stock is default.
template <typename OutType>
struct sm120_blockwise_fp8_config_cooperative_streamk {
  using KernelSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedBlockwiseCooperativeSm120;
  using EpilogueSchedule =
      cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
  using Gemm = cutlass_3x_gemm_fp8_blockwise<
      OutType, 1, 128, 128, TileShape, ClusterShape,
      EpilogueSchedule, KernelSchedule, false,
      cutlass::gemm::StreamKScheduler>;
};

template <typename OutType>
struct sm120_blockwise_fp8_config_cooperative64_streamk {
  using KernelSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedBlockwiseCooperativeSm120;
  using EpilogueSchedule =
      cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_64, _128, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
  using Gemm = cutlass_3x_gemm_fp8_blockwise<
      OutType, 1, 128, 128, TileShape, ClusterShape,
      EpilogueSchedule, KernelSchedule, false,
      cutlass::gemm::StreamKScheduler>;
};

template <typename OutType>
struct sm120_blockwise_fp8_config_swapab_streamk {
  using KernelSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedBlockwiseCooperativeSm120;
  using EpilogueSchedule =
      cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_128, _32, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
  using Gemm = cutlass_3x_gemm_fp8_blockwise<
      OutType, 128, 1, 128, TileShape, ClusterShape,
      EpilogueSchedule, KernelSchedule, true,
      cutlass::gemm::StreamKScheduler>;
};

enum class fixed32_cutlass_wave_variant {
  stock,
  stream_k_cooperative_64,
  stream_k_cooperative_128,
};

static inline fixed32_cutlass_wave_variant fixed32_cutlass_wave_selection() {
  static const fixed32_cutlass_wave_variant selection = [] {
    const char* value = std::getenv("FR13_FIXED32_CUTLASS_WAVE");
    if (value != nullptr && std::strcmp(value, "streamk_coop64") == 0) {
      return fixed32_cutlass_wave_variant::stream_k_cooperative_64;
    }
    if (value != nullptr && std::strcmp(value, "streamk_coop128") == 0) {
      return fixed32_cutlass_wave_variant::stream_k_cooperative_128;
    }
    return fixed32_cutlass_wave_variant::stock;
  }();
  return selection;
}

static inline bool fixed32_cutlass_real_projection(int m, int n, int k) {
  const bool fixed32_rows = m == 32 || m == 64 || m == 96 || m == 128;
  const bool real_projection =
      (n == 34816 && k == 5120) ||
      (n == 5120 && k == 17408) ||
      (n == 5120 && k == 6144) ||
      (n == 16384 && k == 5120) ||
      (n == 8192 && k == 5120);
  return fixed32_rows && real_projection;
}

template <typename Gemm>
void cutlass_gemm_caller_blockwise(torch::stable::Tensor& out, torch::stable::Tensor const& a,
"""

CALLER_ANCHOR = """  c3x::cutlass_gemm_caller<GemmKernel>(a.device(), prob_shape, mainloop_args,
                                       epilogue_args);
"""
CALLER_REPLACEMENT = """  if constexpr (!Gemm::use_stream_k) {
    return c3x::cutlass_gemm_caller<GemmKernel>(
        a.device(), prob_shape, mainloop_args, epilogue_args);
  }

  // Stream-K needs a real SM count to choose the tail decomposition. Cache the
  // device query outside the measured steady state for each candidate kernel.
  static const int sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count();
  TORCH_CHECK(sm_count > 0, "FR13 Stream-K could not query the CUDA SM count");
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = a.device().index();
  hw_info.sm_count = sm_count;

  typename GemmKernel::TileSchedulerArguments scheduler{};
  scheduler.splits = 1;
  scheduler.reduction_mode =
      decltype(scheduler.reduction_mode)::Deterministic;
  scheduler.decomposition_mode =
      decltype(scheduler.decomposition_mode)::Heuristic;
  typename GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm, prob_shape, mainloop_args,
      epilogue_args, hw_info, scheduler};

  using GemmOp = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  GemmOp gemm_op;
  CUTLASS_CHECK(gemm_op.can_implement(args));
  size_t workspace_size = gemm_op.get_workspace_size(args);
  auto workspace = torch::stable::empty(
      workspace_size, torch::headeronly::ScalarType::Byte, std::nullopt,
      a.device());
  auto stream = get_current_cuda_stream(a.device().index());
  CUTLASS_CHECK(gemm_op.run(args, workspace.data_ptr(), stream));
"""

DISPATCH_ANCHOR = """  int M = a.size(0);
  // more heuristic tuning can be done here by checking N/K dimensions as well
  bool swap_ab = (M <= 64) || (M % 4 != 0);

  if (!swap_ab) {
    if (M <= 256) {
      using Gemm = typename sm120_blockwise_fp8_config_pingpong<OutType>::Gemm;
      return cutlass_gemm_caller_blockwise<Gemm>(
          out, a, b, a_scales, b_scales);
    }
    // M > 256: use default 128x128x128 config with Cooperative (Auto) schedule
    using Gemm = typename sm120_blockwise_fp8_config_default<OutType>::Gemm;
    return cutlass_gemm_caller_blockwise<Gemm>(
        out, a, b, a_scales, b_scales);
  } else {
    // Swap A/B for small M to improve performance
    // Use TILE_N=32 as the minimum compatible tile size.
    using Gemm = typename sm120_blockwise_fp8_config_swapab<OutType>::Gemm;
    return cutlass_gemm_caller_blockwise<Gemm>(
        out, a, b, a_scales, b_scales);
  }
"""
DISPATCH_REPLACEMENT = """  int M = a.size(0), N = b.size(1), K = a.size(1);
  fixed32_cutlass_wave_variant wave_variant =
      fixed32_cutlass_real_projection(M, N, K)
          ? fixed32_cutlass_wave_selection()
          : fixed32_cutlass_wave_variant::stock;

  if (wave_variant != fixed32_cutlass_wave_variant::stock) {
    if (M <= 64) {
      using Gemm =
          typename sm120_blockwise_fp8_config_swapab_streamk<OutType>::Gemm;
      return cutlass_gemm_caller_blockwise<Gemm>(
          out, a, b, a_scales, b_scales);
    }
    if (wave_variant ==
        fixed32_cutlass_wave_variant::stream_k_cooperative_128) {
      using Gemm = typename
          sm120_blockwise_fp8_config_cooperative_streamk<OutType>::Gemm;
      return cutlass_gemm_caller_blockwise<Gemm>(
          out, a, b, a_scales, b_scales);
    }
    using Gemm =
        typename sm120_blockwise_fp8_config_cooperative64_streamk<OutType>::Gemm;
    return cutlass_gemm_caller_blockwise<Gemm>(
        out, a, b, a_scales, b_scales);
  }

  // Stock dispatch remains byte-for-byte equivalent when the selector is off.
  bool swap_ab = (M <= 64) || (M % 4 != 0);
  if (!swap_ab) {
    if (M <= 256) {
      using Gemm = typename sm120_blockwise_fp8_config_pingpong<OutType>::Gemm;
      return cutlass_gemm_caller_blockwise<Gemm>(
          out, a, b, a_scales, b_scales);
    }
    using Gemm = typename sm120_blockwise_fp8_config_default<OutType>::Gemm;
    return cutlass_gemm_caller_blockwise<Gemm>(
        out, a, b, a_scales, b_scales);
  }
  using Gemm = typename sm120_blockwise_fp8_config_swapab<OutType>::Gemm;
  return cutlass_gemm_caller_blockwise<Gemm>(
      out, a, b, a_scales, b_scales);
"""


def patch_text(source: str) -> tuple[str, bool]:
    """Return patched dispatch source and whether it changed."""
    if MARKER in source:
        required = (
            INCLUDE_REPLACEMENT,
            TEMPLATE_REPLACEMENT,
            KERNEL_REPLACEMENT,
            CONFIG_REPLACEMENT,
            CALLER_REPLACEMENT,
            DISPATCH_REPLACEMENT,
        )
        if not all(fragment in source for fragment in required):
            raise RuntimeError("partial FR13 fixed32 CUTLASS wave patch found")
        return source, False

    single_anchors = {
        "include": INCLUDE_ANCHOR,
        "GEMM template": TEMPLATE_ANCHOR,
        "kernel scheduler": KERNEL_ANCHOR,
        "candidate insertion": CONFIG_ANCHOR,
        "Stream-K caller": CALLER_ANCHOR,
        "dispatch": DISPATCH_ANCHOR,
    }
    for label, anchor in single_anchors.items():
        count = source.count(anchor)
        if count != 1:
            raise RuntimeError(f"expected exactly one {label} anchor, found {count}")
    patched = source.replace(INCLUDE_ANCHOR, INCLUDE_REPLACEMENT, 1)
    patched = patched.replace(TEMPLATE_ANCHOR, TEMPLATE_REPLACEMENT, 1)
    patched = patched.replace(KERNEL_ANCHOR, KERNEL_REPLACEMENT, 1)
    patched = patched.replace(CONFIG_ANCHOR, CONFIG_REPLACEMENT, 1)
    patched = patched.replace(CALLER_ANCHOR, CALLER_REPLACEMENT, 1)
    patched = patched.replace(DISPATCH_ANCHOR, DISPATCH_REPLACEMENT, 1)
    return patched, True


def patch_source_root(source_root: Path) -> bool:
    target = source_root / TARGET_RELATIVE_PATH
    source = target.read_text(encoding="utf-8")
    if MARKER not in source:
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        if digest != EXPECTED_UNPATCHED_SHA256:
            raise RuntimeError(
                "pinned blockwise dispatch SHA256 mismatch: "
                f"{digest} != {EXPECTED_UNPATCHED_SHA256}"
            )

    cmake_source = (source_root / CMAKE_RELATIVE_PATH).read_text(encoding="utf-8")
    cmake_digest = hashlib.sha256(cmake_source.encode("utf-8")).hexdigest()
    if cmake_digest != EXPECTED_CMAKE_SHA256:
        raise RuntimeError(
            "pinned vLLM CMakeLists SHA256 mismatch: "
            f"{cmake_digest} != {EXPECTED_CMAKE_SHA256}"
        )
    if EXPECTED_CUTLASS_REVISION_LINE not in cmake_source:
        raise RuntimeError("pinned CUTLASS v4.4.2 revision line is missing")

    patched, changed = patch_text(source)
    if changed:
        target.write_text(patched, encoding="utf-8")
    return changed


def validate_cutlass_root(cutlass_root: Path) -> None:
    for relative_path, expected_digest in CUTLASS_REQUIRED_SHA256.items():
        path = cutlass_root / relative_path
        if not path.is_file():
            raise RuntimeError(f"pinned CUTLASS file is missing: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_digest:
            raise RuntimeError(
                f"pinned CUTLASS SHA256 mismatch for {relative_path}: "
                f"{digest} != {expected_digest}"
            )
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(cutlass_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("could not resolve pinned CUTLASS checkout") from error
    if commit != CUTLASS_TAG_COMMIT:
        raise RuntimeError(
            f"pinned CUTLASS commit mismatch: {commit} != {CUTLASS_TAG_COMMIT}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_root",
        type=Path,
        help="root of vLLM fe9c3d6c5f66c873d196800384ed6880687b9e52",
    )
    parser.add_argument(
        "--cutlass-root",
        type=Path,
        required=True,
        help="CUTLASS v4.4.2 checkout at da5e086dab31d63815acafdac9a9c5893b1c69e2",
    )
    args = parser.parse_args()

    validate_cutlass_root(args.cutlass_root)
    changed = patch_source_root(args.source_root)
    state = "patched" if changed else "already patched"
    print(f"[FR13] {state}: {args.source_root / TARGET_RELATIVE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
