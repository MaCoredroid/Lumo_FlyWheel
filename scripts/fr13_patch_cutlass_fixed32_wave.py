#!/usr/bin/env python3
"""Add default-off fixed32 CUTLASS projection candidates to pinned vLLM.

The candidate selector is restricted to the real Qwen3.6 projection histogram.
It does not alter quantization granularity, tile K, accumulation type, or the
BF16/FP16 epilogue. Scheduler-only variants keep each output tile's complete
ordered K reduction. Unset and unknown selector values retain stock dispatch.
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
    Path("include/cutlass/gemm/kernel/sm100_static_tile_scheduler.hpp"):
        "0ed331127afe83d20ec23b1a92e160f7388bb3727c77f05e496cc72371c664fd",
    Path("include/cutlass/gemm/kernel/static_tile_scheduler.hpp"):
        "3cf17407654833666b3be9f7e807f299779ef5f4492bf844d0da180409f2f5f6",
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

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

#include <torch/headeronly/util/shim_utils.h>
"""

SCHEDULER_SPECIALIZATION_ANCHOR = """#include "cutlass_gemm_caller.cuh"

namespace vllm {
"""
SCHEDULER_SPECIALIZATION_REPLACEMENT = r"""#include "cutlass_gemm_caller.cuh"

namespace vllm {
struct fr13_fixed32_m128_static_scheduler {};
struct fr13_fixed32_m128_divisor_static_scheduler {};
struct fr13_fixed32_b1_onen_static_scheduler {};
struct fr13_fixed32_b1_n5120_single_tile_scheduler {};
struct fr13_fixed32_b1_onen_fullgrid_static_scheduler {};
struct fr13_fixed32_b4_twom_static_scheduler {};
struct fr13_fixed32_b4_n5120_single_tile_scheduler {};
struct fr13_fixed32_wide256_recompute_scheduler {};
}  // namespace vllm

namespace cutlass::gemm::kernel::detail {
// The wide256 Stream-K kernel is register-bound enough that nvcc hoists
// threadIdx.x % 128 from fixup and spills it for the full kernel lifetime.
// Keep the scheduler and deterministic reduction unchanged, but materialize
// that exact barrier-group coordinate only at each fixup use.
template <class TileShape, class ClusterShape,
          uint32_t SchedulerPipelineStageCount>
class Fr13Wide256RecomputeTileScheduler
    : public PersistentTileSchedulerSm100StreamK<
          TileShape, ClusterShape, SchedulerPipelineStageCount> {
  using Base = PersistentTileSchedulerSm100StreamK<
      TileShape, ClusterShape, SchedulerPipelineStageCount>;
  using UnderlyingStreamKScheduler =
      PersistentTileSchedulerSm90StreamK<TileShape, ClusterShape>;

 public:
  using Base::Base;
  using Base::fixup;
  using Params = typename Base::Params;
  using WorkTileInfo = typename Base::WorkTileInfo;
  using StreamKParams = typename UnderlyingStreamKScheduler::Params;
  using BarrierType = typename UnderlyingStreamKScheduler::BarrierType;
  using ReductionMode = typename UnderlyingStreamKScheduler::ReductionMode;

  template <class BarrierManager>
  CUTLASS_DEVICE static uint32_t barrier_group_thread_idx() {
    uint32_t thread_idx;
    asm volatile("mov.u32 %0, %%tid.x;" : "=r"(thread_idx));
    return thread_idx % BarrierManager::ThreadCount;
  }

  CUTLASS_HOST_DEVICE static auto tile_peer_range(
      StreamKParams const& params, uint32_t tile_idx,
      WorkTileInfo const& work_tile_info) {
    uint32_t cur_k_tile = static_cast<uint32_t>(work_tile_info.K_idx);
    uint32_t tile_idx_in_cluster_path = params.div_cluster_size(tile_idx);
    uint32_t start_k_tile = params.divmod_tiles_per_output_tile_.divisor *
        tile_idx_in_cluster_path;
    uint32_t end_k_tile =
        start_k_tile + params.divmod_tiles_per_output_tile_.divisor - 1;
    uint32_t big_unit_k_tiles = params.big_units_ *
        (params.divmod_k_tiles_per_sk_unit_.divisor + 1);

    auto adjust_unit = [&](uint32_t k_tile, uint32_t unit_idx,
                           uint32_t unit_k_start, uint32_t unit_k_end) {
      if (k_tile - start_k_tile < StreamKParams::min_iters_per_sk_unit_ &&
          unit_k_end - start_k_tile <
              StreamKParams::min_iters_per_sk_unit_) {
        ++unit_idx;
      }
      if (end_k_tile + 1 - k_tile <
              StreamKParams::min_iters_per_sk_unit_ &&
          end_k_tile + 1 - unit_k_start <
              StreamKParams::min_iters_per_sk_unit_) {
        --unit_idx;
      }
      return unit_idx;
    };

    auto find_unit = [&](uint32_t k_tile) {
      if (k_tile < big_unit_k_tiles) {
        uint32_t unit_idx =
            params.divmod_k_tiles_per_sk_big_unit_.divide(k_tile);
        uint32_t unit_k_start = unit_idx *
            params.divmod_k_tiles_per_sk_big_unit_.divisor;
        uint32_t unit_k_end = unit_k_start +
            params.divmod_k_tiles_per_sk_big_unit_.divisor;
        return static_cast<uint64_t>(
            adjust_unit(k_tile, unit_idx, unit_k_start, unit_k_end));
      }
      uint32_t unit_idx_after_big_units =
          params.divmod_k_tiles_per_sk_unit_.divide(
              k_tile - big_unit_k_tiles);
      uint32_t unit_k_start = unit_idx_after_big_units *
          params.divmod_k_tiles_per_sk_unit_.divisor +
          (params.big_units_ *
           params.divmod_k_tiles_per_sk_big_unit_.divisor);
      uint32_t unit_k_end =
          unit_k_start + params.divmod_k_tiles_per_sk_unit_.divisor;
      uint32_t unit_idx = unit_idx_after_big_units + params.big_units_;
      return static_cast<uint64_t>(
          adjust_unit(k_tile, unit_idx, unit_k_start, unit_k_end));
    };

    return cute::make_tuple(
        find_unit(start_k_tile), find_unit(start_k_tile + cur_k_tile),
        find_unit(end_k_tile));
  }

  template <class FrgTensorC, class BarrierManager>
  CUTLASS_DEVICE static void fixup_helper(
      StreamKParams const& params,
      WorkTileInfo const& work_tile_info, FrgTensorC& accumulators,
      uint32_t num_barriers, uint32_t barrier_idx,
      uint32_t num_accumulator_mtxs = 1,
      uint32_t idx_accumulator_mtxs = 0) {
    using ElementAccumulator = typename FrgTensorC::value_type;

    if (!UnderlyingStreamKScheduler::requires_fixup(
            params, work_tile_info)) {
      return;
    }
    uint64_t tile_idx = UnderlyingStreamKScheduler::output_tile_index(
        params, work_tile_info);
    uint64_t lock_idx = (tile_idx * num_barriers) + barrier_idx;

    uint64_t reduction_tile_idx = tile_idx;
    uint64_t num_peers = 0;
    uint64_t reduction_peer_offset = 0;
    if (params.requires_separate_reduction()) {
      auto [first_peer_id, my_peer_id, last_peer_id] =
          tile_peer_range(params, tile_idx, work_tile_info);
      auto peer_id_in_output_tile = my_peer_id - first_peer_id;
      num_peers = last_peer_id - first_peer_id + 1;
      reduction_tile_idx = tile_idx *
          StreamKParams::max_peers_per_tile(
              params.sk_units_, params.sk_tiles_);
      reduction_peer_offset = peer_id_in_output_tile *
          cute::size<0>(TileShape{}) * cute::size<1>(TileShape{}) *
          num_accumulator_mtxs;
    }

    uint64_t reduction_offset_base =
        (static_cast<uint64_t>(cute::size<0>(TileShape{})) *
         static_cast<uint64_t>(cute::size<1>(TileShape{})) *
         reduction_tile_idx * num_accumulator_mtxs) +
        (static_cast<uint64_t>(size(accumulators)) * barrier_idx *
         BarrierManager::ThreadCount * num_accumulator_mtxs) +
        static_cast<uint64_t>(size(accumulators)) *
            BarrierManager::ThreadCount * idx_accumulator_mtxs;
    uint64_t reduction_offset =
        reduction_offset_base + reduction_peer_offset;

    ElementAccumulator* group_reduction_workspace =
        reinterpret_cast<ElementAccumulator*>(params.reduction_workspace_) +
        reduction_offset;
    using AccumulatorArrayT =
        Array<typename FrgTensorC::value_type, size(FrgTensorC{})>;
    using BlockStripedReduceT =
        BlockStripedReduce<BarrierManager::ThreadCount, AccumulatorArrayT>;
    AccumulatorArrayT* reduction_workspace_array =
        reinterpret_cast<AccumulatorArrayT*>(group_reduction_workspace);
    AccumulatorArrayT* accumulator_array =
        reinterpret_cast<AccumulatorArrayT*>(accumulators.data());

    uint32_t reduction_tiles = 0;
    if (params.divmod_splits_.divisor > 1) {
      reduction_tiles = params.units_per_problem_;
    } else if (params.requires_separate_reduction()) {
      reduction_tiles = params.sk_tiles_ *
          StreamKParams::max_peers_per_tile(
              params.sk_units_, params.sk_tiles_);
    } else {
      reduction_tiles = params.sk_tiles_;
    }

    uint64_t reduction_workspace_size =
        StreamKParams::get_reduction_workspace_size(
            reduction_tiles, to_gemm_coord(TileShape{}),
            sizeof_bits<ElementAccumulator>::value,
            num_accumulator_mtxs);
    BarrierType* lock_workspace = reinterpret_cast<BarrierType*>(
        reinterpret_cast<uint8_t*>(params.reduction_workspace_) +
        reduction_workspace_size);

    if (work_tile_info.is_reduction_unit()) {
      BarrierManager::wait_eq(
          barrier_idx, lock_workspace,
          barrier_group_thread_idx<BarrierManager>(), lock_idx, num_peers);
      UnderlyingStreamKScheduler::template separate_reduction<
          FrgTensorC, BarrierManager>(
          accumulators, num_barriers, group_reduction_workspace,
          barrier_group_thread_idx<BarrierManager>(), num_peers,
          num_accumulator_mtxs);
    } else if (!UnderlyingStreamKScheduler::compute_epilogue(
                   work_tile_info, params)) {
      if (params.requires_separate_reduction() || work_tile_info.K_idx == 0) {
        BlockStripedReduceT::store(
            reduction_workspace_array, *accumulator_array,
            barrier_group_thread_idx<BarrierManager>());
      } else {
        if (params.reduction_mode_ == ReductionMode::Deterministic) {
          BarrierManager::wait_eq(
              barrier_idx, lock_workspace,
              barrier_group_thread_idx<BarrierManager>(), lock_idx,
              work_tile_info.K_idx);
        } else {
          BarrierManager::wait_lt(
              barrier_idx, lock_workspace,
              barrier_group_thread_idx<BarrierManager>(), lock_idx, 1);
        }
        BlockStripedReduceT::reduce(
            reduction_workspace_array, *accumulator_array,
            barrier_group_thread_idx<BarrierManager>());
      }

      uint32_t increment = params.requires_separate_reduction()
          ? 1
          : work_tile_info.k_tile_count;
      if (idx_accumulator_mtxs == (num_accumulator_mtxs - 1)) {
        BarrierManager::arrive_inc(
            barrier_idx, lock_workspace,
            barrier_group_thread_idx<BarrierManager>(), lock_idx, increment);
      }
    } else {
      BarrierManager::wait_eq(
          barrier_idx, lock_workspace,
          barrier_group_thread_idx<BarrierManager>(), lock_idx,
          work_tile_info.K_idx);
      BlockStripedReduceT::load_add(
          *accumulator_array, reduction_workspace_array,
          barrier_group_thread_idx<BarrierManager>());
    }
  }

  template <class FrgTensorC>
  CUTLASS_DEVICE static void fixup(
      Params const& params, WorkTileInfo const& work_tile_info,
      FrgTensorC& accumulators, uint32_t num_barriers,
      uint32_t barrier_idx) {
    static constexpr uint32_t Offset = static_cast<int>(
        cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
    static constexpr uint32_t MaxNumNamedBarriers = 2;
    using BarrierManager = NamedBarrierManager<
        NumThreadsPerWarpGroup, Offset, MaxNumNamedBarriers>;
    return fixup_helper<FrgTensorC, BarrierManager>(
        params.sk_params_, work_tile_info, accumulators,
        num_barriers, barrier_idx);
  }
};

template <class TileShape, class ClusterShape,
          uint32_t SchedulerPipelineStageCount>
struct TileSchedulerSelector<
    vllm::fr13_fixed32_wide256_recompute_scheduler, arch::Sm120,
    TileShape, ClusterShape, SchedulerPipelineStageCount> {
  using Scheduler = Fr13Wide256RecomputeTileScheduler<
      TileShape, ClusterShape, SchedulerPipelineStageCount>;
};

template <class TileShape, class ClusterShape,
          uint32_t SchedulerPipelineStageCount>
struct TileSchedulerSelector<
    vllm::fr13_fixed32_m128_static_scheduler, arch::Sm120,
    TileShape, ClusterShape, SchedulerPipelineStageCount> {
  using Scheduler = StaticPersistentTileScheduler100;
};

// A 48-CTA persistent grid leaves partially occupied final waves on most
// fixed32 projections. Keep every output tile intact, but select the widest
// divisor of the logical tile count down to 28 CTAs so each CTA receives the
// same number of complete tiles.
class Fr13DivisorBalancedStaticTileScheduler100
    : public StaticPersistentTileScheduler100 {
  using Base = StaticPersistentTileScheduler100;

 public:
  using Base::Base;
  using Base::get_grid_shape;
  using Params = typename Base::Params;

  template <class ProblemShapeMNKL, class TileShape, class AtomThrShape,
            class ClusterShape>
  CUTLASS_HOST_DEVICE static dim3 get_grid_shape(
      Params const& params, ProblemShapeMNKL problem_shape_mnkl,
      TileShape tile_shape_mnk, AtomThrShape atom_thr_shape_mnk,
      ClusterShape cluster_shape_mnk, KernelHardwareInfo hw_info) {
    dim3 base_grid = Base::get_grid_shape(
        params, problem_shape_mnkl, tile_shape_mnk, atom_thr_shape_mnk,
        cluster_shape_mnk, hw_info);
    dim3 problem_blocks = Base::get_tiled_cta_shape_mnl(
        problem_shape_mnkl, tile_shape_mnk, atom_thr_shape_mnk,
        cluster_shape_mnk);
    uint64_t logical_tiles = uint64_t(problem_blocks.x) *
        uint64_t(problem_blocks.y) * uint64_t(problem_blocks.z);
    uint32_t base_ctas = base_grid.x * base_grid.y * base_grid.z;
    constexpr uint32_t MinBalancedCtas = 28;
    uint32_t balanced_ctas = base_ctas;
    for (uint32_t candidate = base_ctas;
         candidate >= MinBalancedCtas; --candidate) {
      if (logical_tiles % candidate == 0) {
        balanced_ctas = candidate;
        break;
      }
    }
    if (balanced_ctas == base_ctas) {
      return base_grid;
    }
    // Row-gated candidates use cluster (1,1,1) and L=1. Preserve CUTLASS's
    // one-dimensional launch orientation.
    if (base_grid.y == 1 && base_grid.z == 1) {
      return dim3(balanced_ctas, 1, 1);
    }
    if (base_grid.x == 1 && base_grid.z == 1) {
      return dim3(1, balanced_ctas, 1);
    }
    return base_grid;
  }
};

// Fixed32 B4 with a 64-row tile always has exactly two M tiles, one batch
// plane, cluster (1,1,1), and complete output tiles. Every admitted real
// projection has at least 40 N tiles, so CUTLASS rasterizes AlongM and returns
// an X-only grid. Map that persistent index directly and remove the generic
// grid flattening plus batch/cluster/raster divmods from every tile assignment.
class Fr13B4TwoMStaticTileScheduler100
    : public Fr13DivisorBalancedStaticTileScheduler100 {
  using Base = Fr13DivisorBalancedStaticTileScheduler100;
  uint32_t current_work_linear_idx_ = 0;

  CUTLASS_DEVICE void initialize_linear_work() {
#if defined(__CUDA_ARCH__)
    current_work_linear_idx_ = blockIdx.x;
#endif
  }

  CUTLASS_DEVICE static uint32_t total_grid_size() {
    return gridDim.x;
  }

  CUTLASS_DEVICE uint32_t problem_tiles() const {
    return static_cast<uint32_t>(this->scheduler_params.blocks_per_problem_);
  }

 public:
  using Params = typename Base::Params;
  using WorkTileInfo = typename Base::WorkTileInfo;
  using CLCResponse = typename Base::CLCResponse;

  CUTLASS_DEVICE explicit Fr13B4TwoMStaticTileScheduler100(
      Params const& params) : Base(params) {
    initialize_linear_work();
  }

  CUTLASS_DEVICE explicit Fr13B4TwoMStaticTileScheduler100(
      CLCResponse* response, Params const& params, dim3 block_id_in_cluster)
      : Base(response, params, block_id_in_cluster) {
    initialize_linear_work();
  }

  template <class ClusterShape>
  CUTLASS_DEVICE WorkTileInfo initial_work_tile_info(
      ClusterShape) const {
    return get_current_work();
  }

  CUTLASS_DEVICE WorkTileInfo get_current_work() const {
    return get_current_work_for_linear_idx(current_work_linear_idx_);
  }

  CUTLASS_DEVICE WorkTileInfo get_current_work_for_linear_idx(
      uint32_t linear_idx) const {
    if (linear_idx >= problem_tiles()) {
      return WorkTileInfo::invalid_work_tile();
    }
    return {static_cast<int32_t>(linear_idx & 1),
            static_cast<int32_t>(linear_idx >> 1), 0, true};
  }

  CUTLASS_DEVICE void advance_to_next_work(uint32_t advance_count = 1) {
    current_work_linear_idx_ += total_grid_size() * advance_count;
  }

  CUTLASS_DEVICE bool is_last_tile(
      WorkTileInfo&, uint32_t advance_count = 1) const {
    return current_work_linear_idx_ +
        total_grid_size() * advance_count >= problem_tiles();
  }

  CUTLASS_DEVICE auto fetch_next_work(WorkTileInfo) {
    advance_to_next_work();
    return cute::make_tuple(get_current_work(), true);
  }

  template <class TileSchedulerPipeline, class TileSchedulerPipelineState>
  CUTLASS_DEVICE auto fetch_next_work(
      WorkTileInfo work_tile_info, TileSchedulerPipeline&,
      TileSchedulerPipelineState) {
    return fetch_next_work(work_tile_info);
  }
};

// The two admitted B4 N=5120 projections contain exactly forty scheduler-N
// tiles and launch a (40, 1, 1) grid. Each CTA therefore owns one complete
// output tile. Encode that invariant directly so the cooperative M128 kernel
// does not retain the general persistent scheduler's tile-count state or
// next-work arithmetic.
class Fr13B4N5120SingleTileScheduler100
    : public Fr13DivisorBalancedStaticTileScheduler100 {
  using Base = Fr13DivisorBalancedStaticTileScheduler100;
  static constexpr uint32_t kProblemTiles = 40;

 public:
  using Params = typename Base::Params;
  using WorkTileInfo = typename Base::WorkTileInfo;
  using CLCResponse = typename Base::CLCResponse;

  CUTLASS_DEVICE explicit Fr13B4N5120SingleTileScheduler100(
      Params const&) : Base() {}

  CUTLASS_DEVICE explicit Fr13B4N5120SingleTileScheduler100(
      CLCResponse*, Params const&, dim3) : Base() {}

  template <class ClusterShape>
  CUTLASS_DEVICE WorkTileInfo initial_work_tile_info(ClusterShape) const {
    return get_current_work();
  }

  CUTLASS_DEVICE WorkTileInfo get_current_work() const {
    return {0, static_cast<int32_t>(blockIdx.x), 0, true};
  }

  CUTLASS_DEVICE WorkTileInfo get_current_work_for_linear_idx(
      uint32_t linear_idx) const {
    if (linear_idx >= kProblemTiles) {
      return WorkTileInfo::invalid_work_tile();
    }
    return {0, static_cast<int32_t>(linear_idx), 0, true};
  }

  CUTLASS_DEVICE static auto work_tile_to_cta_coord(
      WorkTileInfo work_tile_info) {
    return cute::make_coord(
        cute::Int<0>{}, work_tile_info.N_idx,
        cute::Underscore{}, cute::Int<0>{});
  }

  CUTLASS_DEVICE static auto work_tile_to_cta_coord(
      WorkTileInfo work_tile_info, dim3) {
    return work_tile_to_cta_coord(work_tile_info);
  }

  CUTLASS_DEVICE bool is_last_tile(WorkTileInfo&, uint32_t = 1) const {
    return true;
  }

  CUTLASS_DEVICE auto fetch_next_work(WorkTileInfo) {
    return cute::make_tuple(WorkTileInfo::invalid_work_tile(), true);
  }

  template <class TileSchedulerPipeline, class TileSchedulerPipelineState>
  CUTLASS_DEVICE auto fetch_next_work(
      WorkTileInfo work_tile_info, TileSchedulerPipeline&,
      TileSchedulerPipelineState) {
    return fetch_next_work(work_tile_info);
  }
};

// Fixed32 B1 swap-AB has exactly one scheduler-N tile, one batch plane,
// cluster (1,1,1), and complete output tiles. Map the persistent linear work
// index directly to scheduler M and remove the generic batch/cluster/raster
// divmods from every tile assignment.
class Fr13B1OneNStaticTileScheduler100
    : public Fr13DivisorBalancedStaticTileScheduler100 {
  using Base = Fr13DivisorBalancedStaticTileScheduler100;
  uint32_t problem_tiles_ = 0;

  CUTLASS_DEVICE void initialize_problem_tiles(typename Base::Params const& params) {
    problem_tiles_ = static_cast<uint32_t>(params.blocks_per_problem_);
  }

  CUTLASS_DEVICE static uint32_t total_grid_size() {
    return gridDim.y;
  }

  CUTLASS_DEVICE uint32_t problem_tiles() const {
    return problem_tiles_;
  }

 public:
  using Params = typename Base::Params;
  using WorkTileInfo = typename Base::WorkTileInfo;
  using CLCResponse = typename Base::CLCResponse;

  CUTLASS_DEVICE explicit Fr13B1OneNStaticTileScheduler100(
      Params const& params) : Base() {
    initialize_problem_tiles(params);
  }

  CUTLASS_DEVICE explicit Fr13B1OneNStaticTileScheduler100(
      CLCResponse*, Params const& params, dim3) : Base() {
    initialize_problem_tiles(params);
  }

  template <class ClusterShape>
  CUTLASS_DEVICE WorkTileInfo initial_work_tile_info(
      ClusterShape) const {
    return get_current_work();
  }

  CUTLASS_DEVICE WorkTileInfo get_current_work() const {
    // CUTLASS's heuristic orients audited one-N grids as (1, CTAs, 1).
    return get_current_work_for_linear_idx(blockIdx.y);
  }

  CUTLASS_DEVICE WorkTileInfo get_current_work_for_linear_idx(
      uint32_t linear_idx) const {
    if (linear_idx >= problem_tiles()) {
      return WorkTileInfo::invalid_work_tile();
    }
    return {static_cast<int32_t>(linear_idx), 0, 0, true};
  }

  CUTLASS_DEVICE static auto work_tile_to_cta_coord(
      WorkTileInfo work_tile_info) {
    return cute::make_coord(
        work_tile_info.M_idx, cute::Int<0>{},
        cute::Underscore{}, cute::Int<0>{});
  }

  CUTLASS_DEVICE static auto work_tile_to_cta_coord(
      WorkTileInfo work_tile_info, dim3) {
    return work_tile_to_cta_coord(work_tile_info);
  }

  CUTLASS_DEVICE bool is_last_tile(
      WorkTileInfo& work_tile_info, uint32_t advance_count = 1) const {
    return static_cast<uint32_t>(work_tile_info.M_idx) +
        total_grid_size() * advance_count >= problem_tiles();
  }

  CUTLASS_DEVICE auto fetch_next_work(WorkTileInfo work_tile_info) {
    uint32_t next_linear_idx = static_cast<uint32_t>(work_tile_info.M_idx) +
        total_grid_size();
    return cute::make_tuple(
        get_current_work_for_linear_idx(next_linear_idx), true);
  }

  template <class TileSchedulerPipeline, class TileSchedulerPipelineState>
  CUTLASS_DEVICE auto fetch_next_work(
      WorkTileInfo work_tile_info, TileSchedulerPipeline&,
      TileSchedulerPipelineState) {
    return fetch_next_work(work_tile_info);
  }
};

// The two admitted B1 N=5120 projections contain exactly forty scheduler-M
// tiles and launch a (1, 40, 1) grid. Each CTA therefore owns one complete
// output tile. Encode that invariant directly so the kernel does not retain
// the general persistent scheduler's tile-count state or next-work arithmetic.
class Fr13B1N5120SingleTileScheduler100
    : public Fr13DivisorBalancedStaticTileScheduler100 {
  using Base = Fr13DivisorBalancedStaticTileScheduler100;
  static constexpr uint32_t kProblemTiles = 40;

 public:
  using Params = typename Base::Params;
  using WorkTileInfo = typename Base::WorkTileInfo;
  using CLCResponse = typename Base::CLCResponse;

  CUTLASS_DEVICE explicit Fr13B1N5120SingleTileScheduler100(
      Params const&) : Base() {}

  CUTLASS_DEVICE explicit Fr13B1N5120SingleTileScheduler100(
      CLCResponse*, Params const&, dim3) : Base() {}

  template <class ClusterShape>
  CUTLASS_DEVICE WorkTileInfo initial_work_tile_info(ClusterShape) const {
    return get_current_work();
  }

  CUTLASS_DEVICE WorkTileInfo get_current_work() const {
    return {static_cast<int32_t>(blockIdx.y), 0, 0, true};
  }

  CUTLASS_DEVICE WorkTileInfo get_current_work_for_linear_idx(
      uint32_t linear_idx) const {
    if (linear_idx >= kProblemTiles) {
      return WorkTileInfo::invalid_work_tile();
    }
    return {static_cast<int32_t>(linear_idx), 0, 0, true};
  }

  CUTLASS_DEVICE static auto work_tile_to_cta_coord(
      WorkTileInfo work_tile_info) {
    return cute::make_coord(
        work_tile_info.M_idx, cute::Int<0>{},
        cute::Underscore{}, cute::Int<0>{});
  }

  CUTLASS_DEVICE static auto work_tile_to_cta_coord(
      WorkTileInfo work_tile_info, dim3) {
    return work_tile_to_cta_coord(work_tile_info);
  }

  CUTLASS_DEVICE bool is_last_tile(WorkTileInfo&, uint32_t = 1) const {
    return true;
  }

  CUTLASS_DEVICE auto fetch_next_work(WorkTileInfo) {
    return cute::make_tuple(WorkTileInfo::invalid_work_tile(), true);
  }

  template <class TileSchedulerPipeline, class TileSchedulerPipelineState>
  CUTLASS_DEVICE auto fetch_next_work(
      WorkTileInfo work_tile_info, TileSchedulerPipeline&,
      TileSchedulerPipelineState) {
    return fetch_next_work(work_tile_info);
  }
};

// The three wider B1 projections have enough complete output tiles to keep all
// 48 SMs resident. Keep CUTLASS's complete static-persistent device contract as
// well as its full host grid: the SM120 ping-pong kernel advances scheduler
// state directly and therefore requires the base cursor and grid size to be
// initialized. N=5120 remains on the separate exact 40-CTA specialization.
class Fr13B1OneNFullGridStaticTileScheduler100
    : public StaticPersistentTileScheduler100 {
  using Base = StaticPersistentTileScheduler100;

 public:
  using Base::Base;
};

template <class TileShape, class ClusterShape,
          uint32_t SchedulerPipelineStageCount>
struct TileSchedulerSelector<
    vllm::fr13_fixed32_m128_divisor_static_scheduler, arch::Sm120,
    TileShape, ClusterShape, SchedulerPipelineStageCount> {
  using Scheduler = Fr13DivisorBalancedStaticTileScheduler100;
};

template <class TileShape, class ClusterShape,
          uint32_t SchedulerPipelineStageCount>
struct TileSchedulerSelector<
    vllm::fr13_fixed32_b1_onen_static_scheduler, arch::Sm120,
    TileShape, ClusterShape, SchedulerPipelineStageCount> {
  using Scheduler = Fr13B1OneNStaticTileScheduler100;
};

template <class TileShape, class ClusterShape,
          uint32_t SchedulerPipelineStageCount>
struct TileSchedulerSelector<
    vllm::fr13_fixed32_b1_n5120_single_tile_scheduler, arch::Sm120,
    TileShape, ClusterShape, SchedulerPipelineStageCount> {
  using Scheduler = Fr13B1N5120SingleTileScheduler100;
};

template <class TileShape, class ClusterShape,
          uint32_t SchedulerPipelineStageCount>
struct TileSchedulerSelector<
    vllm::fr13_fixed32_b1_onen_fullgrid_static_scheduler, arch::Sm120,
    TileShape, ClusterShape, SchedulerPipelineStageCount> {
  using Scheduler = Fr13B1OneNFullGridStaticTileScheduler100;
};

template <class TileShape, class ClusterShape,
          uint32_t SchedulerPipelineStageCount>
struct TileSchedulerSelector<
    vllm::fr13_fixed32_b4_twom_static_scheduler, arch::Sm120,
    TileShape, ClusterShape, SchedulerPipelineStageCount> {
  using Scheduler = Fr13B4TwoMStaticTileScheduler100;
};

template <class TileShape, class ClusterShape,
          uint32_t SchedulerPipelineStageCount>
struct TileSchedulerSelector<
    vllm::fr13_fixed32_b4_n5120_single_tile_scheduler, arch::Sm120,
    TileShape, ClusterShape, SchedulerPipelineStageCount> {
  using Scheduler = Fr13B4N5120SingleTileScheduler100;
};
}  // namespace cutlass::gemm::kernel::detail

namespace vllm {
"""

TEMPLATE_ANCHOR = """          class EpilogueScheduler, class MainloopScheduler,
          bool swap_ab_ = false>
struct cutlass_3x_gemm_fp8_blockwise {
"""

KERNEL_ANCHOR = """  using KernelType = enable_sm120_family<cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>>;
"""

STAGE_COUNT_ANCHOR = (
    "cutlass::gemm::collective::StageCountAutoCarveout<"
    "static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>"
)

STREAMK_CLASS_ANCHOR = KERNEL_ANCHOR + """
  struct GemmKernel : public KernelType {};
};

"""
STREAMK_CLASS_REPLACEMENT = STREAMK_CLASS_ANCHOR + r"""template <
    class OutType, int ScaleGranularityM, int ScaleGranularityN,
    int ScaleGranularityK, class MmaTileShape, class ClusterShape,
    class EpilogueScheduler, class MainloopScheduler, bool swap_ab_,
    class TileScheduler, bool force_stream_k_ = false,
    class MainloopStageCount = cutlass::gemm::collective::StageCountAuto>
struct cutlass_3x_gemm_fp8_blockwise_streamk
    : cutlass_3x_gemm_fp8_blockwise<
          OutType, ScaleGranularityM, ScaleGranularityN, ScaleGranularityK,
          MmaTileShape, ClusterShape, EpilogueScheduler, MainloopScheduler,
          swap_ab_> {
  using Base = cutlass_3x_gemm_fp8_blockwise<
      OutType, ScaleGranularityM, ScaleGranularityN, ScaleGranularityK,
      MmaTileShape, ClusterShape, EpilogueScheduler, MainloopScheduler,
      swap_ab_>;
  static constexpr bool use_stream_k = true;
  static constexpr bool force_stream_k = force_stream_k_;

  using CollectiveEpilogue = typename Base::CollectiveEpilogue;
  using CollectiveMainloop = conditional_t<
      Base::swap_ab,
      typename cutlass::gemm::collective::CollectiveBuilder<
          typename Base::ArchTag, typename Base::OperatorClass,
          typename Base::ElementB,
          cute::tuple<typename Base::LayoutB_Transpose,
                      typename Base::LayoutSFA>,
          Base::AlignmentB, typename Base::ElementA,
          cute::tuple<typename Base::LayoutA_Transpose,
                      typename Base::LayoutSFB>,
          Base::AlignmentA, typename Base::ElementAccumulator, MmaTileShape,
          ClusterShape, MainloopStageCount, MainloopScheduler>::CollectiveOp,
      typename cutlass::gemm::collective::CollectiveBuilder<
          typename Base::ArchTag, typename Base::OperatorClass,
          typename Base::ElementA,
          cute::tuple<typename Base::LayoutA, typename Base::LayoutSFA>,
          Base::AlignmentA, typename Base::ElementB,
          cute::tuple<typename Base::LayoutB, typename Base::LayoutSFB>,
          Base::AlignmentB, typename Base::ElementAccumulator, MmaTileShape,
          ClusterShape, MainloopStageCount, MainloopScheduler>::CollectiveOp>;

  using KernelType = enable_sm120_family<cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue,
      TileScheduler>>;

  struct GemmKernel : public KernelType {};
};

template <
    class OutType, int ScaleGranularityM, int ScaleGranularityN,
    int ScaleGranularityK, class MmaTileShape, class ClusterShape,
    class EpilogueScheduler, class MainloopScheduler, bool swap_ab_>
struct cutlass_3x_gemm_fp8_blockwise_m128_static
    : cutlass_3x_gemm_fp8_blockwise<
          OutType, ScaleGranularityM, ScaleGranularityN, ScaleGranularityK,
          MmaTileShape, ClusterShape, EpilogueScheduler, MainloopScheduler,
          swap_ab_> {
  using Base = cutlass_3x_gemm_fp8_blockwise<
      OutType, ScaleGranularityM, ScaleGranularityN, ScaleGranularityK,
      MmaTileShape, ClusterShape, EpilogueScheduler, MainloopScheduler,
      swap_ab_>;

  using KernelType = enable_sm120_family<cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, typename Base::CollectiveMainloop,
      typename Base::CollectiveEpilogue,
      fr13_fixed32_m128_static_scheduler>>;

  struct GemmKernel : public KernelType {};
};

template <
    class OutType, int ScaleGranularityM, int ScaleGranularityN,
    int ScaleGranularityK, class MmaTileShape, class ClusterShape,
    class EpilogueScheduler, class MainloopScheduler, bool swap_ab_>
struct cutlass_3x_gemm_fp8_blockwise_m128_divisor_static
    : cutlass_3x_gemm_fp8_blockwise<
          OutType, ScaleGranularityM, ScaleGranularityN, ScaleGranularityK,
          MmaTileShape, ClusterShape, EpilogueScheduler, MainloopScheduler,
          swap_ab_> {
  using Base = cutlass_3x_gemm_fp8_blockwise<
      OutType, ScaleGranularityM, ScaleGranularityN, ScaleGranularityK,
      MmaTileShape, ClusterShape, EpilogueScheduler, MainloopScheduler,
      swap_ab_>;

  using KernelType = enable_sm120_family<cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, typename Base::CollectiveMainloop,
      typename Base::CollectiveEpilogue,
      fr13_fixed32_m128_divisor_static_scheduler>>;

  struct GemmKernel : public KernelType {};
};

// Fixed32 projection calls have no source C and always use alpha=1, beta=0.
// Convert the accumulator directly to the output type with the same explicit
// round-to-nearest policy. The fixed-shape configs also use the legal minimum
// two-stage mainloop while preserving tile shape and ordered full-K traversal.
template <
    class OutType, int ScaleGranularityM, int ScaleGranularityN,
    int ScaleGranularityK, class MmaTileShape, class ClusterShape,
    class EpilogueScheduler, class MainloopScheduler, bool swap_ab_,
    class TileScheduler,
    class MainloopStageCount = void>
struct cutlass_3x_gemm_fp8_blockwise_identity_static
    : cutlass_3x_gemm_fp8_blockwise<
          OutType, ScaleGranularityM, ScaleGranularityN, ScaleGranularityK,
          MmaTileShape, ClusterShape, EpilogueScheduler, MainloopScheduler,
          swap_ab_> {
  using Base = cutlass_3x_gemm_fp8_blockwise<
      OutType, ScaleGranularityM, ScaleGranularityN, ScaleGranularityK,
      MmaTileShape, ClusterShape, EpilogueScheduler, MainloopScheduler,
      swap_ab_>;

  using Fr13EpilogueCallbacks = cutlass::epilogue::fusion::Sm90EVT<
      cutlass::epilogue::fusion::Sm90Compute<
          cutlass::epilogue::thread::Identity,
          typename cutlass::detail::get_unpacked_element_type<OutType>::type,
          typename Base::ElementAccumulator,
          cutlass::FloatRoundStyle::round_to_nearest>,
      cutlass::epilogue::fusion::Sm90AccFetch>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          typename Base::ArchTag, typename Base::OperatorClass,
          MmaTileShape, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          typename Base::ElementAccumulator, typename Base::ElementCompute,
          typename Base::ElementC,
          conditional_t<Base::swap_ab, typename Base::LayoutC_Transpose,
                        typename Base::LayoutC>,
          Base::AlignmentC, typename Base::ElementD,
          conditional_t<Base::swap_ab, typename Base::LayoutD_Transpose,
                        typename Base::LayoutD>,
          Base::AlignmentD, EpilogueScheduler,
          Fr13EpilogueCallbacks>::CollectiveOp;

  using ResolvedMainloopStageCount = conditional_t<
      std::is_void_v<MainloopStageCount>,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainloopStageCount>;

  using CollectiveMainloop = conditional_t<
      Base::swap_ab,
      typename cutlass::gemm::collective::CollectiveBuilder<
          typename Base::ArchTag, typename Base::OperatorClass,
          typename Base::ElementB,
          cute::tuple<typename Base::LayoutB_Transpose,
                      typename Base::LayoutSFA>,
          Base::AlignmentB, typename Base::ElementA,
          cute::tuple<typename Base::LayoutA_Transpose,
                      typename Base::LayoutSFB>,
          Base::AlignmentA, typename Base::ElementAccumulator, MmaTileShape,
          ClusterShape, ResolvedMainloopStageCount,
          MainloopScheduler>::CollectiveOp,
      typename cutlass::gemm::collective::CollectiveBuilder<
          typename Base::ArchTag, typename Base::OperatorClass,
          typename Base::ElementA,
          cute::tuple<typename Base::LayoutA, typename Base::LayoutSFA>,
          Base::AlignmentA, typename Base::ElementB,
          cute::tuple<typename Base::LayoutB, typename Base::LayoutSFB>,
          Base::AlignmentB, typename Base::ElementAccumulator, MmaTileShape,
          ClusterShape, ResolvedMainloopStageCount,
          MainloopScheduler>::CollectiveOp>;

  using KernelType = enable_sm120_family<cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop,
      CollectiveEpilogue, TileScheduler>>;

  struct GemmKernel : public KernelType {};
};

"""

CONFIG_ANCHOR = """template <typename Gemm>
void cutlass_gemm_caller_blockwise(torch::stable::Tensor& out, torch::stable::Tensor const& a,
"""
CONFIG_REPLACEMENT = r"""// FR13_FIXED32_CUTLASS_WAVE: source-only candidates; stock is default.
template <typename Gemm, typename = void>
struct fr13_fixed32_streamk_traits {
  static constexpr bool enabled = false;
  static constexpr bool force = false;
};

template <typename Gemm>
struct fr13_fixed32_streamk_traits<
    Gemm, std::void_t<decltype(Gemm::use_stream_k),
                      decltype(Gemm::force_stream_k)>> {
  static constexpr bool enabled = Gemm::use_stream_k;
  static constexpr bool force = Gemm::force_stream_k;
};

template <typename OutType>
struct sm120_blockwise_fp8_config_cooperative_streamk {
  using KernelSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedBlockwiseCooperativeSm120;
  using EpilogueSchedule =
      cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
  using Gemm = cutlass_3x_gemm_fp8_blockwise_streamk<
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
  using Gemm = cutlass_3x_gemm_fp8_blockwise_streamk<
      OutType, 128, 1, 128, TileShape, ClusterShape,
      EpilogueSchedule, KernelSchedule, true,
      cutlass::gemm::StreamKScheduler>;
};

template <typename OutType>
struct sm120_blockwise_fp8_config_swapab_streamk_wide256 {
  using KernelSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedBlockwiseCooperativeSm120;
  using EpilogueSchedule =
      cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_256, _32, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
  using Gemm = cutlass_3x_gemm_fp8_blockwise_streamk<
      OutType, 128, 1, 128, TileShape, ClusterShape,
      EpilogueSchedule, KernelSchedule, true,
      fr13_fixed32_wide256_recompute_scheduler, true,
      cutlass::gemm::collective::StageCount<2>>;
};

// Reuse the audited static scheduler wrapper with the exact stock B1
// swap-AB collective. Despite its legacy type name, the wrapper is generic in
// tile shape and changes only complete-output-tile assignment.
template <typename OutType>
struct sm120_blockwise_fp8_config_b1_static_persistent_stocktile {
  using KernelSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedBlockwiseCooperativeSm120;
  using EpilogueSchedule =
      cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_128, _32, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
  using Gemm = cutlass_3x_gemm_fp8_blockwise_m128_static<
      OutType, 128, 1, 128, TileShape, ClusterShape,
      EpilogueSchedule, KernelSchedule, true>;
};

template <typename OutType>
struct sm120_blockwise_fp8_config_b1_divisor_static_stocktile {
  using KernelSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedBlockwiseCooperativeSm120;
  using EpilogueSchedule =
      cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_128, _32, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
  using Gemm = cutlass_3x_gemm_fp8_blockwise_m128_divisor_static<
      OutType, 128, 1, 128, TileShape, ClusterShape,
      EpilogueSchedule, KernelSchedule, true>;
};

template <typename OutType>
struct sm120_blockwise_fp8_config_b4_persistent_m128 {
  using KernelSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedBlockwiseCooperativeSm120;
  using EpilogueSchedule =
      cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
  using BaseGemm = cutlass_3x_gemm_fp8_blockwise<
      OutType, 1, 128, 128, TileShape, ClusterShape,
      EpilogueSchedule, KernelSchedule, false>;
  struct Gemm : BaseGemm {
    using KernelType = typename BaseGemm::KernelType;
    struct GemmKernel : public KernelType {};
  };
};

// Keep the M128 collective math unchanged and replace only the dynamic CLC
// complete-output-tile allocator with CUTLASS static persistence.
template <typename OutType>
struct sm120_blockwise_fp8_config_b4_persistent_m128_static {
  using KernelSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedBlockwiseCooperativeSm120;
  using EpilogueSchedule =
      cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
  using Gemm = cutlass_3x_gemm_fp8_blockwise_m128_static<
      OutType, 1, 128, 128, TileShape, ClusterShape,
      EpilogueSchedule, KernelSchedule, false>;
};

template <typename OutType>
struct sm120_blockwise_fp8_config_b1_divisor_static_identity_stage2 {
  using KernelSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedBlockwiseCooperativeSm120;
  using EpilogueSchedule =
      cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_128, _32, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
  using Gemm = cutlass_3x_gemm_fp8_blockwise_identity_static<
      OutType, 128, 1, 128, TileShape, ClusterShape,
      EpilogueSchedule, KernelSchedule, true,
      fr13_fixed32_m128_divisor_static_scheduler,
      cutlass::gemm::collective::StageCount<2>>;
};

// Alternate complete output tiles between two consumer warp groups. The
// divisor-balanced B1 grid gives each CTA 4 or 8 tiles on the three wider
// projections, so ping-pong can overlap mainloop and epilogue without split K
// or any change to a tile's accumulation order. The two N=5120 projections
// retain the cooperative kernel because they assign only one tile per CTA.
template <typename OutType>
struct sm120_blockwise_fp8_config_b1_divisor_static_identity_pingpong_stage2 {
  using KernelSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedBlockwisePingpongSm120;
  using EpilogueSchedule =
      cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_128, _32, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
  using Gemm = cutlass_3x_gemm_fp8_blockwise_identity_static<
      OutType, 128, 1, 128, TileShape, ClusterShape,
      EpilogueSchedule, KernelSchedule, true,
      fr13_fixed32_m128_divisor_static_scheduler,
      cutlass::gemm::collective::StageCount<2>>;
};

// Preserve the audited B1 two-stage identity cooperative collective used by the
// two N=5120 projections and replace only its generic static tile-coordinate
// decode. After swap-AB, physical M=32 is exactly one scheduler-N tile.
template <typename OutType>
struct sm120_blockwise_fp8_config_b1_onen_static_identity_stage2 {
  using KernelSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedBlockwiseCooperativeSm120;
  using EpilogueSchedule =
      cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_128, _32, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
  using Gemm = cutlass_3x_gemm_fp8_blockwise_identity_static<
      OutType, 128, 1, 128, TileShape, ClusterShape,
      EpilogueSchedule, KernelSchedule, true,
      fr13_fixed32_b1_onen_static_scheduler,
      cutlass::gemm::collective::StageCount<2>>;
};

// Keep the exact N=5120 cooperative collective and full-K accumulation order,
// changing only the scheduler to the one-output-tile-per-CTA specialization.
template <typename OutType>
struct sm120_blockwise_fp8_config_b1_n5120_single_identity_stage2 {
  using KernelSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedBlockwiseCooperativeSm120;
  using EpilogueSchedule =
      cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_128, _32, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
  using Gemm = cutlass_3x_gemm_fp8_blockwise_identity_static<
      OutType, 128, 1, 128, TileShape, ClusterShape,
      EpilogueSchedule, KernelSchedule, true,
      fr13_fixed32_b1_n5120_single_tile_scheduler,
      cutlass::gemm::collective::StageCount<2>>;
};

// Preserve the audited B1 two-stage identity ping-pong collective used by the
// three wider projections and replace only its generic coordinate decode.
template <typename OutType>
struct sm120_blockwise_fp8_config_b1_onen_static_identity_pingpong_stage2 {
  using KernelSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedBlockwisePingpongSm120;
  using EpilogueSchedule =
      cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_128, _32, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
  using Gemm = cutlass_3x_gemm_fp8_blockwise_identity_static<
      OutType, 128, 1, 128, TileShape, ClusterShape,
      EpilogueSchedule, KernelSchedule, true,
      fr13_fixed32_b1_onen_static_scheduler,
      cutlass::gemm::collective::StageCount<2>>;
};

// Preserve the exact wide-projection math and one-N device mapping, but use
// CUTLASS's full 48-CTA static grid instead of the divisor-balanced host grid.
template <typename OutType>
struct sm120_blockwise_fp8_config_b1_onen_fullgrid_identity_pingpong_stage2 {
  using KernelSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedBlockwisePingpongSm120;
  using EpilogueSchedule =
      cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_128, _32, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
  using Gemm = cutlass_3x_gemm_fp8_blockwise_identity_static<
      OutType, 128, 1, 128, TileShape, ClusterShape,
      EpilogueSchedule, KernelSchedule, true,
      fr13_fixed32_b1_onen_fullgrid_static_scheduler,
      cutlass::gemm::collective::StageCount<2>>;
};

template <typename OutType>
struct sm120_blockwise_fp8_config_b4_m128_static_identity_stage2 {
  using KernelSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedBlockwiseCooperativeSm120;
  using EpilogueSchedule =
      cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
  using Gemm = cutlass_3x_gemm_fp8_blockwise_identity_static<
      OutType, 1, 128, 128, TileShape, ClusterShape,
      EpilogueSchedule, KernelSchedule, false,
      fr13_fixed32_m128_static_scheduler,
      cutlass::gemm::collective::StageCount<2>>;
};

template <typename OutType>
struct sm120_blockwise_fp8_config_b4_m128_n5120_single_identity_stage2 {
  using KernelSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedBlockwiseCooperativeSm120;
  using EpilogueSchedule =
      cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
  using Gemm = cutlass_3x_gemm_fp8_blockwise_identity_static<
      OutType, 1, 128, 128, TileShape, ClusterShape,
      EpilogueSchedule, KernelSchedule, false,
      fr13_fixed32_b4_n5120_single_tile_scheduler,
      cutlass::gemm::collective::StageCount<2>>;
};

// Isolate the identity epilogue on B4 while retaining the stock 64x128x128
// tile, ping-pong mainloop schedule, and dynamic tile scheduler.
template <typename OutType>
struct sm120_blockwise_fp8_config_b4_stockshape_identity {
  using KernelSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedBlockwisePingpongSm120;
  using EpilogueSchedule =
      cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_64, _128, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
  using Gemm = cutlass_3x_gemm_fp8_blockwise_identity_static<
      OutType, 1, 128, 128, TileShape, ClusterShape,
      EpilogueSchedule, KernelSchedule, false, void>;
};

// Isolate a two-stage TMA mainloop on B4 while retaining the stock dynamic
// scheduler, 64x128x128 tile, ping-pong schedule, and identity epilogue.
template <typename OutType>
struct sm120_blockwise_fp8_config_b4_stockshape_identity_stage2 {
  using KernelSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedBlockwisePingpongSm120;
  using EpilogueSchedule =
      cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_64, _128, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
  using Gemm = cutlass_3x_gemm_fp8_blockwise_identity_static<
      OutType, 1, 128, 128, TileShape, ClusterShape,
      EpilogueSchedule, KernelSchedule, false, void,
      cutlass::gemm::collective::StageCount<2>>;
};

template <typename OutType>
struct sm120_blockwise_fp8_config_b4_stockshape_identity_divisor {
  using KernelSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedBlockwisePingpongSm120;
  using EpilogueSchedule =
      cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_64, _128, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
  using Gemm = cutlass_3x_gemm_fp8_blockwise_identity_static<
      OutType, 1, 128, 128, TileShape, ClusterShape,
      EpilogueSchedule, KernelSchedule, false,
      fr13_fixed32_m128_divisor_static_scheduler>;
};

// Preserve the B4 stock tile and divisor-balanced scheduler while limiting the
// mainloop to two stages. This isolates whether lower shared-memory residency
// pressure outweighs the shorter TMA pipeline for the fixed K64 projections.
template <typename OutType>
struct sm120_blockwise_fp8_config_b4_stockshape_identity_divisor_stage2 {
  using KernelSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedBlockwisePingpongSm120;
  using EpilogueSchedule =
      cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_64, _128, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
  using Gemm = cutlass_3x_gemm_fp8_blockwise_identity_static<
      OutType, 1, 128, 128, TileShape, ClusterShape,
      EpilogueSchedule, KernelSchedule, false,
      fr13_fixed32_m128_divisor_static_scheduler,
      cutlass::gemm::collective::StageCount<2>>;
};

template <typename OutType>
struct sm120_blockwise_fp8_config_b4_stockshape_identity_twom {
  using KernelSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedBlockwisePingpongSm120;
  using EpilogueSchedule =
      cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_64, _128, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
  using Gemm = cutlass_3x_gemm_fp8_blockwise_identity_static<
      OutType, 1, 128, 128, TileShape, ClusterShape,
      EpilogueSchedule, KernelSchedule, false,
      fr13_fixed32_b4_twom_static_scheduler,
      cutlass::gemm::collective::StageCount<2>>;
};

enum class fixed32_cutlass_wave_variant {
  stock,
  stream_k_cooperative_128,
  stream_k_cooperative_128_byte_ab,
  stream_k_force_wide256,
  stream_k_force_wide256_byte_ab,
  static_persistent_stocktile,
  static_persistent_stocktile_byte_ab,
  divisor_static_stocktile,
  divisor_static_stocktile_byte_ab,
  persistent_b4_m128,
  persistent_b4_m128_byte_ab,
  persistent_b4_m128_static,
  persistent_b4_m128_static_byte_ab,
  identity_stage2_static,
  identity_stage2_static_byte_ab,
  identity_stage2_pingpong_b1,
  identity_stage2_pingpong_b1_byte_ab,
  identity_onen_b1,
  identity_onen_b1_byte_ab,
  identity_onen_n5120_single_b1,
  identity_onen_n5120_single_b1_byte_ab,
  identity_onen_n5120_fullgrid_b1,
  identity_onen_n5120_fullgrid_b1_byte_ab,
  identity_stockshape_b4,
  identity_stockshape_b4_byte_ab,
  identity_stockshape_stage2_b4,
  identity_stockshape_stage2_b4_byte_ab,
  identity_divisor_b4,
  identity_divisor_b4_byte_ab,
  identity_divisor_stage2_b4,
  identity_divisor_stage2_b4_byte_ab,
  identity_twom_b4,
  identity_twom_b4_byte_ab,
  identity_hybrid_n5120_b4,
  identity_hybrid_n5120_b4_byte_ab,
};

static inline fixed32_cutlass_wave_variant fixed32_cutlass_wave_selection() {
  static const fixed32_cutlass_wave_variant selection = [] {
    const char* environment = std::getenv("FR13_FIXED32_CUTLASS_WAVE");
    std::string value = environment == nullptr ? "" : environment;
    if (value.empty()) {
      std::ifstream selector_file(
          "/logs/fr13_fixed32_cutlass_wave.selector");
      if (selector_file.good()) {
        std::getline(selector_file, value);
      }
    }
    if (value == "streamk_coop128") {
      return fixed32_cutlass_wave_variant::stream_k_cooperative_128;
    }
    if (value == "streamk_coop128_byte_ab") {
      return fixed32_cutlass_wave_variant::stream_k_cooperative_128_byte_ab;
    }
    if (value == "streamk_force_wide256") {
      return fixed32_cutlass_wave_variant::stream_k_force_wide256;
    }
    if (value == "streamk_force_wide256_byte_ab") {
      return fixed32_cutlass_wave_variant::stream_k_force_wide256_byte_ab;
    }
    if (value == "static_persistent_stocktile") {
      return fixed32_cutlass_wave_variant::static_persistent_stocktile;
    }
    if (value == "static_persistent_stocktile_byte_ab") {
      return fixed32_cutlass_wave_variant::static_persistent_stocktile_byte_ab;
    }
    if (value == "divisor_static_stocktile") {
      return fixed32_cutlass_wave_variant::divisor_static_stocktile;
    }
    if (value == "divisor_static_stocktile_byte_ab") {
      return fixed32_cutlass_wave_variant::divisor_static_stocktile_byte_ab;
    }
    if (value == "persistent_b4_m128") {
      return fixed32_cutlass_wave_variant::persistent_b4_m128;
    }
    if (value == "persistent_b4_m128_byte_ab") {
      return fixed32_cutlass_wave_variant::persistent_b4_m128_byte_ab;
    }
    if (value == "persistent_b4_m128_static") {
      return fixed32_cutlass_wave_variant::persistent_b4_m128_static;
    }
    if (value == "persistent_b4_m128_static_byte_ab") {
      return fixed32_cutlass_wave_variant::persistent_b4_m128_static_byte_ab;
    }
    if (value == "identity_stage2_static") {
      return fixed32_cutlass_wave_variant::identity_stage2_static;
    }
    if (value == "identity_stage2_static_byte_ab") {
      return fixed32_cutlass_wave_variant::identity_stage2_static_byte_ab;
    }
    if (value == "identity_stage2_pingpong_b1") {
      return fixed32_cutlass_wave_variant::identity_stage2_pingpong_b1;
    }
    if (value == "identity_stage2_pingpong_b1_byte_ab") {
      return fixed32_cutlass_wave_variant::identity_stage2_pingpong_b1_byte_ab;
    }
    if (value == "identity_onen_b1") {
      return fixed32_cutlass_wave_variant::identity_onen_b1;
    }
    if (value == "identity_onen_b1_byte_ab") {
      return fixed32_cutlass_wave_variant::identity_onen_b1_byte_ab;
    }
    if (value == "identity_onen_n5120_single_b1") {
      return fixed32_cutlass_wave_variant::identity_onen_n5120_single_b1;
    }
    if (value == "identity_onen_n5120_single_b1_byte_ab") {
      return fixed32_cutlass_wave_variant::identity_onen_n5120_single_b1_byte_ab;
    }
    if (value == "identity_onen_n5120_fullgrid_b1") {
      return fixed32_cutlass_wave_variant::identity_onen_n5120_fullgrid_b1;
    }
    if (value == "identity_onen_n5120_fullgrid_b1_byte_ab") {
      return fixed32_cutlass_wave_variant::identity_onen_n5120_fullgrid_b1_byte_ab;
    }
    if (value == "identity_stockshape_b4") {
      return fixed32_cutlass_wave_variant::identity_stockshape_b4;
    }
    if (value == "identity_stockshape_b4_byte_ab") {
      return fixed32_cutlass_wave_variant::identity_stockshape_b4_byte_ab;
    }
    if (value == "identity_stockshape_stage2_b4") {
      return fixed32_cutlass_wave_variant::identity_stockshape_stage2_b4;
    }
    if (value == "identity_stockshape_stage2_b4_byte_ab") {
      return fixed32_cutlass_wave_variant::identity_stockshape_stage2_b4_byte_ab;
    }
    if (value == "identity_divisor_b4") {
      return fixed32_cutlass_wave_variant::identity_divisor_b4;
    }
    if (value == "identity_divisor_b4_byte_ab") {
      return fixed32_cutlass_wave_variant::identity_divisor_b4_byte_ab;
    }
    if (value == "identity_divisor_stage2_b4") {
      return fixed32_cutlass_wave_variant::identity_divisor_stage2_b4;
    }
    if (value == "identity_divisor_stage2_b4_byte_ab") {
      return fixed32_cutlass_wave_variant::identity_divisor_stage2_b4_byte_ab;
    }
    if (value == "identity_twom_b4") {
      return fixed32_cutlass_wave_variant::identity_twom_b4;
    }
    if (value == "identity_twom_b4_byte_ab") {
      return fixed32_cutlass_wave_variant::identity_twom_b4_byte_ab;
    }
    if (value == "identity_hybrid_n5120_b4") {
      return fixed32_cutlass_wave_variant::identity_hybrid_n5120_b4;
    }
    if (value == "identity_hybrid_n5120_b4_byte_ab") {
      return fixed32_cutlass_wave_variant::identity_hybrid_n5120_b4_byte_ab;
    }
    return fixed32_cutlass_wave_variant::stock;
  }();
  return selection;
}

static inline std::string fixed32_cutlass_real_task_marker() {
  constexpr const char* arm_path =
      "/logs/fr13_fixed32_cutlass_streamk.real_event.arm";
  std::ifstream arm(arm_path);
  if (!arm.good()) {
    return "";
  }
  std::string marker;
  std::string trailing;
  std::getline(arm, marker);
  STD_TORCH_CHECK(!marker.empty() && !std::getline(arm, trailing),
                  "FR13 Stream-K real-task arm is malformed");
  constexpr const char* prefix = "swe_verified:";
  constexpr const char* allowed =
      "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-/";
  STD_TORCH_CHECK(
      marker.size() > std::strlen(prefix) && marker.size() <= 256 &&
          marker.rfind(prefix, 0) == 0 &&
          marker.substr(std::strlen(prefix)).find_first_not_of(allowed) ==
              std::string::npos,
      "FR13 Stream-K real-task arm is noncanonical");
  return marker;
}

static inline std::string fixed32_cutlass_b4_real_task_marker() {
  constexpr const char* arm_path =
      "/logs/fr13_fixed32_cutlass_b4_byte_ab.real_event.arm";
  std::ifstream arm(arm_path);
  if (!arm.good()) {
    return "";
  }
  std::string marker;
  std::string trailing;
  std::getline(arm, marker);
  STD_TORCH_CHECK(!marker.empty() && !std::getline(arm, trailing),
                  "FR13 CUTLASS B4 real-task arm is malformed");
  constexpr const char* prefix = "swe_verified:";
  constexpr const char* allowed =
      "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-/";
  STD_TORCH_CHECK(
      marker.size() > std::strlen(prefix) && marker.size() <= 256 &&
          marker.rfind(prefix, 0) == 0 &&
          marker.substr(std::strlen(prefix)).find_first_not_of(allowed) ==
              std::string::npos,
      "FR13 CUTLASS B4 real-task arm is noncanonical");
  return marker;
}

static inline bool fixed32_cutlass_real_projection(int m, int n, int k) {
  const bool fixed32_rows = m == 32 || m == 64 || m == 96 || m == 128;
  const bool real_projection =
      (n == 34816 && k == 5120) ||
      (n == 5120 && k == 17408) ||
      (n == 5120 && k == 6144) ||
      (n == 16384 && k == 5120) ||
      (n == 14336 && k == 5120);
  return fixed32_rows && real_projection;
}

static inline bool fixed32_cutlass_b1_onen_projection(int m, int n, int k) {
  return m == 32 &&
      ((n == 34816 && k == 5120) ||
       (n == 5120 && k == 17408) ||
       (n == 5120 && k == 6144) ||
       (n == 16384 && k == 5120) ||
       (n == 14336 && k == 5120));
}

static inline bool fixed32_cutlass_b4_hybrid_n5120_projection(
    int m, int n, int k) {
  return m == 128 && fixed32_cutlass_real_projection(m, n, k);
}

template <typename Gemm>
void cutlass_gemm_caller_blockwise(torch::stable::Tensor& out, torch::stable::Tensor const& a,
"""

CALLER_ANCHOR = """  c3x::cutlass_gemm_caller<GemmKernel>(a.device(), prob_shape, mainloop_args,
                                       epilogue_args);
"""
CALLER_REPLACEMENT = """  using StreamKTraits = fr13_fixed32_streamk_traits<Gemm>;
  if constexpr (!StreamKTraits::enabled) {
    return c3x::cutlass_gemm_caller<GemmKernel>(
        a.device(), prob_shape, mainloop_args, epilogue_args);
  } else {

  // Stream-K needs a real SM count to choose the tail decomposition. Cache the
  // device query outside the measured steady state for each candidate kernel.
  static const int sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count();
  STD_TORCH_CHECK(sm_count > 0,
                  "FR13 Stream-K could not query the CUDA SM count");
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = a.device().index();
  hw_info.sm_count = sm_count;

  typename GemmKernel::TileSchedulerArguments scheduler{};
  scheduler.splits = 1;
  scheduler.reduction_mode =
      decltype(scheduler.reduction_mode)::Deterministic;
  scheduler.decomposition_mode =
      StreamKTraits::force
          ? decltype(scheduler.decomposition_mode)::StreamK
          : decltype(scheduler.decomposition_mode)::Heuristic;
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
  }
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
  // The normal-layout 128x256 forced-StreamK specialization returned CUTLASS
  // Error::kErrorInternal during real B1 profile warmup. Wide256 is therefore
  // B1-only: both selectors fail safely to stock above 64 physical rows.
  if (M > 64 &&
      (wave_variant ==
           fixed32_cutlass_wave_variant::stream_k_force_wide256 ||
       wave_variant ==
           fixed32_cutlass_wave_variant::stream_k_force_wide256_byte_ab ||
       wave_variant ==
           fixed32_cutlass_wave_variant::static_persistent_stocktile ||
       wave_variant == fixed32_cutlass_wave_variant::
                           static_persistent_stocktile_byte_ab)) {
    wave_variant = fixed32_cutlass_wave_variant::stock;
  }
  if (M != 32 &&
      (wave_variant ==
           fixed32_cutlass_wave_variant::divisor_static_stocktile ||
       wave_variant == fixed32_cutlass_wave_variant::
                           divisor_static_stocktile_byte_ab)) {
    wave_variant = fixed32_cutlass_wave_variant::stock;
  }
  if (M != 128 &&
      (wave_variant ==
           fixed32_cutlass_wave_variant::persistent_b4_m128 ||
       wave_variant ==
           fixed32_cutlass_wave_variant::persistent_b4_m128_byte_ab ||
       wave_variant ==
           fixed32_cutlass_wave_variant::persistent_b4_m128_static ||
       wave_variant ==
           fixed32_cutlass_wave_variant::persistent_b4_m128_static_byte_ab)) {
    wave_variant = fixed32_cutlass_wave_variant::stock;
  }
  if (N > 65536 &&
      (wave_variant == fixed32_cutlass_wave_variant::identity_twom_b4 ||
       wave_variant ==
           fixed32_cutlass_wave_variant::identity_twom_b4_byte_ab)) {
    wave_variant = fixed32_cutlass_wave_variant::stock;
  }
  if (M != 32 && M != 128 &&
      (wave_variant == fixed32_cutlass_wave_variant::identity_stage2_static ||
       wave_variant ==
           fixed32_cutlass_wave_variant::identity_stage2_static_byte_ab)) {
    wave_variant = fixed32_cutlass_wave_variant::stock;
  }
  if (M != 32 &&
      (wave_variant ==
           fixed32_cutlass_wave_variant::identity_stage2_pingpong_b1 ||
       wave_variant == fixed32_cutlass_wave_variant::
                           identity_stage2_pingpong_b1_byte_ab)) {
    wave_variant = fixed32_cutlass_wave_variant::stock;
  }
  if (!fixed32_cutlass_b1_onen_projection(M, N, K) &&
      (wave_variant == fixed32_cutlass_wave_variant::identity_onen_b1 ||
       wave_variant ==
           fixed32_cutlass_wave_variant::identity_onen_b1_byte_ab)) {
    wave_variant = fixed32_cutlass_wave_variant::stock;
  }
  if (!fixed32_cutlass_b1_onen_projection(M, N, K) &&
      (wave_variant == fixed32_cutlass_wave_variant::
                           identity_onen_n5120_single_b1 ||
       wave_variant == fixed32_cutlass_wave_variant::
                           identity_onen_n5120_single_b1_byte_ab)) {
    wave_variant = fixed32_cutlass_wave_variant::stock;
  }
  if (!fixed32_cutlass_b1_onen_projection(M, N, K) &&
      (wave_variant == fixed32_cutlass_wave_variant::
                           identity_onen_n5120_fullgrid_b1 ||
       wave_variant == fixed32_cutlass_wave_variant::
                           identity_onen_n5120_fullgrid_b1_byte_ab)) {
    wave_variant = fixed32_cutlass_wave_variant::stock;
  }
  if (!fixed32_cutlass_b4_hybrid_n5120_projection(M, N, K) &&
      (wave_variant ==
           fixed32_cutlass_wave_variant::identity_hybrid_n5120_b4 ||
       wave_variant == fixed32_cutlass_wave_variant::
                           identity_hybrid_n5120_b4_byte_ab)) {
    wave_variant = fixed32_cutlass_wave_variant::stock;
  }
  if (M != 128 &&
      (wave_variant ==
           fixed32_cutlass_wave_variant::identity_stockshape_b4 ||
       wave_variant == fixed32_cutlass_wave_variant::
                           identity_stockshape_b4_byte_ab ||
       wave_variant == fixed32_cutlass_wave_variant::
                           identity_stockshape_stage2_b4 ||
       wave_variant == fixed32_cutlass_wave_variant::
                           identity_stockshape_stage2_b4_byte_ab)) {
    wave_variant = fixed32_cutlass_wave_variant::stock;
  }
  if (M != 128 &&
      (wave_variant == fixed32_cutlass_wave_variant::identity_divisor_b4 ||
       wave_variant ==
           fixed32_cutlass_wave_variant::identity_divisor_b4_byte_ab ||
       wave_variant ==
           fixed32_cutlass_wave_variant::identity_divisor_stage2_b4 ||
       wave_variant == fixed32_cutlass_wave_variant::
                           identity_divisor_stage2_b4_byte_ab ||
       wave_variant == fixed32_cutlass_wave_variant::identity_twom_b4 ||
       wave_variant ==
           fixed32_cutlass_wave_variant::identity_twom_b4_byte_ab)) {
    wave_variant = fixed32_cutlass_wave_variant::stock;
  }

  auto run_stream_k = [&](torch::stable::Tensor& destination) {
    if (M <= 64) {
      using Gemm =
          typename sm120_blockwise_fp8_config_swapab_streamk<OutType>::Gemm;
      return cutlass_gemm_caller_blockwise<Gemm>(
          destination, a, b, a_scales, b_scales);
    }
    using Gemm = typename
        sm120_blockwise_fp8_config_cooperative_streamk<OutType>::Gemm;
    return cutlass_gemm_caller_blockwise<Gemm>(
        destination, a, b, a_scales, b_scales);
  };

  auto run_stream_k_wide256 = [&](torch::stable::Tensor& destination) {
    using Gemm = typename
        sm120_blockwise_fp8_config_swapab_streamk_wide256<OutType>::Gemm;
    return cutlass_gemm_caller_blockwise<Gemm>(
        destination, a, b, a_scales, b_scales);
  };

  auto run_static_persistent_stocktile =
      [&](torch::stable::Tensor& destination) {
    using Gemm = typename
        sm120_blockwise_fp8_config_b1_static_persistent_stocktile<OutType>::Gemm;
    return cutlass_gemm_caller_blockwise<Gemm>(
        destination, a, b, a_scales, b_scales);
  };

  auto run_divisor_static_stocktile =
      [&](torch::stable::Tensor& destination) {
    using Gemm = typename
        sm120_blockwise_fp8_config_b1_divisor_static_stocktile<OutType>::Gemm;
    return cutlass_gemm_caller_blockwise<Gemm>(
        destination, a, b, a_scales, b_scales);
  };

  auto run_b4_persistent_m128 = [&](torch::stable::Tensor& destination) {
    using Gemm = typename
        sm120_blockwise_fp8_config_b4_persistent_m128<OutType>::Gemm;
    return cutlass_gemm_caller_blockwise<Gemm>(
        destination, a, b, a_scales, b_scales);
  };

  auto run_b4_persistent_m128_static =
      [&](torch::stable::Tensor& destination) {
    using Gemm = typename
        sm120_blockwise_fp8_config_b4_persistent_m128_static<OutType>::Gemm;
    return cutlass_gemm_caller_blockwise<Gemm>(
        destination, a, b, a_scales, b_scales);
  };

  auto run_identity_stage2_static = [&](torch::stable::Tensor& destination) {
    if (M == 32) {
      using Gemm = typename
          sm120_blockwise_fp8_config_b1_divisor_static_identity_stage2<
              OutType>::Gemm;
      return cutlass_gemm_caller_blockwise<Gemm>(
          destination, a, b, a_scales, b_scales);
    }
    using Gemm = typename
        sm120_blockwise_fp8_config_b4_m128_static_identity_stage2<
            OutType>::Gemm;
    return cutlass_gemm_caller_blockwise<Gemm>(
        destination, a, b, a_scales, b_scales);
  };

  auto run_identity_stage2_pingpong_b1 =
      [&](torch::stable::Tensor& destination) {
    if (N == 5120) {
      return run_identity_stage2_static(destination);
    }
    using Gemm = typename
        sm120_blockwise_fp8_config_b1_divisor_static_identity_pingpong_stage2<
            OutType>::Gemm;
    return cutlass_gemm_caller_blockwise<Gemm>(
        destination, a, b, a_scales, b_scales);
  };

  auto run_identity_onen_b1 = [&](torch::stable::Tensor& destination) {
    if (N == 5120) {
      using Gemm = typename
          sm120_blockwise_fp8_config_b1_onen_static_identity_stage2<
              OutType>::Gemm;
      return cutlass_gemm_caller_blockwise<Gemm>(
          destination, a, b, a_scales, b_scales);
    }
    using Gemm = typename
        sm120_blockwise_fp8_config_b1_onen_static_identity_pingpong_stage2<
            OutType>::Gemm;
    return cutlass_gemm_caller_blockwise<Gemm>(
        destination, a, b, a_scales, b_scales);
  };

  auto run_identity_onen_n5120_single_b1 =
      [&](torch::stable::Tensor& destination) {
    if (N == 5120) {
      using Gemm = typename
          sm120_blockwise_fp8_config_b1_n5120_single_identity_stage2<
              OutType>::Gemm;
      return cutlass_gemm_caller_blockwise<Gemm>(
          destination, a, b, a_scales, b_scales);
    }
    return run_identity_onen_b1(destination);
  };

  auto run_identity_onen_n5120_fullgrid_b1 =
      [&](torch::stable::Tensor& destination) {
    if (N == 5120) {
      using Gemm = typename
          sm120_blockwise_fp8_config_b1_n5120_single_identity_stage2<
              OutType>::Gemm;
      return cutlass_gemm_caller_blockwise<Gemm>(
          destination, a, b, a_scales, b_scales);
    }
    using Gemm = typename
        sm120_blockwise_fp8_config_b1_onen_fullgrid_identity_pingpong_stage2<
            OutType>::Gemm;
    return cutlass_gemm_caller_blockwise<Gemm>(
        destination, a, b, a_scales, b_scales);
  };

  auto run_identity_stockshape_b4 =
      [&](torch::stable::Tensor& destination) {
    using Gemm = typename
        sm120_blockwise_fp8_config_b4_stockshape_identity<OutType>::Gemm;
    return cutlass_gemm_caller_blockwise<Gemm>(
        destination, a, b, a_scales, b_scales);
  };

  auto run_identity_stockshape_stage2_b4 =
      [&](torch::stable::Tensor& destination) {
    using Gemm = typename
        sm120_blockwise_fp8_config_b4_stockshape_identity_stage2<
            OutType>::Gemm;
    return cutlass_gemm_caller_blockwise<Gemm>(
        destination, a, b, a_scales, b_scales);
  };

  auto run_identity_divisor_b4 =
      [&](torch::stable::Tensor& destination) {
    using Gemm = typename
        sm120_blockwise_fp8_config_b4_stockshape_identity_divisor<
            OutType>::Gemm;
    return cutlass_gemm_caller_blockwise<Gemm>(
        destination, a, b, a_scales, b_scales);
  };

  auto run_identity_divisor_stage2_b4 =
      [&](torch::stable::Tensor& destination) {
    using Gemm = typename
        sm120_blockwise_fp8_config_b4_stockshape_identity_divisor_stage2<
            OutType>::Gemm;
    return cutlass_gemm_caller_blockwise<Gemm>(
        destination, a, b, a_scales, b_scales);
  };

  auto run_identity_twom_b4 = [&](torch::stable::Tensor& destination) {
    using Gemm = typename
        sm120_blockwise_fp8_config_b4_stockshape_identity_twom<OutType>::Gemm;
    return cutlass_gemm_caller_blockwise<Gemm>(
        destination, a, b, a_scales, b_scales);
  };

  auto run_identity_hybrid_n5120_b4 =
      [&](torch::stable::Tensor& destination) {
    if (N == 5120) {
      using Gemm = typename
          sm120_blockwise_fp8_config_b4_m128_n5120_single_identity_stage2<
              OutType>::Gemm;
      return cutlass_gemm_caller_blockwise<Gemm>(
          destination, a, b, a_scales, b_scales);
    }
    return run_identity_twom_b4(destination);
  };

  auto run_stock = [&](torch::stable::Tensor& destination) {
    bool swap_ab = (M <= 64) || (M % 4 != 0);
    if (!swap_ab) {
      if (M <= 256) {
        using Gemm =
            typename sm120_blockwise_fp8_config_pingpong<OutType>::Gemm;
        return cutlass_gemm_caller_blockwise<Gemm>(
            destination, a, b, a_scales, b_scales);
      }
      using Gemm = typename sm120_blockwise_fp8_config_default<OutType>::Gemm;
      return cutlass_gemm_caller_blockwise<Gemm>(
          destination, a, b, a_scales, b_scales);
    }
    using Gemm = typename sm120_blockwise_fp8_config_swapab<OutType>::Gemm;
    return cutlass_gemm_caller_blockwise<Gemm>(
        destination, a, b, a_scales, b_scales);
  };

  const bool wide256_byte_ab =
      wave_variant ==
      fixed32_cutlass_wave_variant::stream_k_force_wide256_byte_ab;
  const bool static_persistent_byte_ab =
      wave_variant ==
      fixed32_cutlass_wave_variant::static_persistent_stocktile_byte_ab;
  const bool divisor_static_byte_ab =
      wave_variant ==
      fixed32_cutlass_wave_variant::divisor_static_stocktile_byte_ab;
  const bool b4_m128_byte_ab =
      wave_variant ==
      fixed32_cutlass_wave_variant::persistent_b4_m128_byte_ab;
  const bool b4_m128_static_byte_ab =
      wave_variant ==
      fixed32_cutlass_wave_variant::persistent_b4_m128_static_byte_ab;
  const bool identity_stage2_static_byte_ab =
      wave_variant ==
      fixed32_cutlass_wave_variant::identity_stage2_static_byte_ab;
  const bool identity_stage2_pingpong_b1_byte_ab =
      wave_variant == fixed32_cutlass_wave_variant::
                          identity_stage2_pingpong_b1_byte_ab;
  const bool identity_onen_b1_byte_ab =
      wave_variant == fixed32_cutlass_wave_variant::identity_onen_b1_byte_ab;
  const bool identity_onen_n5120_single_b1_byte_ab =
      wave_variant == fixed32_cutlass_wave_variant::
                          identity_onen_n5120_single_b1_byte_ab;
  const bool identity_onen_n5120_fullgrid_b1_byte_ab =
      wave_variant == fixed32_cutlass_wave_variant::
                          identity_onen_n5120_fullgrid_b1_byte_ab;
  const bool identity_stockshape_b4_byte_ab =
      wave_variant == fixed32_cutlass_wave_variant::
                          identity_stockshape_b4_byte_ab;
  const bool identity_stockshape_stage2_b4_byte_ab =
      wave_variant == fixed32_cutlass_wave_variant::
                          identity_stockshape_stage2_b4_byte_ab;
  const bool identity_divisor_b4_byte_ab =
      wave_variant ==
      fixed32_cutlass_wave_variant::identity_divisor_b4_byte_ab;
  const bool identity_divisor_stage2_b4_byte_ab =
      wave_variant == fixed32_cutlass_wave_variant::
                          identity_divisor_stage2_b4_byte_ab;
  const bool identity_twom_b4_byte_ab =
      wave_variant == fixed32_cutlass_wave_variant::identity_twom_b4_byte_ab;
  const bool identity_hybrid_n5120_b4_byte_ab =
      wave_variant == fixed32_cutlass_wave_variant::
                          identity_hybrid_n5120_b4_byte_ab;
  if (wave_variant ==
          fixed32_cutlass_wave_variant::stream_k_cooperative_128_byte_ab ||
      wide256_byte_ab || static_persistent_byte_ab ||
      divisor_static_byte_ab || b4_m128_byte_ab || b4_m128_static_byte_ab ||
      identity_stage2_static_byte_ab || identity_stage2_pingpong_b1_byte_ab ||
      identity_onen_b1_byte_ab || identity_onen_n5120_single_b1_byte_ab ||
      identity_onen_n5120_fullgrid_b1_byte_ab ||
      identity_stockshape_b4_byte_ab ||
      identity_stockshape_stage2_b4_byte_ab || identity_divisor_b4_byte_ab ||
      identity_divisor_stage2_b4_byte_ab || identity_twom_b4_byte_ab ||
      identity_hybrid_n5120_b4_byte_ab) {
    auto run_candidate = [&](torch::stable::Tensor& destination) {
      if (identity_onen_n5120_fullgrid_b1_byte_ab) {
        return run_identity_onen_n5120_fullgrid_b1(destination);
      }
      if (identity_onen_n5120_single_b1_byte_ab) {
        return run_identity_onen_n5120_single_b1(destination);
      }
      if (identity_onen_b1_byte_ab) {
        return run_identity_onen_b1(destination);
      }
      if (identity_hybrid_n5120_b4_byte_ab) {
        return run_identity_hybrid_n5120_b4(destination);
      }
      if (identity_twom_b4_byte_ab) {
        return run_identity_twom_b4(destination);
      }
      if (identity_divisor_stage2_b4_byte_ab) {
        return run_identity_divisor_stage2_b4(destination);
      }
      if (identity_divisor_b4_byte_ab) {
        return run_identity_divisor_b4(destination);
      }
      if (identity_stockshape_stage2_b4_byte_ab) {
        return run_identity_stockshape_stage2_b4(destination);
      }
      if (identity_stockshape_b4_byte_ab) {
        return run_identity_stockshape_b4(destination);
      }
      if (identity_stage2_pingpong_b1_byte_ab) {
        return run_identity_stage2_pingpong_b1(destination);
      }
      if (identity_stage2_static_byte_ab) {
        return run_identity_stage2_static(destination);
      }
      if (b4_m128_static_byte_ab) {
        return run_b4_persistent_m128_static(destination);
      }
      if (b4_m128_byte_ab) {
        return run_b4_persistent_m128(destination);
      }
      if (wide256_byte_ab) {
        return run_stream_k_wide256(destination);
      }
      if (static_persistent_byte_ab) {
        return run_static_persistent_stocktile(destination);
      }
      if (divisor_static_byte_ab) {
        return run_divisor_static_stocktile(destination);
      }
      return run_stream_k(destination);
    };
    // Boot/profile forwards are not authenticated real-task work. Keep them
    // entirely on stock; candidate execution starts only after the arm exists.
    std::string task_marker =
        (b4_m128_byte_ab || b4_m128_static_byte_ab ||
         (identity_stage2_static_byte_ab && M == 128) ||
         identity_stockshape_b4_byte_ab ||
         identity_stockshape_stage2_b4_byte_ab ||
         identity_divisor_b4_byte_ab ||
         identity_divisor_stage2_b4_byte_ab || identity_twom_b4_byte_ab ||
         identity_hybrid_n5120_b4_byte_ab)
            ? fixed32_cutlass_b4_real_task_marker()
            : fixed32_cutlass_real_task_marker();
    if (task_marker.empty()) {
      return run_stock(out);
    }

    // Diagnostic only: compare the first bounded set of armed real-task calls
    // in one process and CUDA stream, then always serve the stock result.
    static std::atomic<int64_t> next_invocation{0};
    constexpr int64_t byte_ab_limit = 320;
    constexpr int64_t b4_m128_byte_ab_limit = 320;
    const int64_t selected_byte_ab_limit =
        (b4_m128_byte_ab || b4_m128_static_byte_ab ||
         (identity_stage2_static_byte_ab && M == 128) ||
         identity_stockshape_b4_byte_ab ||
         identity_stockshape_stage2_b4_byte_ab ||
         identity_divisor_b4_byte_ab ||
         identity_divisor_stage2_b4_byte_ab || identity_twom_b4_byte_ab ||
         identity_hybrid_n5120_b4_byte_ab)
            ? b4_m128_byte_ab_limit
            : byte_ab_limit;
    int64_t invocation = next_invocation.fetch_add(1);
    if (invocation >= selected_byte_ab_limit) {
      return run_stock(out);
    }

    torch::stable::Tensor candidate = torch::stable::empty_like(out);
    run_stock(out);
    run_candidate(candidate);

    const size_t output_bytes =
        static_cast<size_t>(out.numel()) * out.element_size();
    std::vector<unsigned char> stock_host(output_bytes);
    std::vector<unsigned char> candidate_host(output_bytes);
    auto stream = get_current_cuda_stream(a.device().index());
    cudaError_t status = cudaMemcpyAsync(
        stock_host.data(), out.const_data_ptr(), output_bytes,
        cudaMemcpyDeviceToHost, stream);
    STD_TORCH_CHECK(status == cudaSuccess,
                    "FR13 Stream-K stock D2H failed: ",
                    cudaGetErrorString(status));
    status = cudaMemcpyAsync(
        candidate_host.data(), candidate.const_data_ptr(), output_bytes,
        cudaMemcpyDeviceToHost, stream);
    STD_TORCH_CHECK(status == cudaSuccess,
                    "FR13 Stream-K candidate D2H failed: ",
                    cudaGetErrorString(status));
    status = cudaStreamSynchronize(stream);
    STD_TORCH_CHECK(status == cudaSuccess,
                    "FR13 Stream-K byte A/B synchronize failed: ",
                    cudaGetErrorString(status));

    size_t first_mismatch = output_bytes;
    size_t mismatch_count = 0;
    for (size_t index = 0; index < output_bytes; ++index) {
      if (stock_host[index] != candidate_host[index]) {
        if (first_mismatch == output_bytes) {
          first_mismatch = index;
        }
        ++mismatch_count;
      }
    }
    const char* log_path =
        identity_onen_n5120_fullgrid_b1_byte_ab
            ? "/logs/fr13_fixed32_cutlass_identity_onen_n5120_fullgrid_b1_byte_ab.jsonl"
        : identity_onen_n5120_single_b1_byte_ab
            ? "/logs/fr13_fixed32_cutlass_identity_onen_n5120_single_b1_byte_ab.jsonl"
        : identity_onen_b1_byte_ab
            ? "/logs/fr13_fixed32_cutlass_identity_onen_b1_byte_ab.jsonl"
        : identity_hybrid_n5120_b4_byte_ab
            ? "/logs/fr13_fixed32_cutlass_identity_hybrid_n5120_b4_byte_ab.jsonl"
        : identity_twom_b4_byte_ab
            ? "/logs/fr13_fixed32_cutlass_identity_twom_b4_byte_ab.jsonl"
        : identity_divisor_stage2_b4_byte_ab
            ? "/logs/fr13_fixed32_cutlass_identity_divisor_stage2_b4_byte_ab.jsonl"
        : identity_divisor_b4_byte_ab
            ? "/logs/fr13_fixed32_cutlass_identity_divisor_b4_byte_ab.jsonl"
        : identity_stockshape_stage2_b4_byte_ab
            ? "/logs/fr13_fixed32_cutlass_identity_stockshape_stage2_b4_byte_ab.jsonl"
        : identity_stockshape_b4_byte_ab
            ? "/logs/fr13_fixed32_cutlass_identity_stockshape_b4_byte_ab.jsonl"
        : identity_stage2_pingpong_b1_byte_ab
            ? "/logs/fr13_fixed32_cutlass_identity_stage2_pingpong_b1_byte_ab.jsonl"
        : identity_stage2_static_byte_ab
            ? "/logs/fr13_fixed32_cutlass_identity_stage2_static_byte_ab.jsonl"
        : b4_m128_static_byte_ab
            ? "/logs/fr13_fixed32_cutlass_persistent_b4_m128_static_byte_ab.jsonl"
        : b4_m128_byte_ab
            ? "/logs/fr13_fixed32_cutlass_persistent_b4_m128_byte_ab.jsonl"
        : wide256_byte_ab
            ? "/logs/fr13_fixed32_cutlass_streamk_wide256_byte_ab.jsonl"
        : static_persistent_byte_ab
            ? "/logs/fr13_fixed32_cutlass_static_persistent_byte_ab.jsonl"
        : divisor_static_byte_ab
            ? "/logs/fr13_fixed32_cutlass_divisor_static_byte_ab.jsonl"
            : "/logs/fr13_fixed32_cutlass_streamk_byte_ab.jsonl";
    static std::mutex log_mutex;
    {
      std::lock_guard<std::mutex> lock(log_mutex);
      std::ofstream log(log_path, std::ios::app);
      STD_TORCH_CHECK(log.good(),
                      "FR13 Stream-K byte A/B could not open JSONL");
      log << "{\\\"schema\\\":\\\""
          << (identity_onen_n5120_fullgrid_b1_byte_ab
                  ? "fr13.fixed32.cutlass_identity_onen_n5120_fullgrid_b1_byte_ab.v1"
              : identity_onen_n5120_single_b1_byte_ab
                  ? "fr13.fixed32.cutlass_identity_onen_n5120_single_b1_byte_ab.v1"
              : identity_onen_b1_byte_ab
                  ? "fr13.fixed32.cutlass_identity_onen_b1_byte_ab.v1"
              : identity_hybrid_n5120_b4_byte_ab
                  ? "fr13.fixed32.cutlass_identity_hybrid_n5120_b4_byte_ab.v1"
              : identity_twom_b4_byte_ab
                  ? "fr13.fixed32.cutlass_identity_twom_b4_byte_ab.v1"
              : identity_divisor_stage2_b4_byte_ab
                  ? "fr13.fixed32.cutlass_identity_divisor_stage2_b4_byte_ab.v1"
              : identity_divisor_b4_byte_ab
                  ? "fr13.fixed32.cutlass_identity_divisor_b4_byte_ab.v1"
              : identity_stockshape_stage2_b4_byte_ab
                  ? "fr13.fixed32.cutlass_identity_stockshape_stage2_b4_byte_ab.v1"
              : identity_stockshape_b4_byte_ab
                  ? "fr13.fixed32.cutlass_identity_stockshape_b4_byte_ab.v1"
              : identity_stage2_pingpong_b1_byte_ab
                  ? "fr13.fixed32.cutlass_identity_stage2_pingpong_b1_byte_ab.v1"
              : identity_stage2_static_byte_ab
                  ? "fr13.fixed32.cutlass_identity_stage2_static_byte_ab.v1"
              : b4_m128_static_byte_ab
                  ? "fr13.fixed32.cutlass_persistent_b4_m128_static_byte_ab.v1"
              : b4_m128_byte_ab
                  ? "fr13.fixed32.cutlass_persistent_b4_m128_byte_ab.v1"
              : wide256_byte_ab
                  ? "fr13.fixed32.cutlass_streamk_wide256_byte_ab.v1"
              : static_persistent_byte_ab
                  ? "fr13.fixed32.cutlass_static_persistent_byte_ab.v1"
              : divisor_static_byte_ab
                  ? "fr13.fixed32.cutlass_divisor_static_byte_ab.v1"
                  : "fr13.fixed32.cutlass_streamk_byte_ab.v2")
          << "\\\","
          << "\\\"invocation\\\":" << invocation << ","
          << "\\\"task_marker\\\":\\\"" << task_marker << "\\\","
          << "\\\"m\\\":" << M << ",\\\"n\\\":" << N
          << ",\\\"k\\\":" << K << ",\\\"bytes\\\":" << output_bytes
          << ",\\\"byte_equal\\\":"
          << (mismatch_count == 0 ? "true" : "false")
          << ",\\\"mismatch_count\\\":" << mismatch_count
          << ",\\\"first_mismatch\\\":";
      if (first_mismatch == output_bytes) {
        log << "null";
      } else {
        log << first_mismatch;
      }
      log << "}\\n";
      log.flush();
      STD_TORCH_CHECK(log.good(),
                      "FR13 Stream-K byte A/B JSONL write failed");
    }
    return;
  }

  if (wave_variant ==
      fixed32_cutlass_wave_variant::stream_k_cooperative_128) {
    return run_stream_k(out);
  }

  if (wave_variant ==
      fixed32_cutlass_wave_variant::stream_k_force_wide256) {
    return run_stream_k_wide256(out);
  }

  if (wave_variant ==
      fixed32_cutlass_wave_variant::static_persistent_stocktile) {
    return run_static_persistent_stocktile(out);
  }

  if (wave_variant ==
      fixed32_cutlass_wave_variant::divisor_static_stocktile) {
    return run_divisor_static_stocktile(out);
  }

  if (wave_variant ==
      fixed32_cutlass_wave_variant::persistent_b4_m128) {
    return run_b4_persistent_m128(out);
  }

  if (wave_variant ==
      fixed32_cutlass_wave_variant::persistent_b4_m128_static) {
    return run_b4_persistent_m128_static(out);
  }

  if (wave_variant ==
      fixed32_cutlass_wave_variant::identity_stage2_static) {
    return run_identity_stage2_static(out);
  }

  if (wave_variant ==
      fixed32_cutlass_wave_variant::identity_stage2_pingpong_b1) {
    return run_identity_stage2_pingpong_b1(out);
  }

  if (wave_variant == fixed32_cutlass_wave_variant::identity_onen_b1) {
    return run_identity_onen_b1(out);
  }

  if (wave_variant == fixed32_cutlass_wave_variant::
                          identity_onen_n5120_single_b1) {
    return run_identity_onen_n5120_single_b1(out);
  }

  if (wave_variant == fixed32_cutlass_wave_variant::
                          identity_onen_n5120_fullgrid_b1) {
    return run_identity_onen_n5120_fullgrid_b1(out);
  }

  if (wave_variant ==
      fixed32_cutlass_wave_variant::identity_stockshape_b4) {
    return run_identity_stockshape_b4(out);
  }

  if (wave_variant ==
      fixed32_cutlass_wave_variant::identity_stockshape_stage2_b4) {
    return run_identity_stockshape_stage2_b4(out);
  }

  if (wave_variant ==
      fixed32_cutlass_wave_variant::identity_divisor_b4) {
    return run_identity_divisor_b4(out);
  }

  if (wave_variant ==
      fixed32_cutlass_wave_variant::identity_divisor_stage2_b4) {
    return run_identity_divisor_stage2_b4(out);
  }

  if (wave_variant == fixed32_cutlass_wave_variant::identity_twom_b4) {
    return run_identity_twom_b4(out);
  }

  if (wave_variant ==
      fixed32_cutlass_wave_variant::identity_hybrid_n5120_b4) {
    return run_identity_hybrid_n5120_b4(out);
  }

  // Unset/unknown selectors retain the stock kernel and numeric result.
  return run_stock(out);
"""


def patch_text(source: str) -> tuple[str, bool]:
    """Return patched dispatch source and whether it changed."""
    if MARKER in source:
        required = (
            INCLUDE_REPLACEMENT,
            SCHEDULER_SPECIALIZATION_REPLACEMENT,
            STREAMK_CLASS_REPLACEMENT,
            CONFIG_REPLACEMENT,
            CALLER_REPLACEMENT,
            DISPATCH_REPLACEMENT,
        )
        if not all(fragment in source for fragment in required):
            raise RuntimeError("partial FR13 fixed32 CUTLASS wave patch found")
        return source, False

    single_anchors = {
        "include": INCLUDE_ANCHOR,
        "static scheduler specialization": SCHEDULER_SPECIALIZATION_ANCHOR,
        "GEMM template": TEMPLATE_ANCHOR,
        "stock kernel class": STREAMK_CLASS_ANCHOR,
        "candidate insertion": CONFIG_ANCHOR,
        "Stream-K caller": CALLER_ANCHOR,
        "dispatch": DISPATCH_ANCHOR,
    }
    for label, anchor in single_anchors.items():
        count = source.count(anchor)
        if count != 1:
            raise RuntimeError(f"expected exactly one {label} anchor, found {count}")
    stage_count = source.count(STAGE_COUNT_ANCHOR)
    if stage_count != 2:
        raise RuntimeError(
            f"expected exactly two mainloop stage-count anchors, found {stage_count}"
        )
    patched = source.replace(INCLUDE_ANCHOR, INCLUDE_REPLACEMENT, 1)
    patched = patched.replace(
        SCHEDULER_SPECIALIZATION_ANCHOR,
        SCHEDULER_SPECIALIZATION_REPLACEMENT,
        1,
    )
    patched = patched.replace(
        STREAMK_CLASS_ANCHOR, STREAMK_CLASS_REPLACEMENT, 1
    )
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
