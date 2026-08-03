from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest


def _module():
    path = Path("scripts/fr13_patch_cutlass_fixed32_wave.py")
    spec = importlib.util.spec_from_file_location("fr13_cutlass_wave_patch", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_fixture(module) -> str:
    return f"""{module.INCLUDE_ANCHOR}
{module.SCHEDULER_SPECIALIZATION_ANCHOR}
using namespace cute;

template <class OutType, int ScaleGranularityM,
          int ScaleGranularityN, int ScaleGranularityK,
          class MmaTileShape, class ClusterShape,
{module.TEMPLATE_ANCHOR}  static constexpr bool swap_ab = swap_ab_;
  using CollectiveEpilogue = FakeEpilogue;
  using StageCountType = cutlass::gemm::collective::StageCountAuto;
  using CollectiveMainloop = conditional_t<swap_ab,
      FakeBuilder<
          {module.STAGE_COUNT_ANCHOR},
          MainloopScheduler
      >::CollectiveOp,
      FakeBuilder<
          {module.STAGE_COUNT_ANCHOR},
          MainloopScheduler
      >::CollectiveOp>;
{module.KERNEL_ANCHOR}
  struct GemmKernel : public KernelType {{}};
}};

template <typename OutType>
struct sm120_blockwise_fp8_config_swapab {{}};

{module.CONFIG_ANCHOR}                                   torch::stable::Tensor const& b,
                                   torch::stable::Tensor const& a_scales,
                                   torch::stable::Tensor const& b_scales) {{
  using GemmKernel = typename Gemm::GemmKernel;
  auto prob_shape = FakeProblemShape{{}};
  auto mainloop_args = FakeMainloopArguments{{}};
  auto epilogue_args = FakeEpilogueArguments{{}};
{module.CALLER_ANCHOR}}}

template <typename OutType>
void cutlass_gemm_blockwise_sm120_fp8_dispatch(
    torch::stable::Tensor& out, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b, torch::stable::Tensor const& a_scales,
    torch::stable::Tensor const& b_scales) {{
{module.DISPATCH_ANCHOR}}}

}}  // namespace vllm
"""


def test_patch_targets_exact_libtorch_stable_dispatch() -> None:
    module = _module()

    assert module.TARGET_RELATIVE_PATH == Path(
        "csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/"
        "scaled_mm_blockwise_sm120_fp8_dispatch.cuh"
    )
    assert module.EXPECTED_UNPATCHED_SHA256 == (
        "6e1df3f4701f58f233b3831b848c7bbf7936e6cb34b3bc28ded208fd66c48a7f"
    )
    assert module.EXPECTED_CMAKE_SHA256 == (
        "b12cd47f5761442551d6e1966e8a37ad94175382c1b014d2b65f67b74fbb6e3b"
    )
    assert module.CUTLASS_TAG_COMMIT == (
        "da5e086dab31d63815acafdac9a9c5893b1c69e2"
    )


def test_patch_is_default_off_and_shape_gated() -> None:
    module = _module()
    patched, changed = module.patch_text(_source_fixture(module))

    assert changed
    assert 'std::getenv("FR13_FIXED32_CUTLASS_WAVE")' in patched
    assert '"/logs/fr13_fixed32_cutlass_wave.selector"' in patched
    assert 'std::strcmp(value, "streamk_coop64") == 0' not in patched
    assert 'value == "streamk_coop128"' in patched
    assert 'value == "streamk_coop128_byte_ab"' in patched
    assert 'value == "streamk_force_wide256"' in patched
    assert 'value == "streamk_force_wide256_byte_ab"' in patched
    assert 'value == "static_persistent_stocktile"' in patched
    assert 'value == "static_persistent_stocktile_byte_ab"' in patched
    assert 'value == "divisor_static_stocktile"' in patched
    assert 'value == "divisor_static_stocktile_byte_ab"' in patched
    assert 'value == "persistent_b4_m128"' in patched
    assert 'value == "persistent_b4_m128_byte_ab"' in patched
    assert 'value == "persistent_b4_m128_static"' in patched
    assert 'value == "persistent_b4_m128_static_byte_ab"' in patched
    assert 'value == "identity_stage2_static"' in patched
    assert 'value == "identity_stage2_static_byte_ab"' in patched
    assert 'value == "identity_stage2_pingpong_b1"' in patched
    assert 'value == "identity_stage2_pingpong_b1_byte_ab"' in patched
    assert 'value == "identity_onen_b1"' in patched
    assert 'value == "identity_onen_b1_byte_ab"' in patched
    assert 'value == "identity_stockshape_b4"' in patched
    assert 'value == "identity_stockshape_b4_byte_ab"' in patched
    assert 'value == "identity_stockshape_stage2_b4"' in patched
    assert 'value == "identity_stockshape_stage2_b4_byte_ab"' in patched
    assert 'value == "identity_divisor_b4"' in patched
    assert 'value == "identity_divisor_b4_byte_ab"' in patched
    assert 'value == "identity_hybrid_n5120_b4"' in patched
    assert 'value == "identity_hybrid_n5120_b4_byte_ab"' in patched
    assert "return fixed32_cutlass_wave_variant::stock;" in patched
    for rows in (32, 64, 96, 128):
        assert f"m == {rows}" in patched
    for n, k in (
        (34816, 5120),
        (5120, 17408),
        (5120, 6144),
        (16384, 5120),
        (14336, 5120),
    ):
        assert f"n == {n} && k == {k}" in patched
    assert "n == 8192 && k == 5120" not in patched


def test_candidates_keep_scale_k_tile_cluster_and_numeric_math() -> None:
    module = _module()
    patched, _ = module.patch_text(_source_fixture(module))

    assert patched.count("cutlass::gemm::StreamKScheduler") == 2
    assert patched.count("using ClusterShape = Shape<_1, _1, _1>;") == 21
    assert (
        module.CONFIG_REPLACEMENT.count(
            "KernelTmaWarpSpecializedBlockwisePingpongSm120"
        )
        == 8
    )
    assert "OutType, 128, 1, 128, TileShape, ClusterShape" in patched
    assert "using TileShape = Shape<_128, _32, _128>;" in patched
    assert "OutType, 1, 128, 128, TileShape, ClusterShape" in patched
    assert "using TileShape = Shape<_128, _128, _128>;" in patched
    assert "using TileShape = Shape<_64, _256, _128>;" not in patched
    assert "using TileShape = Shape<_256, _32, _128>;" in patched
    assert (
        "OutType, 128, 1, 128, TileShape, ClusterShape,\n"
        "      EpilogueSchedule, KernelSchedule, true,\n"
        "      fr13_fixed32_wide256_recompute_scheduler, true,\n"
        "      cutlass::gemm::collective::StageCount<2>>"
    ) in patched
    assert "using TileShape = Shape<_64, _128, _128>;" in patched
    assert "MainloopStageCount" in patched
    assert "cutlass::gemm::collective::StageCount<2>" in patched
    assert "ElementAccumulator = float" not in module.CONFIG_REPLACEMENT
    assert "ElementD" not in module.CONFIG_REPLACEMENT


def test_b4_m128_preserves_stock_template_and_is_exactly_m128_gated() -> None:
    module = _module()
    patched, _ = module.patch_text(_source_fixture(module))

    config_start = patched.index(
        "struct sm120_blockwise_fp8_config_b4_persistent_m128"
    )
    config_end = patched.index("enum class fixed32_cutlass_wave_variant", config_start)
    config = patched[config_start:config_end]
    assert "using TileShape = Shape<_128, _128, _128>;" in config
    assert "KernelTmaWarpSpecializedBlockwiseCooperativeSm120" in config
    assert "cutlass_3x_gemm_fp8_blockwise<" in config
    assert "cutlass_3x_gemm_fp8_blockwise_streamk" not in config
    assert "StreamKScheduler" not in config
    assert "use_stream_k" not in config
    assert "OutType, 1, 128, 128, TileShape, ClusterShape" in config
    assert "EpilogueSchedule, KernelSchedule, false>" in config
    assert "ElementAccumulator" not in config
    assert "ElementCompute" not in config
    assert "struct Gemm : BaseGemm" in config
    assert "struct GemmKernel : public KernelType" in config
    assert "if (M != 128 &&" in patched
    assert "persistent_b4_m128_byte_ab" in patched
    assert "fixed32_cutlass_b4_real_task_marker()" in patched
    assert (
        '"/logs/fr13_fixed32_cutlass_b4_byte_ab.real_event.arm"' in patched
    )


def test_b1_static_persistent_reuses_stock_collective_and_generic_scheduler() -> None:
    module = _module()
    patched, _ = module.patch_text(_source_fixture(module))

    config_start = patched.index(
        "struct sm120_blockwise_fp8_config_b1_static_persistent_stocktile"
    )
    config_end = patched.index(
        "struct sm120_blockwise_fp8_config_b4_persistent_m128", config_start
    )
    config = patched[config_start:config_end]
    assert "using TileShape = Shape<_128, _32, _128>;" in config
    assert "using ClusterShape = Shape<_1, _1, _1>;" in config
    assert "KernelTmaWarpSpecializedBlockwiseCooperativeSm120" in config
    assert "OutType, 128, 1, 128, TileShape, ClusterShape" in config
    assert "EpilogueSchedule, KernelSchedule, true>" in config
    assert "cutlass_3x_gemm_fp8_blockwise_m128_static" in config
    assert "StreamKScheduler" not in config
    assert "StageCount" not in config

    selector_gate = patched.index("if (M > 64 &&")
    stock_assignment = patched.index(
        "wave_variant = fixed32_cutlass_wave_variant::stock;", selector_gate
    )
    guard = patched[selector_gate:stock_assignment]
    assert "static_persistent_stocktile" in guard
    assert "static_persistent_stocktile_byte_ab" in guard
    assert "run_static_persistent_stocktile" in patched
    assert (
        '"/logs/fr13_fixed32_cutlass_static_persistent_byte_ab.jsonl"'
        in patched
    )
    assert "fr13.fixed32.cutlass_static_persistent_byte_ab.v1" in patched


def test_b1_divisor_static_balances_real_projection_tile_counts() -> None:
    module = _module()
    patched, _ = module.patch_text(_source_fixture(module))

    assert "struct fr13_fixed32_m128_divisor_static_scheduler {};" in patched
    assert "class Fr13DivisorBalancedStaticTileScheduler100" in patched
    assert "constexpr uint32_t MinBalancedCtas = 28;" in patched
    assert "logical_tiles % candidate == 0" in patched
    assert "using Scheduler = Fr13DivisorBalancedStaticTileScheduler100;" in patched
    assert "if (M != 32 &&" in patched
    assert "run_divisor_static_stocktile" in patched
    assert '"/logs/fr13_fixed32_cutlass_divisor_static_byte_ab.jsonl"' in patched
    assert "fr13.fixed32.cutlass_divisor_static_byte_ab.v1" in patched

    # Pinned M32 tile counts and the corresponding widest divisors in [28, 48].
    expected = {40: 40, 112: 28, 128: 32, 272: 34}
    for logical_tiles, grid_ctas in expected.items():
        selected = next(
            candidate
            for candidate in range(min(48, logical_tiles), 27, -1)
            if logical_tiles % candidate == 0
        )
        assert selected == grid_ctas

    config_start = patched.index(
        "struct sm120_blockwise_fp8_config_b1_divisor_static_stocktile"
    )
    config_end = patched.index(
        "struct sm120_blockwise_fp8_config_b4_persistent_m128", config_start
    )
    config = patched[config_start:config_end]
    assert "using TileShape = Shape<_128, _32, _128>;" in config
    assert "using ClusterShape = Shape<_1, _1, _1>;" in config
    assert "OutType, 128, 1, 128, TileShape, ClusterShape" in config
    assert "cutlass_3x_gemm_fp8_blockwise_m128_divisor_static" in config
    assert "StreamKScheduler" not in config


def test_b4_m128_static_changes_only_complete_tile_scheduler() -> None:
    module = _module()
    patched, _ = module.patch_text(_source_fixture(module))

    assert "struct fr13_fixed32_m128_static_scheduler {};" in patched
    assert "using Scheduler = StaticPersistentTileScheduler100;" in patched
    assert "arch::Sm120" in module.SCHEDULER_SPECIALIZATION_REPLACEMENT

    config_start = patched.index(
        "struct sm120_blockwise_fp8_config_b4_persistent_m128_static"
    )
    config_end = patched.index(
        "struct sm120_blockwise_fp8_config_b1_divisor_static_identity_stage2",
        config_start,
    )
    config = patched[config_start:config_end]
    assert "using TileShape = Shape<_128, _128, _128>;" in config
    assert "using ClusterShape = Shape<_1, _1, _1>;" in config
    assert "KernelTmaWarpSpecializedBlockwiseCooperativeSm120" in config
    assert "OutType, 1, 128, 128, TileShape, ClusterShape" in config
    assert "EpilogueSchedule, KernelSchedule, false>" in config
    assert "StreamKScheduler" not in config
    assert "StageCount" not in config
    assert "ElementAccumulator" not in config
    assert "ElementCompute" not in config

    candidate_class_start = patched.index(
        "struct cutlass_3x_gemm_fp8_blockwise_m128_static"
    )
    candidate_class_end = patched.index(module.CONFIG_ANCHOR, candidate_class_start)
    candidate_class = patched[candidate_class_start:candidate_class_end]
    assert "typename Base::CollectiveMainloop" in candidate_class
    assert "typename Base::CollectiveEpilogue" in candidate_class
    assert "fr13_fixed32_m128_static_scheduler" in candidate_class

    m128_gate = patched[patched.index("if (M != 128 &&"):]
    assert "persistent_b4_m128_static" in m128_gate
    assert "persistent_b4_m128_static_byte_ab" in m128_gate
    assert "run_b4_persistent_m128_static" in patched
    assert "fixed32_cutlass_b4_real_task_marker()" in patched
    assert (
        '"/logs/fr13_fixed32_cutlass_persistent_b4_m128_static_byte_ab.jsonl"'
        in patched
    )
    assert (
        "fr13.fixed32.cutlass_persistent_b4_m128_static_byte_ab.v1" in patched
    )


def test_identity_stage2_uses_identity_epilogue_across_batches() -> None:
    module = _module()
    patched, _ = module.patch_text(_source_fixture(module))

    wrapper_start = patched.index(
        "struct cutlass_3x_gemm_fp8_blockwise_identity_static"
    )
    wrapper_end = patched.index(module.CONFIG_ANCHOR, wrapper_start)
    wrapper = patched[wrapper_start:wrapper_end]

    assert "using CollectiveMainloop = conditional_t<" in wrapper
    assert "typename Base::CollectiveMainloop" not in wrapper
    assert "class MainloopStageCount = void>\nstruct cutlass_3x" in patched
    assert "using ResolvedMainloopStageCount = conditional_t<" in wrapper
    assert "std::is_void_v<MainloopStageCount>" in wrapper
    assert "StageCountAutoCarveout<" in wrapper
    assert "sizeof(typename CollectiveEpilogue::SharedStorage)" in wrapper
    assert "ClusterShape, ResolvedMainloopStageCount," in wrapper
    assert "using CollectiveEpilogue" in wrapper
    assert "typename Base::CollectiveEpilogue" not in wrapper
    assert "cutlass::epilogue::thread::Identity" in wrapper
    assert "cutlass::FloatRoundStyle::round_to_nearest" in wrapper
    assert "typename Base::ElementAccumulator" in wrapper
    assert "cutlass::multiplies" not in wrapper
    assert "fr13_fixed32_one_scalar_broadcast" not in patched
    assert "CollectiveEpilogue, TileScheduler" in wrapper
    assert wrapper.index("using CollectiveEpilogue") < wrapper.index(
        "using ResolvedMainloopStageCount"
    ) < wrapper.index("using CollectiveMainloop")

    b1_start = patched.index(
        "struct sm120_blockwise_fp8_config_b1_divisor_static_identity"
    )
    b4_start = patched.index(
        "struct sm120_blockwise_fp8_config_b4_m128_static_identity"
    )
    config_end = patched.index("enum class fixed32_cutlass_wave_variant", b4_start)
    b1 = patched[b1_start:b4_start]
    b4 = patched[b4_start:config_end]
    assert "using TileShape = Shape<_128, _32, _128>;" in b1
    assert "OutType, 128, 1, 128, TileShape, ClusterShape" in b1
    assert "EpilogueSchedule, KernelSchedule, true," in b1
    assert "fr13_fixed32_m128_divisor_static_scheduler" in b1
    assert "cutlass::gemm::collective::StageCount<2>>" in b1
    assert "using TileShape = Shape<_128, _128, _128>;" in b4
    assert "OutType, 1, 128, 128, TileShape, ClusterShape" in b4
    assert "EpilogueSchedule, KernelSchedule, false," in b4
    assert "fr13_fixed32_m128_static_scheduler" in b4
    assert "cutlass::gemm::collective::StageCount<2>>" in b4
    assert "StreamKScheduler" not in b1 + b4

    assert "if (M != 32 && M != 128 &&" in patched
    assert "auto run_identity_stage2_static" in patched
    assert "if (M == 32)" in patched
    assert "identity_stage2_static_byte_ab && M == 128" in patched
    assert (
        '"/logs/fr13_fixed32_cutlass_identity_stage2_static_byte_ab.jsonl"'
        in patched
    )
    assert "fr13.fixed32.cutlass_identity_stage2_static_byte_ab.v1" in patched
    assert (
        "fixed32_cutlass_wave_variant::identity_stage2_static) {\n"
        "    return run_identity_stage2_static(out);"
        in patched
    )


def test_identity_stage2_pingpong_is_b1_only_and_preserves_full_k() -> None:
    module = _module()
    patched, _ = module.patch_text(_source_fixture(module))
    config_start = patched.index(
        "struct sm120_blockwise_fp8_config_b1_divisor_static_identity_pingpong_stage2"
    )
    config_end = patched.index("};", config_start)
    config = patched[config_start:config_end]

    assert "KernelTmaWarpSpecializedBlockwisePingpongSm120" in config
    assert "Shape<_128, _32, _128>" in config
    assert "fr13_fixed32_m128_divisor_static_scheduler" in config
    assert "cutlass::gemm::collective::StageCount<2>" in config
    assert "StreamK" not in config
    assert "if (M != 32 &&" in patched
    assert "auto run_identity_stage2_pingpong_b1" in patched
    assert "if (N == 5120)" in patched
    assert "return run_identity_stage2_static(destination);" in patched
    assert (
        '"/logs/fr13_fixed32_cutlass_identity_stage2_pingpong_b1_byte_ab.jsonl"'
        in patched
    )
    assert (
        "fr13.fixed32.cutlass_identity_stage2_pingpong_b1_byte_ab.v1" in patched
    )
    assert (
        "fixed32_cutlass_wave_variant::identity_stage2_pingpong_b1) {\n"
        "    return run_identity_stage2_pingpong_b1(out);"
        in patched
    )


def test_b4_stockshape_identity_keeps_stock_shape_and_scheduling() -> None:
    module = _module()
    patched, _ = module.patch_text(_source_fixture(module))
    config_start = patched.index(
        "struct sm120_blockwise_fp8_config_b4_stockshape_identity"
    )
    config_end = patched.index(
        "struct sm120_blockwise_fp8_config_b4_stockshape_identity_stage2",
        config_start,
    )
    config = patched[config_start:config_end]

    assert "KernelTmaWarpSpecializedBlockwisePingpongSm120" in config
    assert "using TileShape = Shape<_64, _128, _128>;" in config
    assert "OutType, 1, 128, 128, TileShape, ClusterShape" in config
    assert "EpilogueSchedule, KernelSchedule, false, void>" in config
    assert "StageCount<" not in config
    assert "StreamK" not in config
    assert "fr13_fixed32_m128_static_scheduler" not in config
    assert "fr13_fixed32_m128_divisor_static_scheduler" not in config
    assert "if (M != 128 &&" in patched
    assert "auto run_identity_stockshape_b4" in patched
    assert "fixed32_cutlass_b4_real_task_marker()" in patched
    assert (
        '"/logs/fr13_fixed32_cutlass_identity_stockshape_b4_byte_ab.jsonl"'
        in patched
    )
    assert "fr13.fixed32.cutlass_identity_stockshape_b4_byte_ab.v1" in patched
    assert (
        "fixed32_cutlass_wave_variant::identity_stockshape_b4) {\n"
        "    return run_identity_stockshape_b4(out);"
        in patched
    )


def test_b4_stockshape_stage2_isolates_pipeline_depth() -> None:
    module = _module()
    patched, _ = module.patch_text(_source_fixture(module))
    config_start = patched.index(
        "struct sm120_blockwise_fp8_config_b4_stockshape_identity_stage2"
    )
    config_end = patched.index(
        "struct sm120_blockwise_fp8_config_b4_stockshape_identity_divisor",
        config_start,
    )
    config = patched[config_start:config_end]

    assert "KernelTmaWarpSpecializedBlockwisePingpongSm120" in config
    assert "using TileShape = Shape<_64, _128, _128>;" in config
    assert "OutType, 1, 128, 128, TileShape, ClusterShape" in config
    assert "EpilogueSchedule, KernelSchedule, false, void," in config
    assert "cutlass::gemm::collective::StageCount<2>" in config
    assert "StreamK" not in config
    assert "fr13_fixed32_m128_static_scheduler" not in config
    assert "fr13_fixed32_m128_divisor_static_scheduler" not in config
    assert "auto run_identity_stockshape_stage2_b4" in patched
    assert (
        '"/logs/fr13_fixed32_cutlass_identity_stockshape_stage2_b4_byte_ab.jsonl"'
        in patched
    )
    assert (
        "fr13.fixed32.cutlass_identity_stockshape_stage2_b4_byte_ab.v1"
        in patched
    )
    assert (
        "fixed32_cutlass_wave_variant::identity_stockshape_stage2_b4) {\n"
        "    return run_identity_stockshape_stage2_b4(out);"
        in patched
    )


def test_b4_identity_divisor_balances_stockshape_tile_counts() -> None:
    module = _module()
    patched, _ = module.patch_text(_source_fixture(module))
    config_start = patched.index(
        "struct sm120_blockwise_fp8_config_b4_stockshape_identity_divisor"
    )
    config_end = patched.index(
        "struct sm120_blockwise_fp8_config_b4_stockshape_identity_divisor_stage2",
        config_start,
    )
    config = patched[config_start:config_end]

    assert "KernelTmaWarpSpecializedBlockwisePingpongSm120" in config
    assert "using TileShape = Shape<_64, _128, _128>;" in config
    assert "OutType, 1, 128, 128, TileShape, ClusterShape" in config
    assert "fr13_fixed32_m128_divisor_static_scheduler" in config
    assert "StageCount<" not in config
    assert "StreamK" not in config
    expected = {80: 40, 224: 32, 256: 32, 544: 34}
    for logical_tiles, grid_ctas in expected.items():
        selected = next(
            candidate
            for candidate in range(min(48, logical_tiles), 27, -1)
            if logical_tiles % candidate == 0
        )
        assert selected == grid_ctas
    assert "if (M != 128 &&" in patched
    assert "auto run_identity_divisor_b4" in patched
    assert (
        '"/logs/fr13_fixed32_cutlass_identity_divisor_b4_byte_ab.jsonl"'
        in patched
    )
    assert "fr13.fixed32.cutlass_identity_divisor_b4_byte_ab.v1" in patched
    assert (
        "fixed32_cutlass_wave_variant::identity_divisor_b4) {\n"
        "    return run_identity_divisor_b4(out);"
        in patched
    )


def test_b4_identity_divisor_stage2_preserves_math_and_grid_contract() -> None:
    module = _module()
    patched, _ = module.patch_text(_source_fixture(module))
    config_start = patched.index(
        "struct sm120_blockwise_fp8_config_b4_stockshape_identity_divisor_stage2"
    )
    config_end = patched.index(
        "struct sm120_blockwise_fp8_config_b4_stockshape_identity_twom",
        config_start,
    )
    config = patched[config_start:config_end]

    assert "KernelTmaWarpSpecializedBlockwisePingpongSm120" in config
    assert "using TileShape = Shape<_64, _128, _128>;" in config
    assert "OutType, 1, 128, 128, TileShape, ClusterShape" in config
    assert "fr13_fixed32_m128_divisor_static_scheduler" in config
    assert "cutlass::gemm::collective::StageCount<2>" in config
    assert "StreamK" not in config
    assert "identity_divisor_stage2_b4_byte_ab" in patched
    assert "auto run_identity_divisor_stage2_b4" in patched
    assert (
        '"/logs/fr13_fixed32_cutlass_identity_divisor_stage2_b4_byte_ab.jsonl"'
        in patched
    )
    assert (
        "fr13.fixed32.cutlass_identity_divisor_stage2_b4_byte_ab.v1" in patched
    )
    assert (
        "fixed32_cutlass_wave_variant::identity_divisor_stage2_b4) {\n"
        "    return run_identity_divisor_stage2_b4(out);"
        in patched
    )


def test_b4_twom_scheduler_removes_generic_per_tile_divmods() -> None:
    module = _module()
    patched, _ = module.patch_text(_source_fixture(module))
    scheduler_start = patched.index("class Fr13B4TwoMStaticTileScheduler100")
    scheduler_end = patched.index(
        "class Fr13B1OneNStaticTileScheduler100", scheduler_start
    )
    scheduler = patched[scheduler_start:scheduler_end]

    assert "linear_idx & 1" in scheduler
    assert "linear_idx >> 1" in scheduler
    assert "L_idx" not in scheduler
    assert "divmod_batch_" not in scheduler
    assert "divmod_cluster_shape" not in scheduler
    assert "divmod_cluster_blk_major_" not in scheduler
    assert "raster_order_" not in scheduler
    assert "uint32_t current_work_linear_idx_" in scheduler
    assert "uint32_t total_grid_size_" not in scheduler
    assert "uint32_t problem_tiles_" not in scheduler
    assert "current_work_linear_idx_ = blockIdx.x;" in scheduler
    assert "return gridDim.x;" in scheduler
    assert "blockIdx.y" not in scheduler
    assert "blockIdx.z" not in scheduler
    assert "gridDim.y" not in scheduler
    assert "gridDim.z" not in scheduler
    assert "this->scheduler_params.blocks_per_problem_" in scheduler
    assert "total_grid_size() * advance_count" in scheduler
    assert "uint64_t" not in scheduler
    assert "fr13_fixed32_b4_twom_static_scheduler" in patched
    assert "using Scheduler = Fr13B4TwoMStaticTileScheduler100;" in patched

    # All admitted B4 shapes have two M tiles and at least forty N tiles. The
    # pinned CUTLASS heuristic therefore rasterizes AlongM into an X-only grid.
    projection_n = (34816, 5120, 5120, 16384, 14336)
    assert all(n // 128 >= 40 for n in projection_n)


def test_b1_onen_scheduler_maps_every_audited_tile_without_divmods() -> None:
    module = _module()
    patched, _ = module.patch_text(_source_fixture(module))
    scheduler_start = patched.index("class Fr13B1OneNStaticTileScheduler100")
    scheduler_end = patched.index(
        "template <class TileShape, class ClusterShape,\n"
        "          uint32_t SchedulerPipelineStageCount>\n"
        "struct TileSchedulerSelector<\n"
        "    vllm::fr13_fixed32_m128_divisor_static_scheduler",
        scheduler_start,
    )
    scheduler = patched[scheduler_start:scheduler_end]

    assert "return {static_cast<int32_t>(linear_idx), 0, 0, true};" in scheduler
    assert "work_tile_info.M_idx, cute::Int<0>{}" in scheduler
    assert "cute::Underscore{}, cute::Int<0>{}" in scheduler
    assert "L_idx" not in scheduler
    assert "divmod_batch_" not in scheduler
    assert "divmod_cluster_shape" not in scheduler
    assert "divmod_cluster_blk_major_" not in scheduler
    assert "raster_order_" not in scheduler
    assert "get_current_work_for_linear_idx(blockIdx.y)" in scheduler
    assert "return gridDim.y;" in scheduler
    assert "blockIdx.x" not in scheduler
    assert "gridDim.x" not in scheduler
    assert "uint32_t current_work_linear_idx_" not in scheduler
    assert "uint32_t problem_tiles_" in scheduler
    assert "packed_work_state_" not in scheduler
    assert "params.blocks_per_problem_" in scheduler
    assert "this->scheduler_params" not in scheduler
    assert "work_tile_info.M_idx" in scheduler
    assert "fr13_fixed32_b1_onen_static_scheduler" in patched
    assert "using Scheduler = Fr13B1OneNStaticTileScheduler100;" in patched

    projection_tiles = {
        (34816, 5120): 272,
        (5120, 17408): 40,
        (5120, 6144): 40,
        (16384, 5120): 128,
        (14336, 5120): 112,
    }
    expected_grid_ctas = {
        (34816, 5120): 34,
        (5120, 17408): 40,
        (5120, 6144): 40,
        (16384, 5120): 32,
        (14336, 5120): 28,
    }
    for shape, tile_count in projection_tiles.items():
        grid_ctas = next(
            candidate
            for candidate in range(min(48, tile_count), 27, -1)
            if tile_count % candidate == 0
        )
        assert grid_ctas == expected_grid_ctas[shape]
        problem_tiles_m, problem_tiles_n = tile_count, 1
        raster_order = (
            "AlongM" if problem_tiles_n > problem_tiles_m else "AlongN"
        )
        launch_grid = (1, grid_ctas, 1) if raster_order == "AlongN" else (
            grid_ctas,
            1,
            1,
        )
        assert raster_order == "AlongN"
        assert launch_grid == (1, grid_ctas, 1)
        assigned = [
            (linear_idx, 0, 0)
            for cta in range(grid_ctas)
            for linear_idx in range(cta, tile_count, grid_ctas)
        ]
        assert len(assigned) == tile_count
        assert sorted(assigned) == [(tile, 0, 0) for tile in range(tile_count)]
        for cta in range(grid_ctas):
            work_tile_m = cta
            observed = []
            while work_tile_m < tile_count:
                observed.append(work_tile_m)
                work_tile_m += grid_ctas
            assert observed == list(range(cta, tile_count, grid_ctas))


def test_b1_onen_selector_is_default_off_shape_isolated_and_exact_math() -> None:
    module = _module()
    patched, _ = module.patch_text(_source_fixture(module))
    cooperative_start = patched.index(
        "struct sm120_blockwise_fp8_config_b1_onen_static_identity_stage2"
    )
    cooperative_end = patched.index(
        "struct sm120_blockwise_fp8_config_b1_onen_static_identity_pingpong_stage2",
        cooperative_start,
    )
    cooperative = patched[cooperative_start:cooperative_end]
    config_start = patched.index(
        "struct sm120_blockwise_fp8_config_b1_onen_static_identity_pingpong_stage2"
    )
    config_end = patched.index(
        "struct sm120_blockwise_fp8_config_b4_m128_static_identity_stage2",
        config_start,
    )
    config = patched[config_start:config_end]

    assert "KernelTmaWarpSpecializedBlockwiseCooperativeSm120" in cooperative
    assert "using TileShape = Shape<_128, _32, _128>;" in cooperative
    assert "OutType, 128, 1, 128, TileShape, ClusterShape" in cooperative
    assert "fr13_fixed32_b1_onen_static_scheduler" in cooperative
    assert "cutlass::gemm::collective::StageCount<2>" in cooperative
    assert "KernelTmaWarpSpecializedBlockwisePingpongSm120" in config
    assert "using TileShape = Shape<_128, _32, _128>;" in config
    assert "OutType, 128, 1, 128, TileShape, ClusterShape" in config
    assert "EpilogueSchedule, KernelSchedule, true," in config
    assert "fr13_fixed32_b1_onen_static_scheduler" in config
    assert "cutlass::gemm::collective::StageCount<2>" in config
    assert "StreamK" not in config

    guard_start = patched.index(
        "if (!fixed32_cutlass_b1_onen_projection(M, N, K)"
    )
    guard_end = patched.index(
        "wave_variant = fixed32_cutlass_wave_variant::stock;", guard_start
    )
    guard = patched[guard_start:guard_end]
    assert "identity_onen_b1" in guard
    assert "identity_onen_b1_byte_ab" in guard
    assert "return m == 32" in patched
    for n, k in (
        (34816, 5120),
        (5120, 17408),
        (5120, 6144),
        (16384, 5120),
        (14336, 5120),
    ):
        assert patched.count(f"n == {n} && k == {k}") == 3

    assert 'value == "identity_onen_b1"' in patched
    assert 'value == "identity_onen_b1_byte_ab"' in patched
    assert '"/logs/fr13_fixed32_cutlass_identity_onen_b1_byte_ab.jsonl"' in patched
    assert "fr13.fixed32.cutlass_identity_onen_b1_byte_ab.v1" in patched
    runner_start = patched.index("auto run_identity_onen_b1")
    runner_end = patched.index("auto run_identity_stockshape_b4", runner_start)
    runner = patched[runner_start:runner_end]
    assert "if (N == 5120)" in runner
    assert "b1_onen_static_identity_stage2" in runner
    assert "b1_onen_static_identity_pingpong_stage2" in runner
    assert (
        "fixed32_cutlass_wave_variant::identity_onen_b1) {\n"
        "    return run_identity_onen_b1(out);"
        in patched
    )
    assert patched.count("return fixed32_cutlass_wave_variant::stock;") >= 1


def test_mtp_m1m4_direct_is_diagnostic_only_exact_stock_math() -> None:
    module = _module()
    patched, _ = module.patch_text(_source_fixture(module))

    wrapper_start = patched.index(
        "struct cutlass_3x_gemm_fp8_blockwise_mtp_m1m4_direct"
    )
    wrapper_end = patched.index(
        "struct cutlass_3x_gemm_fp8_blockwise_m128_divisor_static",
        wrapper_start,
    )
    wrapper = patched[wrapper_start:wrapper_end]
    assert "typename Base::CollectiveMainloop" in wrapper
    assert "typename Base::CollectiveEpilogue" in wrapper
    assert "fr13_fixed32_mtp_m1m4_direct_scheduler" in wrapper
    assert "Identity" not in wrapper
    assert "StageCount<" not in wrapper

    selector_start = patched.index(
        "vllm::fr13_fixed32_mtp_m1m4_direct_scheduler"
    )
    selector_end = patched.index("};", selector_start)
    selector = patched[selector_start:selector_end]
    assert "using Scheduler = Fr13B1OneNStaticTileScheduler100;" in selector

    config_start = patched.index(
        "struct sm120_blockwise_fp8_config_swapab_mtp_m1m4_direct"
    )
    config_end = patched.index(
        "enum class fixed32_cutlass_wave_variant", config_start
    )
    config = patched[config_start:config_end]
    assert "KernelTmaWarpSpecializedBlockwiseCooperativeSm120" in config
    assert "using TileShape = Shape<_128, _32, _128>;" in config
    assert "OutType, 128, 1, 128, TileShape, ClusterShape" in config
    assert "EpilogueSchedule, KernelSchedule, true" in config

    assert 'value == "mtp_m1m4_direct_byte_ab"' in patched
    assert 'value == "mtp_m1m4_direct"' not in patched
    assert "mtp_rows = m == 1 || m == 4" in patched
    assert "fixed32_cutlass_mtp_m1m4_projection(M, N, K)" in patched
    assert "mtp_m1m4_direct_selection" in patched
    assert "run_mtp_m1m4_direct(destination)" in patched
    assert (
        '"/logs/fr13_fixed32_cutlass_mtp_m1m4_direct_byte_ab.jsonl"'
        in patched
    )
    assert "fr13.fixed32.cutlass_mtp_m1m4_direct_byte_ab.v1" in patched
    assert "(mtp_m1m4_direct_byte_ab && M == 4)" in patched
    assert "constexpr int64_t mtp_m1m4_byte_ab_limit = 320" in patched
    assert "run_stock(out);\n    run_candidate(candidate);" in patched
    assert "return run_stock(out);" in patched


def test_b1_n5120_single_tile_scheduler_removes_persistent_advance() -> None:
    module = _module()
    patched, _ = module.patch_text(_source_fixture(module))
    scheduler_start = patched.index("class Fr13B1N5120SingleTileScheduler100")
    scheduler_end = patched.index(
        "template <class TileShape, class ClusterShape,\n"
        "          uint32_t SchedulerPipelineStageCount>\n"
        "struct TileSchedulerSelector<\n"
        "    vllm::fr13_fixed32_m128_divisor_static_scheduler",
        scheduler_start,
    )
    scheduler = patched[scheduler_start:scheduler_end]

    assert "static constexpr uint32_t kProblemTiles = 40;" in scheduler
    assert "static_cast<int32_t>(blockIdx.y), 0, 0, true" in scheduler
    assert "return true;" in scheduler
    assert "WorkTileInfo::invalid_work_tile(), true" in scheduler
    assert "problem_tiles_" not in scheduler
    assert "params.blocks_per_problem_" not in scheduler
    assert "gridDim" not in scheduler
    assert "advance_to_next_work" not in scheduler
    assert "total_grid_size" not in scheduler
    assert "fr13_fixed32_b1_n5120_single_tile_scheduler" in patched
    assert "using Scheduler = Fr13B1N5120SingleTileScheduler100;" in patched

    config_start = patched.index(
        "struct sm120_blockwise_fp8_config_b1_n5120_single_identity_stage2"
    )
    config_end = patched.index(
        "struct sm120_blockwise_fp8_config_b1_onen_static_identity_pingpong_stage2",
        config_start,
    )
    config = patched[config_start:config_end]
    assert "KernelTmaWarpSpecializedBlockwiseCooperativeSm120" in config
    assert "using TileShape = Shape<_128, _32, _128>;" in config
    assert "OutType, 128, 1, 128, TileShape, ClusterShape" in config
    assert "fr13_fixed32_b1_n5120_single_tile_scheduler" in config
    assert "cutlass::gemm::collective::StageCount<2>" in config
    assert "StreamK" not in config

    assert 'value == "identity_onen_n5120_single_b1"' in patched
    assert 'value == "identity_onen_n5120_single_b1_byte_ab"' in patched
    assert (
        '"/logs/fr13_fixed32_cutlass_identity_onen_n5120_single_b1_byte_ab.jsonl"'
        in patched
    )
    assert (
        "fr13.fixed32.cutlass_identity_onen_n5120_single_b1_byte_ab.v1"
        in patched
    )
    runner_start = patched.index("auto run_identity_onen_n5120_single_b1")
    runner_end = patched.index("auto run_identity_stockshape_b4", runner_start)
    runner = patched[runner_start:runner_end]
    assert "if (N == 5120)" in runner
    assert "b1_n5120_single_identity_stage2" in runner
    assert "return run_identity_onen_b1(destination);" in runner
    assert (
        "fixed32_cutlass_wave_variant::\n"
        "                          identity_onen_n5120_single_b1) {\n"
        "    return run_identity_onen_n5120_single_b1(out);"
        in patched
    )


def test_b1_n5120_fullgrid_keeps_initialized_static_scheduler_contract() -> None:
    module = _module()
    patched, _ = module.patch_text(_source_fixture(module))

    scheduler_start = patched.index(
        "class Fr13B1OneNFullGridStaticTileScheduler100"
    )
    scheduler_end = patched.index(
        "template <class TileShape, class ClusterShape,\n"
        "          uint32_t SchedulerPipelineStageCount>\n"
        "struct TileSchedulerSelector<\n"
        "    vllm::fr13_fixed32_m128_divisor_static_scheduler",
        scheduler_start,
    )
    scheduler = patched[scheduler_start:scheduler_end]
    assert "public StaticPersistentTileScheduler100" in scheduler
    assert "using Base = StaticPersistentTileScheduler100;" in scheduler
    assert "using Base::Base;" in scheduler
    assert "public Fr13B1OneNStaticTileScheduler100" not in scheduler
    assert "get_current_work" not in scheduler
    assert "advance_to_next_work" not in scheduler
    assert "fetch_next_work" not in scheduler
    assert "get_grid_shape" not in scheduler
    assert "fr13_fixed32_b1_onen_fullgrid_static_scheduler" in patched
    assert "using Scheduler = Fr13B1OneNFullGridStaticTileScheduler100;" in patched

    config_start = patched.index(
        "struct sm120_blockwise_fp8_config_b1_onen_fullgrid_identity_pingpong_stage2"
    )
    config_end = patched.index(
        "struct sm120_blockwise_fp8_config_b4_m128_static_identity_stage2",
        config_start,
    )
    config = patched[config_start:config_end]
    assert "KernelTmaWarpSpecializedBlockwisePingpongSm120" in config
    assert "using TileShape = Shape<_128, _32, _128>;" in config
    assert "fr13_fixed32_b1_onen_fullgrid_static_scheduler" in config
    assert "cutlass::gemm::collective::StageCount<2>" in config

    runner_start = patched.index("auto run_identity_onen_n5120_fullgrid_b1")
    runner_end = patched.index("auto run_identity_stockshape_b4", runner_start)
    runner = patched[runner_start:runner_end]
    assert "if (N == 5120)" in runner
    assert "b1_n5120_single_identity_stage2" in runner
    assert "b1_onen_fullgrid_identity_pingpong_stage2" in runner
    assert 'value == "identity_onen_n5120_fullgrid_b1"' in patched
    assert 'value == "identity_onen_n5120_fullgrid_b1_byte_ab"' in patched
    assert (
        '"/logs/fr13_fixed32_cutlass_identity_onen_n5120_fullgrid_b1_byte_ab.jsonl"'
        in patched
    )
    assert (
        "fr13.fixed32.cutlass_identity_onen_n5120_fullgrid_b1_byte_ab.v1"
        in patched
    )


def test_b4_identity_twom_keeps_complete_tile_math() -> None:
    module = _module()
    patched, _ = module.patch_text(_source_fixture(module))
    config_start = patched.index(
        "struct sm120_blockwise_fp8_config_b4_stockshape_identity_twom"
    )
    config_end = patched.index("enum class fixed32_cutlass_wave_variant", config_start)
    config = patched[config_start:config_end]

    assert "KernelTmaWarpSpecializedBlockwisePingpongSm120" in config
    assert "using TileShape = Shape<_64, _128, _128>;" in config
    assert "OutType, 1, 128, 128, TileShape, ClusterShape" in config
    assert "fr13_fixed32_b4_twom_static_scheduler" in config
    assert "cutlass::gemm::collective::StageCount<2>" in config
    assert "StreamK" not in config
    assert "identity_twom_b4_byte_ab" in patched
    assert "auto run_identity_twom_b4" in patched
    assert '"/logs/fr13_fixed32_cutlass_identity_twom_b4_byte_ab.jsonl"' in patched
    assert "fr13.fixed32.cutlass_identity_twom_b4_byte_ab.v1" in patched
    assert (
        "fixed32_cutlass_wave_variant::identity_twom_b4) {\n"
        "    return run_identity_twom_b4(out);"
        in patched
    )
    assert "if (N > 65536 &&" in patched


def test_b4_hybrid_n5120_routes_only_exact_projections() -> None:
    module = _module()
    patched, _ = module.patch_text(_source_fixture(module))

    predicate_start = patched.index(
        "static inline bool fixed32_cutlass_b4_hybrid_n5120_projection"
    )
    predicate_end = patched.index("template <typename Gemm>", predicate_start)
    predicate = patched[predicate_start:predicate_end]
    assert "m == 128" in predicate
    assert "fixed32_cutlass_real_projection(m, n, k)" in predicate

    guard_start = patched.index(
        "if (!fixed32_cutlass_b4_hybrid_n5120_projection(M, N, K)"
    )
    guard_end = patched.index(
        "wave_variant = fixed32_cutlass_wave_variant::stock;", guard_start
    )
    guard = patched[guard_start:guard_end]
    assert "identity_hybrid_n5120_b4" in guard
    assert "identity_hybrid_n5120_b4_byte_ab" in guard

    runner_start = patched.index("auto run_identity_hybrid_n5120_b4")
    runner_end = patched.index("auto run_stock", runner_start)
    runner = patched[runner_start:runner_end]
    assert "if (N == 5120)" in runner
    assert (
        "sm120_blockwise_fp8_config_b4_m128_n5120_single_identity_stage2"
        in runner
    )
    assert "return run_identity_twom_b4(destination);" in runner
    assert "run_stock" not in runner

    # The two N=5120 projections halve complete scheduler tile assignments.
    # The other three retain the already-qualified two-M assignment count.
    projection_n = (34816, 5120, 5120, 16384, 14336)
    twom_tiles = tuple(2 * n // 128 for n in projection_n)
    hybrid_tiles = tuple(
        n // 128 if n == 5120 else 2 * n // 128 for n in projection_n
    )
    assert twom_tiles == (544, 80, 80, 256, 224)
    assert hybrid_tiles == (544, 40, 40, 256, 224)

    assert 'value == "identity_hybrid_n5120_b4"' in patched
    assert 'value == "identity_hybrid_n5120_b4_byte_ab"' in patched
    assert (
        '"/logs/fr13_fixed32_cutlass_identity_hybrid_n5120_b4_byte_ab.jsonl"'
        in patched
    )
    assert (
        "fr13.fixed32.cutlass_identity_hybrid_n5120_b4_byte_ab.v1" in patched
    )
    assert (
        "fixed32_cutlass_wave_variant::identity_hybrid_n5120_b4) {\n"
        "    return run_identity_hybrid_n5120_b4(out);"
        in patched
    )

    diagnostic_start = patched.index(
        "const bool identity_hybrid_n5120_b4_byte_ab"
    )
    diagnostic_end = patched.index(
        "fixed32_cutlass_wave_variant::stream_k_cooperative_128)",
        diagnostic_start,
    )
    diagnostic = patched[diagnostic_start:diagnostic_end]
    assert "run_stock(out);\n    run_candidate(candidate);" in diagnostic
    assert "\n    return;\n  }\n\n  if (wave_variant ==" in diagnostic


def test_b4_hybrid_n5120_uses_exact_x_axis_single_tile_scheduler() -> None:
    module = _module()
    patched, _ = module.patch_text(_source_fixture(module))

    scheduler_start = patched.index("class Fr13B4N5120SingleTileScheduler100")
    scheduler_end = patched.index(
        "// Fixed32 B1 swap-AB", scheduler_start
    )
    scheduler = patched[scheduler_start:scheduler_end]
    assert ": public StaticPersistentTileScheduler100" in scheduler
    assert "Fr13DivisorBalancedStaticTileScheduler100" not in scheduler
    assert "static constexpr uint32_t kProblemTiles = 40;" in scheduler
    assert "return {0, static_cast<int32_t>(blockIdx.x), 0, true};" in scheduler
    assert "linear_idx >= kProblemTiles" in scheduler
    assert "return {0, static_cast<int32_t>(linear_idx), 0, true};" in scheduler
    assert "bool is_last_tile(WorkTileInfo&, uint32_t = 1) const" in scheduler
    assert "return true;" in scheduler
    assert "WorkTileInfo::invalid_work_tile()" in scheduler
    assert "blockIdx.y" not in scheduler
    assert "gridDim" not in scheduler

    config_start = patched.index(
        "struct sm120_blockwise_fp8_config_b4_m128_n5120_single_identity_stage2"
    )
    config_end = patched.index(
        "// Isolate the identity epilogue on B4", config_start
    )
    config = patched[config_start:config_end]
    assert "KernelTmaWarpSpecializedBlockwiseCooperativeSm120" in config
    assert "using TileShape = Shape<_128, _128, _128>;" in config
    assert "using ClusterShape = Shape<_1, _1, _1>;" in config
    assert "fr13_fixed32_b4_n5120_single_tile_scheduler" in config
    assert "cutlass::gemm::collective::StageCount<2>" in config

    generic_start = patched.index(
        "struct sm120_blockwise_fp8_config_b4_m128_static_identity_stage2"
    )
    generic_end = patched.index("template <typename OutType>", generic_start)
    generic = patched[generic_start:generic_end]
    assert "fr13_fixed32_m128_static_scheduler" in generic
    assert "fr13_fixed32_b4_n5120_single_tile_scheduler" not in generic


def test_wide256_is_b1_only_and_large_rows_fail_to_stock() -> None:
    module = _module()
    patched, _ = module.patch_text(_source_fixture(module))

    selector_gate = patched.index("if (M > 64 &&")
    stock_assignment = patched.index(
        "wave_variant = fixed32_cutlass_wave_variant::stock;", selector_gate
    )
    candidate_dispatch = patched.index("auto run_stream_k_wide256")

    assert selector_gate < stock_assignment < candidate_dispatch
    assert "stream_k_force_wide256_byte_ab" in patched[selector_gate:stock_assignment]
    assert "sm120_blockwise_fp8_config_cooperative_streamk_wide256" not in patched
    assert "sm120_blockwise_fp8_config_swapab_streamk_wide256" in patched


def test_wide256_recomputes_only_fixup_barrier_coordinate() -> None:
    module = _module()
    patched, _ = module.patch_text(_source_fixture(module))

    assert "struct fr13_fixed32_wide256_recompute_scheduler {};" in patched
    assert "class Fr13Wide256RecomputeTileScheduler" in patched
    assert "using Base::Base;" in patched
    assert "using Base::fixup;" in patched
    assert 'asm volatile("mov.u32 %0, %%tid.x;"' in patched
    assert "thread_idx % BarrierManager::ThreadCount" in patched
    assert "uint32_t barrier_group_thread_idx =" not in (
        module.SCHEDULER_SPECIALIZATION_REPLACEMENT
    )
    assert patched.count("barrier_group_thread_idx<BarrierManager>()") == 9
    assert "ReductionMode::Deterministic" in patched
    assert "CUTLASS_HOST_DEVICE static auto tile_peer_range" in patched
    assert "find_unit(start_k_tile + cur_k_tile)" in patched
    assert "params.div_cluster_size(tile_idx)" in patched
    assert "params.divmod_k_tiles_per_sk_big_unit_.divide(k_tile)" in patched
    assert "params.divmod_k_tiles_per_sk_unit_.divide(" in patched
    assert "UnderlyingStreamKScheduler::compute_epilogue" in patched
    assert "UnderlyingStreamKScheduler::template separate_reduction" in patched
    assert "fr13_fixed32_wide256_recompute_scheduler, true" in patched
    assert "cutlass::gemm::StreamKScheduler, true" not in patched


def test_streamk_uses_a_separate_candidate_template() -> None:
    module = _module()
    patched, _ = module.patch_text(_source_fixture(module))

    assert patched.count(module.TEMPLATE_ANCHOR) == 1
    assert "class TileScheduler = void" not in patched
    assert "cutlass_3x_gemm_fp8_blockwise_streamk" in patched
    assert "class TileScheduler, bool force_stream_k_ = false" in patched
    assert "static constexpr bool use_stream_k" in patched
    assert "bool force_stream_k_ = false" in patched
    assert "static constexpr bool force_stream_k = force_stream_k_" in patched
    assert "CollectiveMainloop, CollectiveEpilogue,\n      TileScheduler>>" in patched
    assert "typename GemmKernel::TileSchedulerArguments scheduler{}" in patched
    assert "if constexpr (!StreamKTraits::enabled)" in patched
    assert "} else {" in module.CALLER_REPLACEMENT
    assert "query_device_multiprocessor_count" in patched
    assert "STD_TORCH_CHECK(sm_count > 0" in patched
    assert "\n  TORCH_CHECK(sm_count > 0" not in patched
    assert "Deterministic" in patched
    assert "decltype(scheduler.decomposition_mode)::StreamK" in patched
    assert "decltype(scheduler.decomposition_mode)::Heuristic" in patched
    assert "scheduler.splits = 1" in patched


def test_stock_class_is_textually_unchanged() -> None:
    module = _module()
    source = _source_fixture(module)
    patched, _ = module.patch_text(source)
    stock_start = source.index(module.TEMPLATE_ANCHOR)
    stock_end = source.index(module.STREAMK_CLASS_ANCHOR) + len(
        module.STREAMK_CLASS_ANCHOR
    )
    stock_class = source[stock_start:stock_end]

    assert stock_class in patched
    assert patched.count(module.TEMPLATE_ANCHOR) == 1
    assert patched.count(module.KERNEL_ANCHOR) == 1
    assert patched.count(module.STAGE_COUNT_ANCHOR) == 2
    assert "fr13_fixed32_gemm_universal" not in patched
    assert "class MainloopStageCount = cutlass::gemm::collective::StageCountAuto" in patched
    assert patched.index(stock_class) < patched.index(
        "struct cutlass_3x_gemm_fp8_blockwise_streamk"
    )


def test_stock_caller_uses_default_false_streamk_trait() -> None:
    module = _module()
    patched, _ = module.patch_text(_source_fixture(module))

    assert "template <typename Gemm, typename = void>" in patched
    assert "static constexpr bool enabled = false;" in patched
    assert "std::void_t<decltype(Gemm::use_stream_k)" in patched
    assert "using StreamKTraits = fr13_fixed32_streamk_traits<Gemm>;" in patched
    assert "StreamKTraits::force" in patched


def test_stock_dispatch_retains_original_kernel_configs() -> None:
    module = _module()
    patched, _ = module.patch_text(_source_fixture(module))

    stock = patched[patched.index("auto run_stock"):]
    for config in (
        "sm120_blockwise_fp8_config_pingpong<OutType>::Gemm",
        "sm120_blockwise_fp8_config_default<OutType>::Gemm",
        "sm120_blockwise_fp8_config_swapab<OutType>::Gemm",
    ):
        assert stock.count(config) == 1
    assert "bool swap_ab = (M <= 64) || (M % 4 != 0);" in stock


def test_coop128_geometry_and_heuristic_mode_remain_unchanged() -> None:
    module = _module()
    patched, _ = module.patch_text(_source_fixture(module))

    assert "using TileShape = Shape<_128, _32, _128>;" in patched
    assert "using TileShape = Shape<_128, _128, _128>;" in patched
    assert (
        "cutlass::gemm::StreamKScheduler>;\n};\n\n"
        "template <typename OutType>\n"
        "struct sm120_blockwise_fp8_config_swapab_streamk"
    ) in patched
    assert "StreamKTraits::force\n          ?" in patched


def test_stock_dispatch_text_is_retained() -> None:
    module = _module()
    patched, _ = module.patch_text(_source_fixture(module))

    assert "bool swap_ab = (M <= 64) || (M % 4 != 0);" in patched
    assert "sm120_blockwise_fp8_config_pingpong<OutType>::Gemm" in patched
    assert "sm120_blockwise_fp8_config_default<OutType>::Gemm" in patched
    assert "sm120_blockwise_fp8_config_swapab<OutType>::Gemm" in patched


def test_same_process_byte_ab_is_bounded_and_returns_stock() -> None:
    module = _module()
    patched, _ = module.patch_text(_source_fixture(module))

    assert "constexpr int64_t byte_ab_limit = 320" in patched
    assert "constexpr int64_t b4_m128_byte_ab_limit = 320" in patched
    assert "b4_m128_byte_ab || b4_m128_static_byte_ab" in patched
    assert "? b4_m128_byte_ab_limit" in patched
    assert "invocation >= selected_byte_ab_limit" in patched
    assert "torch::stable::empty_like(out)" in patched
    assert "run_stock(out);\n    run_candidate(candidate);" in patched
    assert "cudaMemcpyDeviceToHost" in patched
    assert "cudaStreamSynchronize(stream)" in patched
    assert "++mismatch_count" in patched
    assert "break;" not in module.DISPATCH_REPLACEMENT
    assert '\\"mismatch_count\\"' in module.DISPATCH_REPLACEMENT
    assert '"/logs/fr13_fixed32_cutlass_streamk_byte_ab.jsonl"' in patched
    assert '"/logs/fr13_fixed32_cutlass_streamk_wide256_byte_ab.jsonl"' in patched
    assert (
        '"/logs/fr13_fixed32_cutlass_static_persistent_byte_ab.jsonl"'
        in patched
    )
    assert (
        '"/logs/fr13_fixed32_cutlass_persistent_b4_m128_byte_ab.jsonl"'
        in patched
    )
    assert (
        '"/logs/fr13_fixed32_cutlass_persistent_b4_m128_static_byte_ab.jsonl"'
        in patched
    )
    assert '\\"byte_equal\\"' in module.DISPATCH_REPLACEMENT
    assert "return run_stock(out);" in patched
    assert "fr13.fixed32.cutlass_streamk_wide256_byte_ab.v1" in patched
    assert "fr13.fixed32.cutlass_static_persistent_byte_ab.v1" in patched
    assert "fr13.fixed32.cutlass_persistent_b4_m128_byte_ab.v1" in patched
    assert "fr13.fixed32.cutlass_persistent_b4_m128_static_byte_ab.v1" in patched


def test_unarmed_boot_warm_cannot_dispatch_candidate_or_consume_gate() -> None:
    module = _module()
    patched, _ = module.patch_text(_source_fixture(module))

    arm_path = '"/logs/fr13_fixed32_cutlass_streamk.real_event.arm"'
    unarmed = patched.index("if (task_marker.empty())")
    counter = patched.index("static std::atomic<int64_t> next_invocation")
    unarmed_path = patched[unarmed:counter]

    assert arm_path in patched
    assert unarmed < counter
    assert "return run_stock(out);" in unarmed_path
    assert "run_candidate" not in unarmed_path
    assert "empty_like" not in unarmed_path
    assert "fixed32_cutlass_real_task_marker();" in patched
    assert 'fr13.fixed32.cutlass_streamk_byte_ab.v2' in patched
    assert '\\"task_marker\\"' in module.DISPATCH_REPLACEMENT


def test_candidate_dispatch_is_after_authenticated_arm_check() -> None:
    module = _module()
    patched, _ = module.patch_text(_source_fixture(module))

    unarmed = patched.index("if (task_marker.empty())")
    unarmed_return = patched.index("return run_stock(out);", unarmed)
    candidate_allocation = patched.index(
        "torch::stable::Tensor candidate = torch::stable::empty_like(out);",
        unarmed_return,
    )
    candidate_dispatch = patched.index("run_candidate(candidate);", candidate_allocation)

    assert unarmed < unarmed_return < candidate_allocation < candidate_dispatch


def test_patch_is_idempotent() -> None:
    module = _module()
    first, changed = module.patch_text(_source_fixture(module))
    second, changed_again = module.patch_text(first)

    assert changed
    assert not changed_again
    assert second == first
    assert first.count(module.MARKER) == 1


def test_patch_fails_closed_on_anchor_drift() -> None:
    module = _module()
    source = _source_fixture(module).replace(
        module.DISPATCH_ANCHOR, "  stock_dispatch_changed();\n"
    )
    with pytest.raises(RuntimeError, match="dispatch anchor"):
        module.patch_text(source)


def test_source_root_fails_closed_on_pinned_digest_drift(tmp_path: Path) -> None:
    module = _module()
    target = tmp_path / module.TARGET_RELATIVE_PATH
    target.parent.mkdir(parents=True)
    target.write_text(_source_fixture(module), encoding="utf-8")

    with pytest.raises(RuntimeError, match="dispatch SHA256 mismatch"):
        module.patch_source_root(tmp_path)


def _write_pinned_cmake(module, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cmake_source = f"project(vllm)\n{module.EXPECTED_CUTLASS_REVISION_LINE}\n"
    (root / module.CMAKE_RELATIVE_PATH).write_text(cmake_source, encoding="utf-8")
    monkeypatch.setattr(
        module,
        "EXPECTED_CMAKE_SHA256",
        hashlib.sha256(cmake_source.encode("utf-8")).hexdigest(),
    )


def test_source_root_applies_exact_pinned_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    source = _source_fixture(module)
    target = tmp_path / module.TARGET_RELATIVE_PATH
    target.parent.mkdir(parents=True)
    target.write_text(source, encoding="utf-8")
    monkeypatch.setattr(
        module,
        "EXPECTED_UNPATCHED_SHA256",
        hashlib.sha256(source.encode("utf-8")).hexdigest(),
    )
    _write_pinned_cmake(module, tmp_path, monkeypatch)

    assert module.patch_source_root(tmp_path)
    assert module.MARKER in target.read_text(encoding="utf-8")
    assert not module.patch_source_root(tmp_path)


def test_source_root_fails_closed_on_cutlass_pin_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    source = _source_fixture(module)
    target = tmp_path / module.TARGET_RELATIVE_PATH
    target.parent.mkdir(parents=True)
    target.write_text(source, encoding="utf-8")
    monkeypatch.setattr(
        module,
        "EXPECTED_UNPATCHED_SHA256",
        hashlib.sha256(source.encode("utf-8")).hexdigest(),
    )
    (tmp_path / module.CMAKE_RELATIVE_PATH).write_text(
        'project(vllm)\nset(CUTLASS_REVISION "v4.4.1")\n', encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="CMakeLists SHA256 mismatch"):
        module.patch_source_root(tmp_path)


def test_cutlass_root_validates_headers_and_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    relative_path = Path("include/cutlass/pinned.hpp")
    content = b"pinned CUTLASS header\n"
    target = tmp_path / relative_path
    target.parent.mkdir(parents=True)
    target.write_bytes(content)
    monkeypatch.setattr(
        module,
        "CUTLASS_REQUIRED_SHA256",
        {relative_path: hashlib.sha256(content).hexdigest()},
    )
    monkeypatch.setattr(
        module.subprocess,
        "check_output",
        lambda *args, **kwargs: module.CUTLASS_TAG_COMMIT + "\n",
    )

    module.validate_cutlass_root(tmp_path)


def test_cutlass_root_fails_closed_on_header_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    relative_path = Path("include/cutlass/pinned.hpp")
    target = tmp_path / relative_path
    target.parent.mkdir(parents=True)
    target.write_bytes(b"modified\n")
    monkeypatch.setattr(
        module,
        "CUTLASS_REQUIRED_SHA256",
        {relative_path: hashlib.sha256(b"expected\n").hexdigest()},
    )

    with pytest.raises(RuntimeError, match="CUTLASS SHA256 mismatch"):
        module.validate_cutlass_root(tmp_path)


def test_cutlass_root_fails_closed_on_commit_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    monkeypatch.setattr(module, "CUTLASS_REQUIRED_SHA256", {})
    monkeypatch.setattr(
        module.subprocess,
        "check_output",
        lambda *args, **kwargs: "0" * 40 + "\n",
    )

    with pytest.raises(RuntimeError, match="CUTLASS commit mismatch"):
        module.validate_cutlass_root(tmp_path)
