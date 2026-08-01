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
    assert "return fixed32_cutlass_wave_variant::stock;" in patched
    for rows in (32, 64, 96, 128):
        assert f"m == {rows}" in patched
    for n, k in (
        (34816, 5120),
        (5120, 17408),
        (5120, 6144),
        (16384, 5120),
        (8192, 5120),
    ):
        assert f"n == {n} && k == {k}" in patched


def test_candidates_keep_scale_k_tile_cluster_and_numeric_math() -> None:
    module = _module()
    patched, _ = module.patch_text(_source_fixture(module))

    assert patched.count("cutlass::gemm::StreamKScheduler") == 3
    assert patched.count("using ClusterShape = Shape<_1, _1, _1>;") == 5
    assert module.CONFIG_REPLACEMENT.count("PingpongSm120") == 1
    assert "OutType, 128, 1, 128, TileShape, ClusterShape" in patched
    assert "using TileShape = Shape<_128, _32, _128>;" in patched
    assert "OutType, 1, 128, 128, TileShape, ClusterShape" in patched
    assert "using TileShape = Shape<_128, _128, _128>;" in patched
    assert "using TileShape = Shape<_128, _256, _128>;" not in patched
    assert "using TileShape = Shape<_256, _32, _128>;" in patched
    assert (
        "OutType, 128, 1, 128, TileShape, ClusterShape,\n"
        "      EpilogueSchedule, KernelSchedule, true,\n"
        "      cutlass::gemm::StreamKScheduler, true,\n"
        "      cutlass::gemm::collective::StageCount<2>>"
    ) in patched
    assert patched.count("using TileShape = Shape<_64, _128, _128>;") == 1
    assert "MainloopStageCount" in patched
    assert "cutlass::gemm::collective::StageCount<2>" in patched
    assert "ElementAccumulator = float" not in module.CONFIG_REPLACEMENT
    assert "ElementD" not in module.CONFIG_REPLACEMENT


def test_static_persistent_candidate_only_changes_tile_allocator() -> None:
    module = _module()
    patched, _ = module.patch_text(_source_fixture(module))

    assert "struct fr13_fixed32_static_persistent_scheduler {};" in patched
    assert (
        "using Scheduler = StaticPersistentTileScheduler100;" in patched
    )
    assert (
        "vllm::fr13_fixed32_static_persistent_scheduler, arch::Sm120" in patched
    )
    assert "cutlass_3x_gemm_fp8_blockwise_static_persistent" in patched
    assert (
        "Shape<int, int, int, int>, typename Base::CollectiveMainloop,\n"
        "      typename Base::CollectiveEpilogue,\n"
        "      fr13_fixed32_static_persistent_scheduler>>"
    ) in patched
    assert (
        "using TileShape = Shape<_128, _32, _128>;" in patched
    )
    assert (
        "using TileShape = Shape<_64, _128, _128>;" in patched
    )
    assert "OutType, 128, 1, 128, TileShape, ClusterShape" in patched
    assert "OutType, 1, 128, 128, TileShape, ClusterShape" in patched
    assert "static_persistent_stocktile" in patched
    assert "scheduler.splits" not in patched[
        patched.index("auto run_static_persistent_stocktile"):
        patched.index("auto run_stock")
    ]


def test_static_persistent_dispatch_covers_b1_and_b4_stock_geometries() -> None:
    module = _module()
    patched, _ = module.patch_text(_source_fixture(module))

    candidate = patched[
        patched.index("auto run_static_persistent_stocktile"):
        patched.index("auto run_stock")
    ]
    assert "if (M <= 64)" in candidate
    assert (
        "sm120_blockwise_fp8_config_swapab_static_persistent<OutType>::Gemm"
        in candidate
    )
    assert (
        "sm120_blockwise_fp8_config_pingpong_static_persistent<OutType>::Gemm"
        in candidate
    )
    assert "M > 64" not in candidate


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
    assert "constexpr int64_t byte_ab_limit = 256" not in patched
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
    assert '\\"byte_equal\\"' in module.DISPATCH_REPLACEMENT
    assert "return run_stock(out);" in patched
    assert "fr13.fixed32.cutlass_streamk_wide256_byte_ab.v1" in patched
    assert "fr13.fixed32.cutlass_static_persistent_byte_ab.v1" in patched


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
    assert "task_marker = fixed32_cutlass_real_task_marker()" in patched
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
