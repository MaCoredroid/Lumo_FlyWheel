from __future__ import annotations

import importlib.util
import math
from functools import cmp_to_key
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
CUDA = REPO / "csrc" / "fr13_dfwd_k64_fp8_mapped_top3.cu"
BUILDER = REPO / "scripts" / "fr13_build_dfwd_k64_fp8_mapped_top3.py"
PATCHER = REPO / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
LAUNCHER = REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"


def _load_builder():
    spec = importlib.util.spec_from_file_location("fr13_dfwd_fp8_builder", BUILDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reference_top3(scores: list[float], id_map: list[int]):
    def compare(lhs: int, rhs: int) -> int:
        lhs_nan = math.isnan(scores[lhs])
        rhs_nan = math.isnan(scores[rhs])
        if lhs_nan != rhs_nan:
            return -1 if lhs_nan else 1
        if scores[lhs] > scores[rhs]:
            return -1
        if scores[lhs] < scores[rhs]:
            return 1
        if lhs < rhs:
            return -1
        if lhs > rhs:
            return 1
        return 0

    subset_ids = sorted(range(len(scores)), key=cmp_to_key(compare))[:3]
    return subset_ids, [id_map[index] for index in subset_ids]


def test_cuda_source_is_two_stage_k64_fp8_mapped_top3() -> None:
    source = CUDA.read_text(encoding="ascii")

    assert "constexpr int kVocab = 65536;" in source
    assert "constexpr int kHidden = 5120;" in source
    assert "constexpr int kGroups = kHidden / kScaleGroup;" in source
    assert "constexpr int kRowsPerPartial = 128;" in source
    assert "constexpr int kPartials = kVocab / kRowsPerPartial;" in source
    assert "constexpr int kMaxBatch = 4;" in source
    assert source.count("<<<") == 2
    assert "<<<kPartials, kThreads, 0, stream>>>" in source
    assert "<<<batch, kThreads, 0, stream>>>" in source
    assert "full_logits" not in source
    assert "const at::Tensor& logits" not in source
    assert "logits.data_ptr" not in source
    assert "cudaMalloc" not in source
    assert "at::empty" not in source
    assert "torch::empty" not in source


def test_stage1_reuses_each_weight_load_across_b4_rows() -> None:
    source = CUDA.read_text(encoding="ascii")

    assert "shared_activation[kMaxBatch * kHidden]" in source
    assert "shared_scale[kMaxBatch * kGroups]" in source
    assert "if (thread < kGroups)" in source
    assert source.count("weight_scale[partial * kGroups + thread]") == 1
    assert "const float tile_weight_scale" in source
    assert "activation_scale[thread * batch + batch_index]" in source
    assert "tile_weight_scale;" in source
    assert "const float weight_value = static_cast<float>(" in source
    assert "for (int batch_index = 0; batch_index < kMaxBatch;" in source
    assert "activation_value," in source
    assert "weight_value * shared_scale" in source
    assert "blockIdx.y" not in source
    assert "batch == 1 || batch == 4" in source


def test_wrapper_fails_closed_on_any_restrict_storage_overlap() -> None:
    source = CUDA.read_text(encoding="ascii")

    assert "struct TensorByteRange" in source
    assert "fr13_dense_byte_range" in source
    assert "is_non_overlapping_and_dense()" in source
    assert "std::array<NamedTensor, 10> restrict_tensors" in source
    assert "fr13_check_no_storage_overlap(restrict_tensors);" in source
    assert "ranges[lhs].end <= ranges[rhs].begin" in source
    assert "ranges[rhs].end <= ranges[lhs].begin" in source
    assert '"FR13 DFWD FP8 restrict tensor storage overlaps: "' in source
    for name in (
        "spine_output",
        "top3_ids",
        "top3_scores",
        "partial_values",
        "partial_indices",
        "activation_q",
        "qweight",
        "activation_scale",
        "weight_scale",
        "id_map",
    ):
        assert f'{{"{name}", &' in source


def test_selection_rounds_to_bf16_then_maps_after_exact_subset_order() -> None:
    source = CUDA.read_text(encoding="ascii")

    assert "const bool lhs_nan = isnan(lhs.value);" in source
    assert "if (lhs_nan != rhs_nan)" in source
    assert "return lhs.index < rhs.index;" in source
    assert "const __nv_bfloat16 rounded = __float2bfloat16_rn(score);" in source
    assert "id_map[block_first.index]" in source
    assert "id_map[block_second.index]" in source
    assert "id_map[block_third.index]" in source

    scores = [3.0, float("nan"), 9.0, float("nan"), 9.0]
    id_map = [100, 7, 900, 5, 800]
    subset_ids, mapped_ids = _reference_top3(scores, id_map)
    assert subset_ids == [1, 3, 2]
    assert mapped_ids == [7, 5, 900]


def test_tensor_contract_is_exact_b1_b4_and_persistent_workspace() -> None:
    source = CUDA.read_text(encoding="ascii")

    assert "at::kFloat8_e4m3fn" in source
    assert '"FR13 DFWD FP8 mapped top3 serves only exact B1 or B4"' in source
    assert "at::IntArrayRef({batch, kHidden})" in source
    assert "at::IntArrayRef({kVocab, kHidden})" in source
    assert "at::IntArrayRef({1, batch})" in source
    assert "at::IntArrayRef({kWeightScaleRows, kGroups})" in source
    assert "at::IntArrayRef({batch, kPartials, kTopK})" in source
    assert "properties->major == 12" in source
    assert "properties->minor == 1" in source
    assert "mapped_top3_out(Tensor(a!) spine_output" in source


def test_physical32_b1_b4_work_model_is_closed() -> None:
    builder = _load_builder()
    b1 = builder.physical32_work_model(1)
    b4 = builder.physical32_work_model(4)

    for model, batch in ((b1, 1), (b4, 4)):
        assert model["served_batch"] == batch
        assert model["physical_tree_nodes"] == 32
        assert model["head_calls_per_event"] == 5
        assert model["partial_blocks_per_head"] == 512
        assert model["kernel_launches_per_head"] == 2
        assert model["qweight_bytes_per_head"] == 335_544_320
        assert model["fp32_weight_scale_bytes_per_head"] == 81_920
        assert model["fp32_weight_scale_elements_per_head"] == 20_480
        assert model["fp32_weight_scale_batch_multiplier"] == 1
        assert model["macs_per_head"] == batch * 65_536 * 5_120
        assert (
            model["removed_full_bf16_logit_write_plus_read_per_head"]
            == batch * 65_536 * 2 * 2
        )
        assert (
            model["partial_bf16_i32_write_plus_read_per_head"]
            == batch * 512 * 3 * (2 + 4) * 2
        )
        assert (
            model["net_intermediate_bytes_removed_per_event"]
            == batch * 1_218_560
        )

    assert b1["qweight_bytes_per_head"] == b4["qweight_bytes_per_head"]
    assert b1["candidate_requested_bytes_per_head"] == 338_348_094
    assert b1["candidate_requested_bytes_per_event"] == 1_691_740_470
    assert b4["candidate_requested_bytes_per_head"] == 346_513_656
    assert b4["candidate_requested_bytes_per_event"] == 1_732_568_280
    try:
        builder.physical32_work_model(2)
    except ValueError:
        pass
    else:
        raise AssertionError("B2 must remain outside the closed work model")


def test_builder_is_pinned_default_off_and_makes_no_live_claim() -> None:
    source = BUILDER.read_text(encoding="ascii")

    assert 'EXPECTED_TORCH = "2.11.0+cu130"' in source
    assert 'EXPECTED_CUDA = "13.0"' in source
    assert 'EXPECTED_ARCH = "12.1a"' in source
    assert '"status": "BUILT_UNQUALIFIED"' in source
    assert '"performance_measurement": False' in source
    assert '"numerical_equality_claim": False' in source
    assert '"real_task_correctness": False' in source
    assert '"production_default_enabled": False' in source
    assert '"runtime_integration_present": False' in source
    assert '"full_logits_materialized": False' in source
    assert '"restrict_overlap_guard": (' in source


def test_fixed32_wide_path_has_no_other_logit_consumer() -> None:
    patcher = PATCHER.read_text(encoding="utf-8")

    assert "(_fr10_wide_choices_ok or _fr13_is_hydra23 or _fr13_is_fixed32)" in patcher
    assert "or _fr10_is_wide\n        ):" in patcher
    assert "_fr10_leaf_steps = frozenset()" in patcher
    assert "_fr10_consumes_root_leaf = (" in patcher
    assert "_fr10_is_cat3w or _fr10_is_cat6root or _fr10_is_cat10" in patcher
    assert "or _fr10_is_333" in patcher
    assert "!= (3, 3, 3, 3, 3)" in patcher
    assert "_fr10_wide_topk[0] = _fr13_root_top3" in patcher
    assert "_fr13_dg_wt = _fr13_step_top3" in patcher


def test_candidate_has_no_runtime_or_launcher_integration() -> None:
    op_name = "fr13_dfwd_fp8_top3"
    assert op_name not in PATCHER.read_text(encoding="utf-8")
    assert op_name not in LAUNCHER.read_text(encoding="utf-8")
