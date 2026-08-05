from __future__ import annotations

import importlib.util
from fnmatch import fnmatchcase
from pathlib import Path


def _module():
    path = Path("scripts/fr13_patch_fa2_tree_bias.py")
    spec = importlib.util.spec_from_file_location("fr13_fa2_patch_pair", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gqa_pair_translation_unit_is_private_unsplit_and_exact_b4() -> None:
    module = _module()
    candidate = module.FIXED32_QUERY_GQA_PAIR32_TRANSLATION_UNIT

    assert candidate.startswith(
        "// FR13 fixed32 B4 qrow32 GQA-pair gate candidate."
    )
    assert "256, 64, 64, 4, false, false" in candidate
    assert "StaticQueryHeadsPerCTA<Fr13Fixed32Qrow32GqaPairKernelTraits>" in candidate
    assert "static constexpr int value = 2" in candidate
    assert "static constexpr int sequences = 4" in candidate
    assert "static constexpr int query_heads_per_kv = 6" in candidate
    assert "__global__ __maxnreg__(254)" in candidate
    assert "false,  // Split" in candidate
    assert "flash_fwd_splitkv_combine_kernel" not in candidate
    assert "params.num_splits" not in candidate
    assert "static_assert(smem_size == 96 * 1024)" in candidate
    assert "3 head pairs * B4 * 4 KV heads = 48 CTAs/layer" in candidate
    assert "StaticLayout::query_heads_per_kv / kHeadsPerCTA" in candidate


def test_gqa_pair_grid_covers_each_b4_query_head_once() -> None:
    observed: list[tuple[int, int]] = []
    kv_passes: dict[tuple[int, int], int] = {}

    for kv_head in range(4):
        for batch in range(4):
            for pair_lane in range(3):
                head_base = kv_head * 6 + pair_lane * 2
                observed.extend((batch, head_base + in_pair) for in_pair in range(2))
                kv_passes[(batch, kv_head)] = kv_passes.get((batch, kv_head), 0) + 1

    expected = [(batch, head) for batch in range(4) for head in range(24)]
    assert sorted(observed) == expected
    assert len(observed) == len(set(observed)) == 96
    assert set(kv_passes.values()) == {3}
    assert sum(kv_passes.values()) == 48


def test_gqa_pair_hierarchical_q_o_lse_addresses_match_scalar_heads() -> None:
    query_rows = 32
    query_heads = 24
    head_dim = 256
    total_q = 4 * query_rows
    q_row_stride = 32 * head_dim
    q_head_stride = head_dim
    o_row_stride = query_heads * head_dim
    o_head_stride = head_dim

    for batch in range(4):
        for kv_head in range(4):
            for pair_lane in range(3):
                head_base = kv_head * 6 + pair_lane * 2
                for logical_row in range(64):
                    row = logical_row % query_rows
                    head = head_base + logical_row // query_rows
                    for column in (0, 127, 255):
                        scalar_q = (
                            batch * query_rows * q_row_stride
                            + row * q_row_stride
                            + head * q_head_stride
                            + column
                        )
                        grouped_q = (
                            batch * query_rows * q_row_stride
                            + head_base * q_head_stride
                            + row * q_row_stride
                            + (logical_row // query_rows) * q_head_stride
                            + column
                        )
                        scalar_o = (
                            batch * query_rows * o_row_stride
                            + row * o_row_stride
                            + head * o_head_stride
                            + column
                        )
                        grouped_o = (
                            batch * query_rows * o_row_stride
                            + head_base * o_head_stride
                            + row * o_row_stride
                            + (logical_row // query_rows) * o_head_stride
                            + column
                        )
                        assert grouped_q == scalar_q
                        assert grouped_o == scalar_o

                    scalar_lse = head * total_q + batch * query_rows + row
                    grouped_lse = (
                        head_base * total_q
                        + batch * query_rows
                        + row
                        + (logical_row // query_rows) * total_q
                    )
                    assert grouped_lse == scalar_lse


def test_gqa_pair_reuses_incumbent_warp_local_row_order() -> None:
    for logical_row in range(64):
        head_in_pair = logical_row // 32
        query_row = logical_row % 32
        pair_warp = logical_row // 16
        pair_warp_row = logical_row % 16
        incumbent_warp = query_row // 16
        incumbent_warp_row = query_row % 16

        assert pair_warp % 2 == incumbent_warp
        assert pair_warp // 2 == head_in_pair
        assert pair_warp_row == incumbent_warp_row
        assert logical_row % 32 == query_row


def test_gqa_pair_source_gate_and_generated_name_are_isolated() -> None:
    module = _module()
    declaration = module.FIXED32_QUERY_GQA_PAIR32_API_DECLARATION
    gate = module.FIXED32_QUERY_GQA_PAIR32_API_GATE
    source = Path("scripts/fr13_patch_fa2_tree_bias.py").read_text()

    assert module.FIXED32_QUERY_GQA_PAIR32_BATCH_STRIDE_SENTINEL == 0x20014
    assert "fr13_run_mha_fwd_fixed32_qrow32_gqa_pair" in declaration
    assert "params.b == 4" in gate
    assert "params.h_h_k_ratio == 6" in gate
    assert "params.seqlen_q == 32" in gate
    assert "params.q_head_stride == 256" in gate
    assert "params.q_row_stride ==" not in gate
    assert "params.unpadded_lse" in gate
    assert "params.num_splits == 0" in gate
    assert "force_split_kernel" in gate
    assert "--fixed32-query-gqa-pair32" in source
    assert "fixed32_query_gqa_pair32: bool = False" in source
    assert fnmatchcase(
        "flash_fwd_fr13_qrow32_gqa_pair_hdim256_bf16_sm80.cu",
        "flash_fwd_*.cu",
    )


def test_gqa_pair_translation_unit_writer_is_exact_and_idempotent(
    tmp_path: Path,
) -> None:
    module = _module()
    stock_path = tmp_path / "flash_fwd_split_hdim256_bf16_sm80.cu"
    stock = "\n".join(
        (
            '#include "namespace_config.h"',
            '#include "flash_fwd_launch_template.h"',
            "namespace FLASH_NAMESPACE {",
            module.STOCK_FIXED32_QUERY_INSTANTIATION,
            "} // namespace FLASH_NAMESPACE",
        )
    )
    stock_path.write_text(stock)

    assert not module._patch_fixed32_query_gqa_pair32_translation_unit(stock_path)
    assert module._patch_fixed32_query_gqa_pair32_translation_unit(
        stock_path,
        fixed32_query_gqa_pair32=True,
    )
    pair_path = stock_path.with_name(
        "flash_fwd_fr13_qrow32_gqa_pair_hdim256_bf16_sm80.cu"
    )
    assert pair_path.read_text() == module.FIXED32_QUERY_GQA_PAIR32_TRANSLATION_UNIT
    assert stock_path.read_text() == stock
    assert not module._patch_fixed32_query_gqa_pair32_translation_unit(
        stock_path,
        fixed32_query_gqa_pair32=True,
    )
