from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = Path("scripts/fr13_patch_fa2_tree_bias.py")
    spec = importlib.util.spec_from_file_location("fr13_fa2_patch", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tile_overlaps_bias(
    n_block: int,
    block_n: int,
    context_len: int,
    bias_k_offset: int,
    bias_cols: int,
) -> bool:
    bias_begin = context_len + bias_k_offset
    bias_end = bias_begin + bias_cols
    block_begin = n_block * block_n
    block_end = block_begin + block_n
    return block_end > bias_begin and block_begin < bias_end


def _tile_has_mutable_column(
    n_block: int,
    block_n: int,
    context_len: int,
    bias_k_offset: int,
    bias_cols: int,
) -> bool:
    for column in range(n_block * block_n, (n_block + 1) * block_n):
        k_rel = column - context_len - bias_k_offset
        if 0 <= k_rel < bias_cols:
            return True
    return False


def test_tree_bias_tile_earlyout_is_independent_and_exact() -> None:
    module = _module()
    baseline = module._tree_bias_helper(tile_earlyout=False)
    candidate = module._tree_bias_helper(tile_earlyout=True)
    guard = module.TREE_BIAS_TILE_OVERLAP_GUARD

    assert "FR13_FA2_TREE_BIAS_TILE_EARLYOUT" not in baseline
    assert "FR13_FA2_TREE_BIAS_TILE_EARLYOUT" in candidate
    assert "block_col_end <= bias_col_begin" in candidate
    assert "block_col_begin >= bias_col_end" in candidate
    assert candidate.count(guard) == 1
    assert candidate.replace(guard, "", 1) == baseline

    for block_n in (32, 64, 128):
        for context_len in (0, 1, 31, 32, 33, 63, 64, 65, 14568):
            for bias_k_offset, bias_cols in ((0, 1), (0, 32), (5, 7), (31, 33)):
                last_column = context_len + bias_k_offset + bias_cols
                last_block = (last_column + block_n - 1) // block_n
                for n_block in range(last_block + 2):
                    assert _tile_overlaps_bias(
                        n_block,
                        block_n,
                        context_len,
                        bias_k_offset,
                        bias_cols,
                    ) == _tile_has_mutable_column(
                        n_block,
                        block_n,
                        context_len,
                        bias_k_offset,
                        bias_cols,
                    )


def _row_mapping(row: int, *, block_m: int, warps: int) -> tuple[int, int, int]:
    assert block_m == 16 * warps
    m_block = row // block_m
    row_in_block = row % block_m
    warp = row_in_block // 16
    return m_block, warp, row_in_block % 16


def test_fixed32_query_tile16_preserves_warp_local_row_mapping(tmp_path: Path) -> None:
    module = _module()
    launch = tmp_path / "flash_fwd_launch_template.h"
    stock = "\n".join(
        (
            module.STOCK_SPLITKV_LAUNCH_SIGNATURE,
            module.STOCK_SPLITKV_COMBINE_GUARD,
            module.STOCK_SPLITKV_DISPATCH,
        )
    )
    launch.write_text(stock)

    assert not module._patch_flash_fwd_launch_template(launch)
    assert launch.read_text() == stock
    assert module._patch_flash_fwd_launch_template(
        launch,
        fixed32_query_tile16=True,
    )
    candidate = launch.read_text()
    assert module.NO_COMBINE_SPLITKV_LAUNCH_SIGNATURE in candidate
    assert module.NO_COMBINE_SPLITKV_COMBINE_GUARD in candidate
    assert module.FIXED32_QUERY_TILE16_DISPATCH in candidate
    assert not module._patch_flash_fwd_launch_template(
        launch,
        fixed32_query_tile16=True,
    )

    assert "std::is_same_v<T, cutlass::bfloat16_t>" in candidate
    assert "Headdim == 256 && !Is_causal" in candidate
    assert "params.tree_bias_ptr != nullptr" in candidate
    assert "params.b == 1" in candidate
    assert "params.d == 256" in candidate
    assert "params.d_rounded == 256" in candidate
    assert "params.h == 24" in candidate
    assert "params.h_k == 4" in candidate
    assert "params.h_h_k_ratio == 6" in candidate
    assert "params.seqlen_q == 32" in candidate
    assert "params.tree_bias_q_offset == 0" in candidate
    assert "params.tree_bias_k_offset == 0" in candidate
    assert "params.cu_seqlens_q != nullptr" in candidate
    assert "!params.seqlenq_ngroups_swapped" in candidate
    assert "params.block_table != nullptr" in candidate
    assert "params.page_block_size == 1024" in candidate
    assert "params.window_size_left < 0" in candidate
    assert "params.window_size_right < 0" in candidate
    assert "params.alibi_slopes_ptr == nullptr" in candidate
    assert "params.knew_ptr == nullptr" in candidate
    assert "params.num_splits == 1" in candidate
    assert "run_flash_splitkv_fwd<TreeKernelTraits, Is_causal, false>" in candidate
    assert "if constexpr (AllowSplit)" in candidate
    assert "kTreeBlockM = 16" in candidate
    assert "kTreeWarps = 1" in candidate
    assert "TreeKernelTraits::kNThreads == 32" in candidate
    assert "TreeKernelTraits::kGmemThreadsPerRow == 8" in candidate
    assert "TreeKernelTraits::kGmemRowsPerThread == 16" in candidate
    assert "1024 % TreeKernelTraits::kGmemRowsPerThread == 0" in candidate
    assert "public FA2 API requires paged-KV blocks divisible by 16" in candidate
    assert "kBlockN" in candidate
    assert "splitkv_combine" not in candidate
    assert "params.num_splits = " not in candidate

    # The CTA id changes for rows 16..31, but the warp-local query-row/lane
    # coordinate is identical to the stock 64-row, four-warp tile.
    for row in range(32):
        _, _, stock_warp_row = _row_mapping(row, block_m=64, warps=4)
        _, candidate_warp, candidate_warp_row = _row_mapping(
            row,
            block_m=16,
            warps=1,
        )
        assert candidate_warp == 0
        assert candidate_warp_row == stock_warp_row


def test_source_build_candidates_are_independent_and_default_off() -> None:
    text = Path("scripts/fr13_patch_fa2_tree_bias.py").read_text()

    assert 'parser.add_argument(\n        "--tree-bias-tile-earlyout",' in text
    assert 'parser.add_argument(\n        "--fixed32-query-tile16",' in text
    assert "tree_bias_tile_earlyout: bool = False" in text
    assert "fixed32_query_tile16: bool = False" in text
    assert "tree_splitkv" not in text
    assert "tree-splitkv" not in text
    assert "FR13_FA2_TREE_SPLITKV" not in text
    assert "params.o_batch_stride = max_seqlen_q * params.o_row_stride" not in text


def test_qrow16_same_boot_gate_uses_real_b1_and_stock_batch_fallbacks() -> None:
    text = Path("scripts/fr13_fa2_qrow16_byte_ab.py").read_text()

    assert 'provenance.get("suite") != "SWE-Verified"' in text
    assert 'provenance.get("concurrency") != 1' in text
    assert 'provenance.get("physical_nodes") != 32' in text
    assert "for copies in (2, 4):" in text
    assert '"output_byte_equal": out_equal' in text
    assert '"lse_byte_equal": lse_equal' in text
    assert "return_softmax_lse=True" in text
    assert "num_splits=1" in text
    assert "block_table=block_table" in text
