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


def test_source_build_has_only_suffix_candidate_and_defaults_off() -> None:
    text = Path("scripts/fr13_patch_fa2_tree_bias.py").read_text()

    assert 'parser.add_argument(\n        "--tree-bias-tile-earlyout",' in text
    assert "tree_bias_tile_earlyout: bool = False" in text
    assert "tree_splitkv" not in text
    assert "tree-splitkv" not in text
    assert "FR13_FA2_TREE_SPLITKV" not in text
    assert "params.o_batch_stride = max_seqlen_q * params.o_row_stride" not in text
