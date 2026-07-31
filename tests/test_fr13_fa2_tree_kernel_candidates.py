from __future__ import annotations

import importlib.util
import math
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


def _num_splits_heuristic(
    batch_nheads_mblocks: int,
    num_sms: int,
    num_n_blocks: int,
    max_splits: int = 128,
) -> int:
    # Mirrors the pinned FA2 C++ helper. It models two 128-thread CTAs per SM.
    wave_slots = num_sms * 2
    if batch_nheads_mblocks >= 0.8 * wave_slots:
        return 1
    max_splits = min(max_splits, wave_slots, num_n_blocks)

    def blocks_per_split(splits: int) -> int:
        return math.ceil(num_n_blocks / splits)

    def eligible(splits: int) -> bool:
        return splits == 1 or blocks_per_split(splits) != blocks_per_split(splits - 1)

    efficiencies = []
    for splits in range(1, max_splits + 1):
        if not eligible(splits):
            efficiencies.append(0.0)
            continue
        waves = batch_nheads_mblocks * splits / wave_slots
        efficiencies.append(waves / math.ceil(waves))

    threshold = 0.85 * max(efficiencies)
    for splits, efficiency in enumerate(efficiencies, start=1):
        if eligible(splits) and efficiency >= threshold:
            return splits
    return 1


def test_tree_bias_tile_earlyout_is_independent_and_exact() -> None:
    module = _module()
    baseline = module._tree_bias_helper(tile_earlyout=False)
    candidate = module._tree_bias_helper(tile_earlyout=True)

    assert "FR13_FA2_TREE_BIAS_TILE_EARLYOUT" not in baseline
    assert "FR13_FA2_TREE_BIAS_TILE_EARLYOUT" in candidate
    assert "block_col_end <= bias_col_begin" in candidate
    assert "block_col_begin >= bias_col_end" in candidate

    for context_len in (0, 1, 31, 32, 33, 14568):
        for bias_k_offset, bias_cols in ((0, 32), (5, 7)):
            last_block = math.ceil((context_len + bias_k_offset + bias_cols) / 64)
            for n_block in range(last_block + 2):
                assert _tile_overlaps_bias(
                    n_block,
                    64,
                    context_len,
                    bias_k_offset,
                    bias_cols,
                ) == _tile_has_mutable_column(
                    n_block,
                    64,
                    context_len,
                    bias_k_offset,
                    bias_cols,
                )


def test_tree_splitkv_source_block_is_independently_reversible() -> None:
    module = _module()
    source = "prefix\n" + module.TREE_SPLITKV_BASE_BLOCK + "suffix\n"

    enabled, changed = module._configure_tree_splitkv(source, enabled=True)
    assert changed
    assert module.TREE_SPLITKV_ENABLED_BLOCK in enabled
    assert module.TREE_SPLITKV_BASE_BLOCK not in enabled

    enabled_again, changed = module._configure_tree_splitkv(enabled, enabled=True)
    assert not changed
    assert enabled_again == enabled

    disabled, changed = module._configure_tree_splitkv(enabled, enabled=False)
    assert changed
    assert disabled == source


def test_tree_splitkv_guard_covers_fixed_dense_varlen_combine() -> None:
    candidate = _module().TREE_SPLITKV_ENABLED_BLOCK

    assert "paged_KV && tree_bias_.has_value()" in candidate
    assert "max_seqlen_q > 1 && max_seqlen_q <= 64" in candidate
    assert "total_q == batch_size * max_seqlen_q" in candidate
    assert (
        "params.o_batch_stride = max_seqlen_q * params.o_row_stride;" in candidate
    )
    assert "seqlenq_ngroups_swapped || fr13_tree_splitkv" in candidate
    assert "set_params_splitkv(" in candidate


def test_gb10_fixed32_split_heuristic_selects_b1_four_b4_one() -> None:
    # Runtime Nsight target info reports 48 GB10 SMs. Fixed32 is one M tile,
    # Qwen has 24 query heads, and a 14.6k context spans 229 N=64 tiles.
    n_blocks = math.ceil(14600 / 64)

    assert _num_splits_heuristic(1 * 24 * 1, 48, n_blocks) == 4
    assert _num_splits_heuristic(4 * 24 * 1, 48, n_blocks) == 1


def test_source_build_flags_are_separate_and_default_off() -> None:
    text = Path("scripts/fr13_patch_fa2_tree_bias.py").read_text()

    assert 'parser.add_argument(\n        "--tree-bias-tile-earlyout",' in text
    assert 'parser.add_argument(\n        "--tree-splitkv",' in text
    assert "tree_bias_tile_earlyout: bool = False" in text
    assert "tree_splitkv: bool = False" in text
