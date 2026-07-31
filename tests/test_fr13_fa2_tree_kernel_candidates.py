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
    translation_unit = tmp_path / "flash_fwd_split_hdim256_bf16_sm80.cu"
    stock = "\n".join(
        (
            '#include "namespace_config.h"',
            '#include "flash_fwd_launch_template.h"',
            "namespace FLASH_NAMESPACE {",
            module.STOCK_FIXED32_QUERY_INSTANTIATION,
            "} // namespace FLASH_NAMESPACE",
        )
    )
    translation_unit.write_text(stock)

    assert not module._patch_fixed32_query_translation_unit(translation_unit)
    assert translation_unit.read_text() == stock
    assert module._patch_fixed32_query_translation_unit(
        translation_unit,
        fixed32_query_tile16=True,
    )
    qrow_translation_unit = translation_unit.with_name(
        "flash_fwd_fr13_qrow16_hdim256_bf16_sm80.cu"
    )
    candidate = qrow_translation_unit.read_text()
    assert translation_unit.read_text() == stock
    assert candidate == module.FIXED32_QUERY_TILE16_TRANSLATION_UNIT
    assert module.STOCK_FIXED32_QUERY_INSTANTIATION not in candidate
    assert not module._patch_fixed32_query_translation_unit(
        translation_unit,
        fixed32_query_tile16=True,
    )

    assert "#include <cstdlib>" not in candidate
    assert "std::getenv" not in candidate
    assert '__attribute__((visibility("hidden")))' in candidate
    assert "fr13_run_mha_fwd_fixed32_qrow16" in candidate
    assert "constexpr size_t smem_size = TreeKernelTraits::kSmemSize" in candidate
    assert "dim3 grid(num_m_block, params.b, params.h)" in candidate
    assert "auto kernel = &flash_fwd_splitkv_kernel<" in candidate
    assert candidate.count("auto kernel = &flash_fwd_splitkv_kernel<") == 1
    assert "false,  // Is_causal" in candidate
    assert "false,  // Is_local" in candidate
    assert "false,  // Has_alibi" in candidate
    assert "false,  // Is_even_MN: paged varlen Q has cu_seqlens_q" in candidate
    assert "true,   // Is_even_K: d == kHeadDim == 256" in candidate
    assert "false,  // Is_softcap" in candidate
    assert "false,  // Split" in candidate
    assert "false   // Append_KV" in candidate
    assert "kernel<<<grid, TreeKernelTraits::kNThreads, smem_size, stream>>>" in candidate
    assert "C10_CUDA_KERNEL_LAUNCH_CHECK();" in candidate
    assert "run_flash_splitkv_fwd<StockKernelTraits, false>" not in candidate
    assert "AllowSplit" not in candidate
    assert "FR13_ALLOW_SPLIT_SWITCH" not in candidate
    assert "kTreeBlockM = 16" in candidate
    assert "kTreeBlockN = 64" in candidate
    assert "kTreeWarps = 1" in candidate
    assert "TreeKernelTraits::kNThreads == 32" in candidate
    assert "TreeKernelTraits::kGmemThreadsPerRow == 8" in candidate
    assert "TreeKernelTraits::kGmemRowsPerThread == 16" in candidate
    assert "1024 % TreeKernelTraits::kGmemRowsPerThread == 0" in candidate
    assert "public FA2 API requires paged-KV blocks divisible by 16" in candidate
    assert "flash_fwd_splitkv_combine_kernel" not in candidate
    assert "params.num_splits = " not in candidate

    api = module.FIXED32_QUERY_TILE16_API_DISPATCH
    stock_body = module.STOCK_RUN_MHA_FWD[
        module.STOCK_RUN_MHA_FWD.index("    FP16_SWITCH") : -1
    ]
    assert stock_body in api
    assert '__attribute__((visibility("hidden")))' in api
    assert "kFr13Qrow16BatchStrideSentinel" in api
    assert str(module.FIXED32_QUERY_TILE16_BATCH_STRIDE_SENTINEL) in api
    assert "TORCH_CHECK(" in api
    assert "internal dispatch reached non-production geometry" in api
    assert "params.tree_bias_ptr != nullptr" in api
    assert "params.is_bf16" in api
    assert "!params.is_causal" in api
    assert "params.b == 1" in api
    assert "params.d == 256" in api
    assert "params.d_rounded == 256" in api
    assert "params.h == 24" in api
    assert "params.h_k == 4" in api
    assert "params.h_h_k_ratio == 6" in api
    assert "params.seqlen_q == 32" in api
    assert "params.tree_bias_q_offset == 0" in api
    assert "params.tree_bias_k_offset == 0" in api
    assert "params.cu_seqlens_q != nullptr" in api
    assert "params.seqused_k != nullptr" in api
    assert "!params.seqlenq_ngroups_swapped" in api
    assert "params.block_table != nullptr" in api
    assert "params.page_block_size == 1024" in api
    assert "params.window_size_left < 0" in api
    assert "params.window_size_right < 0" in api
    assert "params.alibi_slopes_ptr == nullptr" in api
    assert "params.knew_ptr == nullptr" in api
    assert "params.softcap == 0.0f" in api
    assert "params.num_splits == 0" in api
    assert "params.num_splits == 1" not in api
    assert "set_params_fprop zero-initializes this field" in api
    assert "max_seqlen_q==1 q-group split-K setup" in api
    assert "force_split_kernel" in api

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
    assert 'parser.add_argument(\n        "--fixed32-query-tile16-live-ab",' in text
    assert "tree_bias_tile_earlyout: bool = False" in text
    assert "fixed32_query_tile16: bool = False" in text
    assert "tree_splitkv" not in text
    assert "tree-splitkv" not in text
    assert "FR13_FA2_TREE_SPLITKV" not in text
    assert "AllowSplit" not in text
    assert "FR13_ALLOW_SPLIT_SWITCH" not in text
    assert "params.o_batch_stride = max_seqlen_q * params.o_row_stride" not in text


def test_qrow16_capture_checker_is_compile_preflight_only() -> None:
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
    assert "QROW16_BATCH_STRIDE_SENTINEL = 0x46523133" in text
    assert "torch.as_strided(" in text
    assert "FR13_FA2_QROW16_INTERNAL_DISPATCH" not in text
    assert "candidate=True" in text


def test_qrow16_live_gate_uses_retained_paged_operands_after_real_replay() -> None:
    patcher = Path("scripts/fr13_patch_fa2_tree_bias.py").read_text()
    launcher = Path("scripts/fr13_launch_forked_fa2_tree_server.sh").read_text()
    gate_launcher = Path("scripts/fr13_run_b1_kernel_live_gate.sh").read_text()

    assert "FR13_FA2_QROW16_LIVE_PAGED_AB_REPLAY" in patcher
    replay = patcher.index('anchor = "        entry.cudagraph.replay()\\n"')
    gate = patcher.index("_fr13_fa2_qrow16_live_ab_replay(", replay)
    assert replay < gate
    assert '_FR13_FIXED32_CAPTURE_CONTEXT' in patcher
    assert 'int(descriptor.get("num_reqs", -1)) != 1' in patcher
    assert 'tuple(query.shape) == (32, 24, 256)' in patcher
    assert 'tuple(key_cache.shape[1:]) == (1024, 4, 256)' in patcher
    assert '"key_cache": key_cache' in patcher
    assert '"value_cache": value_cache' in patcher
    assert '"block_table": block_table' in patcher
    assert '"seqused_k": seqused_k' in patcher
    assert '"tree_bias": tree_bias' in patcher
    assert "torch.load" not in patcher[patcher.index("FIXED32_QUERY_TILE16_LIVE_AB_HELPERS") :]
    assert "dense_k" not in patcher[patcher.index("FIXED32_QUERY_TILE16_LIVE_AB_HELPERS") :]
    assert 'return_softmax_lse=True' in patcher
    assert 'stock_lse.dtype != torch.float32' in patcher
    assert 'view(torch.uint8)' in patcher
    assert '"raw_byte_mismatches": output_mismatches' in patcher
    assert '"raw_byte_mismatches": lse_mismatches' in patcher
    assert '"served_return": "stock captured graph output unchanged"' in patcher
    assert '"candidate_so_sha256": candidate_so_sha256' in patcher
    assert "entry.output" not in patcher[
        patcher.index("def _fr13_fa2_qrow16_live_ab_replay") :
        patcher.index("def _patch_tree_attn")
    ]

    assert "FR13_FA2_QROW16_LIVE_PAGED_AB=${FR13_FA2_QROW16_LIVE_PAGED_AB:-0}" in launcher
    assert "FR13 qrow16 internal selectors are launcher-private" in launcher
    assert "fixed32 qrow16 candidate FA2 sha256 mismatch" in launcher
    assert "--fixed32-query-tile16-live-ab" in launcher
    assert "FR13_GATE_QROW16=${FR13_GATE_QROW16:-0}" in gate_launcher
    assert 'FR13_FA2_QROW16_LIVE_PAGED_AB="$FR13_GATE_QROW16"' in gate_launcher
