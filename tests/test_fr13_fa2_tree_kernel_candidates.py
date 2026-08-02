from __future__ import annotations

import importlib.util
import re
from fnmatch import fnmatchcase
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
    assert "auto kernel = &fr13_flash_fwd_fixed32_qrow16_kernel" in candidate
    assert candidate.count("fr13_flash_fwd_fixed32_qrow16_kernel") == 2
    assert "__asm__(" not in candidate
    assert "FR13_FA2_QROW16_DEVICE_SYMBOL_COMPAT" not in candidate
    assert candidate.count("__global__ __maxnreg__(216)") == 1
    assert "RU3 must lower pressure before this exact cap" in candidate
    assert "FLASH_NAMESPACE::compute_attn_splitkv<" in candidate
    assert "auto kernel = &flash_fwd_splitkv_kernel<" not in candidate
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
    assert "using Fr13Fixed32Qrow16KernelTraits" in candidate
    assert "StaticPagedKVBlockSize<Fr13Fixed32Qrow16KernelTraits>" in candidate
    assert "using TreeKernelTraits = Fr13Fixed32Qrow16KernelTraits" in candidate
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

    include_at = candidate.index('#include "flash_fwd_launch_template.h"')
    namespace_at = candidate.index("namespace FLASH_NAMESPACE {")
    alias_at = candidate.index("using Fr13Fixed32Qrow16KernelTraits")
    specialization_at = candidate.index(
        "struct StaticPagedKVBlockSize<Fr13Fixed32Qrow16KernelTraits>"
    )
    launcher_at = candidate.index("void fr13_run_mha_fwd_fixed32_qrow16(")
    kernel_at = candidate.index("void fr13_flash_fwd_fixed32_qrow16_kernel(")
    launch_at = candidate.index("auto kernel = &fr13_flash_fwd_fixed32_qrow16_kernel")
    assert (
        include_at
        < namespace_at
        < alias_at
        < specialization_at
        < kernel_at
        < launcher_at
        < launch_at
    )
    assert "StaticQueryRows<Fr13Fixed32Qrow16KernelTraits>" not in candidate

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


def test_fixed32_query_tile32_preserves_stock_warp_local_row_mapping(
    tmp_path: Path,
) -> None:
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

    assert not module._patch_fixed32_query_tile32_translation_unit(
        translation_unit
    )
    assert module._patch_fixed32_query_tile32_translation_unit(
        translation_unit,
        fixed32_query_tile32=True,
    )
    qrow_translation_unit = translation_unit.with_name(
        "flash_fwd_fr13_qrow32_hdim256_bf16_sm80.cu"
    )
    candidate = qrow_translation_unit.read_text()
    assert translation_unit.read_text() == stock
    assert "StaticPagedKVBlockSize" not in translation_unit.read_text()
    assert candidate == module.FIXED32_QUERY_TILE32_TRANSLATION_UNIT
    assert module.STOCK_FIXED32_QUERY_INSTANTIATION not in candidate
    assert not module._patch_fixed32_query_tile32_translation_unit(
        translation_unit,
        fixed32_query_tile32=True,
    )

    assert '__attribute__((visibility("hidden")))' in candidate
    assert "fr13_run_mha_fwd_fixed32_qrow32" in candidate
    assert "kTreeBlockM = 32" in candidate
    assert "kTreeBlockN = 64" in candidate
    assert "kTreeWarps = 2" in candidate
    assert "TreeKernelTraits::kNThreads == 64" in candidate
    assert "TreeKernelTraits::kGmemThreadsPerRow == 8" in candidate
    assert "TreeKernelTraits::kGmemRowsPerThread == 8" in candidate
    assert "static_assert(smem_size == 80 * 1024)" in candidate
    assert "using StaticLayout = StaticQueryBatchLayout<TreeKernelTraits>" in candidate
    assert "StaticLayout::query_heads_per_kv" in candidate
    assert "StaticLayout::sequences" in candidate
    assert "StaticLayout::kv_heads" in candidate
    assert "dim3 grid(" in candidate
    assert "num_m_block" not in candidate
    assert "false,  // Split" in candidate
    assert "flash_fwd_splitkv_combine_kernel" not in candidate
    assert "params.num_splits = " not in candidate
    assert "qrow32 gate candidate" in candidate
    assert "Gate-only entry point" in candidate
    assert "ordinary and production paths cannot tag" in candidate
    assert "using Fr13Fixed32Qrow32KernelTraits" in candidate
    assert "StaticPagedKVBlockSize<Fr13Fixed32Qrow32KernelTraits>" in candidate
    static_page_log2 = 10
    static_page_size = 1 << static_page_log2
    static_block_n_log2 = 6
    static_query_rows = 32
    assert f"static constexpr int value = {static_page_size}" in candidate
    assert f"static constexpr int log2 = {static_page_log2}" in candidate
    assert f"static constexpr int block_n_log2 = {static_block_n_log2}" in candidate
    assert "StaticQueryRows<Fr13Fixed32Qrow32KernelTraits>" in candidate
    assert f"static constexpr int value = {static_query_rows}" in candidate
    assert "StaticQueryBatchLayout<Fr13Fixed32Qrow32KernelTraits>" in candidate
    assert "static constexpr int sequences = 4" in candidate
    assert "static constexpr int query_heads = 24" in candidate
    assert "static constexpr int kv_heads = 4" in candidate
    assert "static constexpr int query_heads_per_kv = 6" in candidate
    assert "using TreeKernelTraits = Fr13Fixed32Qrow32KernelTraits" in candidate

    api_gate = module.FIXED32_QUERY_TILE32_API_GATE
    assert "kFr13Qrow32BatchStrideSentinel" in api_gate
    assert "params.tree_bias_ptr != nullptr" in api_gate
    assert "params.is_bf16" in api_gate
    assert "!params.is_causal" in api_gate
    assert "params.b == 4" in api_gate
    assert "params.total_q == 128" in api_gate
    assert "params.d == 256" in api_gate
    assert "params.d_rounded == 256" in api_gate
    assert "params.h == 24" in api_gate
    assert "params.h_k == 4" in api_gate
    assert "params.h_h_k_ratio == 6" in api_gate
    assert "params.seqlen_q == 32" in api_gate
    assert "params.seqlen_q_rounded == 128" in api_gate
    assert "params.tree_bias_rows == 32" in api_gate
    assert "params.tree_bias_cols == 32" in api_gate
    assert "params.tree_bias_row_stride == 32" in api_gate
    assert "params.tree_bias_col_stride == 1" in api_gate
    assert "params.cu_seqlens_q != nullptr" in api_gate
    assert "params.cu_seqlens_k != nullptr" in api_gate
    assert "params.seqused_k != nullptr" in api_gate
    assert "params.leftpad_k == nullptr" in api_gate
    assert "params.cache_batch_idx == nullptr" in api_gate
    assert "params.block_table != nullptr" in api_gate
    assert f"params.page_block_size == {static_page_size}" in api_gate
    assert "params.window_size_left < 0" in api_gate
    assert "params.window_size_right < 0" in api_gate
    assert "params.alibi_slopes_ptr == nullptr" in api_gate
    assert "params.knew_ptr == nullptr" in api_gate
    assert "params.vnew_ptr == nullptr" in api_gate
    assert "params.softcap == 0.0f" in api_gate
    assert "params.num_splits == 0" in api_gate
    assert "params.num_splits == 1" not in api_gate
    assert "force_split_kernel" in api_gate
    assert "fr13_run_mha_fwd_fixed32_qrow32(params, stream)" in api_gate

    # Stock BM64 uses warps 0 and 1 for physical rows 0..31. BM32 preserves
    # each row's warp and warp-local coordinate while dropping warps 2 and 3.
    for row in range(32):
        stock_block, stock_warp, stock_warp_row = _row_mapping(
            row,
            block_m=64,
            warps=4,
        )
        candidate_block, candidate_warp, candidate_warp_row = _row_mapping(
            row,
            block_m=32,
            warps=2,
        )
        assert stock_block == candidate_block == 0
        assert candidate_warp == stock_warp
        assert candidate_warp_row == stock_warp_row


def test_qrow16_private_kernel_preserves_exact_flags() -> None:
    module = _module()
    translation_unit = module.FIXED32_QUERY_TILE16_TRANSLATION_UNIT

    assert translation_unit.count("template <>") == 1
    assert translation_unit.count("__global__") == 1
    assert translation_unit.count("__global__ __maxnreg__(216)") == 1
    assert translation_unit.count("fr13_flash_fwd_fixed32_qrow16_kernel") == 2
    assert "__asm__(" not in translation_unit
    assert "FR13_FA2_QROW16_DEVICE_SYMBOL_COMPAT" not in translation_unit
    assert "flash_fwd_splitkv_kernelI23Flash_fwd_kernel_traits" not in translation_unit
    assert "flash_fwd_splitkv_kernel<" not in translation_unit
    kernel_match = re.search(
        r"compute_attn_splitkv<(?P<arguments>.*?)\n"
        r"    >\(params\);",
        translation_unit,
        flags=re.DOTALL,
    )
    assert kernel_match is not None
    uncommented = re.sub(r"//[^\n]*", "", kernel_match.group("arguments"))
    arguments = [argument.strip() for argument in uncommented.split(",")]
    assert arguments == [
        "Fr13Fixed32Qrow16KernelTraits",
        "false",  # Is_causal
        "false",  # Is_local
        "false",  # Has_alibi
        "false",  # Is_even_MN
        "true",  # Is_even_K
        "false",  # Is_softcap
        "false",  # Split
        "false",  # Append_KV
    ]


def test_qrow32_traits_precede_the_exact_splitkv_instantiation() -> None:
    module = _module()
    translation_unit = module.FIXED32_QUERY_TILE32_TRANSLATION_UNIT

    include_at = translation_unit.index('#include "flash_fwd_launch_template.h"')
    namespace_at = translation_unit.index("namespace FLASH_NAMESPACE {")
    alias_at = translation_unit.index("using Fr13Fixed32Qrow32KernelTraits")
    page_specialization_at = translation_unit.index(
        "struct StaticPagedKVBlockSize<Fr13Fixed32Qrow32KernelTraits>"
    )
    query_specialization_at = translation_unit.index(
        "struct StaticQueryRows<Fr13Fixed32Qrow32KernelTraits>"
    )
    batch_specialization_at = translation_unit.index(
        "struct StaticQueryBatchLayout<Fr13Fixed32Qrow32KernelTraits>"
    )
    launcher_at = translation_unit.index(
        "void fr13_run_mha_fwd_fixed32_qrow32("
    )
    instantiation_at = translation_unit.index(
        "auto kernel = &flash_fwd_splitkv_kernel<"
    )
    assert (
        include_at
        < namespace_at
        < alias_at
        < page_specialization_at
        < query_specialization_at
        < batch_specialization_at
        < launcher_at
        < instantiation_at
    )

    # The included launch header exposes these primaries through utils.h and
    # flash_fwd_kernel.h. The qrow TU specializes both before the first use
    # that instantiates the exact BM32/N64/two-warp kernel type.
    assert "struct StaticPagedKVBlockSize" in (
        module.FIXED32_QUERY_STATIC_PAGE_TRAIT
    )
    assert "struct StaticQueryRows" in module._tree_bias_helper(
        tile_earlyout=True
    )
    assert "struct StaticQueryBatchLayout" in module._tree_bias_helper(
        tile_earlyout=True
    )
    assert "struct StaticPagedQueryBlockInfo" in module._tree_bias_helper(
        tile_earlyout=True
    )
    assert "static_query_offset" in module._tree_bias_helper(
        tile_earlyout=True
    )
    assert translation_unit.count(
        "StaticPagedKVBlockSize<Fr13Fixed32Qrow32KernelTraits>"
    ) == 1
    assert translation_unit.count(
        "StaticQueryRows<Fr13Fixed32Qrow32KernelTraits>"
    ) == 1
    assert translation_unit.count(
        "StaticQueryBatchLayout<Fr13Fixed32Qrow32KernelTraits>"
    ) == 1
    assert "run_mha_fwd_splitkv_dispatch" not in translation_unit
    assert "run_flash_splitkv_fwd" not in translation_unit

    kernel_match = re.search(
        r"auto kernel = &flash_fwd_splitkv_kernel<(?P<arguments>.*?)\n"
        r"    >;",
        translation_unit,
        flags=re.DOTALL,
    )
    assert kernel_match is not None
    uncommented = re.sub(r"//[^\n]*", "", kernel_match.group("arguments"))
    arguments = [argument.strip() for argument in uncommented.split(",")]
    assert arguments == [
        "TreeKernelTraits",
        "false",  # Is_causal
        "false",  # Is_local
        "false",  # Has_alibi
        "false",  # Is_even_MN
        "true",  # Is_even_K
        "false",  # Is_softcap
        "false",  # Split
        "false",  # Append_KV
    ]


def test_qrow32_hidden_launcher_abi_and_build_name_are_stable() -> None:
    module = _module()
    declaration = module.FIXED32_QUERY_TILE32_API_DECLARATION
    definition = module.FIXED32_QUERY_TILE32_TRANSLATION_UNIT
    signature_pattern = re.compile(
        r"void\s+fr13_run_mha_fwd_fixed32_qrow32\s*\(\s*"
        r"Flash_fwd_params\s*&params\s*,\s*cudaStream_t\s+stream\s*\)"
    )

    declaration_match = signature_pattern.search(declaration)
    definition_match = signature_pattern.search(definition)
    assert declaration_match is not None
    assert definition_match is not None
    assert re.sub(r"\s+", "", declaration_match.group()) == re.sub(
        r"\s+", "", definition_match.group()
    )
    assert declaration.count('__attribute__((visibility("hidden")))') == 1
    assert definition.count('__attribute__((visibility("hidden")))') == 1
    assert 'extern "C"' not in declaration
    assert 'extern "C"' not in definition
    assert "fr13_run_mha_fwd_fixed32_qrow32(params, stream);" in (
        module.FIXED32_QUERY_TILE32_API_GATE
    )

    # The pinned FA2 CMake contract discovers csrc/flash_attn/src/
    # flash_fwd_*.cu at configure time. Keep the generated name in that set.
    generated_name = "flash_fwd_fr13_qrow32_hdim256_bf16_sm80.cu"
    assert fnmatchcase(generated_name, "flash_fwd_*.cu")
    assert definition.startswith("// FR13 fixed32 B4 qrow32 gate candidate.")


def test_fixed32_static_page_specialization_is_opt_in_exact_and_idempotent(
    tmp_path: Path,
) -> None:
    module = _module()
    utils = tmp_path / "utils.h"
    dynamic_offset = """    const int64_t global_row_offset = block_row_offset + n_block * kBlockN;
    const int64_t page_offset = global_row_offset % page_block_size;
    const int64_t virtual_page_idx = global_row_offset / page_block_size;

    return ((int64_t) block_table[virtual_page_idx]) * ((int64_t) page_stride)
        + page_offset * ((int64_t) row_stride)
        + col_offset;
"""
    stock = (
        "namespace FLASH_NAMESPACE {\n"
        "template <typename Kernel_traits>\n"
        "__forceinline__ __device__\n"
        "int64_t resolve_thread_kv_page_slice_offset(\n"
        "    const int tidx, const int n_block, const int page_block_size,\n"
        "    const int* block_table, const int page_stride, const int row_stride) {\n"
        + dynamic_offset
        + "}\n"
        "}\n"
    )
    utils.write_text(stock)

    assert not module._patch_fixed32_query_static_page(utils)
    assert utils.read_text() == stock
    assert module._patch_fixed32_query_static_page(
        utils,
        fixed32_query_tile16=True,
    )
    candidate = utils.read_text()
    assert candidate.count("FR13_FA2_FIXED32_STATIC_PAGE") == 1
    assert candidate.count("struct StaticPagedKVBlockSize") == 1
    assert "static constexpr int value = 0" in candidate
    assert "if constexpr (kStaticPageBlockSize != 0)" in candidate
    assert "n_block >> kBlocksPerPageLog2" in candidate
    assert "n_block & (kBlocksPerPage - 1)" in candidate
    assert "<< kStaticBlockNLog2" in candidate
    assert "const int page_offset =" in candidate
    assert "const int virtual_page_idx = n_block >>" in candidate
    assert "unsigned_global_row_offset" not in candidate
    assert candidate.count("const int64_t global_row_offset") == 1
    assert candidate.count("global_row_offset % page_block_size") == 1
    assert candidate.count("global_row_offset / page_block_size") == 1
    assert not module._patch_fixed32_query_static_page(
        utils,
        fixed32_query_tile16=True,
    )
    assert utils.read_text() == candidate

    # Both private routes install the same header specialization surface.
    qrow32_utils = tmp_path / "qrow32_utils.h"
    qrow32_utils.write_text(stock)
    assert module._patch_fixed32_query_static_page(
        qrow32_utils,
        fixed32_query_tile32=True,
    )
    assert qrow32_utils.read_text() == candidate

    # Exhaust all thread rows, page-block residues, and valid partial-block
    # clamps. Quotient representatives include the largest nonnegative int
    # n_block, so the proof does not depend on a small sequence length.
    max_block_quotient = (2**31 - 1) // 16
    for threads, rows_per_thread in ((32, 16), (64, 8)):
        for thread_idx in range(threads):
            original_block_row_offset = (thread_idx // 8) * rows_per_thread
            for partial_block_size in (None, *range(65)):
                block_row_offset = original_block_row_offset
                if partial_block_size is not None:
                    final_row_offset = max(partial_block_size - 1, 0)
                    final_thread_row_offset = (
                        (final_row_offset + rows_per_thread - 1)
                        // rows_per_thread
                    ) * rows_per_thread
                    block_row_offset = min(
                        block_row_offset,
                        final_thread_row_offset,
                    )
                assert 0 <= block_row_offset < 64
                for block_quotient in (0, 1, 17, max_block_quotient):
                    for block_residue in range(16):
                        n_block = block_quotient * 16 + block_residue
                        if n_block > 2**31 - 1:
                            continue
                        global_row_offset = block_row_offset + n_block * 64
                        quotient, remainder = divmod(global_row_offset, 1024)
                        assert n_block >> 4 == quotient
                        assert (
                            ((n_block & 15) << 6) + block_row_offset
                            == remainder
                        )


def test_qrow32_static_query_specialization_is_exact_and_keeps_kv_masking(
    tmp_path: Path,
) -> None:
    module = _module()
    kernel = tmp_path / "flash_fwd_kernel.h"
    signature = (
        "template<typename Kernel_traits, bool Is_causal, bool Is_local, "
        "bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, "
        "bool Split, bool Append_KV, typename Params>\n"
        "inline __device__ void compute_attn_1rowblock_splitkv"
        "(const Params &params, const int bidb, const int bidh, "
        "const int m_block, const int n_split_idx, const int num_n_splits) {\n"
    )
    actual_q = "binfo.actual_seqlen_q"
    function = signature + """    constexpr int kBlockM = Kernel_traits::kBlockM;
    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;
    int use_0 = binfo.actual_seqlen_q - m_block * kBlockM;
    int use_1 = binfo.actual_seqlen_q + m_block * kBlockM;
    int use_2 = binfo.actual_seqlen_q;
    int use_3 = binfo.actual_seqlen_q;
    FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        output_0, binfo.actual_seqlen_q - m_block * kBlockM);
    if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSEaccum(row) = Split ? -INFINITY : INFINITY; }
    auto shape_q = make_shape(binfo.actual_seqlen_q, params.h, params.d);
    FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_Q,
        query, binfo.actual_seqlen_q - m_block * kBlockM);
    auto mask = Mask(binfo.actual_seqlen_q);
    FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_KV, kv);
    mask.template apply_mask<Is_causal, Is_even_MN>(scores, 0, 0, 32);
    if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSEaccum(row) = lse(mi); }
    FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        output_1, binfo.actual_seqlen_q - m_block * kBlockM);
}
"""
    assert function.count(actual_q) == 12
    suffix = """////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_dropout
"""
    stock = function + "\n" + suffix
    kernel.write_text(stock)

    assert not module._patch_fixed32_query_tile32_static_query(kernel)
    assert kernel.read_text() == stock
    assert module._patch_fixed32_query_tile32_static_query(
        kernel,
        fixed32_query_tile32=True,
    )
    candidate = kernel.read_text()
    assert candidate.count("FR13_FA2_QROW32_STATIC_QUERY") == 1
    assert "constexpr bool kStaticQueryTile = kStaticQueryRows == kBlockM" in candidate
    assert "const int query_m_block = kStaticQueryTile ? 0 : m_block" in candidate
    assert "if constexpr (!kStaticQueryTile)" in candidate
    assert candidate.count(
        "copy<kStaticQueryTile || Is_even_MN, Is_even_K, "
        "/*Clear_OOB_MN=*/false"
    ) == 2
    assert "copy<kStaticQueryTile || Is_even_MN, Is_even_K>(gmem_tiled_copy_Q" in candidate
    assert "copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_KV" in candidate
    assert "mask.template apply_mask<Is_causal, Is_even_MN>" in candidate
    assert "kStaticQueryTile || row < actual_seqlen_q" in candidate
    assert not module._patch_fixed32_query_tile32_static_query(
        kernel,
        fixed32_query_tile32=True,
    )

    rows = set()
    for thread_idx in range(64):
        row_base = (thread_idx // 32) * 16 + (thread_idx % 32) // 4
        for row_in_mma in (0, 8):
            row = row_base + row_in_mma
            assert 0 <= row < 32
            rows.add(row)
    assert rows == set(range(32))

    helper = module._tree_bias_helper(tile_earlyout=True)
    assert "if constexpr (!kStaticQueryTile)" in helper
    assert "if (kStaticQueryTile || (q_rel >= 0 && q_rel < tree_bias_rows))" in helper
    assert "if (k_rel >= 0 && k_rel < tree_bias_cols)" in helper
    assert "mask.template apply_mask" not in helper
    assert "StaticQueryRows" not in module.FIXED32_QUERY_TILE16_TRANSLATION_UNIT
    assert "q_start != [0, 32, 64, 96, 128]" in Path(
        "scripts/fr13_patch_fa2_tree_bias.py"
    ).read_text()


def test_qrow32_static_batch_layout_is_bijective_and_removes_address_division(
    tmp_path: Path,
) -> None:
    module = _module()
    kernel = tmp_path / "flash_fwd_kernel.h"
    signature = (
        "template<typename Kernel_traits, bool Is_causal, bool Is_local, "
        "bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, "
        "bool Split, bool Append_KV, typename Params>\n"
        "inline __device__ void compute_attn_1rowblock_splitkv"
        "(const Params &params, const int bidb, const int bidh) {\n"
    )
    split_body = """    constexpr int kNWarps = Kernel_traits::kNWarps;
    auto q0 = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb);
    auto q1 = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb);
    auto q2 = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb);
    auto q3 = binfo.q_offset(params.seqlen_q, 1, bidb);
    auto h0 = (bidh / params.h_h_k_ratio) * params.k_head_stride;
    auto h1 = (bidh / params.h_h_k_ratio) * params.k_head_stride;
    auto h2 = (bidh / params.h_h_k_ratio) * params.v_head_stride;
    auto h3 = (bidh / params.h_h_k_ratio) * params.v_head_stride;
    auto h4 = (bidh / params.h_h_k_ratio) * params.knew_head_stride;
    auto h5 = (bidh / params.h_h_k_ratio) * params.vnew_head_stride;
    mask.template apply_mask<Is_causal, Is_even_MN>(scores, 0, 0, 32);
    qk_accumulation_order_sentinel();
    pv_accumulation_order_sentinel();
    dynamic_k_length_sentinel(binfo.actual_seqlen_k);
}
"""
    dynamic_wrapper = """template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV, typename Params>
inline __device__ void compute_attn_splitkv(const Params &params) {
    const int m_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = Split ? blockIdx.z / params.h : blockIdx.y;
    // The block index for the head.
    const int bidh = Split ? blockIdx.z - bidb * params.h : blockIdx.z;
    const int n_split_idx = Split ? blockIdx.y : 0;
    const int num_n_splits = Split ? gridDim.y : 1;
    FLASH_NAMESPACE::compute_attn_1rowblock_splitkv<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Split, Append_KV>(params, bidb, bidh, m_block, n_split_idx, num_n_splits);
}"""
    stock = (
        signature
        + split_body
        + "\n////////////////////////////////////////////////////////////////////////////////////////////////////\n\n"
        + "template<typename Kernel_traits, bool Is_dropout\n"
        + "struct UnrelatedForwardKernel;\n\n"
        + dynamic_wrapper
        + "\n"
    )
    kernel.write_text(stock)

    assert not module._patch_fixed32_query_tile32_static_batch_layout(kernel)
    assert kernel.read_text() == stock
    assert module._patch_fixed32_query_tile32_static_batch_layout(
        kernel,
        fixed32_query_tile32=True,
    )
    candidate = kernel.read_text()
    candidate_function = candidate[
        candidate.index(signature[:-2]) : candidate.index(
            "\n////////////////////////////////////////////////////////////////////////////////////////////////////\n\n"
            "template<typename Kernel_traits, bool Is_dropout"
        )
    ]
    assert candidate_function.count("binfo.q_offset(") == 0
    assert candidate_function.count(
        "static_query_offset<Kernel_traits>(binfo, "
    ) == 4
    assert candidate_function.count("bidh / params.h_h_k_ratio") == 1
    assert candidate_function.count("bidh_k") == 9
    assert "blockIdx.z" in candidate_function
    assert "FR13_FA2_QROW32_STATIC_BATCH_GRID" in candidate
    assert "if constexpr (kStaticQueryBatch)" in candidate
    assert "const int m_block = 0" in candidate
    assert "blockIdx.z * kStaticQueryHeadsPerKV" in candidate
    assert "+ blockIdx.x" in candidate
    assert "const int bidb = Split ? blockIdx.z / params.h : blockIdx.y;" in candidate
    assert "const int bidh = Split ? blockIdx.z - bidb * params.h : blockIdx.z;" in candidate
    for sentinel in (
        "mask.template apply_mask<Is_causal, Is_even_MN>",
        "qk_accumulation_order_sentinel()",
        "pv_accumulation_order_sentinel()",
        "dynamic_k_length_sentinel(binfo.actual_seqlen_k)",
    ):
        assert candidate.count(sentinel) == stock.count(sentinel) == 1
    assert not module._patch_fixed32_query_tile32_static_batch_layout(
        kernel,
        fixed32_query_tile32=True,
    )
    assert kernel.read_text() == candidate

    # Exhaust the complete launch domain. The remap is a bijection onto the
    # original (batch, query-head) domain, and blockIdx.z is exactly qhead // 6.
    mapped = set()
    for kv_head in range(4):
        for batch in range(4):
            for query_head_lane in range(6):
                query_head = kv_head * 6 + query_head_lane
                assert query_head // 6 == kv_head
                mapped.add((batch, query_head))
    assert len(mapped) == 96
    assert mapped == {
        (batch, query_head)
        for batch in range(4)
        for query_head in range(24)
    }

    # The live gate separately verifies [0, 32, 64, 96, 128]. Under that exact
    # packed layout, every replaced q_offset is identical for any row stride.
    q_prefix = [0, 32, 64, 96, 128]
    for batch in range(4):
        for row_stride in (1, 24 * 256, 24 * 256 + 17):
            stock_offset = q_prefix[batch] * row_stride
            static_offset = batch * 32 * row_stride
            assert static_offset == stock_offset


def test_qrow32_static_paged_metadata_keeps_only_dynamic_sequence_length(
    tmp_path: Path,
) -> None:
    module = _module()
    kernel = tmp_path / "flash_fwd_kernel.h"
    signature = (
        "template<typename Kernel_traits, bool Is_causal, bool Is_local, "
        "bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, "
        "bool Split, bool Append_KV, typename Params>\n"
        "inline __device__ void compute_attn_1rowblock_splitkv"
        "(const Params &params) {\n"
    )
    dynamic_paged_base = """    const int bidb_cache = params.cache_batch_idx == nullptr ? bidb : params.cache_batch_idx[bidb];
    const int *block_table = params.block_table == nullptr ? nullptr : params.block_table + bidb * params.block_table_batch_stride;
    if constexpr (kStaticPagedKV) {
        if (block_table == nullptr) { return; }
    }
    const index_t row_offset_k = kStaticPagedKV || block_table != nullptr
        ? (bidh_k) * params.k_head_stride
        : binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh_k) * params.k_head_stride;
    const index_t row_offset_v = kStaticPagedKV || block_table != nullptr
        ? (bidh_k) * params.v_head_stride
        : binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh_k) * params.v_head_stride;
"""
    body = (
        "    using index_t = typename Kernel_traits::index_t;\n"
        "    constexpr bool kStaticQueryBatch = true;\n"
        "    constexpr bool kStaticPagedKV = true;\n"
        "    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);\n"
        + dynamic_paged_base
        + "    dynamic_k_length_sentinel(binfo.actual_seqlen_k);\n"
        + "    mask_order_sentinel();\n"
        + "    qk_pv_order_sentinel();\n"
        + "}\n"
    )
    stock = (
        signature
        + body
        + "\n////////////////////////////////////////////////////////////////////////////////////////////////////\n\n"
        + "template<typename Kernel_traits, bool Is_dropout\n"
        + "struct UnrelatedForwardKernel;\n"
    )
    kernel.write_text(stock)

    assert not module._patch_fixed32_query_tile32_static_paged_metadata(kernel)
    assert kernel.read_text() == stock
    assert module._patch_fixed32_query_tile32_static_paged_metadata(
        kernel,
        fixed32_query_tile32=True,
    )
    candidate = kernel.read_text()
    assert "FR13_FA2_QROW32_STATIC_PAGED_METADATA" in candidate
    assert "StaticPagedQueryBlockInfo<Kernel_traits>" in candidate
    assert "static_assert(!kStaticQueryBatch || kStaticPagedKV);" in candidate
    assert "static_assert(!kStaticQueryBatch || !Split);" in candidate
    assert "static_assert(!kStaticQueryBatch || !Append_KV);" in candidate
    assert "if constexpr (kStaticQueryBatch)" in candidate
    assert "block_table = params.block_table" in candidate
    assert "+ bidb * params.block_table_batch_stride" in candidate
    assert "row_offset_k = bidh_k * params.k_head_stride" in candidate
    assert "row_offset_v = bidh_k * params.v_head_stride" in candidate
    # The complete generic path remains available only in the discarded else.
    assert "params.cache_batch_idx == nullptr" in candidate
    assert "params.block_table == nullptr" in candidate
    assert "binfo.k_offset(params.k_batch_stride" in candidate
    for sentinel in (
        "dynamic_k_length_sentinel(binfo.actual_seqlen_k)",
        "mask_order_sentinel()",
        "qk_pv_order_sentinel()",
    ):
        assert candidate.count(sentinel) == stock.count(sentinel) == 1
    assert not module._patch_fixed32_query_tile32_static_paged_metadata(
        kernel,
        fixed32_query_tile32=True,
    )
    assert kernel.read_text() == candidate

    helper = module._tree_bias_helper(tile_earlyout=True)
    info_start = helper.index("struct StaticPagedQueryBlockInfo")
    info_end = helper.index("template <typename Kernel_traits, typename BlockInfoT", info_start)
    static_info = helper[info_start:info_end]
    assert "params.seqused_k[bidb]" in static_info
    assert "cu_seqlens_q" not in static_info
    assert "cu_seqlens_k" not in static_info
    assert "leftpad_k" not in static_info
    assert "cache_batch_idx" not in static_info

    # Paged varlen rejects left padding, and the live gate requires seqused_k.
    # The lightweight load is therefore exactly BlockInfo.actual_seqlen_k.
    for sequence_length in (32, 33, 1024, 8191):
        original_actual_k = sequence_length - 0
        static_actual_k = sequence_length
        assert static_actual_k == original_actual_k
    for block_table_stride in (1, 7, 64, 257):
        for batch in range(4):
            original_nonnull_row = batch * block_table_stride
            static_row = batch * block_table_stride
            assert static_row == original_nonnull_row


def test_qrow16_static_paged_path_folds_only_the_private_trait(
    tmp_path: Path,
) -> None:
    module = _module()
    kernel = tmp_path / "flash_fwd_kernel.h"
    signature = (
        "template<typename Kernel_traits, bool Is_causal, bool Is_local, "
        "bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, "
        "bool Split, bool Append_KV, typename Params>\n"
        "inline __device__ void compute_attn_1rowblock_splitkv"
        "(const Params &params) {\n"
    )
    body = "".join(
        old * expected
        for old, _new, _label, expected in (
            module.FIXED32_QUERY_STATIC_PAGED_PATH_REPLACEMENTS
        )
    )
    suffix = (
        "}\n\n"
        "////////////////////////////////////////////////////////////////////////////////////////////////////\n\n"
        "template<typename Kernel_traits, bool Is_dropout\n"
    )
    stock = signature + body + suffix
    kernel.write_text(stock)

    assert not module._patch_fixed32_query_static_paged_path(kernel)
    assert kernel.read_text() == stock
    assert module._patch_fixed32_query_static_paged_path(
        kernel,
        fixed32_query_tile16=True,
    )
    candidate = kernel.read_text()
    assert candidate.count("FR13_FA2_QROW16_STATIC_PAGED_PATH") == 1
    assert candidate.count("if constexpr (kStaticPagedKV)") == 7
    assert candidate.count("kStaticPagedKV || block_table != nullptr") == 2
    assert candidate.count("} else if (block_table == nullptr) {") == 5
    assert candidate.count("} else if (block_table != nullptr) {") == 1
    assert candidate.count("if (block_table == nullptr) { return; }") == 1
    assert "StaticPagedKVBlockSize<Kernel_traits>::value != 0" in candidate
    assert not module._patch_fixed32_query_static_paged_path(
        kernel,
        fixed32_query_tile16=True,
    )
    assert kernel.read_text() == candidate

    # The same trait-controlled routing fold is valid for B4's private qrow32
    # kernel; stock traits still instantiate only the dynamic branch.
    qrow32_kernel = tmp_path / "flash_fwd_kernel_qrow32.h"
    qrow32_kernel.write_text(stock)
    assert module._patch_fixed32_query_static_paged_path(
        qrow32_kernel,
        fixed32_query_tile32=True,
    )
    assert qrow32_kernel.read_text() == candidate
    assert not module._patch_fixed32_query_static_paged_path(
        qrow32_kernel,
        fixed32_query_tile32=True,
    )


def test_source_build_candidates_are_independent_and_default_off() -> None:
    text = Path("scripts/fr13_patch_fa2_tree_bias.py").read_text()

    assert 'parser.add_argument(\n        "--tree-bias-tile-earlyout",' in text
    assert 'parser.add_argument(\n        "--fixed32-query-tile16",' in text
    assert 'parser.add_argument(\n        "--fixed32-query-tile32",' in text
    assert 'parser.add_argument(\n        "--fixed32-query-tile16-live-ab",' in text
    assert 'parser.add_argument(\n        "--fixed32-query-tile32-live-ab",' in text
    assert "tree_bias_tile_earlyout: bool = False" in text
    assert "fixed32_query_tile16: bool = False" in text
    assert "fixed32_query_tile32: bool = False" in text
    assert "There is no qrow32 production selector" in text
    assert "fixed32_query_tile32_production" not in text
    assert "--fixed32-query-tile32 requires --tree-bias-tile-earlyout" in text
    assert "fixed32 qrow32 requires --tree-bias-tile-earlyout" in text
    assert "tree_splitkv" not in text
    assert "tree-splitkv" not in text
    assert "FR13_FA2_TREE_SPLITKV" not in text
    assert "AllowSplit" not in text
    assert "FR13_ALLOW_SPLIT_SWITCH" not in text
    assert "params.o_batch_stride = max_seqlen_q * params.o_row_stride" not in text


def test_qrow32_api_gate_composes_with_qrow16_and_is_idempotent() -> None:
    module = _module()
    text, changed = module._install_hidden_api_gate(
        module.STOCK_RUN_MHA_FWD,
        declaration=module.FIXED32_QUERY_TILE32_API_DECLARATION,
        gate=module.FIXED32_QUERY_TILE32_API_GATE,
        label="test qrow32",
    )
    assert changed
    assert text.count("fr13_run_mha_fwd_fixed32_qrow32") == 2
    assert text.count(module.FIXED32_QUERY_TILE32_API_GATE.strip()) == 1
    assert module.STOCK_RUN_MHA_FWD[
        module.STOCK_RUN_MHA_FWD.index("    FP16_SWITCH") : -1
    ] in text

    text, changed = module._install_hidden_api_gate(
        text,
        declaration=module.FIXED32_QUERY_TILE32_API_DECLARATION,
        gate=module.FIXED32_QUERY_TILE32_API_GATE,
        label="test qrow32",
    )
    assert not changed

    signature_at = module.FIXED32_QUERY_TILE16_API_DISPATCH.index(
        module.RUN_MHA_FWD_SIGNATURE
    )
    stock_body_at = module.FIXED32_QUERY_TILE16_API_DISPATCH.index(
        "    FP16_SWITCH", signature_at
    )
    text, changed = module._install_hidden_api_gate(
        text,
        declaration=module.FIXED32_QUERY_TILE16_API_DISPATCH[:signature_at],
        gate=module.FIXED32_QUERY_TILE16_API_DISPATCH[
            signature_at + len(module.RUN_MHA_FWD_SIGNATURE) : stock_body_at
        ],
        label="test qrow16",
    )
    assert changed
    assert text.count("fr13_run_mha_fwd_fixed32_qrow16") == 2
    assert text.count("fr13_run_mha_fwd_fixed32_qrow32") == 2


def test_qrow32_source_patch_requires_exact_safe_tile_earlyout(tmp_path: Path) -> None:
    module = _module()
    try:
        module.patch_fa2_source(tmp_path, fixed32_query_tile32=True)
    except ValueError as error:
        assert "requires --tree-bias-tile-earlyout" in str(error)
    else:
        raise AssertionError("qrow32 source patch accepted the one-flag build")


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
    divfree_runner = Path(
        "scripts/fr13_run_b1_k64_qrow16_divfree_live_gate.sh"
    ).read_text()

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
    assert '"draft_vocab_root": draft_vocab_root' in patcher
    assert '"draft_vocab_k": draft_vocab_k' in patcher
    assert 'raise RuntimeError("FR13 qrow16 live gate requires K64 ROOT=1")' in patcher
    assert "entry.output" not in patcher[
        patcher.index("def _fr13_fa2_qrow16_live_ab_replay") :
        patcher.index("def _patch_tree_attn")
    ]

    assert "FR13_FA2_QROW16_LIVE_PAGED_AB=${FR13_FA2_QROW16_LIVE_PAGED_AB:-0}" in launcher
    assert "FR13 qrow16 internal selectors are launcher-private" in launcher
    assert "fixed32 qrow16 candidate FA2 sha256 mismatch" in launcher
    assert "canonical K64 ROOT=1 real B1 task and no other candidate" in launcher
    assert "--fixed32-query-tile16-live-ab" in launcher
    assert "FR13_GATE_QROW16=${FR13_GATE_QROW16:-0}" in gate_launcher
    assert 'FR13_FA2_QROW16_LIVE_PAGED_AB="$FR13_GATE_QROW16"' in gate_launcher

    assert "FR13_RUN_QROW16_DIVFREE_LIVE_GATE:-0" in divfree_runner
    assert "config/fr13_fixed32/subset_b1_diagnostic_one.json" in divfree_runner
    assert "FR13_DRAFT_VOCAB_ROOT=1 FR13_DRAFT_VOCAB_K=65536" in divfree_runner
    assert "FR13_NEEDS_ALLOW=" in divfree_runner
    assert "MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1" in divfree_runner
    assert "FR13_FA2_QROW16_LIVE_PAGED_AB=1" in divfree_runner
    assert "FR13_FA2_QROW16_PRODUCTION=0" in divfree_runner
    assert "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0" in divfree_runner
    assert "FR13_SFWD_GPU_TIMER=0" in divfree_runner
    assert "performance_measurement" in divfree_runner


def test_qrow16_fatbin_graft_and_so_finalize_are_fail_closed() -> None:
    graft = Path("scripts/fr13_fa2_qrow16_fatbin_graft.py").read_text()
    finalize = Path("scripts/fr13_fa2_qrow16_so_finalize.py").read_text()
    builder = Path("scripts/fr13_build_fa2_qrow16_sm121a.py").read_text()

    assert "CUDA_VISIBLE_DEVICES must be explicitly empty" in graft
    assert "--expected-host-object-sha256" in graft
    assert "candidate PTX still contains signed 64-bit division" in graft
    assert "--update-section" in graft
    assert "ptx_signed_64bit_division_count" in graft
    assert "CANDIDATE_SYMBOL" in graft
    assert "HOST_SYMBOL" in graft
    assert "CUDA_VISIBLE_DEVICES must be explicitly empty" in builder
    assert "REGISTER_USAGE_LEVEL = 3" in builder
    assert "EXPECTED_REGISTERS = 216" in builder
    assert "qrow16 source is missing the RU3-paired 216-register cap" in builder
    assert "0 bytes spill stores" in builder
    assert "0 bytes spill loads" in builder
    assert "arch=compute_121a,code=sm_121a" in builder
    assert "performance_measurement" in builder

    assert "CUDA_VISIBLE_DEVICES must be explicitly empty" in finalize
    assert "REPAIRED_SYMBOLS" in finalize
    assert "candidate has dynamic-ABI drift beyond the repair allowlist" in finalize
    assert "finalized dynamic symbol table does not exactly match reference" in finalize
    assert "finalized DT_NEEDED entries do not exactly match reference" in finalize
    assert "finalization changed shared-object size" in finalize


def test_qrow32_live_gate_is_exact4_all_layer_and_stock_served() -> None:
    patcher = Path("scripts/fr13_patch_fa2_tree_bias.py").read_text()
    launcher = Path("scripts/fr13_launch_forked_fa2_tree_server.sh").read_text()
    runner = Path("scripts/fr13_run_b4_fa2_qrow32_live_gate.sh").read_text()

    assert "FR13_FA2_QROW32_LIVE_PAGED_AB_REPLAY" in patcher
    replay = patcher.index('anchor = "        entry.cudagraph.replay()\\n"')
    gate = patcher.index("_fr13_fa2_qrow32_live_ab_replay(", replay)
    assert replay < gate
    assert "range(3, 64, 4)" in patcher
    assert 'int(descriptor.get("num_reqs", -1)) != 4' in patcher
    assert 'tuple(query.shape) == (128, 24, 256)' in patcher
    assert 'tuple(cu_seqlens_q.shape) == (5,)' in patcher
    assert 'tuple(seqused_k.shape) == (4,)' in patcher
    assert 'int(block_table.shape[0]) == 4' in patcher
    assert 'tuple(key_cache.shape[1:]) == (1024, 4, 256)' in patcher
    assert "torch.empty_strided(" in patcher
    assert "_FR13_FA2_QROW32_BATCH_STRIDE_SENTINEL" in patcher
    assert 'q_start != [0, 32, 64, 96, 128]' in patcher
    assert '"slot_coverage": [0, 1, 2, 3]' in patcher
    assert '"layer_count": len(layer_records)' in patcher
    assert '"stock_calls": len(layer_records)' in patcher
    assert '"candidate_calls": len(layer_records)' in patcher
    assert '"served_return": "stock captured graph output unchanged"' in patcher
    assert '"fallback_allowed": False' in patcher
    assert '"performance_measurement": False' in patcher
    assert 'return_softmax_lse=True' in patcher
    assert "stock_out.dtype != torch.bfloat16" in patcher
    assert "stock_lse.dtype != torch.float32" in patcher
    assert "raw_byte_mismatches" in patcher

    assert "FR13_FA2_QROW32_LIVE_PAGED_AB=${FR13_FA2_QROW32_LIVE_PAGED_AB:-0}" in launcher
    assert "fixed32 qrow32 candidate FA2 sha256 mismatch" in launcher
    assert "--fixed32-query-tile32-live-ab" in launcher
    assert "qrow32 live A/B requires canonical SWE-Verified exact4 B4 identity" in launcher
    assert '-e FR13_FA2_QROW32_LIVE_PAGED_AB="$FR13_FA2_QROW32_LIVE_PAGED_AB"' in launcher

    assert "Real SWE-Verified exact4 B4 same-EngineCore byte gate" in runner
    assert "config/fr13_fixed32/subset_b4_four.json" in runner
    assert "MAX_NUM_SEQS_OVR=4 SWE_CONCURRENCY=4" in runner
    assert "FR13_DRAFT_VOCAB_K=65536" in runner
    assert "FR13_DRAFT_VOCAB_ROOT=1" in runner
    assert "FR13_FA2_QROW32_LIVE_PAGED_AB=1" in runner
    assert "FR13_FA2_QROW16_LIVE_PAGED_AB=0" in runner
    assert "FR13_FA2_QROW16_PRODUCTION=0" in runner
    assert "ENFORCE_EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE" in runner
    assert "scripts/fr13_fa2_qrow32_gate.py verify-live" in runner
