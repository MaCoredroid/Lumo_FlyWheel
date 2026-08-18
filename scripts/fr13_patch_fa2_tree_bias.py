#!/usr/bin/env python3
"""Patch vLLM FA2 with an FR13 tree-bias varlen forward op.

The patch intentionally leaves the stock ``varlen_fwd`` op unchanged and adds
``varlen_fwd_tree_bias``.  The new op carries a dense ancestry-bias matrix into
FA2 and adds it to score tiles after QK and before masking/softmax.

The exact-safe ``--tree-bias-tile-earlyout`` source-build candidate skips the
per-score bias walk on K tiles that cannot overlap the tree suffix. It does not
change FA2 launch geometry, split-KV selection, or floating-point reductions.

The independent ``--fixed32-query-tile16`` build includes an underfilled-B1
alternative with two 16-row, one-warp query CTAs per head. It is default-off:
only the private live-gate dispatch selects it. Each CTA still traverses the
complete K sequence; there is no split-K reduction or combine kernel. B4 keeps
the stock geometry, which already supplies 96 CTAs per layer. The candidate is
a hidden launcher in one dedicated production translation unit; the stock
explicit instantiation, shared launcher, and every unrelated CUDA object
remain untouched.

The composable ``--fixed32-query-tile16-static-strides`` specialization fixes
the canonical contiguous K/V page, row, and head strides only in that private
translation unit. The hidden dispatch verifies all six strides before entry;
the stock path and the qualified qrow16 source retain their runtime strides.

The ``--fixed32-query-tile32`` build adds the corresponding B4 specialization:
one 32-row, two-warp CTA per batch/head. It has the same 96-CTA layer grid and
complete ordered K loop as stock BM64 while avoiding the 32 query rows outside
each physical32 batch slot. Its private selector is gate-only and default-off;
it can be tagged only by the canonical retained-live exact4 byte diagnostic.
The independent ``--fixed32-query-tile32-b1`` build uses the same BM32 traits
for exact B1, where its 24 two-warp CTAs replace 48 one-warp qrow16 CTAs and
share one ordered K/V scan across each head's two query warps. It is likewise
gate-only and default-off; the 24-CTA grid is intentionally explicit because
it may underfill a 48-SM GPU even though it preserves 48 resident warps.
The same build emits a separately tagged split-K=2 alternative. Its 48 main
CTAs partition, rather than duplicate, each head's K-block interval and then
use FA2's existing four-warp combine kernel; split scratch must already have
been allocated by the stock ``num_splits=2`` API setup.
Both fixed32 query routes fix the paged-KV block size at 1024, allowing only
their dedicated translation units to resolve pages directly from 64-row
K-block coordinates. The B1 build's independent ``no_split`` and ``split2``
selectors are default-off, require a real K64/root1 B1 byte qualification, and
fail closed on final production geometry or attestation drift.
The composable qrow32-only ``--fixed32-tree-visibility-mask`` candidate keeps
the same physical32 query/KV schedule but replaces the dense 32x32 fp32
ancestry-bias loads with the exact self-plus-ancestor bit rows shared by Tail23
and Hydra27. Inactive physical slots remain in the table; only downstream
validity decides whether their results participate in rejection sampling.
The B4 trait also forms its nonnull block-table row directly and constructs
sequence metadata from only the required dynamic ``seqused_k`` value.
Its canonical contiguous K/V page, row, and head strides are compile-time
constants in the private page resolver and base-head address.
"""

from __future__ import annotations

import argparse
import os
import py_compile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FA2_CANDIDATES = [
    REPO_ROOT
    / "output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260503T204510Z/cutlass_source_workspace/vllm-source/.deps/vllm-flash-attn-src",
    REPO_ROOT
    / "output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260504T053925Z/cutlass_source_workspace/vllm-source/build/lumo_cutlass_research/_deps/vllm-flash-attn-src",
]


TREE_BIAS_TILE_OVERLAP_GUARD = r'''    // FR13_FA2_TREE_BIAS_TILE_EARLYOUT: only the suffix tiles carry bias.
    const int bias_col_begin = context_len + tree_bias_k_offset;
    const int bias_col_end = bias_col_begin + tree_bias_cols;
    const int block_col_begin = n_block * Kernel_traits::kBlockN;
    const int block_col_end = block_col_begin + Kernel_traits::kBlockN;
    if (block_col_end <= bias_col_begin || block_col_begin >= bias_col_end) {
        return;
    }
'''


def _tree_bias_helper(
    tile_earlyout: bool,
    *,
    fixed32_tree_visibility_mask: bool = False,
) -> str:
    overlap_guard = TREE_BIAS_TILE_OVERLAP_GUARD if tile_earlyout else ""
    helper = r'''
// FR13_FA2_TREE_BIAS: add a dense query-suffix ancestry bias after QK.
template <typename Kernel_traits>
struct StaticQueryRows {
    static constexpr int value = 0;
};

template <typename Kernel_traits>
struct StaticQueryBatchLayout {
    static constexpr int sequences = 0;
    static constexpr int query_heads = 0;
    static constexpr int kv_heads = 0;
    static constexpr int query_heads_per_kv = 0;
};

template <typename Kernel_traits>
struct StaticQueryHeadsPerCTA {
    static constexpr int value = 1;
};

template <typename Kernel_traits>
struct StaticPagedQueryBlockInfo {
    template <typename Params>
    __forceinline__ __device__
    StaticPagedQueryBlockInfo(const Params &params, const int bidb)
        : actual_seqlen_q(StaticQueryRows<Kernel_traits>::value)
        , seqlen_k_cache(0)
        , actual_seqlen_k(params.seqused_k[bidb]) {
        static_assert(StaticQueryRows<Kernel_traits>::value == 32);
        static_assert(
            StaticQueryBatchLayout<Kernel_traits>::sequences == 1
            || StaticQueryBatchLayout<Kernel_traits>::sequences == 4);
    }

    const int actual_seqlen_q;
    // Append_KV is forbidden for this trait. Retain the member only so the
    // discarded generic append branch remains syntactically well formed.
    const int seqlen_k_cache;
    const int actual_seqlen_k;
};

template <typename Kernel_traits, typename BlockInfoT,
          typename BatchStrideT, typename RowStrideT>
__forceinline__ __device__
typename Kernel_traits::index_t static_query_offset(
        const BlockInfoT &binfo,
        const BatchStrideT batch_stride,
        const RowStrideT row_stride,
        const int bidb) {
    using index_t = typename Kernel_traits::index_t;
    constexpr int kStaticQueryRows = StaticQueryRows<Kernel_traits>::value;
    if constexpr (kStaticQueryRows != 0) {
        return static_cast<index_t>(bidb) * kStaticQueryRows
            * static_cast<index_t>(row_stride);
    } else {
        return binfo.q_offset(
            static_cast<index_t>(batch_stride),
            static_cast<index_t>(row_stride),
            bidb);
    }
}

template <typename Kernel_traits, typename Engine, typename Layout, typename Params, typename BlockInfoT>
__forceinline__ __device__ void apply_tree_bias(Tensor<Engine, Layout> &tensor_,
                                                const Params &params,
                                                const BlockInfoT &binfo,
                                                const int bidb,
                                                const int n_block,
                                                const int row_idx_offset,
                                                const int warp_row_stride) {
    constexpr int kStaticQueryRows = StaticQueryRows<Kernel_traits>::value;
    constexpr int kStaticQueryHeadsPerCTA =
        StaticQueryHeadsPerCTA<Kernel_traits>::value;
    constexpr bool kStaticQueryTile =
        kStaticQueryRows * kStaticQueryHeadsPerCTA == Kernel_traits::kBlockM;
    static_assert(kStaticQueryHeadsPerCTA == 1 || kStaticQueryHeadsPerCTA == 2);
    static_assert(
        !kStaticQueryTile
        || Kernel_traits::kBlockM == 32 * kStaticQueryHeadsPerCTA);
    static_assert(
        !kStaticQueryTile
        || Kernel_traits::kNWarps == 2 * kStaticQueryHeadsPerCTA);
    static_assert(
        !kStaticQueryTile
        || Kernel_traits::kNThreads == 64 * kStaticQueryHeadsPerCTA);
    if constexpr (!kStaticQueryTile) {
        if (params.tree_bias_ptr == nullptr) { return; }
    }
    static_assert(Layout::rank == 3, "Only support 3D Tensor");
    static_assert(decltype(size<0>(tensor_))::value == 4, "First dimension must be 4");
    Tensor tensor = make_tensor(tensor_.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(tensor_.layout()));
    static_assert(!kStaticQueryTile || decltype(size<0, 0>(tensor))::value == 2);
    static_assert(!kStaticQueryTile || decltype(size<0, 1>(tensor))::value == 1);
    const float *tree_bias = reinterpret_cast<const float *>(params.tree_bias_ptr)
        + bidb * params.tree_bias_batch_stride;
    const int query_rows = kStaticQueryTile ? kStaticQueryRows : binfo.actual_seqlen_q;
    const int tree_bias_rows = kStaticQueryTile ? 32 : params.tree_bias_rows;
    const int tree_bias_cols = kStaticQueryTile ? 32 : params.tree_bias_cols;
    const int tree_bias_q_offset = kStaticQueryTile ? 0 : params.tree_bias_q_offset;
    const int tree_bias_k_offset = kStaticQueryTile ? 0 : params.tree_bias_k_offset;
    const int64_t tree_bias_row_stride = kStaticQueryTile ? 32 : params.tree_bias_row_stride;
    const int64_t tree_bias_col_stride = kStaticQueryTile ? 1 : params.tree_bias_col_stride;
    const int context_len = binfo.actual_seqlen_k - query_rows;
''' + overlap_guard + r'''    const int lane_id = threadIdx.x % 32;
    const int col_idx_offset = n_block * Kernel_traits::kBlockN + (lane_id % 4) * 2;
    const int fixed_row_idx_offset =
        (threadIdx.x / 32) * 16 + (threadIdx.x % 32) / 4;
    const int query_row_idx_offset =
        kStaticQueryTile ? fixed_row_idx_offset : row_idx_offset;
    const int query_warp_row_stride =
        kStaticQueryTile ? 32 : warp_row_stride;
    #pragma unroll
    for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
        const int row_idx_base = query_row_idx_offset + mi * query_warp_row_stride;
        #pragma unroll
        for (int i = 0; i < size<0, 0>(tensor); ++i) {
            const int logical_row = row_idx_base + i * 8;
            const int q_rel = (
                kStaticQueryTile && kStaticQueryHeadsPerCTA == 2
                    ? logical_row & (kStaticQueryRows - 1)
                    : logical_row
            ) - tree_bias_q_offset;
            if (kStaticQueryTile || (q_rel >= 0 && q_rel < tree_bias_rows)) {
                #pragma unroll
                for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                    const int col_idx_base = col_idx_offset + nj * 8;
                    #pragma unroll
                    for (int j = 0; j < size<1, 0>(tensor); ++j) {
                        const int k_rel = col_idx_base + j - context_len - tree_bias_k_offset;
                        if (k_rel >= 0 && k_rel < tree_bias_cols) {
                            const float bias = tree_bias[
                                q_rel * tree_bias_row_stride
                                + k_rel * tree_bias_col_stride
                            ];
                            if (bias == -INFINITY) {
                                tensor(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
                            } else {
                                tensor(make_coord(i, mi), make_coord(j, nj)) += bias / params.scale_softmax;
                            }
                        }
                    }
                }
            }
        }
    }
}

'''
    if not fixed32_tree_visibility_mask:
        return helper

    visibility_trait = r'''
template <typename Kernel_traits>
struct StaticTreeVisibility {
    static constexpr bool enabled = false;

    __forceinline__ __device__
    static unsigned int row_mask(const int) { return 0U; }
};

'''
    trait_anchor = r'''template <typename Kernel_traits>
struct StaticQueryHeadsPerCTA {
    static constexpr int value = 1;
};

'''
    if helper.count(trait_anchor) != 1:
        raise RuntimeError("fixed32 visibility trait anchor drifted")
    helper = helper.replace(
        trait_anchor,
        trait_anchor + visibility_trait,
        1,
    )

    static_shape_anchor = r'''    constexpr bool kStaticQueryTile =
        kStaticQueryRows * kStaticQueryHeadsPerCTA == Kernel_traits::kBlockM;
'''
    static_shape_replacement = static_shape_anchor + r'''    constexpr bool kStaticTreeVisibility =
        StaticTreeVisibility<Kernel_traits>::enabled;
    constexpr bool kStaticTreeShape =
        kStaticQueryTile || kStaticTreeVisibility;
'''
    if helper.count(static_shape_anchor) != 1:
        raise RuntimeError("fixed32 visibility shape anchor drifted")
    helper = helper.replace(
        static_shape_anchor,
        static_shape_replacement,
        1,
    )
    helper = helper.replace(
        "    if constexpr (!kStaticQueryTile) {\n",
        "    if constexpr (!kStaticTreeShape) {\n",
        1,
    )

    dense_metadata = r'''    const float *tree_bias = reinterpret_cast<const float *>(params.tree_bias_ptr)
        + bidb * params.tree_bias_batch_stride;
    const int query_rows = kStaticQueryTile ? kStaticQueryRows : binfo.actual_seqlen_q;
    const int tree_bias_rows = kStaticQueryTile ? 32 : params.tree_bias_rows;
    const int tree_bias_cols = kStaticQueryTile ? 32 : params.tree_bias_cols;
    const int tree_bias_q_offset = kStaticQueryTile ? 0 : params.tree_bias_q_offset;
    const int tree_bias_k_offset = kStaticQueryTile ? 0 : params.tree_bias_k_offset;
    const int64_t tree_bias_row_stride = kStaticQueryTile ? 32 : params.tree_bias_row_stride;
    const int64_t tree_bias_col_stride = kStaticQueryTile ? 1 : params.tree_bias_col_stride;
'''
    static_metadata = r'''    const float *tree_bias = nullptr;
    if constexpr (!kStaticTreeVisibility) {
        tree_bias = reinterpret_cast<const float *>(params.tree_bias_ptr)
            + bidb * params.tree_bias_batch_stride;
    }
    const int query_rows = kStaticTreeShape ? 32 : binfo.actual_seqlen_q;
    const int tree_bias_rows = kStaticTreeShape ? 32 : params.tree_bias_rows;
    const int tree_bias_cols = kStaticTreeShape ? 32 : params.tree_bias_cols;
    const int tree_bias_q_offset = kStaticTreeShape ? 0 : params.tree_bias_q_offset;
    const int tree_bias_k_offset = kStaticTreeShape ? 0 : params.tree_bias_k_offset;
    const int64_t tree_bias_row_stride = kStaticTreeShape ? 32 : params.tree_bias_row_stride;
    const int64_t tree_bias_col_stride = kStaticTreeShape ? 1 : params.tree_bias_col_stride;
'''
    if helper.count(dense_metadata) != 1:
        raise RuntimeError("fixed32 visibility metadata anchor drifted")
    helper = helper.replace(dense_metadata, static_metadata, 1)

    q_bound = "            if (kStaticQueryTile || (q_rel >= 0 && q_rel < tree_bias_rows)) {\n"
    q_bound_replacement = r'''            if (kStaticTreeShape || (q_rel >= 0 && q_rel < tree_bias_rows)) {
                const unsigned int tree_visibility = kStaticTreeVisibility
                    ? StaticTreeVisibility<Kernel_traits>::row_mask(q_rel)
                    : 0U;
'''
    if helper.count(q_bound) != 1:
        raise RuntimeError("fixed32 visibility query bound anchor drifted")
    helper = helper.replace(q_bound, q_bound_replacement, 1)

    dense_bias = r'''                            const float bias = tree_bias[
                                q_rel * tree_bias_row_stride
                                + k_rel * tree_bias_col_stride
                            ];
'''
    static_bias = r'''                            const float bias = kStaticTreeVisibility
                                ? ((tree_visibility & (1U << k_rel)) != 0U
                                    ? 0.0f : -INFINITY)
                                : tree_bias[
                                    q_rel * tree_bias_row_stride
                                    + k_rel * tree_bias_col_stride
                                ];
'''
    if helper.count(dense_bias) != 1:
        raise RuntimeError("fixed32 visibility bias-load anchor drifted")
    helper = helper.replace(dense_bias, static_bias, 1)
    return helper


TREE_BIAS_HELPER = _tree_bias_helper(tile_earlyout=False)


STOCK_FIXED32_QUERY_INSTANTIATION = r'''template void run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 256, false>(Flash_fwd_params &params, cudaStream_t stream);'''


FIXED32_QUERY_TILE16_BATCH_STRIDE_SENTINEL = 0x46523133
FIXED32_QUERY_TILE32_B1_BATCH_STRIDE_SENTINEL = 0x46523134
FIXED32_QUERY_TILE32_B1_SPLIT2_BATCH_STRIDE_SENTINEL = 0x46523135
# The B1 family tags its private dispatches with a four-byte ASCII run
# ("FR13", "FR14", "FR15", ...). The GQA-pair B1 arm takes the next tag.
FIXED32_QUERY_GQA_PAIR32_B1_BATCH_STRIDE_SENTINEL = 0x46523136
# FR14 split-K (Tier-B). Same GQA-pair traits, same K-block order per split,
# but the context walk is partitioned across blockIdx.y and re-reduced by
# FA2's own combine kernel -- so this arm is deliberately NOT byte-identical
# to 0x46523136 and carries its own sentinel so it can never be reached by an
# operand tagged for a byte-qualified arm.
FIXED32_QUERY_GQA_PAIR32_SPLITK_B1_BATCH_STRIDE_SENTINEL = 0x46523137
# 4 context splits x 3 head pairs x 4 KV heads = 48 CTAs at B1, one per SM on
# GB10. The value is a compile-time constant in the launcher, in the API gate
# and in the scratch allocation, so a drift in any one of them fails closed.
FIXED32_QUERY_GQA_PAIR32_SPLITK_B1_CONTEXT_SPLITS = 4


# Unlike B1, B4 dereferences the tree-bias batch stride. Keep this sentinel
# large enough to be private but small enough for the gate's deliberately
# padded four-batch diagnostic tensor (about 1.6 MiB of BF32 storage).
FIXED32_QUERY_TILE32_BATCH_STRIDE_SENTINEL = 0x20013
FIXED32_QUERY_GQA_PAIR32_BATCH_STRIDE_SENTINEL = 0x20014


# A shared sentinel would silently route one arm's traffic into another arm's
# kernel, so every private dispatch tag must stay distinct by construction.
_FIXED32_BATCH_STRIDE_SENTINELS = (
    FIXED32_QUERY_TILE16_BATCH_STRIDE_SENTINEL,
    FIXED32_QUERY_TILE32_B1_BATCH_STRIDE_SENTINEL,
    FIXED32_QUERY_TILE32_B1_SPLIT2_BATCH_STRIDE_SENTINEL,
    FIXED32_QUERY_GQA_PAIR32_B1_BATCH_STRIDE_SENTINEL,
    FIXED32_QUERY_GQA_PAIR32_SPLITK_B1_BATCH_STRIDE_SENTINEL,
    FIXED32_QUERY_TILE32_BATCH_STRIDE_SENTINEL,
    FIXED32_QUERY_GQA_PAIR32_BATCH_STRIDE_SENTINEL,
)
assert len(set(_FIXED32_BATCH_STRIDE_SENTINELS)) == len(
    _FIXED32_BATCH_STRIDE_SENTINELS
), "FR13 private dispatch sentinels must be pairwise distinct"


FIXED32_PHYSICAL_PARENT = (
    -1,
    0, 0, 0,
    1, 1, 1,
    2,
    3,
    4, 4, 4,
    7,
    8,
    9, 9, 9,
    12,
    13,
    14, 14, 14,
    17,
    18,
    19,
    23,
    24,
    25,
    26,
    28,
    29,
    30,
)


def _fixed32_visibility_masks() -> tuple[int, ...]:
    masks = []
    for node in range(len(FIXED32_PHYSICAL_PARENT)):
        mask = 0
        cursor = node
        while cursor >= 0:
            mask |= 1 << cursor
            cursor = FIXED32_PHYSICAL_PARENT[cursor]
        masks.append(mask)
    return tuple(masks)


FIXED32_TREE_VISIBILITY_MASKS = _fixed32_visibility_masks()
assert len(FIXED32_TREE_VISIBILITY_MASKS) == 32


def _with_fixed32_tree_visibility(
    source: str,
    *,
    trait: str,
    symbol: str,
    max_registers: int,
) -> str:
    marker = "// FR13_FA2_FIXED32_TREE_VISIBILITY_MASK"
    if marker in source:
        raise RuntimeError("fixed32 tree visibility specialization duplicated")
    initializer = ",\n".join(
        f"    0x{mask:08x}U" for mask in FIXED32_TREE_VISIBILITY_MASKS
    )
    specialization = f'''{marker}: the physical32 parent topology is shared by
// Tail23 and Hydra27. Inactive slots remain present and retain their exact
// self-plus-ancestor visibility; the valid-node mask is consumed downstream.
static __device__ __constant__ unsigned int {symbol}[32] = {{
{initializer}
}};

template <>
struct StaticTreeVisibility<{trait}> {{
    static constexpr bool enabled = true;

    __forceinline__ __device__
    static unsigned int row_mask(const int row) {{
        return {symbol}[row];
    }}
}};

'''
    register_anchor = "__global__ __maxnreg__(254)"
    if source.count(register_anchor) != 1:
        raise RuntimeError("fixed32 tree visibility register cap drifted")
    source = source.replace(
        register_anchor,
        f"__global__ __maxnreg__({max_registers})",
        1,
    )
    kernel_anchor = "__global__ __maxnreg__("
    if source.count(kernel_anchor) != 1:
        raise RuntimeError("fixed32 tree visibility kernel anchor drifted")
    return source.replace(kernel_anchor, specialization + kernel_anchor, 1)


FIXED32_QUERY_STATIC_PAGE_TRAIT_LEGACY = r'''// FR13_FA2_FIXED32_STATIC_PAGE: stock traits retain the dynamic page size.
template <typename Kernel_traits>
struct StaticPagedKVBlockSize {
    static constexpr int value = 0;
    static constexpr int log2 = 0;
    static constexpr int block_n_log2 = 0;
};

'''


FIXED32_QUERY_STATIC_PAGE_TRAIT = r'''// FR13_FA2_FIXED32_STATIC_PAGE: stock traits retain the dynamic page size.
template <typename Kernel_traits>
struct StaticPagedKVBlockSize {
    static constexpr int value = 0;
    static constexpr int log2 = 0;
    static constexpr int block_n_log2 = 0;
};

template <typename Kernel_traits>
struct StaticPagedKVStrides {
    static constexpr int64_t page = 0;
    static constexpr int64_t row = 0;
    static constexpr int64_t head = 0;
};

'''


FIXED32_QUERY_STATIC_PAGE_OFFSET_LEGACY = r'''    constexpr int kStaticPageBlockSize = StaticPagedKVBlockSize<Kernel_traits>::value;
    if constexpr (kStaticPageBlockSize != 0) {
        constexpr int kStaticPageBlockLog2 = StaticPagedKVBlockSize<Kernel_traits>::log2;
        constexpr int kStaticBlockNLog2 = StaticPagedKVBlockSize<Kernel_traits>::block_n_log2;
        constexpr int kBlocksPerPageLog2 = kStaticPageBlockLog2 - kStaticBlockNLog2;
        constexpr int kBlocksPerPage = 1U << kBlocksPerPageLog2;
        static_assert(kStaticPageBlockSize > 0);
        static_assert(kStaticPageBlockSize == (1U << kStaticPageBlockLog2));
        static_assert(Kernel_traits::kBlockN == (1U << kStaticBlockNLog2));
        static_assert(kStaticPageBlockLog2 >= kStaticBlockNLog2);
        static_assert(Kernel_traits::kNThreads % Kernel_traits::kGmemThreadsPerRow == 0);
        static_assert(
            Kernel_traits::kNThreads / Kernel_traits::kGmemThreadsPerRow
                * Kernel_traits::kGmemRowsPerThread
            == Kernel_traits::kBlockN);
        // n_block is active and nonnegative. Each fixed32 thread starts below
        // kBlockN, and the partial-block clamp can only lower that row offset.
        // Resolve the page in 32-bit block coordinates before address math.
        const int page_offset =
            ((n_block & (kBlocksPerPage - 1)) << kStaticBlockNLog2)
            + static_cast<int>(block_row_offset);
        const int virtual_page_idx = n_block >> kBlocksPerPageLog2;
        return ((int64_t) block_table[virtual_page_idx]) * ((int64_t) page_stride)
            + ((int64_t) page_offset) * ((int64_t) row_stride)
            + col_offset;
    } else {
        const int64_t global_row_offset = block_row_offset + n_block * kBlockN;
        const int64_t page_offset = global_row_offset % page_block_size;
        const int64_t virtual_page_idx = global_row_offset / page_block_size;
        return ((int64_t) block_table[virtual_page_idx]) * ((int64_t) page_stride)
            + page_offset * ((int64_t) row_stride)
            + col_offset;
    }
'''


FIXED32_QUERY_STATIC_PAGE_OFFSET = r'''    constexpr int kStaticPageBlockSize = StaticPagedKVBlockSize<Kernel_traits>::value;
    if constexpr (kStaticPageBlockSize != 0) {
        constexpr int kStaticPageBlockLog2 = StaticPagedKVBlockSize<Kernel_traits>::log2;
        constexpr int kStaticBlockNLog2 = StaticPagedKVBlockSize<Kernel_traits>::block_n_log2;
        constexpr int kBlocksPerPageLog2 = kStaticPageBlockLog2 - kStaticBlockNLog2;
        constexpr int kBlocksPerPage = 1U << kBlocksPerPageLog2;
        constexpr int64_t kStaticPageStride = StaticPagedKVStrides<Kernel_traits>::page;
        constexpr int64_t kStaticRowStride = StaticPagedKVStrides<Kernel_traits>::row;
        constexpr bool kStaticStrides = kStaticPageStride != 0;
        static_assert(kStaticPageBlockSize > 0);
        static_assert(kStaticPageBlockSize == (1U << kStaticPageBlockLog2));
        static_assert(Kernel_traits::kBlockN == (1U << kStaticBlockNLog2));
        static_assert(kStaticPageBlockLog2 >= kStaticBlockNLog2);
        static_assert((kStaticPageStride == 0) == (kStaticRowStride == 0));
        static_assert(Kernel_traits::kNThreads % Kernel_traits::kGmemThreadsPerRow == 0);
        static_assert(
            Kernel_traits::kNThreads / Kernel_traits::kGmemThreadsPerRow
                * Kernel_traits::kGmemRowsPerThread
            == Kernel_traits::kBlockN);
        // n_block is active and nonnegative. Each fixed32 thread starts below
        // kBlockN, and the partial-block clamp can only lower that row offset.
        // Resolve the page in 32-bit block coordinates before address math.
        const int page_offset =
            ((n_block & (kBlocksPerPage - 1)) << kStaticBlockNLog2)
            + static_cast<int>(block_row_offset);
        const int virtual_page_idx = n_block >> kBlocksPerPageLog2;
        if constexpr (kStaticStrides) {
            return ((int64_t) block_table[virtual_page_idx]) * kStaticPageStride
                + ((int64_t) page_offset) * kStaticRowStride
                + col_offset;
        } else {
            return ((int64_t) block_table[virtual_page_idx]) * ((int64_t) page_stride)
                + ((int64_t) page_offset) * ((int64_t) row_stride)
                + col_offset;
        }
    } else {
        const int64_t global_row_offset = block_row_offset + n_block * kBlockN;
        const int64_t page_offset = global_row_offset % page_block_size;
        const int64_t virtual_page_idx = global_row_offset / page_block_size;
        return ((int64_t) block_table[virtual_page_idx]) * ((int64_t) page_stride)
            + page_offset * ((int64_t) row_stride)
            + col_offset;
    }
'''


FIXED32_QUERY_STATIC_PAGED_PATH_REPLACEMENTS = (
    (
        r'''    constexpr int kNWarps = Kernel_traits::kNWarps;
''',
        r'''    constexpr int kNWarps = Kernel_traits::kNWarps;
    // FR13_FA2_QROW16_STATIC_PAGED_PATH: remove dynamic page routing only for
    // the private trait whose physical page size is fixed at compile time.
    constexpr bool kStaticPagedKV =
        FLASH_NAMESPACE::StaticPagedKVBlockSize<Kernel_traits>::value != 0;
''',
        "static paged-path trait",
        1,
    ),
    (
        r'''    const int *block_table = params.block_table == nullptr ? nullptr : params.block_table + bidb * params.block_table_batch_stride;
    const index_t row_offset_k = block_table == nullptr
        ? binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride
        : (bidh / params.h_h_k_ratio) * params.k_head_stride; // block addresses are later resolved per-thread
    const index_t row_offset_v = block_table == nullptr
        ? binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride
        : (bidh / params.h_h_k_ratio) * params.v_head_stride;
''',
        r'''    const int *block_table = params.block_table == nullptr ? nullptr : params.block_table + bidb * params.block_table_batch_stride;
    if constexpr (kStaticPagedKV) {
        if (block_table == nullptr) { return; }
    }
    const index_t row_offset_k = kStaticPagedKV || block_table != nullptr
        ? (bidh / params.h_h_k_ratio) * params.k_head_stride
        : binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = kStaticPagedKV || block_table != nullptr
        ? (bidh / params.h_h_k_ratio) * params.v_head_stride
        : binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
''',
        "static paged-path base offsets",
        1,
    ),
    (
        r'''    if (block_table != nullptr) {
        auto final_block_size = binfo.actual_seqlen_k - (n_block_max - 1) * kBlockN;
        tKgK.data() = gK.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block_max - 1, params.page_block_size,
            block_table, params.k_batch_stride, params.k_row_stride, final_block_size);
        tVgV.data() = gV.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block_max - 1, params.page_block_size,
            block_table, params.v_batch_stride, params.v_row_stride, final_block_size);
    }
''',
        r'''    if constexpr (kStaticPagedKV) {
        auto final_block_size = binfo.actual_seqlen_k - (n_block_max - 1) * kBlockN;
        tKgK.data() = gK.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block_max - 1, params.page_block_size,
            block_table, params.k_batch_stride, params.k_row_stride, final_block_size);
        tVgV.data() = gV.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block_max - 1, params.page_block_size,
            block_table, params.v_batch_stride, params.v_row_stride, final_block_size);
    } else if (block_table != nullptr) {
        auto final_block_size = binfo.actual_seqlen_k - (n_block_max - 1) * kBlockN;
        tKgK.data() = gK.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block_max - 1, params.page_block_size,
            block_table, params.k_batch_stride, params.k_row_stride, final_block_size);
        tVgV.data() = gV.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block_max - 1, params.page_block_size,
            block_table, params.v_batch_stride, params.v_row_stride, final_block_size);
    }
''',
        "static paged-path initial tile",
        1,
    ),
    (
        r'''            if (block_table == nullptr) {
                tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
                tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
            } else {
                if (n_block > n_block_copy_min) {
                    tVgV.data() = gV.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block - 1, params.page_block_size,
                        block_table, params.v_batch_stride, params.v_row_stride);
                    tKgK.data() = gK.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block - 1, params.page_block_size,
                        block_table, params.k_batch_stride, params.k_row_stride);
                }
            }
''',
        r'''            if constexpr (kStaticPagedKV) {
                if (n_block > n_block_copy_min) {
                    tVgV.data() = gV.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block - 1, params.page_block_size,
                        block_table, params.v_batch_stride, params.v_row_stride);
                    tKgK.data() = gK.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block - 1, params.page_block_size,
                        block_table, params.k_batch_stride, params.k_row_stride);
                }
            } else if (block_table == nullptr) {
                tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
                tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
            } else {
                if (n_block > n_block_copy_min) {
                    tVgV.data() = gV.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block - 1, params.page_block_size,
                        block_table, params.v_batch_stride, params.v_row_stride);
                    tKgK.data() = gK.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block - 1, params.page_block_size,
                        block_table, params.k_batch_stride, params.k_row_stride);
                }
            }
''',
        "static paged-path append-KV advance",
        1,
    ),
    (
        r'''            if (block_table == nullptr) {
                tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
            } else {
                tVgV.data() = gV.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block, params.page_block_size,
                    block_table, params.v_batch_stride, params.v_row_stride);
            }
''',
        r'''            if constexpr (kStaticPagedKV) {
                tVgV.data() = gV.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block, params.page_block_size,
                    block_table, params.v_batch_stride, params.v_row_stride);
            } else if (block_table == nullptr) {
                tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
            } else {
                tVgV.data() = gV.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block, params.page_block_size,
                    block_table, params.v_batch_stride, params.v_row_stride);
            }
''',
        "static paged-path masked V advance",
        1,
    ),
    (
        r'''        if (block_table == nullptr) {
            tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
        } else {
            tVgV.data() = gV.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block, params.page_block_size,
                block_table, params.v_batch_stride, params.v_row_stride);
        }
''',
        r'''        if constexpr (kStaticPagedKV) {
            tVgV.data() = gV.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block, params.page_block_size,
                block_table, params.v_batch_stride, params.v_row_stride);
        } else if (block_table == nullptr) {
            tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
        } else {
            tVgV.data() = gV.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block, params.page_block_size,
                block_table, params.v_batch_stride, params.v_row_stride);
        }
''',
        "static paged-path unmasked V advance",
        1,
    ),
    (
        r'''            if (block_table == nullptr) {
                tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
            } else {
                tKgK.data() = gK.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block - 1, params.page_block_size,
                    block_table, params.k_batch_stride, params.k_row_stride);
            }
''',
        r'''            if constexpr (kStaticPagedKV) {
                tKgK.data() = gK.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block - 1, params.page_block_size,
                    block_table, params.k_batch_stride, params.k_row_stride);
            } else if (block_table == nullptr) {
                tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
            } else {
                tKgK.data() = gK.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block - 1, params.page_block_size,
                    block_table, params.k_batch_stride, params.k_row_stride);
            }
''',
        "static paged-path K advances",
        2,
    ),
)


FIXED32_QUERY_TILE16_TRANSLATION_UNIT = r'''// FR13 fixed32 B1 qrow16 internal kernel.
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

using Fr13Fixed32Qrow16KernelTraits = Flash_fwd_kernel_traits<
    256, 16, 64, 1, false, false, cutlass::bfloat16_t>;

template <>
struct StaticPagedKVBlockSize<Fr13Fixed32Qrow16KernelTraits> {
    static constexpr int value = 1024;
    static constexpr int log2 = 10;
    static constexpr int block_n_log2 = 6;
};

// RU3 must lower pressure before this exact cap; the cap alone spills.
// This non-templated wrapper is private to the fail-closed B1 launcher.
__global__ __maxnreg__(216)
void fr13_flash_fwd_fixed32_qrow16_kernel(
        KERNEL_PARAM_MODIFIER const Flash_fwd_params params) {
#if defined(ARCH_SUPPORTS_FLASH)
    FLASH_NAMESPACE::compute_attn_splitkv<
        Fr13Fixed32Qrow16KernelTraits,
        false,  // Is_causal
        false,  // Is_local
        false,  // Has_alibi
        false,  // Is_even_MN: paged varlen Q has cu_seqlens_q
        true,   // Is_even_K: d == kHeadDim == 256
        false,  // Is_softcap
        false,  // Split
        false   // Append_KV
    >(params);
#else
    FLASH_UNSUPPORTED_ARCH
#endif
}

// Internal caller-only entry point. This separate TU keeps the stock HD256
// BF16 explicit-instantiation object byte-for-byte unchanged.
__attribute__((visibility("hidden")))
void fr13_run_mha_fwd_fixed32_qrow16(
        Flash_fwd_params &params, cudaStream_t stream) {
    // FR13_FA2_FIXED32_QUERY_TILE16: two query CTAs fill all 48 GB10 SMs.
    // This is query partitioning, not split-K: each real row keeps one warp's
    // complete, ordered K-block loop and no combine kernel is launched.
    constexpr static int kTreeBlockM = 16;
    constexpr static int kTreeBlockN = 64;
    constexpr static int kTreeWarps = 1;
    static_assert(kTreeBlockM == 16 * kTreeWarps);
    using TreeKernelTraits = Fr13Fixed32Qrow16KernelTraits;
    static_assert(TreeKernelTraits::kBlockM == kTreeBlockM);
    static_assert(TreeKernelTraits::kBlockN == kTreeBlockN);
    static_assert(TreeKernelTraits::kNWarps == kTreeWarps);
    static_assert(TreeKernelTraits::kNThreads == 32);
    static_assert(TreeKernelTraits::kGmemThreadsPerRow == 8);
    // The public FA2 API requires paged-KV blocks divisible by 16;
    // production fixes the physical page at 1024 rows.
    static_assert(TreeKernelTraits::kGmemRowsPerThread == 16);
    static_assert(1024 % TreeKernelTraits::kGmemRowsPerThread == 0);
    constexpr size_t smem_size = TreeKernelTraits::kSmemSize;
    const int num_m_block =
        (params.seqlen_q + TreeKernelTraits::kBlockM - 1)
        / TreeKernelTraits::kBlockM;
    dim3 grid(num_m_block, params.b, params.h);
    auto kernel = &fr13_flash_fwd_fixed32_qrow16_kernel;
    if (smem_size >= 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size));
    }
    kernel<<<grid, TreeKernelTraits::kNThreads, smem_size, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace FLASH_NAMESPACE
'''


FIXED32_QUERY_TILE16_STATIC_STRIDES_TRAIT = r'''
template <>
struct StaticPagedKVStrides<Fr13Fixed32Qrow16KernelTraits> {
    static constexpr int64_t page = 1024 * 4 * 256;
    static constexpr int64_t row = 4 * 256;
    static constexpr int64_t head = 256;
};
'''


_FIXED32_QUERY_TILE16_STATIC_STRIDES_ANCHOR = r'''template <>
struct StaticPagedKVBlockSize<Fr13Fixed32Qrow16KernelTraits> {
    static constexpr int value = 1024;
    static constexpr int log2 = 10;
    static constexpr int block_n_log2 = 6;
};
'''


FIXED32_QUERY_TILE16_STATIC_STRIDES_TRANSLATION_UNIT = (
    FIXED32_QUERY_TILE16_TRANSLATION_UNIT.replace(
        _FIXED32_QUERY_TILE16_STATIC_STRIDES_ANCHOR,
        _FIXED32_QUERY_TILE16_STATIC_STRIDES_ANCHOR
        + FIXED32_QUERY_TILE16_STATIC_STRIDES_TRAIT,
        1,
    )
)


FIXED32_QUERY_TILE32_TRANSLATION_UNIT = r'''// FR13 fixed32 B4 qrow32 gate candidate.
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

using Fr13Fixed32Qrow32KernelTraits = Flash_fwd_kernel_traits<
    256, 32, 64, 2, false, false, cutlass::bfloat16_t>;

template <>
struct StaticPagedKVBlockSize<Fr13Fixed32Qrow32KernelTraits> {
    static constexpr int value = 1024;
    static constexpr int log2 = 10;
    static constexpr int block_n_log2 = 6;
};

template <>
struct StaticPagedKVStrides<Fr13Fixed32Qrow32KernelTraits> {
    static constexpr int64_t page = 2 * 1024 * 4 * 256;
    static constexpr int64_t row = 4 * 256;
    static constexpr int64_t head = 256;
};

template <>
struct StaticQueryRows<Fr13Fixed32Qrow32KernelTraits> {
    static constexpr int value = 32;
};

template <>
struct StaticQueryBatchLayout<Fr13Fixed32Qrow32KernelTraits> {
    static constexpr int sequences = 4;
    static constexpr int query_heads = 24;
    static constexpr int kv_heads = 4;
    static constexpr int query_heads_per_kv = 6;
};

// The exact non-templated entry point gives ptxas a tighter register
// allocation than the generic split-K kernel. Keep the hard ceiling below the
// rejected 255-register allocation; the static gate also rejects any spill.
__global__ __maxnreg__(254)
void fr13_flash_fwd_fixed32_qrow32_kernel(
        KERNEL_PARAM_MODIFIER const Flash_fwd_params params) {
#if defined(ARCH_SUPPORTS_FLASH)
    FLASH_NAMESPACE::compute_attn_splitkv<
        Fr13Fixed32Qrow32KernelTraits,
        false,  // Is_causal
        false,  // Is_local
        false,  // Has_alibi
        false,  // Is_even_MN: paged varlen Q has cu_seqlens_q
        true,   // Is_even_K: d == kHeadDim == 256
        false,  // Is_softcap
        false,  // Split
        false   // Append_KV
    >(params);
#else
    FLASH_UNSUPPORTED_ARCH
#endif
}

// Gate-only entry point. The ordinary and production paths cannot tag the
// exact4 diagnostic bias layout that selects this hidden function.
__attribute__((visibility("hidden")))
void fr13_run_mha_fwd_fixed32_qrow32(
        Flash_fwd_params &params, cudaStream_t stream) {
    // One physical32 query CTA per batch/head: B4 * H24 = 96 CTAs/layer.
    // This is query tiling, not split-K. Every real row retains one warp's
    // complete ordered K-block loop and no combine kernel is launched.
    constexpr static int kTreeBlockM = 32;
    constexpr static int kTreeBlockN = 64;
    constexpr static int kTreeWarps = 2;
    static_assert(kTreeBlockM == 16 * kTreeWarps);
    using TreeKernelTraits = Fr13Fixed32Qrow32KernelTraits;
    static_assert(TreeKernelTraits::kBlockM == kTreeBlockM);
    static_assert(TreeKernelTraits::kBlockN == kTreeBlockN);
    static_assert(TreeKernelTraits::kNWarps == kTreeWarps);
    static_assert(TreeKernelTraits::kNThreads == 64);
    static_assert(TreeKernelTraits::kGmemThreadsPerRow == 8);
    static_assert(TreeKernelTraits::kGmemRowsPerThread == 8);
    static_assert(1024 % TreeKernelTraits::kGmemRowsPerThread == 0);
    constexpr size_t smem_size = TreeKernelTraits::kSmemSize;
    static_assert(smem_size == 80 * 1024);
    using StaticLayout = StaticQueryBatchLayout<TreeKernelTraits>;
    static_assert(StaticLayout::sequences == 4);
    static_assert(StaticLayout::query_heads == 24);
    static_assert(StaticLayout::kv_heads == 4);
    static_assert(StaticLayout::query_heads_per_kv == 6);
    static_assert(
        StaticLayout::query_heads
        == StaticLayout::kv_heads * StaticLayout::query_heads_per_kv);
    TORCH_CHECK(
        params.tree_bias_ptr != nullptr
        && params.is_bf16
        && !params.is_causal
        && params.b == 4
        && params.total_q == 128
        && params.d == 256
        && params.d_rounded == 256
        && params.h == 24
        && params.h_k == 4
        && params.h_h_k_ratio == 6
        && params.seqlen_q == 32
        && params.seqlen_q_rounded == 128
        && params.q_head_stride == 256
        && params.k_batch_stride == 2 * 1024 * 4 * 256
        && params.k_row_stride == 4 * 256
        && params.k_head_stride == 256
        && params.v_batch_stride == 2 * 1024 * 4 * 256
        && params.v_row_stride == 4 * 256
        && params.v_head_stride == 256
        && params.o_head_stride == 256
        && params.tree_bias_rows == 32
        && params.tree_bias_cols == 32
        && params.tree_bias_row_stride == 32
        && params.tree_bias_col_stride == 1
        && params.tree_bias_q_offset == 0
        && params.tree_bias_k_offset == 0
        && params.cu_seqlens_q != nullptr
        && params.cu_seqlens_k != nullptr
        && params.seqused_k != nullptr
        && params.is_seqlens_k_cumulative
        && !params.seqlenq_ngroups_swapped
        && params.leftpad_k == nullptr
        && params.cache_batch_idx == nullptr
        && params.block_table != nullptr
        && params.block_table_batch_stride > 0
        && params.page_block_size == 1024
        && params.window_size_left < 0
        && params.window_size_right < 0
        && params.alibi_slopes_ptr == nullptr
        && params.knew_ptr == nullptr
        && params.vnew_ptr == nullptr
        && params.p_ptr == nullptr
        && params.softmax_lse_ptr != nullptr
        && params.p_dropout == 1.0f
        && params.softcap == 0.0f
        && params.num_splits == 0,
        "FR13 qrow32 B4 launcher reached non-canonical geometry");
    // blockIdx.x is the query-head lane within a six-head GQA group;
    // blockIdx.z is therefore already the KV head. This remains 96 CTAs.
    dim3 grid(
        StaticLayout::query_heads_per_kv,
        StaticLayout::sequences,
        StaticLayout::kv_heads);
    auto kernel = &fr13_flash_fwd_fixed32_qrow32_kernel;
    if (smem_size >= 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size));
    }
    kernel<<<grid, TreeKernelTraits::kNThreads, smem_size, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace FLASH_NAMESPACE
'''


FIXED32_QUERY_GQA_PAIR32_TRANSLATION_UNIT = r'''// FR13 fixed32 B4 qrow32 GQA-pair gate candidate.
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

using Fr13Fixed32Qrow32GqaPairKernelTraits = Flash_fwd_kernel_traits<
    256, 64, 64, 4, false, false, cutlass::bfloat16_t>;

template <>
struct StaticPagedKVBlockSize<Fr13Fixed32Qrow32GqaPairKernelTraits> {
    static constexpr int value = 1024;
    static constexpr int log2 = 10;
    static constexpr int block_n_log2 = 6;
};

template <>
struct StaticPagedKVStrides<Fr13Fixed32Qrow32GqaPairKernelTraits> {
    static constexpr int64_t page = 2 * 1024 * 4 * 256;
    static constexpr int64_t row = 4 * 256;
    static constexpr int64_t head = 256;
};

template <>
struct StaticQueryRows<Fr13Fixed32Qrow32GqaPairKernelTraits> {
    static constexpr int value = 32;
};

template <>
struct StaticQueryBatchLayout<Fr13Fixed32Qrow32GqaPairKernelTraits> {
    static constexpr int sequences = 4;
    static constexpr int query_heads = 24;
    static constexpr int kv_heads = 4;
    static constexpr int query_heads_per_kv = 6;
};

template <>
struct StaticQueryHeadsPerCTA<Fr13Fixed32Qrow32GqaPairKernelTraits> {
    static constexpr int value = 2;
};

__global__ __maxnreg__(254)
void fr13_flash_fwd_fixed32_qrow32_gqa_pair_kernel(
        KERNEL_PARAM_MODIFIER const Flash_fwd_params params) {
#if defined(ARCH_SUPPORTS_FLASH)
    FLASH_NAMESPACE::compute_attn_splitkv<
        Fr13Fixed32Qrow32GqaPairKernelTraits,
        false,  // Is_causal
        false,  // Is_local
        false,  // Has_alibi
        false,  // Is_even_MN: paged varlen Q has cu_seqlens_q
        true,   // Is_even_K: d == kHeadDim == 256
        false,  // Is_softcap
        false,  // Split
        false   // Append_KV
    >(params);
#else
    FLASH_UNSUPPORTED_ARCH
#endif
}

// Two adjacent query heads in one GQA group share each staged K/V tile. The
// logical M coordinate is ((query_row, head_in_pair), column), so each head
// retains 32 independent rows and the same reverse-ordered K loop.
__attribute__((visibility("hidden")))
void fr13_run_mha_fwd_fixed32_qrow32_gqa_pair(
        Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int kTreeBlockM = 64;
    constexpr static int kTreeBlockN = 64;
    constexpr static int kTreeWarps = 4;
    constexpr static int kHeadsPerCTA = 2;
    using TreeKernelTraits = Fr13Fixed32Qrow32GqaPairKernelTraits;
    using StaticLayout = StaticQueryBatchLayout<TreeKernelTraits>;
    static_assert(TreeKernelTraits::kBlockM == kTreeBlockM);
    static_assert(TreeKernelTraits::kBlockN == kTreeBlockN);
    static_assert(TreeKernelTraits::kNWarps == kTreeWarps);
    static_assert(TreeKernelTraits::kNThreads == 128);
    static_assert(TreeKernelTraits::kGmemThreadsPerRow == 8);
    static_assert(TreeKernelTraits::kGmemRowsPerThread == 4);
    static_assert(StaticQueryHeadsPerCTA<TreeKernelTraits>::value == kHeadsPerCTA);
    static_assert(StaticLayout::sequences == 4);
    static_assert(StaticLayout::query_heads == 24);
    static_assert(StaticLayout::kv_heads == 4);
    static_assert(StaticLayout::query_heads_per_kv == 6);
    static_assert(StaticLayout::query_heads_per_kv % kHeadsPerCTA == 0);
    constexpr size_t smem_size = TreeKernelTraits::kSmemSize;
    static_assert(smem_size == 96 * 1024);
    // 3 head pairs * B4 * 4 KV heads = 48 CTAs/layer. There is no split-K or
    // combine launch, and each CTA stages one K/V tile for both query heads.
    dim3 grid(
        StaticLayout::query_heads_per_kv / kHeadsPerCTA,
        StaticLayout::sequences,
        StaticLayout::kv_heads);
    auto kernel = &fr13_flash_fwd_fixed32_qrow32_gqa_pair_kernel;
    if (smem_size >= 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size));
    }
    kernel<<<grid, TreeKernelTraits::kNThreads, smem_size, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace FLASH_NAMESPACE
'''


FIXED32_QUERY_TILE32_B1_TRANSLATION_UNIT = r'''// FR13 fixed32 B1 qrow32 gate candidate.
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

using Fr13Fixed32Qrow32B1KernelTraits = Flash_fwd_kernel_traits<
    256, 32, 64, 2, false, false, cutlass::bfloat16_t>;

template <>
struct StaticPagedKVBlockSize<Fr13Fixed32Qrow32B1KernelTraits> {
    static constexpr int value = 1024;
    static constexpr int log2 = 10;
    static constexpr int block_n_log2 = 6;
};

template <>
struct StaticPagedKVStrides<Fr13Fixed32Qrow32B1KernelTraits> {
    static constexpr int64_t page = 2 * 1024 * 4 * 256;
    static constexpr int64_t row = 4 * 256;
    static constexpr int64_t head = 256;
};

template <>
struct StaticQueryRows<Fr13Fixed32Qrow32B1KernelTraits> {
    static constexpr int value = 32;
};

template <>
struct StaticQueryBatchLayout<Fr13Fixed32Qrow32B1KernelTraits> {
    static constexpr int sequences = 1;
    static constexpr int query_heads = 24;
    static constexpr int kv_heads = 4;
    static constexpr int query_heads_per_kv = 6;
};

__global__ __maxnreg__(254)
void fr13_flash_fwd_fixed32_qrow32_b1_kernel(
        KERNEL_PARAM_MODIFIER const Flash_fwd_params params) {
#if defined(ARCH_SUPPORTS_FLASH)
    FLASH_NAMESPACE::compute_attn_splitkv<
        Fr13Fixed32Qrow32B1KernelTraits,
        false,  // Is_causal
        false,  // Is_local
        false,  // Has_alibi
        false,  // Is_even_MN: paged varlen Q has cu_seqlens_q
        true,   // Is_even_K: d == kHeadDim == 256
        false,  // Is_softcap
        false,  // Split
        false   // Append_KV
    >(params);
#else
    FLASH_UNSUPPORTED_ARCH
#endif
}

__attribute__((visibility("hidden")))
void fr13_run_mha_fwd_fixed32_qrow32_b1(
        Flash_fwd_params &params, cudaStream_t stream) {
    // One physical32 query CTA per head: B1 * H24 = 24 CTAs/layer.
    // Both query warps share one complete ordered K/V scan; this is not split-K.
    constexpr static int kTreeBlockM = 32;
    constexpr static int kTreeBlockN = 64;
    constexpr static int kTreeWarps = 2;
    static_assert(kTreeBlockM == 16 * kTreeWarps);
    using TreeKernelTraits = Fr13Fixed32Qrow32B1KernelTraits;
    static_assert(TreeKernelTraits::kBlockM == kTreeBlockM);
    static_assert(TreeKernelTraits::kBlockN == kTreeBlockN);
    static_assert(TreeKernelTraits::kNWarps == kTreeWarps);
    static_assert(TreeKernelTraits::kNThreads == 64);
    static_assert(TreeKernelTraits::kGmemThreadsPerRow == 8);
    static_assert(TreeKernelTraits::kGmemRowsPerThread == 8);
    static_assert(1024 % TreeKernelTraits::kGmemRowsPerThread == 0);
    constexpr size_t smem_size = TreeKernelTraits::kSmemSize;
    static_assert(smem_size == 80 * 1024);
    using StaticLayout = StaticQueryBatchLayout<TreeKernelTraits>;
    static_assert(StaticLayout::sequences == 1);
    static_assert(StaticLayout::query_heads == 24);
    static_assert(StaticLayout::kv_heads == 4);
    static_assert(StaticLayout::query_heads_per_kv == 6);
    static_assert(
        StaticLayout::query_heads
        == StaticLayout::kv_heads * StaticLayout::query_heads_per_kv);
    TORCH_CHECK(
        params.tree_bias_batch_stride == 1179791668
        && params.tree_bias_ptr != nullptr
        && params.is_bf16
        && !params.is_causal
        && params.b == 1
        && params.total_q == 32
        && params.d == 256
        && params.d_rounded == 256
        && params.h == 24
        && params.h_k == 4
        && params.h_h_k_ratio == 6
        && params.seqlen_q == 32
        && params.seqlen_q_rounded == 128
        && params.q_head_stride == 256
        && params.k_batch_stride == 2 * 1024 * 4 * 256
        && params.k_row_stride == 4 * 256
        && params.k_head_stride == 256
        && params.v_batch_stride == 2 * 1024 * 4 * 256
        && params.v_row_stride == 4 * 256
        && params.v_head_stride == 256
        && params.o_head_stride == 256
        && params.tree_bias_rows == 32
        && params.tree_bias_cols == 32
        && params.tree_bias_row_stride == 32
        && params.tree_bias_col_stride == 1
        && params.tree_bias_q_offset == 0
        && params.tree_bias_k_offset == 0
        && params.cu_seqlens_q != nullptr
        && params.seqused_k != nullptr
        && !params.seqlenq_ngroups_swapped
        && params.leftpad_k == nullptr
        && params.cache_batch_idx == nullptr
        && params.block_table != nullptr
        && params.block_table_batch_stride > 0
        && params.page_block_size == 1024
        && params.window_size_left < 0
        && params.window_size_right < 0
        && params.alibi_slopes_ptr == nullptr
        && params.knew_ptr == nullptr
        && params.vnew_ptr == nullptr
        && params.p_ptr == nullptr
        && params.softmax_lse_ptr != nullptr
        && params.p_dropout == 1.0f
        && params.softcap == 0.0f
        && params.num_splits == 0,
        "FR13 qrow32 B1 launcher reached non-canonical geometry");
    dim3 grid(
        StaticLayout::query_heads_per_kv,
        StaticLayout::sequences,
        StaticLayout::kv_heads);
    auto kernel = &fr13_flash_fwd_fixed32_qrow32_b1_kernel;
    if (smem_size >= 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size));
    }
    kernel<<<grid, TreeKernelTraits::kNThreads, smem_size, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace FLASH_NAMESPACE
'''


FIXED32_QUERY_TILE32_B1_SPLIT2_TRANSLATION_UNIT = r'''// FR13 fixed32 B1 qrow32 split2 gate candidate.
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

using Fr13Fixed32Qrow32B1Split2KernelTraits = Flash_fwd_kernel_traits<
    256, 32, 64, 2, false, false, cutlass::bfloat16_t>;
using Fr13Fixed32Qrow32B1Split2CombineTraits = Flash_fwd_kernel_traits<
    256, 64, 64, 4, false, false, cutlass::bfloat16_t>;

template <>
struct StaticPagedKVBlockSize<Fr13Fixed32Qrow32B1Split2KernelTraits> {
    static constexpr int value = 1024;
    static constexpr int log2 = 10;
    static constexpr int block_n_log2 = 6;
};

template <>
struct StaticPagedKVStrides<Fr13Fixed32Qrow32B1Split2KernelTraits> {
    static constexpr int64_t page = 2 * 1024 * 4 * 256;
    static constexpr int64_t row = 4 * 256;
    static constexpr int64_t head = 256;
};

template <>
struct StaticQueryRows<Fr13Fixed32Qrow32B1Split2KernelTraits> {
    static constexpr int value = 32;
};

template <>
struct StaticQueryBatchLayout<Fr13Fixed32Qrow32B1Split2KernelTraits> {
    static constexpr int sequences = 1;
    static constexpr int query_heads = 24;
    static constexpr int kv_heads = 4;
    static constexpr int query_heads_per_kv = 6;
};

__global__ __maxnreg__(254)
void fr13_flash_fwd_fixed32_qrow32_b1_split2_kernel(
        KERNEL_PARAM_MODIFIER const Flash_fwd_params params) {
#if defined(ARCH_SUPPORTS_FLASH)
    FLASH_NAMESPACE::compute_attn_splitkv<
        Fr13Fixed32Qrow32B1Split2KernelTraits,
        false,  // Is_causal
        false,  // Is_local
        false,  // Has_alibi
        false,  // Is_even_MN: paged varlen Q has cu_seqlens_q
        true,   // Is_even_K: d == kHeadDim == 256
        false,  // Is_softcap
        true,   // Split: blockIdx.y partitions K blocks exactly twice
        false   // Append_KV
    >(params);
#else
    FLASH_UNSUPPORTED_ARCH
#endif
}

__attribute__((visibility("hidden")))
void fr13_run_mha_fwd_fixed32_qrow32_b1_split2(
        Flash_fwd_params &params, cudaStream_t stream) {
    // Two disjoint context partitions per head: B1 * H24 * split2 = 48 CTAs.
    constexpr static int kTreeBlockM = 32;
    constexpr static int kTreeBlockN = 64;
    constexpr static int kTreeWarps = 2;
    constexpr static int kContextSplits = 2;
    static_assert(kTreeBlockM == 16 * kTreeWarps);
    using TreeKernelTraits = Fr13Fixed32Qrow32B1Split2KernelTraits;
    static_assert(TreeKernelTraits::kBlockM == kTreeBlockM);
    static_assert(TreeKernelTraits::kBlockN == kTreeBlockN);
    static_assert(TreeKernelTraits::kNWarps == kTreeWarps);
    static_assert(TreeKernelTraits::kNThreads == 64);
    static_assert(TreeKernelTraits::kGmemThreadsPerRow == 8);
    static_assert(TreeKernelTraits::kGmemRowsPerThread == 8);
    static_assert(1024 % TreeKernelTraits::kGmemRowsPerThread == 0);
    constexpr size_t smem_size = TreeKernelTraits::kSmemSize;
    static_assert(smem_size == 80 * 1024);
    using StaticLayout = StaticQueryBatchLayout<TreeKernelTraits>;
    static_assert(StaticLayout::sequences == 1);
    static_assert(StaticLayout::query_heads == 24);
    static_assert(StaticLayout::kv_heads == 4);
    static_assert(StaticLayout::query_heads_per_kv == 6);
    static_assert(
        StaticLayout::query_heads
        == StaticLayout::kv_heads * StaticLayout::query_heads_per_kv);
    TORCH_CHECK(
        params.tree_bias_batch_stride == 1179791669
        && params.tree_bias_ptr != nullptr
        && params.is_bf16
        && !params.is_causal
        && params.b == 1
        && params.total_q == 32
        && params.d == 256
        && params.d_rounded == 256
        && params.h == 24
        && params.h_k == 4
        && params.h_h_k_ratio == 6
        && params.seqlen_q == 32
        && params.seqlen_q_rounded == 128
        && params.q_head_stride == 256
        && params.k_batch_stride == 2 * 1024 * 4 * 256
        && params.k_row_stride == 4 * 256
        && params.k_head_stride == 256
        && params.v_batch_stride == 2 * 1024 * 4 * 256
        && params.v_row_stride == 4 * 256
        && params.v_head_stride == 256
        && params.o_head_stride == 256
        && params.tree_bias_rows == 32
        && params.tree_bias_cols == 32
        && params.tree_bias_row_stride == 32
        && params.tree_bias_col_stride == 1
        && params.tree_bias_q_offset == 0
        && params.tree_bias_k_offset == 0
        && params.cu_seqlens_q != nullptr
        && params.seqused_k != nullptr
        && !params.seqlenq_ngroups_swapped
        && params.leftpad_k == nullptr
        && params.cache_batch_idx == nullptr
        && params.block_table != nullptr
        && params.block_table_batch_stride > 0
        && params.page_block_size == 1024
        && params.window_size_left < 0
        && params.window_size_right < 0
        && params.alibi_slopes_ptr == nullptr
        && params.knew_ptr == nullptr
        && params.vnew_ptr == nullptr
        && params.p_ptr == nullptr
        && params.softmax_lse_ptr != nullptr
        && params.p_dropout == 1.0f
        && params.softcap == 0.0f
        && params.num_splits == kContextSplits
        && params.oaccum_ptr != nullptr
        && params.softmax_lseaccum_ptr != nullptr,
        "FR13 B1 qrow32 split2 launcher geometry or scratch drifted");
    dim3 grid(
        StaticLayout::query_heads_per_kv,
        kContextSplits,
        StaticLayout::kv_heads);
    auto kernel = &fr13_flash_fwd_fixed32_qrow32_b1_split2_kernel;
    if (smem_size >= 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size));
    }
    kernel<<<grid, TreeKernelTraits::kNThreads, smem_size, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Reuse FA2's exact combine implementation with its required four-warp
    // traits. The main attention kernel remains the two-warp BM32 trait.
    using CombineTraits = Fr13Fixed32Qrow32B1Split2CombineTraits;
    static_assert(CombineTraits::kNThreads == 128);
    constexpr int kCombineBlockM = 4;
    constexpr int kLogMaxSplits = 1;
    constexpr bool kEvenK = true;
    dim3 combine_grid(
        (StaticLayout::sequences * StaticLayout::query_heads * 32
         + kCombineBlockM - 1) / kCombineBlockM);
    flash_fwd_splitkv_combine_kernel<
        CombineTraits, kCombineBlockM, kLogMaxSplits, kEvenK>
        <<<combine_grid, CombineTraits::kNThreads, 0, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace FLASH_NAMESPACE
'''


# The B1 GQA-pair translation unit is derived mechanically from the B4 GQA-pair
# unit rather than retyped. Every trait the two share -- the fused paged-KV page
# stride, the 1024-entry page, the 32 static query rows, the two query heads per
# CTA, kBlockM=64 / kBlockN=64 / kNWarps=4, Split=false -- therefore stays
# byte-identical to the unit that carried the dual byte gate, and the only
# deltas are the ones B1 actually forces: StaticQueryBatchLayout::sequences and
# the private symbol names. Each substitution below is anchored and counted, so
# any future drift in the B4 unit fails here instead of silently forking the two
# kernels apart.
_FIXED32_QUERY_GQA_PAIR32_B1_TRANSLATION_UNIT_SUBSTITUTIONS = (
    (
        "// FR13 fixed32 B4 qrow32 GQA-pair gate candidate.",
        "// FR13 fixed32 B1 qrow32 GQA-pair gate candidate.",
        1,
    ),
    (
        "Fr13Fixed32Qrow32GqaPairKernelTraits",
        "Fr13Fixed32Qrow32GqaPairB1KernelTraits",
        8,
    ),
    (
        "fr13_flash_fwd_fixed32_qrow32_gqa_pair_kernel",
        "fr13_flash_fwd_fixed32_qrow32_gqa_pair_b1_kernel",
        2,
    ),
    (
        "fr13_run_mha_fwd_fixed32_qrow32_gqa_pair(",
        "fr13_run_mha_fwd_fixed32_qrow32_gqa_pair_b1(",
        1,
    ),
    (
        "    static constexpr int sequences = 4;",
        "    static constexpr int sequences = 1;",
        1,
    ),
    (
        "    static_assert(StaticLayout::sequences == 4);",
        "    static_assert(StaticLayout::sequences == 1);",
        1,
    ),
    (
        "    // 3 head pairs * B4 * 4 KV heads = 48 CTAs/layer. There is no "
        "split-K or\n"
        "    // combine launch, and each CTA stages one K/V tile for both "
        "query heads.",
        "    // 3 head pairs * B1 * 4 KV heads = 12 CTAs/layer. There is no "
        "split-K or\n"
        "    // combine launch, and each CTA stages one K/V tile for both "
        "query heads.\n"
        "    // The incumbent B1 qrow16 kernel spends 48 single-warp CTAs "
        "re-staging the\n"
        "    // same KV; pairing the GQA heads removes half of that redundant "
        "staging\n"
        "    // at sequences=1.",
        1,
    ),
)


def _fixed32_query_gqa_pair32_b1_translation_unit() -> str:
    unit = FIXED32_QUERY_GQA_PAIR32_TRANSLATION_UNIT
    for anchor, replacement, expected in (
        _FIXED32_QUERY_GQA_PAIR32_B1_TRANSLATION_UNIT_SUBSTITUTIONS
    ):
        if unit.count(anchor) != expected:
            raise RuntimeError(
                "B4 GQA-pair translation unit drifted at the B1 derivation "
                f"anchor {anchor!r}: expected {expected}, found "
                f"{unit.count(anchor)}"
            )
        unit = unit.replace(anchor, replacement)
    for survivor in (
        "sequences = 4",
        "Fr13Fixed32Qrow32GqaPairKernelTraits",
        "B4",
    ):
        if survivor in unit:
            raise RuntimeError(
                f"B1 GQA-pair translation unit still carries B4 text: {survivor!r}"
            )
    return unit


FIXED32_QUERY_GQA_PAIR32_B1_TRANSLATION_UNIT = (
    _fixed32_query_gqa_pair32_b1_translation_unit()
)


# The FR14 split-K unit is derived mechanically from the promoted B1 GQA-pair
# unit, on the same discipline that derived that unit from B4: every trait the
# two share -- kBlockM=64 / kBlockN=64 / kNWarps=4, two query heads per CTA, the
# 1024-row fused paged-KV layout, 32 static query rows, 96 KiB smem, REG budget
# -- therefore stays byte-identical to the arm that carries the byte gate, and
# the only deltas are the ones split-K actually forces: Split=true, a
# blockIdx.y context partition, and the combine launch. Each substitution is
# anchored and counted, so a drift in the promoted unit fails HERE rather than
# silently forking the two kernels apart.
#
# THIS IS A TIER-B ARM. Splitting the context walk changes the order in which a
# row's softmax denominator and weighted sum are accumulated: each split keeps
# its own running max and partial sums, and the combine rescales by
# exp(m_split - m_global) before summing. The per-row arithmetic is therefore
# NOT bit-identical to the promoted arm and no byte gate can qualify it -- see
# _fr13_fa2_qrow32_b1_require_same_reduction, which refuses to compare arms of
# differing reduction topology at all.
_FIXED32_QUERY_GQA_PAIR32_SPLITK_B1_LAUNCH_TAIL = rf'''    // SPLIT-K (Tier-B). 3 head pairs * {FIXED32_QUERY_GQA_PAIR32_SPLITK_B1_CONTEXT_SPLITS} context splits * 4 KV heads
    // = 48 CTAs/layer at B1 -- one per SM on this device -- against the 12 the
    // promoted GQA-pair arm launches. Each CTA still stages one K/V tile for
    // both of its query heads and still walks its own tiles in the same reverse
    // order; what changes is that it walks only 1/{FIXED32_QUERY_GQA_PAIR32_SPLITK_B1_CONTEXT_SPLITS} of them and writes a PARTIAL
    // attention -- running max, partial denominator, partial weighted sum --
    // into the stock split accumulators. FA2's own combine kernel then rescales
    // each split by exp(m_split - m_global), sums, and blends.
    TORCH_CHECK(
        params.tree_bias_batch_stride == {FIXED32_QUERY_GQA_PAIR32_SPLITK_B1_BATCH_STRIDE_SENTINEL}
        && params.num_splits == kContextSplits
        && params.oaccum_ptr != nullptr
        && params.softmax_lseaccum_ptr != nullptr
        && params.b == 1
        && params.total_q == 32
        && params.h == 24
        && params.h_k == 4
        && params.seqlen_q == 32
        && params.d_rounded == 256
        && params.unpadded_lse,
        "FR14 B1 qrow32 GQA-pair split-K launcher geometry or scratch drifted");
    dim3 grid(
        StaticLayout::query_heads_per_kv / kHeadsPerCTA,
        kContextSplits,
        StaticLayout::kv_heads);
    auto kernel = &fr13_flash_fwd_fixed32_qrow32_gqa_pair_b1_kernel;
    if (smem_size >= 48 * 1024) {{
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size));
    }}
    kernel<<<grid, TreeKernelTraits::kNThreads, smem_size, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Reuse FA2's exact combine implementation, exactly as the qualified
    // qrow32 B1 split2 unit does. The combine asserts 128 threads; the promoted
    // GQA-pair traits already ARE four-warp / 128-thread, so the same traits
    // carry both launches and no second trait instantiation is introduced.
    //
    // DETERMINISM. The combine is a fixed reduction: every row's splits are
    // visited in index order 0..num_splits-1 by one thread, the cross-split max
    // and sum go through Allreduce<>'s fixed butterfly of __shfl_xor_sync, and
    // nothing is accumulated with an atomic. Same inputs therefore give bitwise
    // the same outputs on every run.
    using CombineTraits = TreeKernelTraits;
    static_assert(CombineTraits::kNThreads == 128);
    constexpr int kCombineBlockM = 4;
    constexpr int kLogMaxSplits = 2;
    static_assert((1 << kLogMaxSplits) == kContextSplits);
    constexpr bool kEvenK = true;
    dim3 combine_grid(
        (StaticLayout::sequences * StaticLayout::query_heads * 32
         + kCombineBlockM - 1) / kCombineBlockM);
    flash_fwd_splitkv_combine_kernel<
        CombineTraits, kCombineBlockM, kLogMaxSplits, kEvenK>
        <<<combine_grid, CombineTraits::kNThreads, 0, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}}
'''


_FIXED32_QUERY_GQA_PAIR32_SPLITK_B1_TRANSLATION_UNIT_SUBSTITUTIONS = (
    (
        "// FR13 fixed32 B1 qrow32 GQA-pair gate candidate.",
        "// FR14 fixed32 B1 qrow32 GQA-pair SPLIT-K gate candidate (Tier-B).",
        1,
    ),
    (
        "    // 3 head pairs * B1 * 4 KV heads = 12 CTAs/layer. There is no "
        "split-K or\n"
        "    // combine launch, and each CTA stages one K/V tile for both "
        "query heads.\n"
        "    // The incumbent B1 qrow16 kernel spends 48 single-warp CTAs "
        "re-staging the\n"
        "    // same KV; pairing the GQA heads removes half of that redundant "
        "staging\n"
        "    // at sequences=1.\n"
        "    dim3 grid(\n"
        "        StaticLayout::query_heads_per_kv / kHeadsPerCTA,\n"
        "        StaticLayout::sequences,\n"
        "        StaticLayout::kv_heads);\n"
        "    auto kernel = &fr13_flash_fwd_fixed32_qrow32_gqa_pair_b1_kernel;\n"
        "    if (smem_size >= 48 * 1024) {\n"
        "        C10_CUDA_CHECK(cudaFuncSetAttribute(\n"
        "            kernel,\n"
        "            cudaFuncAttributeMaxDynamicSharedMemorySize,\n"
        "            smem_size));\n"
        "    }\n"
        "    kernel<<<grid, TreeKernelTraits::kNThreads, smem_size, stream>>>"
        "(params);\n"
        "    C10_CUDA_KERNEL_LAUNCH_CHECK();\n"
        "}\n",
        _FIXED32_QUERY_GQA_PAIR32_SPLITK_B1_LAUNCH_TAIL,
        1,
    ),
    (
        "    constexpr static int kHeadsPerCTA = 2;\n",
        "    constexpr static int kHeadsPerCTA = 2;\n"
        "    constexpr static int kContextSplits = "
        f"{FIXED32_QUERY_GQA_PAIR32_SPLITK_B1_CONTEXT_SPLITS};\n",
        1,
    ),
    (
        "        false,  // Split\n",
        "        true,   // Split: blockIdx.y partitions the context walk\n",
        1,
    ),
    (
        "Fr13Fixed32Qrow32GqaPairB1KernelTraits",
        "Fr13Fixed32Qrow32GqaPairSplitKB1KernelTraits",
        8,
    ),
    (
        "fr13_flash_fwd_fixed32_qrow32_gqa_pair_b1_kernel",
        "fr13_flash_fwd_fixed32_qrow32_gqa_pair_splitk_b1_kernel",
        2,
    ),
    (
        "fr13_run_mha_fwd_fixed32_qrow32_gqa_pair_b1(",
        "fr13_run_mha_fwd_fixed32_qrow32_gqa_pair_splitk_b1(",
        1,
    ),
)


def _fixed32_query_gqa_pair32_splitk_b1_translation_unit() -> str:
    unit = FIXED32_QUERY_GQA_PAIR32_B1_TRANSLATION_UNIT
    for anchor, replacement, expected in (
        _FIXED32_QUERY_GQA_PAIR32_SPLITK_B1_TRANSLATION_UNIT_SUBSTITUTIONS
    ):
        if unit.count(anchor) != expected:
            raise RuntimeError(
                "B1 GQA-pair translation unit drifted at the split-K "
                f"derivation anchor {anchor!r}: expected {expected}, found "
                f"{unit.count(anchor)}"
            )
        unit = unit.replace(anchor, replacement)
    # The split-K unit must actually BE split-K, and must not still carry the
    # promoted arm's private symbols (which would collide at link time) or its
    # sentinel (which would let a byte-gated operand reach this kernel).
    for required in (
        "        true,   // Split: blockIdx.y partitions the context walk\n",
        "        kContextSplits,\n",
        "    flash_fwd_splitkv_combine_kernel<\n",
        f"        params.tree_bias_batch_stride == "
        f"{FIXED32_QUERY_GQA_PAIR32_SPLITK_B1_BATCH_STRIDE_SENTINEL}\n",
        "        && params.num_splits == kContextSplits\n",
        "        && params.oaccum_ptr != nullptr\n",
        "        && params.softmax_lseaccum_ptr != nullptr\n",
        "    static_assert(smem_size == 96 * 1024);\n",
    ):
        if unit.count(required) != 1:
            raise RuntimeError(
                f"B1 GQA-pair split-K unit lost a required line: {required!r}"
            )
    for survivor in (
        "Fr13Fixed32Qrow32GqaPairB1KernelTraits",
        "fr13_flash_fwd_fixed32_qrow32_gqa_pair_b1_kernel",
        "fr13_run_mha_fwd_fixed32_qrow32_gqa_pair_b1(",
        "false,  // Split",
        "There is no split-K",
    ):
        if survivor in unit:
            raise RuntimeError(
                "B1 GQA-pair split-K unit still carries the promoted arm's "
                f"text: {survivor!r}"
            )
    return unit


FIXED32_QUERY_GQA_PAIR32_SPLITK_B1_TRANSLATION_UNIT = (
    _fixed32_query_gqa_pair32_splitk_b1_translation_unit()
)


RUN_MHA_FWD_SIGNATURE = (
    "void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream, "
    "bool force_split_kernel=false) {\n"
)


FIXED32_QUERY_TILE32_API_DECLARATION = rf'''constexpr int64_t kFr13Qrow32BatchStrideSentinel =
    {FIXED32_QUERY_TILE32_BATCH_STRIDE_SENTINEL};

__attribute__((visibility("hidden")))
void fr13_run_mha_fwd_fixed32_qrow32(
    Flash_fwd_params &params, cudaStream_t stream);

'''


FIXED32_QUERY_TILE32_API_GATE = r'''    if (params.tree_bias_batch_stride == kFr13Qrow32BatchStrideSentinel) {
        TORCH_CHECK(
            params.tree_bias_ptr != nullptr
            && params.is_bf16
            && !params.is_causal
            && params.b == 4
            && params.total_q == 128
            && params.d == 256
            && params.d_rounded == 256
            && params.h == 24
            && params.h_k == 4
            && params.h_h_k_ratio == 6
            && params.seqlen_q == 32
            && params.seqlen_q_rounded == 128
            && params.q_head_stride == 256
            && params.k_batch_stride == 2 * 1024 * 4 * 256
            && params.k_row_stride == 4 * 256
            && params.k_head_stride == 256
            && params.v_batch_stride == 2 * 1024 * 4 * 256
            && params.v_row_stride == 4 * 256
            && params.v_head_stride == 256
            && params.o_head_stride == 256
            && params.tree_bias_rows == 32
            && params.tree_bias_cols == 32
            && params.tree_bias_row_stride == 32
            && params.tree_bias_col_stride == 1
            && params.tree_bias_q_offset == 0
            && params.tree_bias_k_offset == 0
            && params.cu_seqlens_q != nullptr
            && params.cu_seqlens_k != nullptr
            && params.seqused_k != nullptr
            && params.is_seqlens_k_cumulative
            && !params.seqlenq_ngroups_swapped
            && params.leftpad_k == nullptr
            && params.cache_batch_idx == nullptr
            && params.block_table != nullptr
            && params.block_table_batch_stride > 0
            && params.page_block_size == 1024
            && params.window_size_left < 0
            && params.window_size_right < 0
            && params.alibi_slopes_ptr == nullptr
            && params.knew_ptr == nullptr
            && params.vnew_ptr == nullptr
            && params.p_ptr == nullptr
            && params.softmax_lse_ptr != nullptr
            && params.p_dropout == 1.0f
            && params.softcap == 0.0f
            // set_params_fprop zero-initializes this field. Varlen q=32 does
            // not enter the max_seqlen_q==1 q-group split-K setup. Paged KV
            // reaches this family through force_split_kernel instead.
            && params.num_splits == 0
            && force_split_kernel,
            "FR13 qrow32 gate dispatch reached non-canonical B4 geometry");
        fr13_run_mha_fwd_fixed32_qrow32(params, stream);
        return;
    }
'''


FIXED32_QUERY_GQA_PAIR32_API_DECLARATION = rf'''constexpr int64_t kFr13Qrow32GqaPairBatchStrideSentinel =
    {FIXED32_QUERY_GQA_PAIR32_BATCH_STRIDE_SENTINEL};

__attribute__((visibility("hidden")))
void fr13_run_mha_fwd_fixed32_qrow32_gqa_pair(
    Flash_fwd_params &params, cudaStream_t stream);

'''


FIXED32_QUERY_GQA_PAIR32_API_GATE = r'''    if (params.tree_bias_batch_stride == kFr13Qrow32GqaPairBatchStrideSentinel) {
        TORCH_CHECK(
            params.tree_bias_ptr != nullptr
            && params.is_bf16
            && !params.is_causal
            && params.b == 4
            && params.total_q == 128
            && params.d == 256
            && params.d_rounded == 256
            && params.h == 24
            && params.h_k == 4
            && params.h_h_k_ratio == 6
            && params.seqlen_q == 32
            && params.seqlen_q_rounded == 128
            && params.q_head_stride == 256
            && params.k_batch_stride == 2 * 1024 * 4 * 256
            && params.k_row_stride == 4 * 256
            && params.k_head_stride == 256
            && params.v_batch_stride == 2 * 1024 * 4 * 256
            && params.v_row_stride == 4 * 256
            && params.v_head_stride == 256
            && params.o_head_stride == 256
            && params.tree_bias_rows == 32
            && params.tree_bias_cols == 32
            && params.tree_bias_row_stride == 32
            && params.tree_bias_col_stride == 1
            && params.tree_bias_q_offset == 0
            && params.tree_bias_k_offset == 0
            && params.cu_seqlens_q != nullptr
            && params.cu_seqlens_k != nullptr
            && params.seqused_k != nullptr
            && params.is_seqlens_k_cumulative
            && params.unpadded_lse
            && !params.seqlenq_ngroups_swapped
            && params.leftpad_k == nullptr
            && params.cache_batch_idx == nullptr
            && params.block_table != nullptr
            && params.block_table_batch_stride > 0
            && params.page_block_size == 1024
            && params.window_size_left < 0
            && params.window_size_right < 0
            && params.alibi_slopes_ptr == nullptr
            && params.knew_ptr == nullptr
            && params.vnew_ptr == nullptr
            && params.p_ptr == nullptr
            && params.softmax_lse_ptr != nullptr
            && params.p_dropout == 1.0f
            && params.softcap == 0.0f
            && params.num_splits == 0
            && force_split_kernel,
            "FR13 qrow32 GQA-pair gate reached non-canonical B4 geometry");
        fr13_run_mha_fwd_fixed32_qrow32_gqa_pair(params, stream);
        return;
    }
'''


FIXED32_QUERY_GQA_PAIR32_B1_API_DECLARATION = rf'''constexpr int64_t kFr13Qrow32GqaPairB1BatchStrideSentinel =
    {FIXED32_QUERY_GQA_PAIR32_B1_BATCH_STRIDE_SENTINEL};

__attribute__((visibility("hidden")))
void fr13_run_mha_fwd_fixed32_qrow32_gqa_pair_b1(
    Flash_fwd_params &params, cudaStream_t stream);

'''


# Same derivation discipline as the translation unit: the B1 GQA-pair gate is
# the B4 GQA-pair gate with the two operands B1 actually changes. Everything
# else -- including `unpadded_lse` and `is_seqlens_k_cumulative`, which the
# paired LSE layout depends on because it addresses softmax_lse as
# [head, total_q] -- is inherited verbatim from the operand set the B4 dual byte
# gate exercised. mha_varlen_fwd_impl passes /*unpadded_lse*/true and
# set_params_fprop sets is_seqlens_k_cumulative=true unconditionally at
# fa2_head 29210221863736a08f71a866459e368ad1ac4a95, and force_split_kernel is
# `paged_KV`, which the inherited `block_table != nullptr` already pins.
_FIXED32_QUERY_GQA_PAIR32_B1_API_GATE_SUBSTITUTIONS = (
    (
        "kFr13Qrow32GqaPairBatchStrideSentinel",
        "kFr13Qrow32GqaPairB1BatchStrideSentinel",
        1,
    ),
    ("            && params.b == 4\n", "            && params.b == 1\n", 1),
    (
        "            && params.total_q == 128\n",
        "            && params.total_q == 32\n",
        1,
    ),
    (
        "        fr13_run_mha_fwd_fixed32_qrow32_gqa_pair(params, stream);",
        "        fr13_run_mha_fwd_fixed32_qrow32_gqa_pair_b1(params, stream);",
        1,
    ),
    (
        '"FR13 qrow32 GQA-pair gate reached non-canonical B4 geometry"',
        '"FR13 qrow32 GQA-pair gate reached non-canonical B1 geometry"',
        1,
    ),
)


def _fixed32_query_gqa_pair32_b1_api_gate() -> str:
    gate = FIXED32_QUERY_GQA_PAIR32_API_GATE
    for anchor, replacement, expected in (
        _FIXED32_QUERY_GQA_PAIR32_B1_API_GATE_SUBSTITUTIONS
    ):
        if gate.count(anchor) != expected:
            raise RuntimeError(
                "B4 GQA-pair API gate drifted at the B1 derivation anchor "
                f"{anchor!r}: expected {expected}, found {gate.count(anchor)}"
            )
        gate = gate.replace(anchor, replacement)
    # The B1 shapes must actually be the ones the B1 lineage live-validates.
    for required in (
        "            && params.b == 1\n",
        "            && params.total_q == 32\n",
        "            && params.seqlen_q == 32\n",
        "            && params.seqlen_q_rounded == 128\n",
        "            && params.k_batch_stride == 2 * 1024 * 4 * 256\n",
        "            && params.v_batch_stride == 2 * 1024 * 4 * 256\n",
        "            && params.page_block_size == 1024\n",
        "            && params.unpadded_lse\n",
        "            && params.num_splits == 0\n",
    ):
        if gate.count(required) != 1:
            raise RuntimeError(
                f"B1 GQA-pair API gate lost a required operand: {required!r}"
            )
    for forbidden in ("B4", "params.b == 4", "params.total_q == 128"):
        if forbidden in gate:
            raise RuntimeError(
                "B1 GQA-pair API gate still carries B4 geometry: "
                f"{forbidden!r}"
            )
    return gate


FIXED32_QUERY_GQA_PAIR32_B1_API_GATE = _fixed32_query_gqa_pair32_b1_api_gate()


FIXED32_QUERY_GQA_PAIR32_SPLITK_B1_API_DECLARATION = rf'''constexpr int64_t kFr14Qrow32GqaPairSplitKB1BatchStrideSentinel =
    {FIXED32_QUERY_GQA_PAIR32_SPLITK_B1_BATCH_STRIDE_SENTINEL};

__attribute__((visibility("hidden")))
void fr13_run_mha_fwd_fixed32_qrow32_gqa_pair_splitk_b1(
    Flash_fwd_params &params, cudaStream_t stream);

'''


# The split-K gate is the promoted B1 GQA-pair gate with the three operands
# split-K actually changes -- the sentinel, the split count, and the two stock
# accumulators the combine reads -- and NOTHING else. Every other operand,
# including `unpadded_lse` (which the paired LSE layout depends on because it
# addresses softmax_lse as [head, total_q]) and `is_seqlens_k_cumulative`, is
# inherited verbatim from the operand set the promoted arm's byte gate
# exercised, by counted substitution, so a drift in that gate fails here
# instead of silently forking the two.
_FIXED32_QUERY_GQA_PAIR32_SPLITK_B1_API_GATE_SUBSTITUTIONS = (
    (
        "kFr13Qrow32GqaPairB1BatchStrideSentinel",
        "kFr14Qrow32GqaPairSplitKB1BatchStrideSentinel",
        1,
    ),
    (
        "            && params.num_splits == 0\n",
        "            && params.num_splits == "
        f"{FIXED32_QUERY_GQA_PAIR32_SPLITK_B1_CONTEXT_SPLITS}\n"
        "            && params.oaccum_ptr != nullptr\n"
        "            && params.softmax_lseaccum_ptr != nullptr\n",
        1,
    ),
    (
        "        fr13_run_mha_fwd_fixed32_qrow32_gqa_pair_b1(params, stream);",
        "        fr13_run_mha_fwd_fixed32_qrow32_gqa_pair_splitk_b1("
        "params, stream);",
        1,
    ),
    (
        '"FR13 qrow32 GQA-pair gate reached non-canonical B1 geometry"',
        '"FR14 qrow32 GQA-pair split-K gate reached non-canonical B1 geometry"',
        1,
    ),
)


def _fixed32_query_gqa_pair32_splitk_b1_api_gate() -> str:
    gate = FIXED32_QUERY_GQA_PAIR32_B1_API_GATE
    for anchor, replacement, expected in (
        _FIXED32_QUERY_GQA_PAIR32_SPLITK_B1_API_GATE_SUBSTITUTIONS
    ):
        if gate.count(anchor) != expected:
            raise RuntimeError(
                "B1 GQA-pair API gate drifted at the split-K derivation anchor "
                f"{anchor!r}: expected {expected}, found {gate.count(anchor)}"
            )
        gate = gate.replace(anchor, replacement)
    for required in (
        "            && params.b == 1\n",
        "            && params.total_q == 32\n",
        "            && params.seqlen_q == 32\n",
        "            && params.unpadded_lse\n",
        "            && params.is_seqlens_k_cumulative\n",
        "            && params.page_block_size == 1024\n",
        "            && params.num_splits == "
        f"{FIXED32_QUERY_GQA_PAIR32_SPLITK_B1_CONTEXT_SPLITS}\n",
        "            && params.oaccum_ptr != nullptr\n",
        "            && params.softmax_lseaccum_ptr != nullptr\n",
    ):
        if gate.count(required) != 1:
            raise RuntimeError(
                f"B1 GQA-pair split-K API gate lost a required operand: "
                f"{required!r}"
            )
    for forbidden in (
        "params.num_splits == 0",
        "kFr13Qrow32GqaPairB1BatchStrideSentinel",
    ):
        if forbidden in gate:
            raise RuntimeError(
                "B1 GQA-pair split-K API gate still carries the promoted arm's "
                f"contract: {forbidden!r}"
            )
    return gate


FIXED32_QUERY_GQA_PAIR32_SPLITK_B1_API_GATE = (
    _fixed32_query_gqa_pair32_splitk_b1_api_gate()
)


# Same shape as the qualified split2 scratch patch: the private fixed32 route
# is not `seqlenq_ngroups_swapped`, so stock would never allocate the split
# accumulators for it. num_splits is pinned to the launcher's compile-time
# kContextSplits here as well, so a mismatch fails in the API before a kernel
# is reached rather than reading uninitialised accumulators.
FIXED32_QUERY_GQA_PAIR32_SPLITK_B1_ALLOCATION = rf'''    const bool fr14_qrow32_gqa_pair_splitk_b1 =
        params.tree_bias_batch_stride ==
        kFr14Qrow32GqaPairSplitKB1BatchStrideSentinel;
    TORCH_CHECK(
        !fr14_qrow32_gqa_pair_splitk_b1
        || num_splits == {FIXED32_QUERY_GQA_PAIR32_SPLITK_B1_CONTEXT_SPLITS},
        "FR14 B1 qrow32 GQA-pair split-K scratch setup requires num_splits="
        "{FIXED32_QUERY_GQA_PAIR32_SPLITK_B1_CONTEXT_SPLITS}");
    if (seqlenq_ngroups_swapped || fr14_qrow32_gqa_pair_splitk_b1) {{
        // Stock applies split-K only to decoding. The private fixed32 route
        // also needs the stock-owned accumulation buffers for qlen 32.
'''


FIXED32_QUERY_TILE32_B1_API_DECLARATION = rf'''constexpr int64_t kFr13Qrow32B1BatchStrideSentinel =
    {FIXED32_QUERY_TILE32_B1_BATCH_STRIDE_SENTINEL};

__attribute__((visibility("hidden")))
void fr13_run_mha_fwd_fixed32_qrow32_b1(
    Flash_fwd_params &params, cudaStream_t stream);

'''


FIXED32_QUERY_TILE32_B1_API_GATE = r'''    if (params.tree_bias_batch_stride == kFr13Qrow32B1BatchStrideSentinel) {
        // The hidden launcher revalidates every canonical field before launch.
        fr13_run_mha_fwd_fixed32_qrow32_b1(params, stream);
        return;
    }
'''


FIXED32_QUERY_TILE32_B1_SPLIT2_API_DECLARATION = rf'''constexpr int64_t kFr13Qrow32B1Split2BatchStrideSentinel =
    {FIXED32_QUERY_TILE32_B1_SPLIT2_BATCH_STRIDE_SENTINEL};

__attribute__((visibility("hidden")))
void fr13_run_mha_fwd_fixed32_qrow32_b1_split2(
    Flash_fwd_params &params, cudaStream_t stream);

'''


FIXED32_QUERY_TILE32_B1_SPLIT2_API_GATE = r'''    if (params.tree_bias_batch_stride == kFr13Qrow32B1Split2BatchStrideSentinel) {
        // The hidden launcher revalidates geometry and stock split scratch.
        fr13_run_mha_fwd_fixed32_qrow32_b1_split2(params, stream);
        return;
    }
'''


STOCK_VARLEN_SPLITKV_ALLOCATION = r'''    if (seqlenq_ngroups_swapped) {
        // Only apply split-k for decoding
'''


FIXED32_QUERY_TILE32_B1_SPLIT2_ALLOCATION = r'''    const bool fr13_qrow32_b1_split2 =
        params.tree_bias_batch_stride ==
        kFr13Qrow32B1Split2BatchStrideSentinel;
    TORCH_CHECK(
        !fr13_qrow32_b1_split2 || num_splits == 2,
        "FR13 B1 qrow32 split2 scratch setup requires num_splits=2");
    if (seqlenq_ngroups_swapped || fr13_qrow32_b1_split2) {
        // Stock applies split-K only to decoding. The private fixed32 route
        // also needs the stock-owned accumulation buffers for qlen 32.
'''


FIXED32_QUERY_TILE16_API_DISPATCH = rf'''constexpr int64_t kFr13Qrow16BatchStrideSentinel =
    {FIXED32_QUERY_TILE16_BATCH_STRIDE_SENTINEL};

__attribute__((visibility("hidden")))
void fr13_run_mha_fwd_fixed32_qrow16(
    Flash_fwd_params &params, cudaStream_t stream);

void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel=false) {{
    if (params.tree_bias_batch_stride == kFr13Qrow16BatchStrideSentinel) {{
        TORCH_CHECK(
            params.tree_bias_ptr != nullptr
            && params.is_bf16
            && !params.is_causal
            && params.b == 1
            && params.d == 256
            && params.d_rounded == 256
            && params.h == 24
            && params.h_k == 4
            && params.h_h_k_ratio == 6
            && params.seqlen_q == 32
            && params.tree_bias_rows == 32
            && params.tree_bias_cols == 32
            && params.tree_bias_q_offset == 0
            && params.tree_bias_k_offset == 0
            && params.cu_seqlens_q != nullptr
            && params.seqused_k != nullptr
            && !params.seqlenq_ngroups_swapped
            && params.block_table != nullptr
            && params.page_block_size == 1024
            && params.window_size_left < 0
            && params.window_size_right < 0
            && params.alibi_slopes_ptr == nullptr
            && params.knew_ptr == nullptr
            && params.softcap == 0.0f
            // set_params_fprop zero-initializes this field. Varlen q=32 does
            // not enter the max_seqlen_q==1 q-group split-K setup.
            && params.num_splits == 0
            && force_split_kernel,
            "FR13 qrow16 internal dispatch reached non-production geometry");
        fr13_run_mha_fwd_fixed32_qrow16(params, stream);
        return;
    }}
    FP16_SWITCH(!params.is_bf16, [&] {{
        HEADDIM_SWITCH(params.d, [&] {{
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {{
                if (params.num_splits <= 1 && !force_split_kernel) {{  // If we don't set it num_splits == 0
                    run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
                }} else {{
                    run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
                }}
            }});
        }});
    }});
}}
'''


_FIXED32_QUERY_TILE16_STATIC_STRIDES_API_ANCHOR = r'''            && params.page_block_size == 1024
'''


FIXED32_QUERY_TILE16_STATIC_STRIDES_API_DISPATCH = (
    FIXED32_QUERY_TILE16_API_DISPATCH.replace(
        _FIXED32_QUERY_TILE16_STATIC_STRIDES_API_ANCHOR,
        r'''            && params.k_batch_stride == 1024 * 4 * 256
            && params.k_row_stride == 4 * 256
            && params.k_head_stride == 256
            && params.v_batch_stride == 1024 * 4 * 256
            && params.v_row_stride == 4 * 256
            && params.v_head_stride == 256
'''
        + _FIXED32_QUERY_TILE16_STATIC_STRIDES_API_ANCHOR,
        1,
    )
)


_FIXED32_QUERY_TILE16_B1_REFERENCE_EXACT_GEOMETRY = (
    (
        r'''            && params.seqlen_q == 32
''',
        r'''            && params.total_q == 32
            && params.seqlen_q == 32
            && params.seqlen_q_rounded == 128
            && params.q_head_stride == 256
''',
    ),
    (
        r'''            && params.tree_bias_cols == 32
''',
        r'''            && params.o_head_stride == 256
            && params.tree_bias_cols == 32
            && params.tree_bias_row_stride == 32
            && params.tree_bias_col_stride == 1
''',
    ),
    (
        r'''            && !params.seqlenq_ngroups_swapped
''',
        r'''            && !params.seqlenq_ngroups_swapped
            && params.leftpad_k == nullptr
            && params.cache_batch_idx == nullptr
''',
    ),
    (
        r'''            && params.block_table != nullptr
''',
        r'''            && params.block_table != nullptr
            && params.block_table_batch_stride > 0
''',
    ),
    (
        r'''            && params.knew_ptr == nullptr
''',
        r'''            && params.knew_ptr == nullptr
            && params.vnew_ptr == nullptr
            && params.p_ptr == nullptr
            && params.softmax_lse_ptr != nullptr
            && params.p_dropout == 1.0f
''',
    ),
    (
        "FR13 qrow16 internal dispatch reached non-production geometry",
        "FR13 qrow16 reference dispatch reached non-canonical B1 geometry",
    ),
)


def _fixed32_query_tile16_b1_reference_api_dispatch() -> str:
    # Gate A uses interleaved K/V storage. The retained qrow16 object consumes
    # runtime strides, so keep the static contract's row/head checks while
    # pinning its page stride to the actual B1 view.
    dispatch = FIXED32_QUERY_TILE16_STATIC_STRIDES_API_DISPATCH.replace(
        "params.k_batch_stride == 1024 * 4 * 256",
        "params.k_batch_stride == 2 * 1024 * 4 * 256",
        1,
    ).replace(
        "params.v_batch_stride == 1024 * 4 * 256",
        "params.v_batch_stride == 2 * 1024 * 4 * 256",
        1,
    )
    for anchor, replacement in _FIXED32_QUERY_TILE16_B1_REFERENCE_EXACT_GEOMETRY:
        if dispatch.count(anchor) != 1:
            raise RuntimeError(
                "qrow16 B1 reference API geometry anchor is not unique: "
                f"{anchor!r}"
            )
        dispatch = dispatch.replace(anchor, replacement, 1)
    return dispatch


FIXED32_QUERY_TILE16_B1_REFERENCE_API_DISPATCH = (
    _fixed32_query_tile16_b1_reference_api_dispatch()
)


STOCK_RUN_MHA_FWD = r'''void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel=false) {
    FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(params.d, [&] {
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                if (params.num_splits <= 1 && !force_split_kernel) {  // If we don't set it num_splits == 0
                    run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
                } else {
                    run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
                }
            });
        });
    });
}'''


def _replace_once(text: str, old: str, new: str, label: str) -> tuple[str, bool]:
    if new in text:
        return text, False
    if old not in text:
        raise RuntimeError(f"anchor not found for {label}")
    return text.replace(old, new, 1), True


def _insert_once(text: str, marker: str, insert: str, label: str) -> tuple[str, bool]:
    if insert.strip() in text:
        return text, False
    if marker not in text:
        raise RuntimeError(f"marker not found for {label}")
    return text.replace(marker, insert + marker, 1), True


def _install_hidden_api_gate(
    text: str,
    *,
    declaration: str,
    gate: str,
    label: str,
) -> tuple[str, bool]:
    """Install a private dispatch without replacing the stock function body."""
    changed = False
    text, did = _insert_once(
        text,
        RUN_MHA_FWD_SIGNATURE,
        declaration,
        f"{label} declaration",
    )
    changed = changed or did
    if gate.strip() not in text:
        if RUN_MHA_FWD_SIGNATURE not in text:
            raise RuntimeError(f"anchor not found for {label} body")
        text = text.replace(
            RUN_MHA_FWD_SIGNATURE,
            RUN_MHA_FWD_SIGNATURE + gate,
            1,
        )
        changed = True
    return text, changed


def _install_qrow16_api_dispatch(
    text: str,
    dispatch: str,
    *,
    label: str,
) -> tuple[str, bool]:
    signature_at = dispatch.index(RUN_MHA_FWD_SIGNATURE)
    stock_body_at = dispatch.index("    FP16_SWITCH", signature_at)
    return _install_hidden_api_gate(
        text,
        declaration=dispatch[:signature_at],
        gate=dispatch[
            signature_at + len(RUN_MHA_FWD_SIGNATURE) : stock_body_at
        ],
        label=label,
    )


def _patch_flash_h(path: Path) -> bool:
    text = path.read_text()
    insert = """    // FR13_FA2_TREE_BIAS: dense [q_suffix, q_suffix] or [batch, q_suffix, q_suffix] fp32 bias.\n    void * __restrict__ tree_bias_ptr;\n    index_t tree_bias_batch_stride;\n    index_t tree_bias_row_stride;\n    index_t tree_bias_col_stride;\n    int tree_bias_rows;\n    int tree_bias_cols;\n    int tree_bias_q_offset;\n    int tree_bias_k_offset;\n\n"""
    if "int tree_bias_cols;\n" in text and "int tree_bias_q_offset;\n" not in text:
        text = text.replace(
            "    int tree_bias_cols;\n",
            "    int tree_bias_cols;\n    int tree_bias_q_offset;\n    int tree_bias_k_offset;\n",
            1,
        )
        path.write_text(text)
        return True
    text, changed = _insert_once(
        text,
        "    void * __restrict__ alibi_slopes_ptr;\n",
        insert,
        "Flash_fwd_params tree bias fields",
    )
    if changed:
        path.write_text(text)
    return changed


def _patch_flash_fwd_kernel(
    path: Path,
    *,
    tile_earlyout: bool = False,
    fixed32_tree_visibility_mask: bool = False,
) -> bool:
    text = path.read_text()
    changed = False
    tree_bias_helper = _tree_bias_helper(
        tile_earlyout,
        fixed32_tree_visibility_mask=fixed32_tree_visibility_mask,
    )
    helper_marker = "// FR13_FA2_TREE_BIAS: add a dense query-suffix ancestry bias after QK."
    helper_end_marker = "template<typename Kernel_traits, bool Is_dropout"
    if helper_marker in text:
        start = text.index(helper_marker)
        end = text.index(helper_end_marker, start)
        if text[start:end] != tree_bias_helper.lstrip():
            text = text[:start] + tree_bias_helper.lstrip() + text[end:]
            changed = True
    else:
        text, did = _insert_once(
            text,
            helper_end_marker,
            tree_bias_helper,
            "tree bias helper",
        )
        changed = changed or did
    anchors = [
        (
            """        if constexpr (Is_softcap){\n            FLASH_NAMESPACE::apply_softcap(acc_s, params.softcap);\n        }\n\n        mask.template apply_mask<Is_causal, Is_even_MN>(\n""",
            """        if constexpr (Is_softcap){\n            FLASH_NAMESPACE::apply_softcap(acc_s, params.softcap);\n        }\n        FLASH_NAMESPACE::apply_tree_bias<Kernel_traits>(\n            acc_s, params, binfo, bidb, n_block,\n            m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4,\n            kNWarps * 16\n        );\n\n        mask.template apply_mask<Is_causal, Is_even_MN>(\n""",
            "standard masking-loop tree bias",
        ),
        (
            """        if constexpr (Is_softcap){\n            FLASH_NAMESPACE::apply_softcap(acc_s, params.softcap);\n        }\n\n        FLASH_NAMESPACE::cp_async_wait<0>();\n""",
            """        if constexpr (Is_softcap){\n            FLASH_NAMESPACE::apply_softcap(acc_s, params.softcap);\n        }\n        FLASH_NAMESPACE::apply_tree_bias<Kernel_traits>(\n            acc_s, params, binfo, bidb, n_block,\n            m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4,\n            kNWarps * 16\n        );\n\n        FLASH_NAMESPACE::cp_async_wait<0>();\n""",
            "standard unmasked-loop tree bias",
        ),
        (
            """        if constexpr (Is_softcap){\n            FLASH_NAMESPACE::apply_softcap(acc_s, params.softcap);\n        }\n\n\n        mask.template apply_mask<Is_causal, Is_even_MN>(\n""",
            """        if constexpr (Is_softcap){\n            FLASH_NAMESPACE::apply_softcap(acc_s, params.softcap);\n        }\n        FLASH_NAMESPACE::apply_tree_bias<Kernel_traits>(\n            acc_s, params, binfo, bidb, n_block,\n            m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4,\n            kNWarps * 16\n        );\n\n\n        mask.template apply_mask<Is_causal, Is_even_MN>(\n""",
            "split masking-loop tree bias",
        ),
        (
            """        if constexpr (Is_softcap){\n            FLASH_NAMESPACE::apply_softcap(acc_s, params.softcap);\n        }\n\n        FLASH_NAMESPACE::cp_async_wait<0>();\n""",
            """        if constexpr (Is_softcap){\n            FLASH_NAMESPACE::apply_softcap(acc_s, params.softcap);\n        }\n        FLASH_NAMESPACE::apply_tree_bias<Kernel_traits>(\n            acc_s, params, binfo, bidb, n_block,\n            m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4,\n            kNWarps * 16\n        );\n\n        FLASH_NAMESPACE::cp_async_wait<0>();\n""",
            "split unmasked-loop tree bias",
        ),
    ]
    for old, new, label in anchors:
        # Two anchors intentionally share text; keep replacing until the target
        # insertion count reaches the four forward score loops.
        if text.count("FR13_FA2_TREE_BIAS") >= 5 and "split" in label:
            continue
        if old in text:
            text = text.replace(old, new, 1)
            changed = True
    if text.count("apply_tree_bias<Kernel_traits>") < 4:
        raise RuntimeError(
            "expected tree bias calls in all four FA2 forward loops, found "
            f"{text.count('apply_tree_bias<Kernel_traits>')}"
        )
    if changed:
        path.write_text(text)
    return changed


def _patch_fixed32_query_static_page(
    path: Path,
    *,
    fixed32_query_tile16: bool = False,
    fixed32_query_tile32: bool = False,
) -> bool:
    if not (fixed32_query_tile16 or fixed32_query_tile32):
        return False
    text = path.read_text()
    changed = False
    helper_anchor = (
        "template <typename Kernel_traits>\n"
        "__forceinline__ __device__\n"
        "int64_t resolve_thread_kv_page_slice_offset(\n"
    )
    trait_marker = "// FR13_FA2_FIXED32_STATIC_PAGE: stock traits retain the dynamic page size."
    if trait_marker in text:
        if text.count(trait_marker) != 1:
            raise RuntimeError("fixed32 static paged-KV trait drifted")
        if FIXED32_QUERY_STATIC_PAGE_TRAIT not in text:
            if text.count(FIXED32_QUERY_STATIC_PAGE_TRAIT_LEGACY) != 1:
                raise RuntimeError("fixed32 static paged-KV trait drifted")
            text = text.replace(
                FIXED32_QUERY_STATIC_PAGE_TRAIT_LEGACY,
                FIXED32_QUERY_STATIC_PAGE_TRAIT,
                1,
            )
            changed = True
    else:
        text, did = _insert_once(
            text,
            helper_anchor,
            FIXED32_QUERY_STATIC_PAGE_TRAIT,
            "fixed32 static paged-KV trait",
        )
        changed = changed or did

    dynamic_offset = r'''    const int64_t global_row_offset = block_row_offset + n_block * kBlockN;
    const int64_t page_offset = global_row_offset % page_block_size;
    const int64_t virtual_page_idx = global_row_offset / page_block_size;

    return ((int64_t) block_table[virtual_page_idx]) * ((int64_t) page_stride)
        + page_offset * ((int64_t) row_stride)
        + col_offset;
'''
    if FIXED32_QUERY_STATIC_PAGE_OFFSET in text:
        if text.count(FIXED32_QUERY_STATIC_PAGE_OFFSET) != 1:
            raise RuntimeError("fixed32 static paged-KV offset was duplicated")
    elif FIXED32_QUERY_STATIC_PAGE_OFFSET_LEGACY in text:
        if text.count(FIXED32_QUERY_STATIC_PAGE_OFFSET_LEGACY) != 1:
            raise RuntimeError("fixed32 static paged-KV legacy offset was duplicated")
        text = text.replace(
            FIXED32_QUERY_STATIC_PAGE_OFFSET_LEGACY,
            FIXED32_QUERY_STATIC_PAGE_OFFSET,
            1,
        )
        changed = True
    else:
        text, did = _replace_once(
            text,
            dynamic_offset,
            FIXED32_QUERY_STATIC_PAGE_OFFSET,
            "fixed32 static paged-KV offset",
        )
        changed = changed or did
    if changed:
        path.write_text(text)
    return changed


def _patch_fixed32_query_static_paged_path(
    path: Path,
    *,
    fixed32_query_tile16: bool = False,
    fixed32_query_tile32: bool = False,
) -> bool:
    if not (fixed32_query_tile16 or fixed32_query_tile32):
        return False
    text = path.read_text()
    signature = (
        "template<typename Kernel_traits, bool Is_causal, bool Is_local, "
        "bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, "
        "bool Split, bool Append_KV, typename Params>\n"
        "inline __device__ void compute_attn_1rowblock_splitkv"
    )
    function_start = text.index(signature)
    function_end = text.index(
        "\n////////////////////////////////////////////////////////////////////////////////////////////////////\n\n"
        "template<typename Kernel_traits, bool Is_dropout",
        function_start,
    )
    function = text[function_start:function_end]
    function = "\n".join(line.rstrip() for line in function.split("\n"))
    marker = "FR13_FA2_QROW16_STATIC_PAGED_PATH"
    if marker in function:
        required_counts = {
            marker: 1,
            "if constexpr (kStaticPagedKV)": 7,
            "kStaticPagedKV || block_table != nullptr": 2,
            "} else if (block_table == nullptr) {": 5,
            "} else if (block_table != nullptr) {": 1,
            "if (block_table == nullptr) { return; }": 1,
        }
        for snippet, expected in required_counts.items():
            if function.count(snippet) != expected:
                raise RuntimeError("fixed32 qrow16 static paged path drifted")
        return False

    for old, new, label, expected in FIXED32_QUERY_STATIC_PAGED_PATH_REPLACEMENTS:
        if function.count(old) != expected:
            raise RuntimeError(
                f"{label} anchor count drifted: expected {expected}, "
                f"found {function.count(old)}"
            )
        function = function.replace(old, new)
    text = text[:function_start] + function + text[function_end:]
    path.write_text(text)
    return True


def _patch_fixed32_query_tile16_static_kv_head_stride(
    path: Path,
    *,
    fixed32_query_tile16_static_strides: bool = False,
) -> bool:
    if not fixed32_query_tile16_static_strides:
        return False
    text = path.read_text()
    function_signature = (
        "template<typename Kernel_traits, bool Is_causal, bool Is_local, "
        "bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, "
        "bool Split, bool Append_KV, typename Params>\n"
        "inline __device__ void compute_attn_1rowblock_splitkv"
    )
    function_start = text.index(function_signature)
    function_end = text.index(
        "\n////////////////////////////////////////////////////////////////////////////////////////////////////\n\n"
        "template<typename Kernel_traits, bool Is_dropout",
        function_start,
    )
    function = text[function_start:function_end]
    marker = "// FR13_FA2_QROW16_STATIC_KV_HEAD_STRIDE"
    if marker in function:
        required = (
            "StaticPagedKVStrides<Kernel_traits>::head",
            "static_assert(kStaticKVHeadStride == 0 || kStaticKVHeadStride == 256);",
            "kStaticKVHeadStride != 0",
            "* kStaticKVHeadStride",
        )
        if function.count(marker) != 1 or any(
            item not in function for item in required
        ):
            raise RuntimeError("qrow16 static KV head stride drifted")
        return False

    dynamic_base = r'''    const index_t row_offset_k = kStaticPagedKV || block_table != nullptr
        ? (bidh / params.h_h_k_ratio) * params.k_head_stride
        : binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = kStaticPagedKV || block_table != nullptr
        ? (bidh / params.h_h_k_ratio) * params.v_head_stride
        : binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
'''
    static_base = r'''    // FR13_FA2_QROW16_STATIC_KV_HEAD_STRIDE
    constexpr int64_t kStaticKVHeadStride =
        StaticPagedKVStrides<Kernel_traits>::head;
    static_assert(kStaticKVHeadStride == 0 || kStaticKVHeadStride == 256);
    const index_t row_offset_k = kStaticKVHeadStride != 0
        ? (bidh / params.h_h_k_ratio) * kStaticKVHeadStride
        : kStaticPagedKV || block_table != nullptr
            ? (bidh / params.h_h_k_ratio) * params.k_head_stride
            : binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb_cache)
              + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = kStaticKVHeadStride != 0
        ? (bidh / params.h_h_k_ratio) * kStaticKVHeadStride
        : kStaticPagedKV || block_table != nullptr
            ? (bidh / params.h_h_k_ratio) * params.v_head_stride
            : binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb_cache)
              + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
'''
    if function.count(dynamic_base) != 1:
        raise RuntimeError("qrow16 static KV head-stride anchor drifted")
    function = function.replace(dynamic_base, static_base, 1)
    text = text[:function_start] + function + text[function_end:]
    path.write_text(text)
    return True


def _patch_fixed32_query_tile32_static_query(
    path: Path,
    *,
    fixed32_query_tile32: bool = False,
) -> bool:
    if not fixed32_query_tile32:
        return False
    text = path.read_text()
    signature = (
        "template<typename Kernel_traits, bool Is_causal, bool Is_local, "
        "bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, "
        "bool Split, bool Append_KV, typename Params>\n"
        "inline __device__ void compute_attn_1rowblock_splitkv"
    )
    function_start = text.index(signature)
    function_end = text.index(
        "\n////////////////////////////////////////////////////////////////////////////////////////////////////\n\n"
        "template<typename Kernel_traits, bool Is_dropout",
        function_start,
    )
    function = text[function_start:function_end]
    marker = "// FR13_FA2_QROW32_STATIC_QUERY: exact one-tile varlen query extent."
    if marker in function:
        required = (
            "kStaticQueryRows * kStaticHeadGroupSize == kBlockM;",
            "const int query_m_block = kStaticQueryTile ? 0 : m_block;",
            "FLASH_NAMESPACE::copy<kStaticQueryTile || Is_even_MN, Is_even_K>",
            "mask.template apply_mask<Is_causal, Is_even_MN>",
        )
        if function.count(marker) != 1 or any(item not in function for item in required):
            raise RuntimeError("qrow32 static query specialization drifted")
        return False

    function_prefix, function_body = function.split("{\n", 1)
    if function_body.count("binfo.actual_seqlen_q") != 12:
        raise RuntimeError("split-KV query-length use count drifted")
    function_body = function_body.replace(
        "binfo.actual_seqlen_q",
        "actual_seqlen_q",
    )
    function_body = function_body.replace("m_block", "query_m_block")
    binfo_anchor = "    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);\n"
    static_query = r'''    // FR13_FA2_QROW32_STATIC_QUERY: exact one-tile varlen query extent.
    constexpr int kStaticQueryRows = StaticQueryRows<Kernel_traits>::value;
    constexpr int kStaticHeadGroupSize =
        StaticQueryHeadsPerCTA<Kernel_traits>::value;
    constexpr bool kStaticQueryTile =
        kStaticQueryRows * kStaticHeadGroupSize == kBlockM;
    static_assert(kStaticHeadGroupSize == 1 || kStaticHeadGroupSize == 2);
    static_assert(kStaticQueryRows == 0 || kStaticQueryTile);
    const int actual_seqlen_q =
        kStaticQueryTile ? kStaticQueryRows : binfo.actual_seqlen_q;
    const int query_m_block = kStaticQueryTile ? 0 : m_block;
'''
    if function_body.count(binfo_anchor) != 1:
        raise RuntimeError("split-KV BlockInfo anchor drifted")
    function_body = function_body.replace(
        binfo_anchor,
        binfo_anchor + static_query,
        1,
    )
    early_exit = "    if (query_m_block * kBlockM >= actual_seqlen_q) return;\n"
    static_early_exit = r'''    if constexpr (!kStaticQueryTile) {
        if (query_m_block * kBlockM >= actual_seqlen_q) return;
    }
'''
    if function_body.count(early_exit) != 1:
        raise RuntimeError("split-KV query early-exit anchor drifted")
    function_body = function_body.replace(early_exit, static_early_exit, 1)

    output_copy = (
        "FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, "
        "/*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>("
    )
    if function_body.count(output_copy) != 2:
        raise RuntimeError("split-KV output predicate count drifted")
    function_body = function_body.replace(
        output_copy,
        "FLASH_NAMESPACE::copy<kStaticQueryTile || Is_even_MN, Is_even_K, "
        "/*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(",
    )
    query_copy = (
        "FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K>"
        "(gmem_tiled_copy_Q,"
    )
    if function_body.count(query_copy) != 1:
        raise RuntimeError("split-KV query predicate anchor drifted")
    function_body = function_body.replace(
        query_copy,
        "FLASH_NAMESPACE::copy<kStaticQueryTile || Is_even_MN, Is_even_K>"
        "(gmem_tiled_copy_Q,",
        1,
    )
    early_lse = (
        "if (row < actual_seqlen_q - query_m_block * kBlockM "
        "&& get<1>(tOcO(0, m, 0)) == 0)"
    )
    if function_body.count(early_lse) != 1:
        raise RuntimeError("split-KV empty-output LSE predicate drifted")
    function_body = function_body.replace(
        early_lse,
        "if ((kStaticQueryTile "
        "|| row < actual_seqlen_q - query_m_block * kBlockM) "
        "&& get<1>(tOcO(0, m, 0)) == 0)",
        1,
    )
    epilogue_lse = (
        "if (row < actual_seqlen_q - query_m_block * kBlockM) "
        "{ gLSEaccum(row) = lse(mi); }"
    )
    if function_body.count(epilogue_lse) != 1:
        raise RuntimeError("split-KV epilogue LSE predicate drifted")
    function_body = function_body.replace(
        epilogue_lse,
        "if (kStaticQueryTile "
        "|| row < actual_seqlen_q - query_m_block * kBlockM) "
        "{ gLSEaccum(row) = lse(mi); }",
        1,
    )
    function = function_prefix + "{\n" + function_body
    text = text[:function_start] + function + text[function_end:]
    path.write_text(text)
    return True


def _patch_fixed32_query_tile32_static_batch_layout(
    path: Path,
    *,
    fixed32_query_tile32: bool = False,
) -> bool:
    if not fixed32_query_tile32:
        return False
    text = path.read_text()
    function_signature = (
        "template<typename Kernel_traits, bool Is_causal, bool Is_local, "
        "bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, "
        "bool Split, bool Append_KV, typename Params>\n"
        "inline __device__ void compute_attn_1rowblock_splitkv"
    )
    function_start = text.index(function_signature)
    function_end = text.index(
        "\n////////////////////////////////////////////////////////////////////////////////////////////////////\n\n"
        "template<typename Kernel_traits, bool Is_dropout",
        function_start,
    )
    function = text[function_start:function_end]
    marker = "// FR13_FA2_QROW32_STATIC_BATCH_LAYOUT: fixed exact4 CTA coordinates."
    wrapper_marker = (
        "// FR13_FA2_QROW32_STATIC_BATCH_GRID: blockIdx.x selects the GQA lane."
    )
    if marker in function:
        pair_layout_marker = (
            "// FR13_FA2_QROW32_GQA_PAIR_LAYOUT: two heads share the M tile."
        )
        query_offset_count = 3 if pair_layout_marker in function else 4
        required_counts = {
            marker: 1,
            "static_query_offset<Kernel_traits>(binfo, ": query_offset_count,
            "bidh / params.h_h_k_ratio": 1,
            "int bidh_k;": 1,
        }
        for snippet, expected in required_counts.items():
            if function.count(snippet) != expected:
                raise RuntimeError("qrow32 static batch layout drifted")
        if text.count(wrapper_marker) != 1:
            raise RuntimeError("qrow32 static batch grid drifted")
        return False

    query_offset = "binfo.q_offset("
    head_division = "bidh / params.h_h_k_ratio"
    if function.count(query_offset) != 4:
        raise RuntimeError(
            "split-KV query-offset use count drifted: expected 4, found "
            f"{function.count(query_offset)}"
        )
    if function.count(head_division) != 6:
        raise RuntimeError(
            "split-KV GQA-head division count drifted: expected 6, found "
            f"{function.count(head_division)}"
        )
    function = function.replace(
        query_offset,
        "static_query_offset<Kernel_traits>(binfo, ",
    )
    function = function.replace(head_division, "bidh_k")

    warp_anchor = "    constexpr int kNWarps = Kernel_traits::kNWarps;\n"
    static_layout = r'''    // FR13_FA2_QROW32_STATIC_BATCH_LAYOUT: fixed exact4 CTA coordinates.
    constexpr int kStaticSequences =
        StaticQueryBatchLayout<Kernel_traits>::sequences;
    constexpr int kStaticQueryHeads =
        StaticQueryBatchLayout<Kernel_traits>::query_heads;
    constexpr int kStaticKVHeads =
        StaticQueryBatchLayout<Kernel_traits>::kv_heads;
    constexpr int kStaticQueryHeadsPerKV =
        StaticQueryBatchLayout<Kernel_traits>::query_heads_per_kv;
    constexpr int kStaticQueryHeadsPerCTA =
        StaticQueryHeadsPerCTA<Kernel_traits>::value;
    constexpr bool kStaticQueryBatch = kStaticSequences != 0;
    static_assert(
        !kStaticQueryBatch || kStaticSequences == 1 || kStaticSequences == 4);
    static_assert(!kStaticQueryBatch || kStaticQueryHeads == 24);
    static_assert(!kStaticQueryBatch || kStaticKVHeads == 4);
    static_assert(!kStaticQueryBatch || kStaticQueryHeadsPerKV == 6);
    static_assert(
        !kStaticQueryBatch
        || (kStaticQueryHeadsPerCTA == 1 || kStaticQueryHeadsPerCTA == 2));
    static_assert(
        !kStaticQueryBatch
        || kStaticQueryHeadsPerKV % kStaticQueryHeadsPerCTA == 0);
    static_assert(
        !kStaticQueryBatch
        || kStaticQueryHeads == kStaticKVHeads * kStaticQueryHeadsPerKV);
    int bidh_k;
    if constexpr (kStaticQueryBatch) {
        bidh_k = static_cast<int>(blockIdx.z);
    } else {
        bidh_k = bidh / params.h_h_k_ratio;
    }
'''
    if function.count(warp_anchor) != 1:
        raise RuntimeError("split-KV warp trait anchor drifted")
    function = function.replace(
        warp_anchor,
        warp_anchor + static_layout,
        1,
    )
    text = text[:function_start] + function + text[function_end:]

    dynamic_wrapper = r'''template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV, typename Params>
inline __device__ void compute_attn_splitkv(const Params &params) {
    const int m_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = Split ? blockIdx.z / params.h : blockIdx.y;
    // The block index for the head.
    const int bidh = Split ? blockIdx.z - bidb * params.h : blockIdx.z;
    const int n_split_idx = Split ? blockIdx.y : 0;
    const int num_n_splits = Split ? gridDim.y : 1;
    FLASH_NAMESPACE::compute_attn_1rowblock_splitkv<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Split, Append_KV>(params, bidb, bidh, m_block, n_split_idx, num_n_splits);
}'''
    static_wrapper = r'''template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV, typename Params>
inline __device__ void compute_attn_splitkv(const Params &params) {
    // FR13_FA2_QROW32_STATIC_BATCH_GRID: blockIdx.x selects the GQA lane.
    constexpr int kStaticSequences =
        StaticQueryBatchLayout<Kernel_traits>::sequences;
    constexpr int kStaticQueryHeads =
        StaticQueryBatchLayout<Kernel_traits>::query_heads;
    constexpr int kStaticKVHeads =
        StaticQueryBatchLayout<Kernel_traits>::kv_heads;
    constexpr int kStaticQueryHeadsPerKV =
        StaticQueryBatchLayout<Kernel_traits>::query_heads_per_kv;
    constexpr int kStaticQueryHeadsPerCTA =
        StaticQueryHeadsPerCTA<Kernel_traits>::value;
    constexpr bool kStaticQueryBatch = kStaticSequences != 0;
    static_assert(!kStaticQueryBatch || !Split || kStaticSequences == 1);
    static_assert(
        !kStaticQueryBatch || kStaticSequences == 1 || kStaticSequences == 4);
    static_assert(!kStaticQueryBatch || kStaticQueryHeads == 24);
    static_assert(!kStaticQueryBatch || kStaticKVHeads == 4);
    static_assert(!kStaticQueryBatch || kStaticQueryHeadsPerKV == 6);
    static_assert(
        !kStaticQueryBatch
        || (kStaticQueryHeadsPerCTA == 1 || kStaticQueryHeadsPerCTA == 2));
    static_assert(
        !kStaticQueryBatch
        || kStaticQueryHeadsPerKV % kStaticQueryHeadsPerCTA == 0);
    static_assert(
        !kStaticQueryBatch
        || kStaticQueryHeads == kStaticKVHeads * kStaticQueryHeadsPerKV);
    if constexpr (kStaticQueryBatch) {
        const int m_block = 0;
        const int bidb = Split ? 0 : blockIdx.y;
        const int bidh = blockIdx.z * kStaticQueryHeadsPerKV
            + blockIdx.x * kStaticQueryHeadsPerCTA;
        const int n_split_idx = Split ? blockIdx.y : 0;
        const int num_n_splits = Split ? gridDim.y : 1;
        FLASH_NAMESPACE::compute_attn_1rowblock_splitkv<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Split, Append_KV>(params, bidb, bidh, m_block, n_split_idx, num_n_splits);
    } else {
        const int m_block = blockIdx.x;
        // The block index for the batch.
        const int bidb = Split ? blockIdx.z / params.h : blockIdx.y;
        // The block index for the head.
        const int bidh = Split ? blockIdx.z - bidb * params.h : blockIdx.z;
        const int n_split_idx = Split ? blockIdx.y : 0;
        const int num_n_splits = Split ? gridDim.y : 1;
        FLASH_NAMESPACE::compute_attn_1rowblock_splitkv<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Split, Append_KV>(params, bidb, bidh, m_block, n_split_idx, num_n_splits);
    }
}'''
    if text.count(dynamic_wrapper) != 1:
        raise RuntimeError(
            "split-KV CTA-coordinate wrapper drifted: expected one stock body, "
            f"found {text.count(dynamic_wrapper)}"
        )
    text = text.replace(dynamic_wrapper, static_wrapper, 1)
    path.write_text(text)
    return True


def _patch_fixed32_query_tile32_static_paged_metadata(
    path: Path,
    *,
    fixed32_query_tile32: bool = False,
) -> bool:
    if not fixed32_query_tile32:
        return False
    text = path.read_text()
    function_signature = (
        "template<typename Kernel_traits, bool Is_causal, bool Is_local, "
        "bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, "
        "bool Split, bool Append_KV, typename Params>\n"
        "inline __device__ void compute_attn_1rowblock_splitkv"
    )
    function_start = text.index(function_signature)
    function_end = text.index(
        "\n////////////////////////////////////////////////////////////////////////////////////////////////////\n\n"
        "template<typename Kernel_traits, bool Is_dropout",
        function_start,
    )
    function = text[function_start:function_end]
    marker = (
        "// FR13_FA2_QROW32_STATIC_PAGED_METADATA: direct exact4 sequence metadata."
    )
    if marker in function:
        required = (
            "StaticPagedQueryBlockInfo<Kernel_traits>",
            "static_assert(!kStaticQueryBatch || kStaticPagedKV);",
            "static_assert(!kStaticQueryBatch || !Append_KV);",
            "block_table = params.block_table",
            "if constexpr (kStaticQueryBatch)",
        )
        if function.count(marker) != 1 or any(item not in function for item in required):
            raise RuntimeError("qrow32 static paged metadata specialization drifted")
        return False

    binfo_anchor = (
        "    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);\n"
    )
    static_binfo = r'''    // FR13_FA2_QROW32_STATIC_PAGED_METADATA: direct exact4 sequence metadata.
    static_assert(!kStaticQueryBatch || kStaticPagedKV);
    static_assert(!kStaticQueryBatch || !Split || kStaticSequences == 1);
    static_assert(!kStaticQueryBatch || !Append_KV);
    using QueryBlockInfo = std::conditional_t<
        kStaticQueryBatch,
        StaticPagedQueryBlockInfo<Kernel_traits>,
        BlockInfo</*Varlen=*/!Is_even_MN>>;
    const QueryBlockInfo binfo(params, bidb);
'''
    if function.count(binfo_anchor) != 1:
        raise RuntimeError("split-KV BlockInfo construction anchor drifted")
    function = function.replace(binfo_anchor, static_binfo, 1)

    dynamic_paged_base = r'''    const int bidb_cache = params.cache_batch_idx == nullptr ? bidb : params.cache_batch_idx[bidb];
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
'''
    static_paged_base = r'''    const int *block_table;
    index_t row_offset_k;
    index_t row_offset_v;
    if constexpr (kStaticQueryBatch) {
        // The gate requires a paged table and the varlen paged API forbids a
        // cache-batch remap, so form the only reachable row directly.
        block_table = params.block_table
            + bidb * params.block_table_batch_stride;
        row_offset_k = bidh_k * params.k_head_stride;
        row_offset_v = bidh_k * params.v_head_stride;
    } else {
        const int bidb_cache = params.cache_batch_idx == nullptr
            ? bidb : params.cache_batch_idx[bidb];
        block_table = params.block_table == nullptr
            ? nullptr
            : params.block_table + bidb * params.block_table_batch_stride;
        if constexpr (kStaticPagedKV) {
            if (block_table == nullptr) { return; }
        }
        row_offset_k = kStaticPagedKV || block_table != nullptr
            ? (bidh_k) * params.k_head_stride
            : binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb_cache)
              + (n_block_max - 1) * kBlockN * params.k_row_stride
              + (bidh_k) * params.k_head_stride;
        row_offset_v = kStaticPagedKV || block_table != nullptr
            ? (bidh_k) * params.v_head_stride
            : binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb_cache)
              + (n_block_max - 1) * kBlockN * params.v_row_stride
              + (bidh_k) * params.v_head_stride;
    }
'''
    if function.count(dynamic_paged_base) != 1:
        raise RuntimeError(
            "split-KV paged metadata/address anchor drifted: expected one, found "
            f"{function.count(dynamic_paged_base)}"
        )
    function = function.replace(dynamic_paged_base, static_paged_base, 1)
    text = text[:function_start] + function + text[function_end:]
    path.write_text(text)
    return True


def _patch_fixed32_query_tile32_static_kv_strides(
    path: Path,
    *,
    fixed32_query_tile32: bool = False,
) -> bool:
    if not fixed32_query_tile32:
        return False
    text = path.read_text()
    function_signature = (
        "template<typename Kernel_traits, bool Is_causal, bool Is_local, "
        "bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, "
        "bool Split, bool Append_KV, typename Params>\n"
        "inline __device__ void compute_attn_1rowblock_splitkv"
    )
    function_start = text.index(function_signature)
    function_end = text.index(
        "\n////////////////////////////////////////////////////////////////////////////////////////////////////\n\n"
        "template<typename Kernel_traits, bool Is_dropout",
        function_start,
    )
    function = text[function_start:function_end]
    marker = (
        "// FR13_FA2_QROW32_STATIC_KV_STRIDES: canonical contiguous page layout."
    )
    if marker in function:
        required = (
            "StaticPagedKVStrides<Kernel_traits>::head",
            "static_assert(kStaticKVHeadStride == 256);",
            "row_offset_k = bidh_k * kStaticKVHeadStride;",
            "row_offset_v = bidh_k * kStaticKVHeadStride;",
        )
        if function.count(marker) != 1 or any(item not in function for item in required):
            raise RuntimeError("qrow32 static KV stride specialization drifted")
        return False

    static_head_base = r'''        block_table = params.block_table
            + bidb * params.block_table_batch_stride;
        row_offset_k = bidh_k * params.k_head_stride;
        row_offset_v = bidh_k * params.v_head_stride;
'''
    fixed_head_base = r'''        // FR13_FA2_QROW32_STATIC_KV_STRIDES: canonical contiguous page layout.
        constexpr int64_t kStaticKVHeadStride =
            StaticPagedKVStrides<Kernel_traits>::head;
        static_assert(kStaticKVHeadStride == 256);
        block_table = params.block_table
            + bidb * params.block_table_batch_stride;
        row_offset_k = bidh_k * kStaticKVHeadStride;
        row_offset_v = bidh_k * kStaticKVHeadStride;
'''
    if function.count(static_head_base) != 1:
        raise RuntimeError(
            "split-KV static KV head-stride anchor drifted: expected one, found "
            f"{function.count(static_head_base)}"
        )
    function = function.replace(static_head_base, fixed_head_base, 1)
    text = text[:function_start] + function + text[function_end:]
    path.write_text(text)
    return True


def _patch_fixed32_query_tile32_fused_initial_kv_page(
    path: Path,
    *,
    fixed32_query_tile32: bool = False,
) -> bool:
    if not fixed32_query_tile32:
        return False
    text = path.read_text()
    function_signature = (
        "template<typename Kernel_traits, bool Is_causal, bool Is_local, "
        "bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, "
        "bool Split, bool Append_KV, typename Params>\n"
        "inline __device__ void compute_attn_1rowblock_splitkv"
    )
    function_start = text.index(function_signature)
    function_end = text.index(
        "\n////////////////////////////////////////////////////////////////////////////////////////////////////\n\n"
        "template<typename Kernel_traits, bool Is_dropout",
        function_start,
    )
    function = text[function_start:function_end]
    marker = (
        "// FR13_FA2_QROW32_FUSED_INITIAL_KV_PAGE: reuse the gated K/V page address."
    )
    if marker in function:
        required = (
            "StaticPagedKVStrides<Kernel_traits>::page",
            "StaticPagedKVStrides<Kernel_traits>::row",
            "const int64_t initial_kv_page_offset",
            "tKgK.data() = gK.data() + initial_kv_page_offset;",
            "tVgV.data() = gV.data() + initial_kv_page_offset;",
        )
        if function.count(marker) != 1 or any(item not in function for item in required):
            raise RuntimeError("qrow32 fused initial K/V page specialization drifted")
        return False

    prerequisite = (
        "// FR13_FA2_QROW32_STATIC_KV_STRIDES: canonical contiguous page layout."
    )
    if function.count(prerequisite) != 1:
        raise RuntimeError(
            "qrow32 fused initial K/V page requires the static KV stride specialization"
        )

    initial_page = r'''    if constexpr (kStaticPagedKV) {
        auto final_block_size = binfo.actual_seqlen_k - (n_block_max - 1) * kBlockN;
        tKgK.data() = gK.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block_max - 1, params.page_block_size,
            block_table, params.k_batch_stride, params.k_row_stride, final_block_size);
        tVgV.data() = gV.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block_max - 1, params.page_block_size,
            block_table, params.v_batch_stride, params.v_row_stride, final_block_size);
    } else if (block_table != nullptr) {
        auto final_block_size = binfo.actual_seqlen_k - (n_block_max - 1) * kBlockN;
        tKgK.data() = gK.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block_max - 1, params.page_block_size,
            block_table, params.k_batch_stride, params.k_row_stride, final_block_size);
        tVgV.data() = gV.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block_max - 1, params.page_block_size,
            block_table, params.v_batch_stride, params.v_row_stride, final_block_size);
    }
'''
    fused_initial_page = r'''    if constexpr (kStaticQueryBatch) {
        // FR13_FA2_QROW32_FUSED_INITIAL_KV_PAGE: reuse the gated K/V page address.
        constexpr int kStaticPageBlockSize =
            StaticPagedKVBlockSize<Kernel_traits>::value;
        constexpr int kStaticKVPageStride = static_cast<int>(
            StaticPagedKVStrides<Kernel_traits>::page);
        constexpr int kStaticKVRowStride = static_cast<int>(
            StaticPagedKVStrides<Kernel_traits>::row);
        static_assert(kStaticPageBlockSize == 1024);
        static_assert(
            kStaticKVPageStride
            == 2 * 1024 * 4 * 256);
        static_assert(kStaticKVRowStride == 4 * 256);
        auto final_block_size = binfo.actual_seqlen_k - (n_block_max - 1) * kBlockN;
        const int64_t initial_kv_page_offset =
            flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(
                tidx, n_block_max - 1, kStaticPageBlockSize, block_table,
                kStaticKVPageStride, kStaticKVRowStride, final_block_size);
        tKgK.data() = gK.data() + initial_kv_page_offset;
        tVgV.data() = gV.data() + initial_kv_page_offset;
    } else if constexpr (kStaticPagedKV) {
        auto final_block_size = binfo.actual_seqlen_k - (n_block_max - 1) * kBlockN;
        tKgK.data() = gK.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block_max - 1, params.page_block_size,
            block_table, params.k_batch_stride, params.k_row_stride, final_block_size);
        tVgV.data() = gV.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block_max - 1, params.page_block_size,
            block_table, params.v_batch_stride, params.v_row_stride, final_block_size);
    } else if (block_table != nullptr) {
        auto final_block_size = binfo.actual_seqlen_k - (n_block_max - 1) * kBlockN;
        tKgK.data() = gK.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block_max - 1, params.page_block_size,
            block_table, params.k_batch_stride, params.k_row_stride, final_block_size);
        tVgV.data() = gV.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block_max - 1, params.page_block_size,
            block_table, params.v_batch_stride, params.v_row_stride, final_block_size);
    }
'''
    if function.count(initial_page) != 1:
        raise RuntimeError(
            "split-KV initial K/V page anchor drifted: expected one, found "
            f"{function.count(initial_page)}"
        )
    function = function.replace(initial_page, fused_initial_page, 1)
    text = text[:function_start] + function + text[function_end:]
    path.write_text(text)
    return True


def _patch_fixed32_query_tile32_carry_kv_page_address(
    path: Path,
    *,
    fixed32_query_tile32: bool = False,
) -> bool:
    if not fixed32_query_tile32:
        return False
    text = path.read_text()
    function_signature = (
        "template<typename Kernel_traits, bool Is_causal, bool Is_local, "
        "bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, "
        "bool Split, bool Append_KV, typename Params>\n"
        "inline __device__ void compute_attn_1rowblock_splitkv"
    )
    function_start = text.index(function_signature)
    function_end = text.index(
        "\n////////////////////////////////////////////////////////////////////////////////////////////////////\n\n"
        "template<typename Kernel_traits, bool Is_dropout",
        function_start,
    )
    function = text[function_start:function_end]
    marker = (
        "// FR13_FA2_QROW32_CARRY_KV_PAGE: V reuses the preceding K advance."
    )
    if marker in function:
        required_counts = {
            marker: 1,
            "const int64_t next_kv_page_offset": 2,
            "tKgK.data() = gK.data() + next_kv_page_offset;": 2,
            "tVgV.data() = gV.data() + next_kv_page_offset;": 2,
            "static_assert(n_masking_steps == 1);": 3,
        }
        for snippet, expected in required_counts.items():
            if function.count(snippet) != expected:
                raise RuntimeError("qrow32 carried K/V page address drifted")
        return False

    prerequisites = (
        "// FR13_FA2_QROW32_STATIC_KV_STRIDES: canonical contiguous page layout.",
        "// FR13_FA2_QROW32_FUSED_INITIAL_KV_PAGE: reuse the gated K/V page address.",
    )
    if any(function.count(prerequisite) != 1 for prerequisite in prerequisites):
        raise RuntimeError(
            "qrow32 carried K/V page address requires fused static K/V addressing"
        )

    k_advance = r'''            if constexpr (kStaticPagedKV) {
                tKgK.data() = gK.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block - 1, params.page_block_size,
                    block_table, params.k_batch_stride, params.k_row_stride);
            } else if (block_table == nullptr) {
                tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
            } else {
                tKgK.data() = gK.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block - 1, params.page_block_size,
                    block_table, params.k_batch_stride, params.k_row_stride);
            }
'''
    carried_kv_advance = r'''            if constexpr (kStaticQueryBatch) {
                static_assert(!Is_causal && !Is_local);
                static_assert(n_masking_steps == 1);
                const int64_t next_kv_page_offset =
                    flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(
                        tidx, n_block - 1,
                        StaticPagedKVBlockSize<Kernel_traits>::value, block_table,
                        static_cast<int>(StaticPagedKVStrides<Kernel_traits>::page),
                        static_cast<int>(StaticPagedKVStrides<Kernel_traits>::row));
                tKgK.data() = gK.data() + next_kv_page_offset;
                tVgV.data() = gV.data() + next_kv_page_offset;
            } else if constexpr (kStaticPagedKV) {
                tKgK.data() = gK.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block - 1, params.page_block_size,
                    block_table, params.k_batch_stride, params.k_row_stride);
            } else if (block_table == nullptr) {
                tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
            } else {
                tKgK.data() = gK.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block - 1, params.page_block_size,
                    block_table, params.k_batch_stride, params.k_row_stride);
            }
'''
    if function.count(k_advance) != 2:
        raise RuntimeError(
            "split-KV K-advance anchors drifted: expected two, found "
            f"{function.count(k_advance)}"
        )
    function = function.replace(k_advance, carried_kv_advance)

    unmasked_v_advance = r'''        if constexpr (kStaticPagedKV) {
            tVgV.data() = gV.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block, params.page_block_size,
                block_table, params.v_batch_stride, params.v_row_stride);
        } else if (block_table == nullptr) {
            tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
        } else {
            tVgV.data() = gV.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block, params.page_block_size,
                block_table, params.v_batch_stride, params.v_row_stride);
        }
'''
    carried_v_advance = r'''        if constexpr (kStaticQueryBatch) {
            // FR13_FA2_QROW32_CARRY_KV_PAGE: V reuses the preceding K advance.
            static_assert(!Is_causal && !Is_local);
            static_assert(n_masking_steps == 1);
        } else if constexpr (kStaticPagedKV) {
            tVgV.data() = gV.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block, params.page_block_size,
                block_table, params.v_batch_stride, params.v_row_stride);
        } else if (block_table == nullptr) {
            tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
        } else {
            tVgV.data() = gV.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block, params.page_block_size,
                block_table, params.v_batch_stride, params.v_row_stride);
        }
'''
    if function.count(unmasked_v_advance) != 1:
        raise RuntimeError(
            "split-KV unmasked V-advance anchor drifted: expected one, found "
            f"{function.count(unmasked_v_advance)}"
        )
    function = function.replace(unmasked_v_advance, carried_v_advance, 1)
    text = text[:function_start] + function + text[function_end:]
    path.write_text(text)
    return True


def _patch_fixed32_query_gqa_pair_layout(
    path: Path,
    *,
    fixed32_query_gqa_pair32: bool = False,
) -> bool:
    if not fixed32_query_gqa_pair32:
        return False
    text = path.read_text()
    function_signature = (
        "template<typename Kernel_traits, bool Is_causal, bool Is_local, "
        "bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, "
        "bool Split, bool Append_KV, typename Params>\n"
        "inline __device__ void compute_attn_1rowblock_splitkv"
    )
    function_start = text.index(function_signature)
    function_end = text.index(
        "\n////////////////////////////////////////////////////////////////////////////////////////////////////\n\n"
        "template<typename Kernel_traits, bool Is_dropout",
        function_start,
    )
    function = text[function_start:function_end]
    marker = "// FR13_FA2_QROW32_GQA_PAIR_LAYOUT: two heads share the M tile."
    splitk_marker = (
        "// FR14_FA2_QROW32_GQA_PAIR_SPLITK_LAYOUT: the same paired"
    )
    if marker in function:
        required_counts = {
            marker: 3,
            "Int<kStaticQueryRows>{},": 5,
            "Int<kStaticHeadGroupSize>{})": 5,
            "params.q_row_stride,\n                            "
            "params.q_head_stride)": 1,
        }
        if splitk_marker in function:
            # The split-K arm rewrites the paired O/LSE addressing of BOTH
            # output sites to reach the stock split accumulators. The paired
            # Q tile and the M-tile shape above are untouched by that rewrite,
            # so they are still checked; the four rewritten expressions are
            # checked in their split-K form by
            # _patch_fixed32_query_gqa_pair_splitk_layout's own guard.
            required_counts[splitk_marker] = 4
        else:
            required_counts.update({
                "params.o_row_stride,\n                                "
                "params.o_head_stride)": 1,
                "params.o_row_stride,\n                            "
                "params.o_head_stride)": 1,
                "make_stride(_1{}, params.total_q)": 2,
            })
        for snippet, expected in required_counts.items():
            if function.count(snippet) != expected:
                raise RuntimeError("qrow32 GQA-pair address layout drifted")
        return False

    q_tensor = r'''    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr) + static_query_offset<Kernel_traits>(binfo, params.q_batch_stride, params.q_row_stride, bidb)),
                            make_shape(actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(query_m_block, 0));  // (kBlockM, kHeadDim)
'''
    paired_q_tensor = r'''    // FR13_FA2_QROW32_GQA_PAIR_LAYOUT: two heads share the M tile.
    Tensor gQ = [&] {
        if constexpr (kStaticHeadGroupSize == 2) {
            static_assert(kStaticQueryRows == 32);
            static_assert(kBlockM == kStaticQueryRows * kStaticHeadGroupSize);
            auto q_ptr = make_gmem_ptr(
                reinterpret_cast<Element*>(params.q_ptr)
                + static_query_offset<Kernel_traits>(
                    binfo, params.q_batch_stride, params.q_row_stride, bidb)
                + bidh * params.q_head_stride);
            return make_tensor(
                q_ptr,
                make_layout(
                    make_shape(
                        make_shape(
                            Int<kStaticQueryRows>{},
                            Int<kStaticHeadGroupSize>{}),
                        Int<kHeadDim>{}),
                    make_stride(
                        make_stride(
                            params.q_row_stride,
                            params.q_head_stride),
                        _1{})));
        } else {
            Tensor mQ = make_tensor(
                make_gmem_ptr(
                    reinterpret_cast<Element*>(params.q_ptr)
                    + static_query_offset<Kernel_traits>(
                        binfo,
                        params.q_batch_stride,
                        params.q_row_stride,
                        bidb)),
                make_shape(actual_seqlen_q, params.h, params.d),
                make_stride(
                    params.q_row_stride, params.q_head_stride, _1{}));
            return local_tile(
                mQ(_, bidh, _),
                Shape<Int<kBlockM>, Int<kHeadDim>>{},
                make_coord(query_m_block, 0));
        }
    }();  // (kBlockM, kHeadDim)
'''
    if function.count(q_tensor) != 1:
        raise RuntimeError("qrow32 GQA-pair Q-tensor anchor drifted")
    function = function.replace(q_tensor, paired_q_tensor, 1)

    early_output = r'''        Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split ? params.oaccum_ptr : params.o_ptr) + (Split ? row_offset_oaccum : row_offset_o)),
                                      Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                     make_stride(Split ? kHeadDim : params.o_row_stride, _1{}));
        Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split ? params.softmax_lseaccum_ptr : params.softmax_lse_ptr) + row_offset_lseaccum),
                                      Shape<Int<kBlockM>>{}, Stride<_1>{});
'''
    paired_early_output = r'''        // FR13_FA2_QROW32_GQA_PAIR_LAYOUT: two heads share the M tile.
        Tensor gOaccum = [&] {
            if constexpr (kStaticHeadGroupSize == 2) {
                static_assert(!Split);
                auto o_ptr = make_gmem_ptr(
                    reinterpret_cast<ElementO *>(params.o_ptr)
                    + static_query_offset<Kernel_traits>(
                        binfo,
                        params.o_batch_stride,
                        params.o_row_stride,
                        bidb)
                    + bidh * params.o_head_stride);
                return make_tensor(
                    o_ptr,
                    make_layout(
                        make_shape(
                            make_shape(
                                Int<kStaticQueryRows>{},
                                Int<kStaticHeadGroupSize>{}),
                            Int<kHeadDim>{}),
                        make_stride(
                            make_stride(
                                params.o_row_stride,
                                params.o_head_stride),
                            _1{})));
            } else {
                return make_tensor(
                    make_gmem_ptr(
                        reinterpret_cast<ElementO *>(
                            Split ? params.oaccum_ptr : params.o_ptr)
                        + (Split ? row_offset_oaccum : row_offset_o)),
                    Shape<Int<kBlockM>, Int<kHeadDim>>{},
                    make_stride(
                        Split ? kHeadDim : params.o_row_stride, _1{}));
            }
        }();
        Tensor gLSEaccum = [&] {
            if constexpr (kStaticHeadGroupSize == 2) {
                static_assert(!Split);
                auto lse_ptr = make_gmem_ptr(
                    reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr)
                    + bidh * params.total_q
                    + static_query_offset<Kernel_traits>(
                        binfo, params.seqlen_q, 1, bidb));
                return make_tensor(
                    lse_ptr,
                    make_layout(
                        make_shape(
                            Int<kStaticQueryRows>{},
                            Int<kStaticHeadGroupSize>{}),
                        make_stride(_1{}, params.total_q)));
            } else {
                return make_tensor(
                    make_gmem_ptr(
                        reinterpret_cast<ElementAccum *>(
                            Split
                                ? params.softmax_lseaccum_ptr
                                : params.softmax_lse_ptr)
                        + row_offset_lseaccum),
                    Shape<Int<kBlockM>>{}, Stride<_1>{});
            }
        }();
'''
    if function.count(early_output) != 1:
        raise RuntimeError("qrow32 GQA-pair early-output anchor drifted")
    function = function.replace(early_output, paired_early_output, 1)

    epilogue_output = r'''    Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split ? params.oaccum_ptr : params.o_ptr) + (Split ? row_offset_oaccum : row_offset_o)),
                                 Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                 make_stride(Split ? kHeadDim : params.o_row_stride, _1{}));
    Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split ? params.softmax_lseaccum_ptr : params.softmax_lse_ptr) + row_offset_lseaccum),
                                   Shape<Int<kBlockM>>{}, Stride<_1>{});
'''
    paired_epilogue_output = r'''    // FR13_FA2_QROW32_GQA_PAIR_LAYOUT: two heads share the M tile.
    Tensor gOaccum = [&] {
        if constexpr (kStaticHeadGroupSize == 2) {
            static_assert(!Split);
            auto o_ptr = make_gmem_ptr(
                reinterpret_cast<ElementO *>(params.o_ptr)
                + static_query_offset<Kernel_traits>(
                    binfo,
                    params.o_batch_stride,
                    params.o_row_stride,
                    bidb)
                + bidh * params.o_head_stride);
            return make_tensor(
                o_ptr,
                make_layout(
                    make_shape(
                        make_shape(
                            Int<kStaticQueryRows>{},
                            Int<kStaticHeadGroupSize>{}),
                        Int<kHeadDim>{}),
                    make_stride(
                        make_stride(
                            params.o_row_stride,
                            params.o_head_stride),
                        _1{})));
        } else {
            return make_tensor(
                make_gmem_ptr(
                    reinterpret_cast<ElementO *>(
                        Split ? params.oaccum_ptr : params.o_ptr)
                    + (Split ? row_offset_oaccum : row_offset_o)),
                Shape<Int<kBlockM>, Int<kHeadDim>>{},
                make_stride(
                    Split ? kHeadDim : params.o_row_stride, _1{}));
        }
    }();
    Tensor gLSEaccum = [&] {
        if constexpr (kStaticHeadGroupSize == 2) {
            static_assert(!Split);
            auto lse_ptr = make_gmem_ptr(
                reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr)
                + bidh * params.total_q
                + static_query_offset<Kernel_traits>(
                    binfo, params.seqlen_q, 1, bidb));
            return make_tensor(
                lse_ptr,
                make_layout(
                    make_shape(
                        Int<kStaticQueryRows>{},
                        Int<kStaticHeadGroupSize>{}),
                    make_stride(_1{}, params.total_q)));
        } else {
            return make_tensor(
                make_gmem_ptr(
                    reinterpret_cast<ElementAccum *>(
                        Split
                            ? params.softmax_lseaccum_ptr
                            : params.softmax_lse_ptr)
                    + row_offset_lseaccum),
                Shape<Int<kBlockM>>{}, Stride<_1>{});
        }
    }();
'''
    if function.count(epilogue_output) != 1:
        raise RuntimeError("qrow32 GQA-pair epilogue-output anchor drifted")
    function = function.replace(epilogue_output, paired_epilogue_output, 1)

    text = text[:function_start] + function + text[function_end:]
    path.write_text(text)
    return True


def _reindent(block: str, pad: str) -> str:
    """Re-indent a whole block, leaving blank lines untouched."""
    return "".join(
        (pad + line) if line.strip() else line
        for line in block.splitlines(keepends=True)
    )


# The paired ((query_row, head_in_pair), column) O/LSE tensors are written at
# two sites -- the empty-split early-out and the epilogue -- whose bodies are
# identical apart from four spaces of indentation. Both are pinned to
# `static_assert(!Split)` by the promoted arm, because the promoted arm has no
# split. Split-K needs exactly the same paired M tile addressed into the stock
# split accumulators instead, and nothing else: same shape, same head pairing,
# same 32 rows. The offsets it uses (`row_offset_oaccum`, `row_offset_lseaccum`)
# are the STOCK ones already computed a few lines above each site, so the
# accumulator layout the combine kernel reads is stock by construction and not
# a re-derived equivalent.
_GQA_PAIR_SPLITK_LAYOUT_SUBSTITUTIONS = (
    (
        """            static_assert(!Split);
            auto o_ptr = make_gmem_ptr(
                reinterpret_cast<ElementO *>(params.o_ptr)
                + static_query_offset<Kernel_traits>(
                    binfo,
                    params.o_batch_stride,
                    params.o_row_stride,
                    bidb)
                + bidh * params.o_head_stride);
""",
        """            // FR14_FA2_QROW32_GQA_PAIR_SPLITK_LAYOUT: the same paired
            // M tile, addressed into the stock split accumulator when Split.
            auto o_ptr = make_gmem_ptr(
                reinterpret_cast<ElementO *>(
                    Split ? params.oaccum_ptr : params.o_ptr)
                + (Split
                    ? row_offset_oaccum
                    : static_query_offset<Kernel_traits>(
                          binfo,
                          params.o_batch_stride,
                          params.o_row_stride,
                          bidb)
                      + bidh * params.o_head_stride));
""",
    ),
    (
        """                    make_stride(
                        make_stride(
                            params.o_row_stride,
                            params.o_head_stride),
                        _1{})));
""",
        """                    make_stride(
                        make_stride(
                            Split ? static_cast<index_t>(params.d_rounded)
                                  : params.o_row_stride,
                            Split ? static_cast<index_t>(params.seqlen_q)
                                        * params.d_rounded
                                  : params.o_head_stride),
                        _1{})));
""",
    ),
    (
        """            static_assert(!Split);
            auto lse_ptr = make_gmem_ptr(
                reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr)
                + bidh * params.total_q
                + static_query_offset<Kernel_traits>(
                    binfo, params.seqlen_q, 1, bidb));
""",
        """            // FR14_FA2_QROW32_GQA_PAIR_SPLITK_LAYOUT: the same paired
            // M tile, addressed into the stock split accumulator when Split.
            auto lse_ptr = make_gmem_ptr(
                reinterpret_cast<ElementAccum *>(
                    Split ? params.softmax_lseaccum_ptr
                          : params.softmax_lse_ptr)
                + (Split
                    ? row_offset_lseaccum
                    : bidh * params.total_q
                      + static_query_offset<Kernel_traits>(
                            binfo, params.seqlen_q, 1, bidb)));
""",
    ),
    (
        """                    make_stride(_1{}, params.total_q)));
""",
        """                    make_stride(
                        _1{},
                        Split ? params.seqlen_q : params.total_q)));
""",
    ),
)


def _patch_fixed32_combine_static_geometry(
    path: Path,
    *,
    fixed32_query_gqa_pair32_splitk_b1: bool = False,
) -> bool:
    """Give FA2's combine kernel the static geometry it is already pinned to.

    FOUND BY THE GUARD, NOT AROUND IT. The promoted build script's SASS contract
    forbids LDL/STL/CALL, and the split-K TU's first compile tripped it with
    seven CALL.REL.NOINC into a 64-bit integer-division helper -- all of them
    inside FA2's own combine kernel, which the promoted arm never instantiates
    because it has no combine. STACK and LOCAL were both 0, so nothing had
    spilled; the calls were runtime division, and the honest fix is to remove
    the division rather than to widen the contract that caught it.

    Two sources, both pure ADDRESSING -- no arithmetic on any accumulated value
    changes, and no store moves to a different element:

    1. The unpadded-LSE tensor is built by composing three CuTe layouts whose
       flat extent is an index_t, so every store evaluates an int64 div/mod. At
       sequences == 1 that composition reduces exactly to the identity: the
       composed offset is q + h*seqlen_q*b + b_idx*seqlen_q, which at b == 1 is
       q + h*seqlen_q, and the flat index being stored IS h*seqlen_q + q. The
       two branches therefore address the SAME element, and the padded branch
       needs no division at all.

    2. The output epilogue decomposes the flat index with two runtime
       divisions. At sequences == 1 the batch index is 0, and kStaticQueryRows
       is a power of two, so the head and row fall out of a shift and a mask.

    Neither reduction is an assumption: the split-K launcher TORCH_CHECKs
    b == 1, h == 24 and seqlen_q == 32 before it launches anything, so a
    geometry that broke the reduction cannot reach this kernel.
    """
    if not fixed32_query_gqa_pair32_splitk_b1:
        return False
    text = path.read_text()
    marker = "// FR14_FA2_COMBINE_STATIC_GEOMETRY"
    if marker in text:
        if text.count(marker) != 2:
            raise RuntimeError("combine static geometry drifted")
        return False

    traits = r'''    static_assert(kNThreads == 128, "We assume that each block has 128 threads");
'''
    static_traits = r'''    static_assert(kNThreads == 128, "We assume that each block has 128 threads");
    // FR14_FA2_COMBINE_STATIC_GEOMETRY: the split-K arm's traits carry the
    // exact served geometry, and its launcher refuses to run at any other.
    constexpr int kStaticCombineRows = StaticQueryRows<Kernel_traits>::value;
    constexpr int kStaticCombineSequences =
        StaticQueryBatchLayout<Kernel_traits>::sequences;
    constexpr bool kStaticCombineBatch =
        kStaticCombineSequences == 1 && kStaticCombineRows != 0;
    static_assert(
        !kStaticCombineBatch
        || (kStaticCombineRows & (kStaticCombineRows - 1)) == 0,
        "static combine geometry requires a power-of-two query-row count");
'''
    if text.count(traits) != 1:
        raise RuntimeError("combine trait anchor drifted")
    text = text.replace(traits, static_traits, 1)

    lse_store = r'''        if (params.unpadded_lse) {
            const index_t lse_offset = row_offset_lse + tidx / kRowsPerLoadTranspose;
            if (lse_offset < lse_size) {
                gLSE_unpadded(lse_offset) = lse_logsum;
            }
        } else {
'''
    static_lse_store = r'''        if constexpr (kStaticCombineBatch) {
            // FR14_FA2_COMBINE_STATIC_GEOMETRY: at b == 1 the composed
            // unpadded layout and the flat layout name the same element, so
            // the store is the same store without the int64 division.
            const index_t lse_offset = row_offset_lse + tidx / kRowsPerLoadTranspose;
            if (lse_offset < lse_size) {
                gLSE(tidx / kRowsPerLoadTranspose) = lse_logsum;
            }
        } else if (params.unpadded_lse) {
            const index_t lse_offset = row_offset_lse + tidx / kRowsPerLoadTranspose;
            if (lse_offset < lse_size) {
                gLSE_unpadded(lse_offset) = lse_logsum;
            }
        } else {
'''
    if text.count(lse_store) != 1:
        raise RuntimeError("combine unpadded-LSE store anchor drifted")
    text = text.replace(lse_store, static_lse_store, 1)

    decompose = r'''            const int batch_idx = idx / (params.h * params.seqlen_q);
            const int head_idx = (idx - batch_idx * (params.h * params.seqlen_q)) / params.seqlen_q;
            // The index to the rows of Q
            const int row = idx - batch_idx * (params.h * params.seqlen_q) - head_idx * params.seqlen_q;
'''
    static_decompose = r'''            int batch_idx, head_idx, row;
            if constexpr (kStaticCombineBatch) {
                batch_idx = 0;
                head_idx = idx / kStaticCombineRows;
                row = idx & (kStaticCombineRows - 1);
            } else {
                batch_idx = idx / (params.h * params.seqlen_q);
                head_idx = (idx - batch_idx * (params.h * params.seqlen_q)) / params.seqlen_q;
                // The index to the rows of Q
                row = idx - batch_idx * (params.h * params.seqlen_q) - head_idx * params.seqlen_q;
            }
'''
    if text.count(decompose) != 1:
        raise RuntimeError("combine index-decomposition anchor drifted")
    text = text.replace(decompose, static_decompose, 1)
    path.write_text(text)
    return True


def _patch_fixed32_query_gqa_pair_splitk_layout(
    path: Path,
    *,
    fixed32_query_gqa_pair32_splitk_b1: bool = False,
) -> bool:
    """Let the paired M tile address the stock split accumulators.

    Runs AFTER _patch_fixed32_query_gqa_pair_layout and only for the split-K
    arm, so the promoted GQA-pair source closure is untouched: with the flag
    off this function writes nothing at all.
    """
    if not fixed32_query_gqa_pair32_splitk_b1:
        return False
    text = path.read_text()
    function_signature = (
        "template<typename Kernel_traits, bool Is_causal, bool Is_local, "
        "bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, "
        "bool Split, bool Append_KV, typename Params>\n"
        "inline __device__ void compute_attn_1rowblock_splitkv"
    )
    function_start = text.index(function_signature)
    function_end = text.index(
        "\n////////////////////////////////////////////////////////////////////////////////////////////////////\n\n"
        "template<typename Kernel_traits, bool Is_dropout",
        function_start,
    )
    function = text[function_start:function_end]
    marker = (
        "// FR14_FA2_QROW32_GQA_PAIR_SPLITK_LAYOUT: the same paired"
    )
    if marker in function:
        if function.count(marker) != 4 or "static_assert(!Split);" in function:
            raise RuntimeError("qrow32 GQA-pair split-K address layout drifted")
        return False
    if "FR13_FA2_QROW32_GQA_PAIR_LAYOUT: two heads share the M tile." not in function:
        raise RuntimeError(
            "qrow32 GQA-pair split-K layout requires the paired address layout"
        )
    # Both sites, in one pass each: the epilogue at its own indentation and the
    # early-out four spaces deeper. Counted, so a drift in either one fails.
    for anchor, replacement in _GQA_PAIR_SPLITK_LAYOUT_SUBSTITUTIONS:
        # The leading newline makes every anchor line-anchored, so the
        # epilogue's pattern cannot match mid-line inside the deeper-indented
        # early-out block.
        anchor, replacement = "\n" + anchor, "\n" + replacement
        for pad in ("", "    "):
            padded_anchor = _reindent(anchor, pad)
            if function.count(padded_anchor) != 1:
                raise RuntimeError(
                    "qrow32 GQA-pair split-K layout anchor drifted at "
                    f"indent {len(pad)}: expected one, found "
                    f"{function.count(padded_anchor)}"
                )
            function = function.replace(
                padded_anchor, _reindent(replacement, pad), 1
            )
    if "static_assert(!Split);" in function:
        raise RuntimeError(
            "qrow32 GQA-pair split-K layout left a !Split assertion behind"
        )
    text = text[:function_start] + function + text[function_end:]
    path.write_text(text)
    return True


def _patch_fixed32_query_translation_unit(
    path: Path,
    *,
    fixed32_query_tile16: bool = False,
    fixed32_query_tile16_static_strides: bool = False,
) -> bool:
    if not (fixed32_query_tile16 or fixed32_query_tile16_static_strides):
        return False
    if fixed32_query_tile16_static_strides and not fixed32_query_tile16:
        raise ValueError("qrow16 static strides require the qrow16 private kernel")
    stock_text = path.read_text()
    if STOCK_FIXED32_QUERY_INSTANTIATION not in stock_text:
        raise RuntimeError("stock fixed32 FA2 explicit instantiation drifted")
    if "FR13_FA2_FIXED32_QUERY_TILE16" in stock_text:
        raise RuntimeError("qrow16 must not share the stock instantiation TU")
    qrow_path = path.with_name("flash_fwd_fr13_qrow16_hdim256_bf16_sm80.cu")
    expected = (
        FIXED32_QUERY_TILE16_STATIC_STRIDES_TRANSLATION_UNIT
        if fixed32_query_tile16_static_strides
        else FIXED32_QUERY_TILE16_TRANSLATION_UNIT
    )
    if qrow_path.exists():
        if qrow_path.read_text() != expected:
            raise RuntimeError("existing qrow16 translation unit drifted")
        return False
    qrow_path.write_text(expected)
    return True


def _patch_fixed32_query_tile32_translation_unit(
    path: Path,
    *,
    fixed32_query_tile32: bool = False,
    fixed32_tree_visibility_mask: bool = False,
) -> bool:
    if not fixed32_query_tile32:
        return False
    stock_text = path.read_text()
    if STOCK_FIXED32_QUERY_INSTANTIATION not in stock_text:
        raise RuntimeError("stock fixed32 FA2 explicit instantiation drifted")
    if "FR13_FA2_FIXED32_QUERY_TILE32" in stock_text:
        raise RuntimeError("qrow32 must not share the stock instantiation TU")
    qrow_path = path.with_name("flash_fwd_fr13_qrow32_hdim256_bf16_sm80.cu")
    expected = FIXED32_QUERY_TILE32_TRANSLATION_UNIT
    if fixed32_tree_visibility_mask:
        expected = _with_fixed32_tree_visibility(
            expected,
            trait="Fr13Fixed32Qrow32KernelTraits",
            symbol="fr13_fixed32_qrow32_tree_visibility",
            max_registers=252,
        )
    if qrow_path.exists():
        if qrow_path.read_text() != expected:
            raise RuntimeError("existing qrow32 translation unit drifted")
        return False
    qrow_path.write_text(expected)
    return True


def _patch_fixed32_query_gqa_pair32_translation_unit(
    path: Path,
    *,
    fixed32_query_gqa_pair32: bool = False,
    fixed32_tree_visibility_mask: bool = False,
) -> bool:
    if not fixed32_query_gqa_pair32:
        return False
    stock_text = path.read_text()
    if STOCK_FIXED32_QUERY_INSTANTIATION not in stock_text:
        raise RuntimeError("stock fixed32 FA2 explicit instantiation drifted")
    if "FR13_FA2_FIXED32_QUERY_GQA_PAIR32" in stock_text:
        raise RuntimeError("qrow32 GQA pair must not share the stock instantiation TU")
    pair_path = path.with_name(
        "flash_fwd_fr13_qrow32_gqa_pair_hdim256_bf16_sm80.cu"
    )
    expected = FIXED32_QUERY_GQA_PAIR32_TRANSLATION_UNIT
    if fixed32_tree_visibility_mask:
        expected = _with_fixed32_tree_visibility(
            expected,
            trait="Fr13Fixed32Qrow32GqaPairKernelTraits",
            symbol="fr13_fixed32_qrow32_gqa_pair_tree_visibility",
            max_registers=252,
        )
    if pair_path.exists():
        if pair_path.read_text() != expected:
            raise RuntimeError("existing qrow32 GQA-pair translation unit drifted")
        return False
    pair_path.write_text(expected)
    return True


def _patch_fixed32_query_gqa_pair32_b1_translation_unit(
    path: Path,
    *,
    fixed32_query_gqa_pair32_b1: bool = False,
    fixed32_tree_visibility_mask: bool = False,
) -> bool:
    if not fixed32_query_gqa_pair32_b1:
        return False
    stock_text = path.read_text()
    if STOCK_FIXED32_QUERY_INSTANTIATION not in stock_text:
        raise RuntimeError("stock fixed32 FA2 explicit instantiation drifted")
    if "FR13_FA2_FIXED32_QUERY_GQA_PAIR32" in stock_text:
        raise RuntimeError(
            "qrow32 B1 GQA pair must not share the stock instantiation TU"
        )
    pair_path = path.with_name(
        "flash_fwd_fr13_qrow32_gqa_pair_b1_hdim256_bf16_sm80.cu"
    )
    expected = FIXED32_QUERY_GQA_PAIR32_B1_TRANSLATION_UNIT
    if fixed32_tree_visibility_mask:
        expected = _with_fixed32_tree_visibility(
            expected,
            trait="Fr13Fixed32Qrow32GqaPairB1KernelTraits",
            symbol="fr13_fixed32_qrow32_gqa_pair_b1_tree_visibility",
            max_registers=252,
        )
    if pair_path.exists():
        if pair_path.read_text() != expected:
            raise RuntimeError(
                "existing qrow32 B1 GQA-pair translation unit drifted"
            )
        return False
    pair_path.write_text(expected)
    return True


def _patch_fixed32_query_gqa_pair32_splitk_b1_translation_unit(
    path: Path,
    *,
    fixed32_query_gqa_pair32_splitk_b1: bool = False,
) -> bool:
    if not fixed32_query_gqa_pair32_splitk_b1:
        return False
    stock_text = path.read_text()
    if STOCK_FIXED32_QUERY_INSTANTIATION not in stock_text:
        raise RuntimeError("stock fixed32 FA2 explicit instantiation drifted")
    if "FR13_FA2_FIXED32_QUERY_GQA_PAIR32" in stock_text:
        raise RuntimeError(
            "qrow32 B1 GQA pair split-K must not share the stock "
            "instantiation TU"
        )
    pair_path = path.with_name(
        "flash_fwd_fr13_qrow32_gqa_pair_splitk_b1_hdim256_bf16_sm80.cu"
    )
    expected = FIXED32_QUERY_GQA_PAIR32_SPLITK_B1_TRANSLATION_UNIT
    if pair_path.exists():
        if pair_path.read_text() != expected:
            raise RuntimeError(
                "existing qrow32 B1 GQA-pair split-K translation unit drifted"
            )
        return False
    pair_path.write_text(expected)
    return True


def _patch_fixed32_query_tile32_b1_translation_unit(
    path: Path,
    *,
    fixed32_query_tile32_b1: bool = False,
    fixed32_tree_visibility_mask: bool = False,
) -> bool:
    if not fixed32_query_tile32_b1:
        return False
    stock_text = path.read_text()
    if STOCK_FIXED32_QUERY_INSTANTIATION not in stock_text:
        raise RuntimeError("stock fixed32 FA2 explicit instantiation drifted")
    if "FR13_FA2_FIXED32_QUERY_TILE32" in stock_text:
        raise RuntimeError("qrow32 B1 must not share the stock instantiation TU")
    candidates = (
        (
            path.with_name(
                "flash_fwd_fr13_qrow32_b1_hdim256_bf16_sm80.cu"
            ),
            FIXED32_QUERY_TILE32_B1_TRANSLATION_UNIT,
            "qrow32 B1 no-split",
        ),
        (
            path.with_name(
                "flash_fwd_fr13_qrow32_b1_split2_hdim256_bf16_sm80.cu"
            ),
            FIXED32_QUERY_TILE32_B1_SPLIT2_TRANSLATION_UNIT,
            "qrow32 B1 split2",
        ),
    )
    changed = False
    for qrow_path, expected, label in candidates:
        if fixed32_tree_visibility_mask and label == "qrow32 B1 no-split":
            expected = _with_fixed32_tree_visibility(
                expected,
                trait="Fr13Fixed32Qrow32B1KernelTraits",
                symbol="fr13_fixed32_qrow32_b1_tree_visibility",
                max_registers=252,
            )
        if qrow_path.exists():
            if qrow_path.read_text() != expected:
                raise RuntimeError(f"existing {label} translation unit drifted")
        else:
            qrow_path.write_text(expected)
            changed = True
    return changed


def _patch_flash_api_cpp(
    path: Path,
    *,
    fixed32_query_tile16: bool = False,
    fixed32_query_tile16_static_strides: bool = False,
    fixed32_query_tile32: bool = False,
    fixed32_query_gqa_pair32: bool = False,
    fixed32_query_tile32_b1: bool = False,
    fixed32_query_gqa_pair32_b1: bool = False,
    fixed32_query_gqa_pair32_splitk_b1: bool = False,
) -> bool:
    text = path.read_text()
    changed = False
    helper = r'''
// FR13_FA2_TREE_BIAS
void set_params_tree_bias(Flash_fwd_params &params,
                          std::optional<at::Tensor> &tree_bias_,
                          int batch_size,
                          int max_seqlen_q) {
    if (!tree_bias_.has_value()) {
        params.tree_bias_ptr = nullptr;
        params.tree_bias_batch_stride = 0;
        params.tree_bias_row_stride = 0;
        params.tree_bias_col_stride = 0;
        params.tree_bias_rows = 0;
        params.tree_bias_cols = 0;
        params.tree_bias_q_offset = 0;
        params.tree_bias_k_offset = 0;
        return;
    }
    at::Tensor tree_bias = tree_bias_.value();
    CHECK_DEVICE(tree_bias);
    TORCH_CHECK(tree_bias.dtype() == torch::kFloat32, "tree_bias must be fp32");
    TORCH_CHECK(tree_bias.stride(-1) == 1, "tree_bias last dimension must be contiguous");
    TORCH_CHECK(tree_bias.dim() == 2 || tree_bias.dim() == 3, "tree_bias must have shape [q, q] or [batch, q, q]");
    if (tree_bias.dim() == 3) {
        TORCH_CHECK(tree_bias.size(0) == batch_size, "batched tree_bias batch dimension mismatch");
    }
    TORCH_CHECK(tree_bias.size(-2) > 0, "tree_bias rows must be non-empty");
    TORCH_CHECK(tree_bias.size(-1) > 0, "tree_bias cols must be non-empty");
    params.tree_bias_ptr = tree_bias.data_ptr();
    params.tree_bias_batch_stride = tree_bias.dim() == 3 ? tree_bias.stride(0) : 0;
    params.tree_bias_row_stride = tree_bias.stride(-2);
    params.tree_bias_col_stride = tree_bias.stride(-1);
    params.tree_bias_rows = tree_bias.size(-2);
    params.tree_bias_cols = tree_bias.size(-1);
    params.tree_bias_q_offset = max_seqlen_q > tree_bias.size(-2) ? max_seqlen_q - tree_bias.size(-2) : 0;
    params.tree_bias_k_offset = max_seqlen_q > tree_bias.size(-1) ? max_seqlen_q - tree_bias.size(-1) : 0;
}

'''
    helper_marker = "// FR13_FA2_TREE_BIAS\nvoid set_params_tree_bias"
    helper_end_marker = "std::vector<at::Tensor>\nmha_fwd("
    if helper_marker in text:
        start = text.index(helper_marker)
        end = text.index(helper_end_marker, start)
        if text[start:end] != helper.lstrip():
            text = text[:start] + helper.lstrip() + text[end:]
            changed = True
    else:
        text, did = _insert_once(
            text,
            helper_end_marker,
            helper,
            "set_params_tree_bias helper",
        )
        changed = changed or did
    text, did = _replace_once(
        text,
        "std::vector<at::Tensor>\nmha_varlen_fwd(",
        "std::vector<at::Tensor>\nmha_varlen_fwd_impl(",
        "rename varlen impl",
    )
    changed = changed or did
    old_params = """               const bool return_softmax,\n               int num_splits,\n               std::optional<at::Generator> gen_) {\n"""
    new_params = """               const bool return_softmax,\n               int num_splits,\n               std::optional<at::Tensor> &tree_bias_,\n               std::optional<at::Generator> gen_) {\n"""
    text, did = _replace_once(text, old_params, new_params, "impl tree_bias param")
    changed = changed or did
    old_call = """    params.page_block_size = page_block_size;\n    // Keep references to these tensors to extend their lifetime\n"""
    new_call = """    params.page_block_size = page_block_size;\n    set_params_tree_bias(params, tree_bias_, batch_size, max_seqlen_q);\n    // Keep references to these tensors to extend their lifetime\n"""
    text, did = _replace_once(text, old_call, new_call, "set tree bias params")
    changed = changed or did
    wrappers = r'''
std::vector<at::Tensor>
mha_varlen_fwd(at::Tensor &q,
               const at::Tensor &k,
               const at::Tensor &v,
               std::optional<at::Tensor> &out_,
               const at::Tensor &cu_seqlens_q,
               const at::Tensor &cu_seqlens_k,
               std::optional<at::Tensor> &seqused_k,
               std::optional<const at::Tensor> &leftpad_k_,
               std::optional<at::Tensor> &block_table_,
               std::optional<at::Tensor> &alibi_slopes_,
               int max_seqlen_q,
               const int max_seqlen_k,
               const float p_dropout,
               const float softmax_scale,
               const bool zero_tensors,
               bool is_causal,
               int window_size_left,
               int window_size_right,
               const float softcap,
               const bool return_softmax,
               int num_splits,
               std::optional<at::Generator> gen_) {
    std::optional<at::Tensor> tree_bias_;
    return mha_varlen_fwd_impl(q, k, v, out_, cu_seqlens_q, cu_seqlens_k,
                               seqused_k, leftpad_k_, block_table_,
                               alibi_slopes_, max_seqlen_q, max_seqlen_k,
                               p_dropout, softmax_scale, zero_tensors,
                               is_causal, window_size_left, window_size_right,
                               softcap, return_softmax, num_splits,
                               tree_bias_, gen_);
}

// FR13_FA2_TREE_BIAS
std::vector<at::Tensor>
mha_varlen_fwd_tree_bias(at::Tensor &q,
                         const at::Tensor &k,
                         const at::Tensor &v,
                         std::optional<at::Tensor> &out_,
                         const at::Tensor &cu_seqlens_q,
                         const at::Tensor &cu_seqlens_k,
                         std::optional<at::Tensor> &seqused_k,
                         std::optional<const at::Tensor> &leftpad_k_,
                         std::optional<at::Tensor> &block_table_,
                         std::optional<at::Tensor> &alibi_slopes_,
                         int max_seqlen_q,
                         const int max_seqlen_k,
                         const float p_dropout,
                         const float softmax_scale,
                         const bool zero_tensors,
                         bool is_causal,
                         int window_size_left,
                         int window_size_right,
                         const float softcap,
                         const bool return_softmax,
                         int num_splits,
                         std::optional<at::Tensor> &tree_bias_,
                         std::optional<at::Generator> gen_) {
    return mha_varlen_fwd_impl(q, k, v, out_, cu_seqlens_q, cu_seqlens_k,
                               seqused_k, leftpad_k_, block_table_,
                               alibi_slopes_, max_seqlen_q, max_seqlen_k,
                               p_dropout, softmax_scale, zero_tensors,
                               is_causal, window_size_left, window_size_right,
                               softcap, return_softmax, num_splits,
                               tree_bias_, gen_);
}

'''
    text, did = _insert_once(
        text,
        "void run_mha_bwd",
        wrappers,
        "varlen wrappers",
    )
    changed = changed or did
    if fixed32_query_tile16:
        qrow16_dispatch = (
            FIXED32_QUERY_TILE16_STATIC_STRIDES_API_DISPATCH
            if fixed32_query_tile16_static_strides
            else FIXED32_QUERY_TILE16_API_DISPATCH
        )
        text, did = _install_qrow16_api_dispatch(
            text,
            qrow16_dispatch,
            label="fixed32 FA2 query tile16 hidden API dispatch",
        )
        changed = changed or did
    if fixed32_query_tile32:
        text, did = _install_hidden_api_gate(
            text,
            declaration=FIXED32_QUERY_TILE32_API_DECLARATION,
            gate=FIXED32_QUERY_TILE32_API_GATE,
            label="fixed32 FA2 query tile32 gate-only API dispatch",
        )
        changed = changed or did
    if fixed32_query_gqa_pair32:
        text, did = _install_hidden_api_gate(
            text,
            declaration=FIXED32_QUERY_GQA_PAIR32_API_DECLARATION,
            gate=FIXED32_QUERY_GQA_PAIR32_API_GATE,
            label="fixed32 FA2 qrow32 GQA-pair gate-only API dispatch",
        )
        changed = changed or did
    if fixed32_query_tile32_b1:
        text, did = _install_qrow16_api_dispatch(
            text,
            FIXED32_QUERY_TILE16_B1_REFERENCE_API_DISPATCH,
            label="fixed32 FA2 query tile16 B1 reference hidden API dispatch",
        )
        changed = changed or did
        text, did = _install_hidden_api_gate(
            text,
            declaration=FIXED32_QUERY_TILE32_B1_API_DECLARATION,
            gate=FIXED32_QUERY_TILE32_B1_API_GATE,
            label="fixed32 FA2 query tile32 B1 hidden API dispatch",
        )
        changed = changed or did
        text, did = _install_hidden_api_gate(
            text,
            declaration=FIXED32_QUERY_TILE32_B1_SPLIT2_API_DECLARATION,
            gate=FIXED32_QUERY_TILE32_B1_SPLIT2_API_GATE,
            label="fixed32 FA2 query tile32 B1 split2 hidden API dispatch",
        )
        changed = changed or did
        text, did = _replace_once(
            text,
            STOCK_VARLEN_SPLITKV_ALLOCATION,
            FIXED32_QUERY_TILE32_B1_SPLIT2_ALLOCATION,
            "fixed32 FA2 query tile32 B1 split2 scratch allocation",
        )
        changed = changed or did
    if fixed32_query_gqa_pair32_b1:
        # The B1 byte gate compares against the qrow16 incumbent, so the
        # reference sentinel needs its dispatch in the same binary. This is the
        # identical reference dispatch the qrow32 B1 arms install; no split-K
        # scratch patch is installed because the GQA-pair arm keeps
        # num_splits=0.
        text, did = _install_qrow16_api_dispatch(
            text,
            FIXED32_QUERY_TILE16_B1_REFERENCE_API_DISPATCH,
            label="fixed32 FA2 query tile16 B1 reference hidden API dispatch",
        )
        changed = changed or did
        text, did = _install_hidden_api_gate(
            text,
            declaration=FIXED32_QUERY_GQA_PAIR32_B1_API_DECLARATION,
            gate=FIXED32_QUERY_GQA_PAIR32_B1_API_GATE,
            label="fixed32 FA2 qrow32 B1 GQA-pair gate-only API dispatch",
        )
        changed = changed or did
    if fixed32_query_gqa_pair32_splitk_b1:
        # The split-K arm cannot be byte-gated against qrow16 (it is Tier-B by
        # construction), but the characterization harness still needs both the
        # qrow16 incumbent and the promoted GQA-pair arm in the same binary to
        # measure against, so all three dispatches are installed here.
        text, did = _install_qrow16_api_dispatch(
            text,
            FIXED32_QUERY_TILE16_B1_REFERENCE_API_DISPATCH,
            label="fixed32 FA2 query tile16 B1 reference hidden API dispatch",
        )
        changed = changed or did
        text, did = _install_hidden_api_gate(
            text,
            declaration=FIXED32_QUERY_GQA_PAIR32_B1_API_DECLARATION,
            gate=FIXED32_QUERY_GQA_PAIR32_B1_API_GATE,
            label="fixed32 FA2 qrow32 B1 GQA-pair gate-only API dispatch",
        )
        changed = changed or did
        text, did = _install_hidden_api_gate(
            text,
            declaration=FIXED32_QUERY_GQA_PAIR32_SPLITK_B1_API_DECLARATION,
            gate=FIXED32_QUERY_GQA_PAIR32_SPLITK_B1_API_GATE,
            label=(
                "fixed32 FA2 qrow32 B1 GQA-pair split-K gate-only API dispatch"
            ),
        )
        changed = changed or did
        text, did = _replace_once(
            text,
            STOCK_VARLEN_SPLITKV_ALLOCATION,
            FIXED32_QUERY_GQA_PAIR32_SPLITK_B1_ALLOCATION,
            "fixed32 FA2 qrow32 B1 GQA-pair split-K scratch allocation",
        )
        changed = changed or did
    if changed:
        path.write_text(text)
    return changed


def _patch_torch_lib(path: Path) -> bool:
    text = path.read_text()
    changed = False
    decl = r'''
// FR13_FA2_TREE_BIAS
std::vector<at::Tensor>
mha_varlen_fwd_tree_bias(at::Tensor &q,
                         const at::Tensor &k,
                         const at::Tensor &v,
                         std::optional<at::Tensor> &out_,
                         const at::Tensor &cu_seqlens_q,
                         const at::Tensor &cu_seqlens_k,
                         std::optional<at::Tensor> &seqused_k,
                         std::optional<const at::Tensor> &leftpad_k_,
                         std::optional<at::Tensor> &block_table_,
                         std::optional<at::Tensor> &alibi_slopes_,
                         int max_seqlen_q,
                         const int max_seqlen_k,
                         const float p_dropout,
                         const float softmax_scale,
                         const bool zero_tensors,
                         bool is_causal,
                         int window_size_left,
                         int window_size_right,
                         const float softcap,
                         const bool return_softmax,
                         int num_splits,
                         std::optional<at::Tensor> &tree_bias_,
                         std::optional<at::Generator> gen_);

'''
    text, did = _insert_once(
        text,
        "std::vector<at::Tensor>\nmha_fwd_kvcache",
        decl,
        "tree bias declaration",
    )
    changed = changed or did
    schema = r'''    ops.def("varlen_fwd_tree_bias(Tensor! q, Tensor k, Tensor v, Tensor!? out, Tensor cu_seqlens_q, "
            "Tensor cu_seqlens_k, Tensor? seqused_k, Tensor? leftpad_k, Tensor? block_table, Tensor? alibi_slopes, "
            "int max_seqlen_q, int max_seqlen_k, float p_dropout, float softmax_scale, bool zero_tensors, "
            "bool is_causal, int window_size_left, int window_size_right, float softcap, bool return_softmax, "
            "int num_splits, Tensor? tree_bias, Generator? gen) -> Tensor[]");
    ops.impl("varlen_fwd_tree_bias", torch::kCUDA, make_pytorch_shim(&mha_varlen_fwd_tree_bias));

'''
    text, did = _insert_once(
        text,
        "    ops.def(\"fwd_kvcache",
        schema,
        "tree bias torch schema",
    )
    changed = changed or did
    if changed:
        path.write_text(text)
    return changed


def patch_fa2_source(
    fa2_src: Path,
    *,
    tree_bias_tile_earlyout: bool = False,
    fixed32_query_tile16: bool = False,
    fixed32_query_tile16_static_strides: bool = False,
    fixed32_query_tile32: bool = False,
    fixed32_query_gqa_pair32: bool = False,
    fixed32_query_tile32_b1: bool = False,
    fixed32_query_gqa_pair32_b1: bool = False,
    fixed32_query_gqa_pair32_splitk_b1: bool = False,
    fixed32_tree_visibility_mask: bool = False,
) -> dict[str, bool]:
    if fixed32_query_tile16_static_strides and not fixed32_query_tile16:
        raise ValueError(
            "fixed32 qrow16 static strides require --fixed32-query-tile16"
        )
    if fixed32_query_gqa_pair32_splitk_b1 and fixed32_tree_visibility_mask:
        raise ValueError(
            "the split-K arm has no tree-visibility variant"
        )
    qrow32_builds = sum(
        bool(value)
        for value in (
            fixed32_query_tile32,
            fixed32_query_gqa_pair32,
            fixed32_query_tile32_b1,
            fixed32_query_gqa_pair32_b1,
            fixed32_query_gqa_pair32_splitk_b1,
        )
    )
    # Every GQA-pair candidate needs the paired ((row, head), column) address
    # layout in the shared kernel header; only the trait geometry differs.
    gqa_pair_layout = bool(
        fixed32_query_gqa_pair32
        or fixed32_query_gqa_pair32_b1
        or fixed32_query_gqa_pair32_splitk_b1
    )
    if qrow32_builds > 1:
        raise ValueError("fixed32 qrow32 source builds are mutually exclusive")
    fixed32_query_tile32_any = bool(qrow32_builds)
    if fixed32_tree_visibility_mask and not fixed32_query_tile32_any:
        raise ValueError(
            "fixed32 tree visibility requires a private qrow32 kernel"
        )
    if fixed32_query_tile32_any and not tree_bias_tile_earlyout:
        raise ValueError(
            "fixed32 qrow32 requires --tree-bias-tile-earlyout in the same "
            "source patch invocation"
        )
    files = {
        "flash.h": fa2_src / "csrc/flash_attn/src/flash.h",
        "flash_fwd_kernel.h": fa2_src / "csrc/flash_attn/src/flash_fwd_kernel.h",
        "utils.h": fa2_src / "csrc/flash_attn/src/utils.h",
        "flash_fwd_split_hdim256_bf16_sm80.cu": fa2_src
        / "csrc/flash_attn/src/flash_fwd_split_hdim256_bf16_sm80.cu",
        "flash_api.cpp": fa2_src / "csrc/flash_attn/flash_api.cpp",
        "flash_api_torch_lib.cpp": fa2_src / "csrc/flash_attn/flash_api_torch_lib.cpp",
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing FA2 source files: " + ", ".join(missing))
    flash_fwd_kernel_changed = _patch_flash_fwd_kernel(
        files["flash_fwd_kernel.h"],
        tile_earlyout=tree_bias_tile_earlyout,
        fixed32_tree_visibility_mask=fixed32_tree_visibility_mask,
    )
    flash_fwd_kernel_changed = (
        _patch_fixed32_query_static_paged_path(
            files["flash_fwd_kernel.h"],
            fixed32_query_tile16=fixed32_query_tile16,
            fixed32_query_tile32=fixed32_query_tile32_any,
        )
        or flash_fwd_kernel_changed
    )
    flash_fwd_kernel_changed = (
        _patch_fixed32_query_tile16_static_kv_head_stride(
            files["flash_fwd_kernel.h"],
            fixed32_query_tile16_static_strides=(
                fixed32_query_tile16_static_strides
            ),
        )
        or flash_fwd_kernel_changed
    )
    flash_fwd_kernel_changed = (
        _patch_fixed32_query_tile32_static_query(
            files["flash_fwd_kernel.h"],
            fixed32_query_tile32=fixed32_query_tile32_any,
        )
        or flash_fwd_kernel_changed
    )
    flash_fwd_kernel_changed = (
        _patch_fixed32_query_tile32_static_batch_layout(
            files["flash_fwd_kernel.h"],
            fixed32_query_tile32=fixed32_query_tile32_any,
        )
        or flash_fwd_kernel_changed
    )
    flash_fwd_kernel_changed = (
        _patch_fixed32_query_tile32_static_paged_metadata(
            files["flash_fwd_kernel.h"],
            fixed32_query_tile32=fixed32_query_tile32_any,
        )
        or flash_fwd_kernel_changed
    )
    flash_fwd_kernel_changed = (
        _patch_fixed32_query_tile32_static_kv_strides(
            files["flash_fwd_kernel.h"],
            fixed32_query_tile32=fixed32_query_tile32_any,
        )
        or flash_fwd_kernel_changed
    )
    flash_fwd_kernel_changed = (
        _patch_fixed32_query_tile32_fused_initial_kv_page(
            files["flash_fwd_kernel.h"],
            fixed32_query_tile32=fixed32_query_tile32_any,
        )
        or flash_fwd_kernel_changed
    )
    flash_fwd_kernel_changed = (
        _patch_fixed32_query_tile32_carry_kv_page_address(
            files["flash_fwd_kernel.h"],
            fixed32_query_tile32=fixed32_query_tile32_any,
        )
        or flash_fwd_kernel_changed
    )
    flash_fwd_kernel_changed = (
        _patch_fixed32_query_gqa_pair_layout(
            files["flash_fwd_kernel.h"],
            fixed32_query_gqa_pair32=gqa_pair_layout,
        )
        or flash_fwd_kernel_changed
    )
    flash_fwd_kernel_changed = (
        _patch_fixed32_query_gqa_pair_splitk_layout(
            files["flash_fwd_kernel.h"],
            fixed32_query_gqa_pair32_splitk_b1=(
                fixed32_query_gqa_pair32_splitk_b1
            ),
        )
        or flash_fwd_kernel_changed
    )
    flash_fwd_kernel_changed = (
        _patch_fixed32_combine_static_geometry(
            files["flash_fwd_kernel.h"],
            fixed32_query_gqa_pair32_splitk_b1=(
                fixed32_query_gqa_pair32_splitk_b1
            ),
        )
        or flash_fwd_kernel_changed
    )
    b1_translation_units_changed = (
        _patch_fixed32_query_tile32_b1_translation_unit(
            files["flash_fwd_split_hdim256_bf16_sm80.cu"],
            fixed32_query_tile32_b1=fixed32_query_tile32_b1,
            fixed32_tree_visibility_mask=fixed32_tree_visibility_mask,
        )
    )
    return {
        "flash.h": _patch_flash_h(files["flash.h"]),
        "flash_fwd_kernel.h": flash_fwd_kernel_changed,
        "utils.h": _patch_fixed32_query_static_page(
            files["utils.h"],
            fixed32_query_tile16=fixed32_query_tile16,
            fixed32_query_tile32=fixed32_query_tile32_any,
        ),
        "flash_fwd_fr13_qrow16_hdim256_bf16_sm80.cu": _patch_fixed32_query_translation_unit(
            files["flash_fwd_split_hdim256_bf16_sm80.cu"],
            fixed32_query_tile16=fixed32_query_tile16,
            fixed32_query_tile16_static_strides=(
                fixed32_query_tile16_static_strides
            ),
        ),
        "flash_fwd_fr13_qrow32_hdim256_bf16_sm80.cu": _patch_fixed32_query_tile32_translation_unit(
            files["flash_fwd_split_hdim256_bf16_sm80.cu"],
            fixed32_query_tile32=fixed32_query_tile32,
            fixed32_tree_visibility_mask=fixed32_tree_visibility_mask,
        ),
        "flash_fwd_fr13_qrow32_gqa_pair_hdim256_bf16_sm80.cu": (
            _patch_fixed32_query_gqa_pair32_translation_unit(
                files["flash_fwd_split_hdim256_bf16_sm80.cu"],
                fixed32_query_gqa_pair32=fixed32_query_gqa_pair32,
                fixed32_tree_visibility_mask=fixed32_tree_visibility_mask,
            )
        ),
        "flash_fwd_fr13_qrow32_b1_hdim256_bf16_sm80.cu": (
            b1_translation_units_changed
        ),
        "flash_fwd_fr13_qrow32_b1_split2_hdim256_bf16_sm80.cu": (
            b1_translation_units_changed
        ),
        # The split-K binary carries the PROMOTED unit too, and deliberately.
        # Its API gate installs the promoted arm's dispatch so the
        # characterization can measure against it, which means the promoted
        # launcher must resolve at link time -- but the deeper reason is
        # evidential: a reference arm compiled from the same source with the
        # same flags into the same binary is the only way to know the baseline
        # is the served kernel and not a rebuild of it. The build script
        # asserts that unit's SASS digest against the SEALED kernel's pin, so
        # a baseline that drifted fails the build instead of quietly becoming
        # the thing every ULP number is measured from.
        "flash_fwd_fr13_qrow32_gqa_pair_b1_hdim256_bf16_sm80.cu": (
            _patch_fixed32_query_gqa_pair32_b1_translation_unit(
                files["flash_fwd_split_hdim256_bf16_sm80.cu"],
                fixed32_query_gqa_pair32_b1=(
                    fixed32_query_gqa_pair32_b1
                    or fixed32_query_gqa_pair32_splitk_b1
                ),
                fixed32_tree_visibility_mask=fixed32_tree_visibility_mask,
            )
        ),
        "flash_fwd_fr13_qrow32_gqa_pair_splitk_b1_hdim256_bf16_sm80.cu": (
            _patch_fixed32_query_gqa_pair32_splitk_b1_translation_unit(
                files["flash_fwd_split_hdim256_bf16_sm80.cu"],
                fixed32_query_gqa_pair32_splitk_b1=(
                    fixed32_query_gqa_pair32_splitk_b1
                ),
            )
        ),
        "flash_api.cpp": _patch_flash_api_cpp(
            files["flash_api.cpp"],
            fixed32_query_tile16=fixed32_query_tile16,
            fixed32_query_tile16_static_strides=(
                fixed32_query_tile16_static_strides
            ),
            fixed32_query_tile32=fixed32_query_tile32,
            fixed32_query_gqa_pair32=fixed32_query_gqa_pair32,
            fixed32_query_tile32_b1=fixed32_query_tile32_b1,
            fixed32_query_gqa_pair32_b1=fixed32_query_gqa_pair32_b1,
            fixed32_query_gqa_pair32_splitk_b1=(
                fixed32_query_gqa_pair32_splitk_b1
            ),
        ),
        "flash_api_torch_lib.cpp": _patch_torch_lib(files["flash_api_torch_lib.cpp"]),
    }


FR13_FA2_QROW32_B1_SPLIT2_INTERFACE_HELPER = r'''# FR13_FA2_QROW32_B1_SPLIT2_INTERFACE
# The private B1 routes that legitimately carry num_splits > 1, as
# (num_splits, tree-bias batch-stride tag) pairs. 1179791669 is the qrow32 B1
# split2 gate instrument; 1179791671 is the FR14 GQA-pair split-K arm, whose
# launcher and API gate both require exactly 4. Neither is in
# _FR13_FA2_QROW32_B1_PRODUCTION_ARMS, so neither can answer real traffic; this
# only stops the stock interface from refusing a split the fork does implement.
# The pairing matters: a sentinel is not enough on its own, because a split
# count the kernel was not compiled for would read the wrong accumulators.
_FR13_FA2_QROW32_B1_SPLIT_ROUTES = ((2, 1179791669), (4, 1179791671))


def _fr13_fa2_qrow32_b1_split2_interface_allowed(num_splits, tree_bias):
    if (
        tree_bias is None
        or not tree_bias.is_cuda
        or tree_bias.dtype != torch.float32
        or tuple(tree_bias.shape) != (1, 32, 32)
    ):
        return False
    stride = tuple(tree_bias.stride())
    return (
        stride[1:] == (32, 1)
        and (int(num_splits), int(stride[0])) in _FR13_FA2_QROW32_B1_SPLIT_ROUTES
    )


'''


def _patch_flash_attn_interface(path: Path) -> bool:
    text = path.read_text()
    changed = False
    text, did = _insert_once(
        text,
        "DEFAULT_FA_VERSION = 2\n",
        FR13_FA2_QROW32_B1_SPLIT2_INTERFACE_HELPER,
        "flash_attn_interface private B1 split2 guard",
    )
    changed = changed or did
    text, did = _replace_once(
        text,
        "    s_aux=None,\n    cp_world_size=1,\n",
        "    s_aux=None,\n    tree_bias=None,\n    cp_world_size=1,\n",
        "flash_attn_interface tree_bias parameter",
    )
    changed = changed or did
    old = """        out, softmax_lse = torch.ops._vllm_fa2_C.varlen_fwd(\n            q,\n            k,\n            v,\n            out,\n"""
    new = """        _fr13_fa2_op = (\n            torch.ops._vllm_fa2_C.varlen_fwd_tree_bias\n            if tree_bias is not None\n            else torch.ops._vllm_fa2_C.varlen_fwd\n        )\n        out, softmax_lse = _fr13_fa2_op(\n            q,\n            k,\n            v,\n            out,\n"""
    text, did = _replace_once(text, old, new, "flash_attn_interface choose tree op")
    changed = changed or did
    old_guard = '''        if num_splits > 1:
            raise NotImplementedError("FA2 does not support num_splits > 1")
'''
    new_guard = '''        if (
            num_splits > 1
            and not _fr13_fa2_qrow32_b1_split2_interface_allowed(
                num_splits, tree_bias
            )
        ):
            raise NotImplementedError("FA2 does not support num_splits > 1")
'''
    text, did = _replace_once(
        text,
        old_guard,
        new_guard,
        "flash_attn_interface private B1 split2 route",
    )
    changed = changed or did
    old_tail = """            return_softmax_lse and dropout_p > 0,\n            num_splits,\n            None,\n        )\n"""
    new_tail = """            return_softmax_lse and dropout_p > 0,\n            num_splits,\n            *(([tree_bias] if tree_bias is not None else [])),\n            None,\n        )\n"""
    text, did = _replace_once(text, old_tail, new_tail, "flash_attn_interface pass tree_bias")
    changed = changed or did
    if changed:
        path.write_text(text)
        py_compile.compile(path, doraise=True)
    return changed


FIXED32_QUERY_TILE16_LIVE_AB_HELPERS = r'''# FR13_FA2_QROW16_LIVE_PAGED_AB
_FR13_FA2_QROW16_LIVE_AB_GRAPHS = {}
_FR13_FA2_QROW16_LIVE_AB_ATTEMPTED = False
_FR13_FA2_QROW16_LIVE_AB_PASSED = False
_FR13_FA2_QROW16_BATCH_STRIDE_SENTINEL = 1179791667


def _fr13_fa2_qrow16_candidate_tree_bias(tree_bias):
    """Tag an exact B1 bias using its semantically inert batch stride."""
    if tree_bias.dtype != torch.float32:
        raise RuntimeError("FR13 qrow16 tree bias is not FP32")
    if tuple(tree_bias.shape) not in ((32, 32), (1, 32, 32)):
        raise RuntimeError("FR13 qrow16 tree bias shape drifted")
    base = tree_bias[0] if tree_bias.ndim == 3 else tree_bias
    if int(base.stride(-1)) != 1:
        raise RuntimeError("FR13 qrow16 tree bias columns are not contiguous")
    return torch.as_strided(
        base,
        size=(1, 32, 32),
        stride=(
            _FR13_FA2_QROW16_BATCH_STRIDE_SENTINEL,
            int(base.stride(-2)),
            1,
        ),
    )


def _fr13_fa2_qrow16_live_ab_register(
    *,
    layer,
    flash_fn,
    query,
    key_cache,
    value_cache,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    num_splits,
    tree_bias,
):
    """Retain the first final B1 graph's exact live paged operands."""
    if os.environ.get("FR13_FA2_QROW16_LIVE_PAGED_AB", "0") != "1":
        return
    if not (
        torch.cuda.is_available()
        and torch.cuda.is_current_stream_capturing()
    ):
        return
    from vllm.model_executor.layers.mamba import gdn_linear_attn as _fr13_qrow_gdn

    context = getattr(_fr13_qrow_gdn, "_FR13_FIXED32_CAPTURE_CONTEXT", None)
    if not isinstance(context, dict):
        # Memory-profile graphs deliberately have no final capture context.
        return
    descriptor = context.get("descriptor")
    if not isinstance(descriptor, dict) or int(descriptor.get("num_reqs", -1)) != 1:
        return
    graph_id = int(context.get("graph_id", 0))
    if graph_id <= 0 or graph_id in _FR13_FA2_QROW16_LIVE_AB_GRAPHS:
        return

    exact = (
        query.dtype == torch.bfloat16
        and tuple(query.shape) == (32, 24, 256)
        and key_cache.dtype == torch.bfloat16
        and value_cache.dtype == torch.bfloat16
        and tuple(key_cache.shape[1:]) == (1024, 4, 256)
        and tuple(value_cache.shape) == tuple(key_cache.shape)
        and cu_seqlens_q.dtype == torch.int32
        and tuple(cu_seqlens_q.shape) == (2,)
        and seqused_k.dtype == torch.int32
        and tuple(seqused_k.shape) == (1,)
        and block_table.dtype == torch.int32
        and block_table.ndim == 2
        and int(block_table.shape[0]) == 1
        and tree_bias.dtype == torch.float32
        and tuple(tree_bias.shape) in ((32, 32), (1, 32, 32))
        and int(max_seqlen_q) == 32
        and int(max_seqlen_k) > 0
        and not bool(causal)
        and int(num_splits) in (0, 1)
    )
    if not exact:
        raise RuntimeError("FR13 qrow16 live gate saw non-production B1 geometry")
    if window_size is not None and tuple(int(x) for x in window_size) != (-1, -1):
        raise RuntimeError("FR13 qrow16 live gate requires a full attention window")

    _FR13_FA2_QROW16_LIVE_AB_GRAPHS[graph_id] = {
        "layer_name": str(getattr(layer, "layer_name", "")),
        "flash_fn": flash_fn,
        "query": query,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "cu_seqlens_q": cu_seqlens_q,
        "max_seqlen_q": int(max_seqlen_q),
        "seqused_k": seqused_k,
        "max_seqlen_k": int(max_seqlen_k),
        "softmax_scale": float(softmax_scale),
        "causal": bool(causal),
        "window_size": None if window_size is None else list(window_size),
        "block_table": block_table,
        "softcap": float(softcap),
        "num_splits": int(num_splits),
        "tree_bias": tree_bias,
    }
    logger.info(
        "FR13 qrow16 live paged A/B registered graph=%d layer=%s",
        graph_id,
        _FR13_FA2_QROW16_LIVE_AB_GRAPHS[graph_id]["layer_name"],
    )


def _fr13_fa2_qrow16_live_ab_call(bundle, out, *, candidate=False):
    tree_bias = bundle["tree_bias"]
    if candidate:
        tree_bias = _fr13_fa2_qrow16_candidate_tree_bias(tree_bias)
    return bundle["flash_fn"](
        q=bundle["query"],
        k=bundle["key_cache"],
        v=bundle["value_cache"],
        out=out,
        cu_seqlens_q=bundle["cu_seqlens_q"],
        max_seqlen_q=bundle["max_seqlen_q"],
        seqused_k=bundle["seqused_k"],
        max_seqlen_k=bundle["max_seqlen_k"],
        softmax_scale=bundle["softmax_scale"],
        causal=bundle["causal"],
        alibi_slopes=None,
        window_size=bundle["window_size"],
        block_table=bundle["block_table"],
        softcap=bundle["softcap"],
        scheduler_metadata=None,
        fa_version=2,
        num_splits=bundle["num_splits"],
        s_aux=None,
        tree_bias=tree_bias,
        return_softmax_lse=True,
    )


def _fr13_fa2_qrow16_raw_bytes(tensor):
    return (
        tensor.detach()
        .contiguous()
        .view(torch.uint8)
        .cpu()
        .numpy()
        .tobytes()
    )


def _fr13_fa2_qrow16_live_ab_write(record):
    import json as _json
    from pathlib import Path as _Path

    path = _Path(
        os.environ.get(
            "FR13_FA2_QROW16_LIVE_PAGED_AB_JSON",
            "/logs/fr13_fa2_qrow16_live_paged_ab.json",
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        _json.dumps(record, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    temporary.replace(path)


def _fr13_fa2_qrow16_live_ab_replay(graph_id, runtime_mode, batch_size):
    """Run the one-shot byte gate after the first real stock B1 replay."""
    global _FR13_FA2_QROW16_LIVE_AB_ATTEMPTED
    global _FR13_FA2_QROW16_LIVE_AB_PASSED

    if os.environ.get("FR13_FA2_QROW16_LIVE_PAGED_AB", "0") != "1":
        return
    if _FR13_FA2_QROW16_LIVE_AB_ATTEMPTED:
        return
    if str(runtime_mode).upper() != "FULL" or int(batch_size) != 1:
        return
    from vllm.model_executor.layers.mamba import gdn_linear_attn as _fr13_qrow_gdn

    if not _fr13_qrow_gdn._fr13_fixed32_observed_event_active():
        return
    event = getattr(_fr13_qrow_gdn, "_FR13_FIXED32_OBSERVED_CURRENT", None)
    if not isinstance(event, dict) or int(event.get("batch_size", -1)) != 1:
        raise RuntimeError("FR13 qrow16 live gate has no exact B1 observed event")
    bundle = _FR13_FA2_QROW16_LIVE_AB_GRAPHS.get(int(graph_id))
    if not isinstance(bundle, dict):
        raise RuntimeError("FR13 qrow16 live gate has no operands for replayed graph")
    instance_id = os.environ.get("FR13_FA2_QROW16_LIVE_PAGED_AB_INSTANCE_ID", "")
    if not instance_id:
        raise RuntimeError("FR13 qrow16 live gate has no SWE-Verified instance id")
    candidate_so_sha256 = os.environ.get("FR13_FA2_QROW16_SO_SHA256", "")
    if (
        len(candidate_so_sha256) != 64
        or any(char not in "0123456789abcdef" for char in candidate_so_sha256)
    ):
        raise RuntimeError("FR13 qrow16 live gate has no candidate SO digest")
    draft_vocab_root = int(os.environ.get("FR13_DRAFT_VOCAB_ROOT", "0"))
    draft_vocab_k = int(os.environ.get("FR13_DRAFT_VOCAB_K", "0"))
    if draft_vocab_root != 1 or draft_vocab_k != 65536:
        raise RuntimeError("FR13 qrow16 live gate requires K64 ROOT=1")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("FR13 qrow16 live gate ran inside CUDA capture")
    _FR13_FA2_QROW16_LIVE_AB_ATTEMPTED = True
    # The retained query is produced inside the replay. Synchronizing here
    # makes its current real-task bytes, plus the live cache metadata, stable
    # before the two diagnostic recalls.
    torch.cuda.synchronize()
    q_start = [int(x) for x in bundle["cu_seqlens_q"].cpu().tolist()]
    seq_lens = [int(x) for x in bundle["seqused_k"].cpu().tolist()]
    if q_start != [0, 32] or len(seq_lens) != 1 or seq_lens[0] <= 0:
        raise RuntimeError("FR13 qrow16 live sequence metadata drifted")
    if seq_lens[0] > bundle["max_seqlen_k"]:
        raise RuntimeError("FR13 qrow16 live sequence exceeds max_seqlen_k")

    stock_buf = torch.empty_like(bundle["query"])
    candidate_buf = torch.empty_like(bundle["query"])
    stock_out, stock_lse = _fr13_fa2_qrow16_live_ab_call(bundle, stock_buf)
    candidate_out, candidate_lse = _fr13_fa2_qrow16_live_ab_call(
        bundle, candidate_buf, candidate=True
    )
    torch.cuda.synchronize()
    if stock_lse.dtype != torch.float32 or candidate_lse.dtype != torch.float32:
        raise RuntimeError("FR13 qrow16 live gate LSE is not FP32")

    stock_output_bytes = _fr13_fa2_qrow16_raw_bytes(stock_out)
    candidate_output_bytes = _fr13_fa2_qrow16_raw_bytes(candidate_out)
    stock_lse_bytes = _fr13_fa2_qrow16_raw_bytes(stock_lse)
    candidate_lse_bytes = _fr13_fa2_qrow16_raw_bytes(candidate_lse)
    output_mismatches = (
        abs(len(stock_output_bytes) - len(candidate_output_bytes))
        + sum(
            left != right
            for left, right in zip(stock_output_bytes, candidate_output_bytes)
        )
    )
    lse_mismatches = (
        abs(len(stock_lse_bytes) - len(candidate_lse_bytes))
        + sum(
            left != right
            for left, right in zip(stock_lse_bytes, candidate_lse_bytes)
        )
    )
    passed = output_mismatches == 0 and lse_mismatches == 0
    import hashlib as _hashlib

    record = {
        "schema": "fr13.fixed32.fa2_qrow16_live_paged_ab.v1",
        "status": "PASS" if passed else "FAIL",
        "suite": "SWE-Verified",
        "instance_id": instance_id,
        "concurrency": 1,
        "physical_rows": 32,
        "draft_vocab_root": draft_vocab_root,
        "draft_vocab_k": draft_vocab_k,
        "candidate_so_sha256": candidate_so_sha256,
        "graph_id": int(graph_id),
        "runtime_mode": str(runtime_mode).upper(),
        "layer_name": bundle["layer_name"],
        "operands": {
            "query_shape": list(bundle["query"].shape),
            "key_cache_shape": list(bundle["key_cache"].shape),
            "value_cache_shape": list(bundle["value_cache"].shape),
            "block_table_shape": list(bundle["block_table"].shape),
            "query_start_loc": q_start,
            "seq_lens": seq_lens,
            "max_seqlen_k": bundle["max_seqlen_k"],
            "tree_bias_shape": list(bundle["tree_bias"].shape),
        },
        "output": {
            "dtype": str(stock_out.dtype),
            "bytes": len(stock_output_bytes),
            "raw_byte_mismatches": output_mismatches,
            "stock_sha256": _hashlib.sha256(stock_output_bytes).hexdigest(),
            "candidate_sha256": _hashlib.sha256(candidate_output_bytes).hexdigest(),
        },
        "lse": {
            "dtype": str(stock_lse.dtype),
            "bytes": len(stock_lse_bytes),
            "raw_byte_mismatches": lse_mismatches,
            "stock_sha256": _hashlib.sha256(stock_lse_bytes).hexdigest(),
            "candidate_sha256": _hashlib.sha256(candidate_lse_bytes).hexdigest(),
        },
        "candidate_dispatch": "qrow16 internal exact-geometry require",
        "served_return": "stock captured graph output unchanged",
        "performance_measurement": False,
    }
    _fr13_fa2_qrow16_live_ab_write(record)
    if not passed:
        raise RuntimeError(
            "FR13 qrow16 live paged byte A/B mismatch: "
            f"output_bytes={output_mismatches} lse_bytes={lse_mismatches}"
        )
    _FR13_FA2_QROW16_LIVE_AB_PASSED = True
    logger.warning(
        "[FR13_FA2_QROW16_LIVE_PAGED_AB] PASS instance=%s layer=%s "
        "output_byte_mismatches=0 lse_byte_mismatches=0 stock_served=1",
        instance_id,
        bundle["layer_name"],
    )


'''


FIXED32_QUERY_TILE32_LIVE_AB_HELPERS = r'''# FR13_FA2_QROW32_LIVE_PAGED_AB
_FR13_FA2_QROW32_LIVE_AB_GRAPHS = {}
_FR13_FA2_QROW32_LIVE_AB_ATTEMPTED = False
_FR13_FA2_QROW32_LIVE_AB_PASSED = False
# FR13_FA2_QROW32_B34_PADDED (Mark's ruling 2026-08-13). The gate qualifies
# TWO operating points now: the native canonical width 4, and width 3 served
# by PADDING to the canonical (b == 4, total_q == 128) geometry with one inert
# shadow request in slot 3. These constants are declared here rather than
# imported from the B4 production block because the two helper blocks are
# installed into tree_attn.py by MUTUALLY EXCLUSIVE patcher flags -- the gate
# build never has the production block, and a cross-block name would be a
# NameError at the first width-3 replay.
_FR13_FA2_QROW32_LIVE_AB_WIDTHS = (3, 4)
_FR13_FA2_QROW32_LIVE_AB_CANONICAL_WIDTH = 4
_FR13_FA2_QROW32_LIVE_AB_ROWS = 32
_FR13_FA2_QROW32_LIVE_AB_CANONICAL_ROWS = 128
_FR13_FA2_QROW32_LIVE_AB_NULL_BLOCK_ID = 0
# A page index that CANNOT exist. If the kernel ever dereferences the shadow
# row -- i.e. if the n_block_min >= n_block_max early exit at
# flash_fwd_kernel.h:759 is not taken -- this faults instead of silently
# reading somebody else's KV page. Loud is the entire point.
_FR13_FA2_QROW32_LIVE_AB_POISON_BLOCK_ID = 0x7FFFFFF0
_FR13_FA2_QROW32_LIVE_AB_STAGING = {}
# Once per qualified width, not once per process: at concurrency 4 the width-4
# graph replays first and the width-3 graph follows as tasks stall or finish,
# and BOTH have to be gated.
_FR13_FA2_QROW32_LIVE_AB_WIDTHS_ATTEMPTED = set()
_FR13_FA2_QROW32_LIVE_AB_WIDTHS_PASSED = set()
_FR13_FA2_QROW32_BATCH_STRIDE_SENTINEL = 131092
_FR13_FA2_QROW32_LIVE_AB_ARMS = {
    "qrow32": {
        "sentinel": _FR13_FA2_QROW32_BATCH_STRIDE_SENTINEL,
        "num_splits": 0,
        "candidate_dispatch": "qrow32 BM32 exact B4 geometry; no fallback",
        "candidate_so_sha256": (
            "77f3fb22c19d0eb2ac0ec28230cf9401221425692a505efde62aa838760d81ce"
        ),
        "candidate_so_size": 299876120,
        "fa2_head": "29210221863736a08f71a866459e368ad1ac4a95",
        "fa2_source_closure_sha256": (
            "dd3bebd047b8ccc2248b0d0e75b9db1f23747c486592ec2a5c72ee96581e10dc"
        ),
    },
    "gqa_pair": {
        "sentinel": 131092,
        "num_splits": 0,
        "candidate_dispatch": "qrow32 GQA-pair exact geometry; no fallback",
        "candidate_so_sha256": (
            "af9e9f24335db899468032f5b5a3eba100febe294932533cb9b87163ce2b3fdb"
        ),
        "candidate_so_size": 299813360,
        "fa2_head": "29210221863736a08f71a866459e368ad1ac4a95",
        "fa2_source_closure_sha256": (
            "9c3f9e751da7b783e9d07d8e40d5bc2234b99e719a1048668bd6c82244ed2d81"
        ),
    },
    "visibility": {
        "sentinel": 131092,
        "num_splits": 0,
        "candidate_dispatch": (
            "qrow32 fixed32 visibility-mask exact B4 geometry; no fallback"
        ),
        "candidate_so_sha256": (
            "805635d6881dbf73287d66c10541880b7cf93bcb6bf7b04e50efd3d32728b0aa"
        ),
        "candidate_so_size": 299810632,
        "fa2_head": "29210221863736a08f71a866459e368ad1ac4a95",
        "fa2_source_closure_sha256": (
            "1dac8f7fd910a564c5c3b792770029f0013e2df48c25c89376e4d5e7da949ced"
        ),
    },
}
_FR13_FA2_QROW32_TARGET_LAYERS = tuple(
    f"language_model.model.layers.{index}.self_attn.attn"
    for index in range(3, 64, 4)
)
_FR13_FA2_QROW32_CANONICAL_TASK_IDS = (
    "astropy__astropy-12907",
    "astropy__astropy-13033",
    "astropy__astropy-13236",
    "astropy__astropy-13398",
)
_FR13_FA2_QROW32_EXACT4_SUBSET_SHA256 = (
    "0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5"
)


def _fr13_fa2_qrow32_live_ab_contract():
    arm = os.environ.get("FR13_FA2_QROW32_LIVE_PAGED_AB_ARM", "")
    contract = _FR13_FA2_QROW32_LIVE_AB_ARMS.get(arm)
    if contract is None:
        raise RuntimeError("FR13 qrow32 live gate arm is not explicit")
    return arm, contract


def _fr13_fa2_qrow32_candidate_tree_bias(tree_bias):
    """Copy the live mask into the private, semantically exact B4 layout.

    Always four planes, because the tagged batch stride IS the dispatch key
    and the sealed .so serves exactly one grid. At width 3 the incoming mask
    is either the (32,32) broadcast tile -- which expands to four identical
    planes exactly as at width 4 -- or a (3,32,32) per-slot mask, whose fourth
    plane is filled from plane 0. The shadow never reads it (seqused_k == 0
    exits before the mask is touched); filling it deterministically rather
    than leaving it uninitialised is the fail-closed choice.
    """
    _, contract = _fr13_fa2_qrow32_live_ab_contract()
    sentinel = int(contract["sentinel"])
    canonical = _FR13_FA2_QROW32_LIVE_AB_CANONICAL_WIDTH
    if tree_bias.dtype != torch.float32:
        raise RuntimeError("FR13 qrow32 tree bias is not FP32")
    if tuple(tree_bias.shape) not in (
        (32, 32),
        (canonical, 32, 32),
    ) + tuple((width, 32, 32) for width in _FR13_FA2_QROW32_LIVE_AB_WIDTHS):
        raise RuntimeError("FR13 qrow32 tree bias shape drifted")
    if int(tree_bias.stride(-1)) != 1:
        raise RuntimeError("FR13 qrow32 tree bias columns are not contiguous")
    if tree_bias.ndim == 2:
        source = tree_bias.unsqueeze(0).expand(canonical, -1, -1)
    elif int(tree_bias.shape[0]) == canonical:
        source = tree_bias
    else:
        source = torch.cat(
            (
                tree_bias,
                tree_bias[:1].expand(
                    canonical - int(tree_bias.shape[0]), -1, -1
                ),
            ),
            dim=0,
        )
    tagged = torch.empty_strided(
        (4, 32, 32),
        (sentinel, 32, 1),
        dtype=tree_bias.dtype,
        device=tree_bias.device,
    )
    tagged.copy_(source)
    if tuple(tagged.stride()) != (
        sentinel,
        32,
        1,
    ):
        raise RuntimeError("FR13 qrow32 selector stride was not preserved")
    return tagged


def _fr13_fa2_qrow32_live_ab_register(
    *,
    layer,
    flash_fn,
    query,
    key_cache,
    value_cache,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    num_splits,
    tree_bias,
):
    """Retain every target layer from a qualified exact4 FULL graph.

    Qualified means width 4 (the canonical geometry) or width 3 (served by
    padding to it). The registration is width-parameterised so the gate can
    compare a NATIVE width-3 stock call against the PADDED width-4 candidate
    call on exactly the operands the runtime captured.
    """
    if os.environ.get("FR13_FA2_QROW32_LIVE_PAGED_AB", "0") != "1":
        return
    if not (
        torch.cuda.is_available()
        and torch.cuda.is_current_stream_capturing()
    ):
        return
    from vllm.model_executor.layers.mamba import gdn_linear_attn as _fr13_qrow_gdn

    context = getattr(_fr13_qrow_gdn, "_FR13_FIXED32_CAPTURE_CONTEXT", None)
    if not isinstance(context, dict):
        # Memory-profile graphs deliberately have no final capture context.
        return
    descriptor = context.get("descriptor")
    if not isinstance(descriptor, dict):
        return
    width = int(descriptor.get("num_reqs", -1))
    if width not in _FR13_FA2_QROW32_LIVE_AB_WIDTHS:
        return
    rows = _FR13_FA2_QROW32_LIVE_AB_ROWS * width
    graph_id = int(context.get("graph_id", 0))
    if graph_id <= 0:
        raise RuntimeError("FR13 qrow32 live gate has no final graph identity")
    layer_name = str(getattr(layer, "layer_name", ""))
    if layer_name not in _FR13_FA2_QROW32_TARGET_LAYERS:
        raise RuntimeError("FR13 qrow32 live gate reached a non-target layer")

    _, candidate_contract = _fr13_fa2_qrow32_live_ab_contract()
    # Each predicate carries its observed value so a drift names itself. A bare
    # conjunction here is undiagnosable: it cannot distinguish a genuinely wrong
    # engine shape from a dummy-run tensor that merely has not been filled yet.
    _checks = (
        ("query.dtype", query.dtype == torch.bfloat16, query.dtype),
        ("query.shape", tuple(query.shape) == (rows, 24, 256), tuple(query.shape)),
        ("query.stride(-1)", int(query.stride(-1)) == 1, int(query.stride(-1))),
        ("query.stride(-2)", int(query.stride(-2)) == 256, int(query.stride(-2))),
        ("key_cache.dtype", key_cache.dtype == torch.bfloat16, key_cache.dtype),
        ("value_cache.dtype", value_cache.dtype == torch.bfloat16, value_cache.dtype),
        (
            "key_cache.shape[1:]",
            tuple(key_cache.shape[1:]) == (1024, 4, 256),
            tuple(key_cache.shape),
        ),
        (
            "value_cache.shape",
            tuple(value_cache.shape) == tuple(key_cache.shape),
            tuple(value_cache.shape),
        ),
        (
            "key_cache.stride",
            tuple(key_cache.stride()) == (2 * 1024 * 4 * 256, 4 * 256, 256, 1),
            tuple(key_cache.stride()),
        ),
        (
            "value_cache.stride",
            tuple(value_cache.stride()) == tuple(key_cache.stride()),
            tuple(value_cache.stride()),
        ),
        ("cu_seqlens_q.dtype", cu_seqlens_q.dtype == torch.int32, cu_seqlens_q.dtype),
        (
            "cu_seqlens_q.shape",
            tuple(cu_seqlens_q.shape) == (width + 1,),
            tuple(cu_seqlens_q.shape),
        ),
        ("seqused_k.dtype", seqused_k.dtype == torch.int32, seqused_k.dtype),
        (
            "seqused_k.shape",
            tuple(seqused_k.shape) == (width,),
            tuple(seqused_k.shape),
        ),
        ("block_table.dtype", block_table.dtype == torch.int32, block_table.dtype),
        ("block_table.ndim", block_table.ndim == 2, block_table.ndim),
        (
            "block_table.shape[0]",
            int(block_table.shape[0]) == width,
            tuple(block_table.shape),
        ),
        ("tree_bias.dtype", tree_bias.dtype == torch.float32, tree_bias.dtype),
        (
            "tree_bias.shape",
            tuple(tree_bias.shape) in ((32, 32), (width, 32, 32)),
            tuple(tree_bias.shape),
        ),
        (
            "tree_bias.stride(-1)",
            int(tree_bias.stride(-1)) == 1,
            int(tree_bias.stride(-1)),
        ),
        ("max_seqlen_q", int(max_seqlen_q) == 32, int(max_seqlen_q)),
        ("max_seqlen_k", int(max_seqlen_k) > 0, int(max_seqlen_k)),
        ("causal", not bool(causal), bool(causal)),
        ("softcap", float(softcap) == 0.0, float(softcap)),
        (
            "num_splits",
            int(num_splits) == int(candidate_contract["num_splits"]),
            int(num_splits),
        ),
    )
    _failed = [(name, observed) for name, ok, observed in _checks if not ok]
    if _failed:
        raise RuntimeError(
            "FR13 qrow32 live gate saw non-canonical B4 geometry: "
            + "; ".join(f"{name}={observed!r}" for name, observed in _failed)
        )
    if window_size is not None and tuple(int(x) for x in window_size) != (-1, -1):
        raise RuntimeError("FR13 qrow32 live gate requires a full attention window")

    graph = _FR13_FA2_QROW32_LIVE_AB_GRAPHS.setdefault(graph_id, {})
    if layer_name in graph:
        raise RuntimeError("FR13 qrow32 target layer executed twice in capture")
    graph[layer_name] = {
        "batch_size": width,
        "layer_name": layer_name,
        "flash_fn": flash_fn,
        "query": query,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "cu_seqlens_q": cu_seqlens_q,
        "max_seqlen_q": int(max_seqlen_q),
        "seqused_k": seqused_k,
        "max_seqlen_k": int(max_seqlen_k),
        "softmax_scale": float(softmax_scale),
        "causal": bool(causal),
        "window_size": None if window_size is None else list(window_size),
        "block_table": block_table,
        "softcap": float(softcap),
        "num_splits": int(num_splits),
        "tree_bias": tree_bias,
    }
    logger.info(
        "FR13 qrow32 live paged A/B registered graph=%d layer=%s count=%d",
        graph_id,
        layer_name,
        len(graph),
    )


def _fr13_fa2_qrow32_live_ab_call(bundle, out, *, candidate=False):
    tree_bias = bundle["tree_bias"]
    if candidate:
        tree_bias = _fr13_fa2_qrow32_candidate_tree_bias(tree_bias)
    return bundle["flash_fn"](
        q=bundle["query"],
        k=bundle["key_cache"],
        v=bundle["value_cache"],
        out=out,
        cu_seqlens_q=bundle["cu_seqlens_q"],
        max_seqlen_q=bundle["max_seqlen_q"],
        seqused_k=bundle["seqused_k"],
        max_seqlen_k=bundle["max_seqlen_k"],
        softmax_scale=bundle["softmax_scale"],
        causal=bundle["causal"],
        alibi_slopes=None,
        window_size=bundle["window_size"],
        block_table=bundle["block_table"],
        softcap=bundle["softcap"],
        scheduler_metadata=None,
        fa_version=2,
        num_splits=bundle["num_splits"],
        s_aux=None,
        tree_bias=tree_bias,
        return_softmax_lse=True,
    )


def _fr13_fa2_qrow32_live_ab_staging(device, block_columns):
    """The gate's own copy of the production staging, same shapes, same rules.

    The gate runs OUTSIDE capture (it asserts so before it starts), so this
    allocation is unconditionally safe here. It is deliberately a separate
    registry from the production one: the gate build does not install the
    production helper block at all, and a gate that borrowed production state
    would be testing its own bookkeeping instead of the kernel.
    """
    key = (str(device), int(block_columns))
    staged = _FR13_FA2_QROW32_LIVE_AB_STAGING.get(key)
    if staged is not None:
        return staged
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "FR13 qrow32 live gate staging must not be allocated in capture"
        )
    canonical = _FR13_FA2_QROW32_LIVE_AB_CANONICAL_ROWS
    width = _FR13_FA2_QROW32_LIVE_AB_CANONICAL_WIDTH
    rows = _FR13_FA2_QROW32_LIVE_AB_ROWS
    staged = {
        "query": torch.zeros(
            (canonical, 24, 256), dtype=torch.bfloat16, device=device
        ),
        "out": torch.zeros(
            (canonical, 24, 256), dtype=torch.bfloat16, device=device
        ),
        "cu_seqlens_q": torch.arange(
            0, canonical + rows, rows, dtype=torch.int32, device=device
        ),
        "seqused_k": torch.zeros((width,), dtype=torch.int32, device=device),
        "block_table": torch.full(
            (width, int(block_columns)),
            _FR13_FA2_QROW32_LIVE_AB_NULL_BLOCK_ID,
            dtype=torch.int32,
            device=device,
        ),
    }
    _FR13_FA2_QROW32_LIVE_AB_STAGING[key] = staged
    return staged


def _fr13_fa2_qrow32_live_ab_padded_call(bundle, *, poison):
    """Serve the width-3 bundle to the sealed .so as canonical width 4.

    Returns (staged_out, staged_lse) at the FULL canonical extent, so the
    caller can check BOTH halves: the real rows against the native stock call,
    and the shadow rows against the kernel's early-return contract (zeros in
    O, +INF in LSE).

    With poison=True the shadow's Q rows are filled with a NaN/garbage bit
    pattern AND the shadow block-table row is pointed at a page index that
    cannot exist. Under seqused_k[shadow] == 0 the kernel reads neither, so
    the real rows must come back BIT-IDENTICAL to the clean-shadow run. If the
    early exit is ever not taken, this either faults on the impossible page or
    poisons the output -- both loud, neither silent.
    """
    width = int(bundle["batch_size"])
    canonical_width = _FR13_FA2_QROW32_LIVE_AB_CANONICAL_WIDTH
    rows = _FR13_FA2_QROW32_LIVE_AB_ROWS
    real_rows = width * rows
    shadow = canonical_width - 1
    if width == canonical_width:
        raise RuntimeError("FR13 qrow32 live gate padded the canonical width")
    block_table = bundle["block_table"]
    staged = _fr13_fa2_qrow32_live_ab_staging(
        bundle["query"].device, int(block_table.shape[1])
    )
    staged["query"].zero_()
    staged["query"][:real_rows].copy_(bundle["query"])
    staged["seqused_k"].zero_()
    staged["seqused_k"][:width].copy_(bundle["seqused_k"])
    staged["block_table"].fill_(_FR13_FA2_QROW32_LIVE_AB_NULL_BLOCK_ID)
    staged["block_table"][:width].copy_(block_table)
    if poison:
        staged["query"][real_rows:] = float("nan")
        staged["block_table"][shadow].fill_(
            _FR13_FA2_QROW32_LIVE_AB_POISON_BLOCK_ID
        )
    if int(staged["seqused_k"][shadow].item()) != 0:
        raise RuntimeError("FR13 qrow32 live gate shadow seqused_k is not zero")
    staged["out"].zero_()
    padded = dict(bundle)
    padded["query"] = staged["query"]
    padded["cu_seqlens_q"] = staged["cu_seqlens_q"]
    padded["seqused_k"] = staged["seqused_k"]
    padded["block_table"] = staged["block_table"]
    out, lse = _fr13_fa2_qrow32_live_ab_call(
        padded, staged["out"], candidate=True
    )
    if tuple(out.shape) != (
        _FR13_FA2_QROW32_LIVE_AB_CANONICAL_ROWS,
        24,
        256,
    ):
        raise RuntimeError("FR13 qrow32 live gate padded output extent drifted")
    return out, lse


def _fr13_fa2_qrow32_live_ab_shadow_mismatches(out, lse, real_rows):
    """The kernel's early-return contract, stated as bytes.

    compute_attn_1rowblock_splitkv at n_block_min >= n_block_max writes ZERO
    to every O element it owns and +INF to every LSE entry, then returns --
    before Q is read and before the block table is formed. Anything else in
    the shadow half means the shadow was not inert.
    """
    canonical = _FR13_FA2_QROW32_LIVE_AB_CANONICAL_ROWS
    shadow_out = out[real_rows:canonical]
    shadow_lse = lse[..., real_rows:canonical]
    failures = []
    if int(shadow_out.numel()) == 0 or int(shadow_lse.numel()) == 0:
        failures.append("shadow_extent=empty")
    if not bool((shadow_out.to(torch.float32) == 0.0).all().item()):
        failures.append("shadow_output_not_zero")
    if not bool(torch.isinf(shadow_lse).all().item()):
        failures.append("shadow_lse_not_infinite")
    elif not bool((shadow_lse > 0).all().item()):
        failures.append("shadow_lse_not_positive_infinity")
    return tuple(failures)


def _fr13_fa2_qrow32_raw_bytes(tensor):
    return tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()


def _fr13_fa2_qrow32_byte_summary(stock, candidate):
    import hashlib as _hashlib

    if stock.dtype != candidate.dtype or tuple(stock.shape) != tuple(candidate.shape):
        raise RuntimeError("FR13 qrow32 comparison tensor contract drifted")
    stock_bytes = _fr13_fa2_qrow32_raw_bytes(stock)
    candidate_bytes = _fr13_fa2_qrow32_raw_bytes(candidate)
    mismatches = abs(len(stock_bytes) - len(candidate_bytes)) + sum(
        left != right
        for left, right in zip(stock_bytes, candidate_bytes)
    )
    return {
        "dtype": str(stock.dtype),
        "shape": list(stock.shape),
        "bytes": len(stock_bytes),
        "raw_byte_mismatches": mismatches,
        "stock_sha256": _hashlib.sha256(stock_bytes).hexdigest(),
        "candidate_sha256": _hashlib.sha256(candidate_bytes).hexdigest(),
    }


def _fr13_fa2_qrow32_live_ab_write(record, batch_size=None):
    import json as _json
    from pathlib import Path as _Path

    path = _Path(
        os.environ.get(
            "FR13_FA2_QROW32_LIVE_PAGED_AB_JSON",
            "/logs/fr13_fa2_qrow32_live_paged_ab.json",
        )
    )
    if (
        batch_size is not None
        and int(batch_size) != _FR13_FA2_QROW32_LIVE_AB_CANONICAL_WIDTH
    ):
        # The canonical width keeps the original filename -- every banked
        # verifier reads exactly it. A padded width lands beside it, so the
        # two gate results can never overwrite one another and no new
        # environment name enters the runner contract.
        path = path.with_name(
            path.stem + "_b" + str(int(batch_size)) + path.suffix
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        _json.dumps(record, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    temporary.replace(path)


def _fr13_fa2_qrow32_live_ab_replay(graph_id, runtime_mode, batch_size):
    """Run the all-layer byte gate after the first real stock replay.

    Once per QUALIFIED WIDTH. At width 4 this is the sealed exact4 comparison,
    clause for clause as before. At width 3 the incumbent is the NATIVE
    width-3 stock call and the candidate is the PADDED canonical width-4 call,
    plus a poisoned-shadow repeat that must be bit-identical to the clean one.
    """
    global _FR13_FA2_QROW32_LIVE_AB_ATTEMPTED
    global _FR13_FA2_QROW32_LIVE_AB_PASSED

    if os.environ.get("FR13_FA2_QROW32_LIVE_PAGED_AB", "0") != "1":
        return
    if (
        str(runtime_mode).upper() != "FULL"
        or int(batch_size) not in _FR13_FA2_QROW32_LIVE_AB_WIDTHS
    ):
        return
    width = int(batch_size)
    if width in _FR13_FA2_QROW32_LIVE_AB_WIDTHS_ATTEMPTED:
        return
    padded = width != _FR13_FA2_QROW32_LIVE_AB_CANONICAL_WIDTH
    if padded and os.environ.get(
        "FR13_FA2_QROW32_LIVE_PAGED_AB_B3", "0"
    ) != "1":
        # The width-3 padded arm is default-off and separately declared: it
        # presents a synthetic 4th request to a sealed kernel, and that is a
        # thing a runner asks for explicitly or not at all.
        return
    from vllm.model_executor.layers.mamba import gdn_linear_attn as _fr13_qrow_gdn

    if not _fr13_qrow_gdn._fr13_fixed32_observed_event_active():
        return
    event = getattr(_fr13_qrow_gdn, "_FR13_FIXED32_OBSERVED_CURRENT", None)
    if not isinstance(event, dict) or int(
        event.get("batch_size", -1)
    ) != int(batch_size):
        raise RuntimeError("FR13 qrow32 live gate has no exact b3/b4 observed event")
    graph = _FR13_FA2_QROW32_LIVE_AB_GRAPHS.get(int(graph_id))
    if not isinstance(graph, dict):
        raise RuntimeError("FR13 qrow32 live gate has no retained graph operands")
    if set(graph) != set(_FR13_FA2_QROW32_TARGET_LAYERS):
        raise RuntimeError(
            "FR13 qrow32 live gate did not retain all 16 target tree layers"
        )
    task_ids = tuple(
        item
        for item in os.environ.get(
            "FR13_FA2_QROW32_LIVE_PAGED_AB_TASK_IDS", ""
        ).split(",")
        if item
    )
    if task_ids != _FR13_FA2_QROW32_CANONICAL_TASK_IDS:
        raise RuntimeError("FR13 qrow32 live gate task identity is not canonical exact4")
    if (
        os.environ.get("FR13_FA2_QROW32_LIVE_PAGED_AB_SUBSET_SHA256", "")
        != _FR13_FA2_QROW32_EXACT4_SUBSET_SHA256
    ):
        raise RuntimeError("FR13 qrow32 live gate subset digest drifted")
    fixed32_mode = os.environ.get("FR13_FIXED32_MODE", "")
    if fixed32_mode not in ("tail6_fixed32", "hydra27_fixed32"):
        raise RuntimeError("FR13 qrow32 live gate topology mode drifted")
    candidate_so_sha256 = os.environ.get("FR13_FA2_QROW32_SO_SHA256", "")
    if (
        len(candidate_so_sha256) != 64
        or any(char not in "0123456789abcdef" for char in candidate_so_sha256)
    ):
        raise RuntimeError("FR13 qrow32 live gate has no candidate SO digest")
    candidate_arm, candidate_contract = _fr13_fa2_qrow32_live_ab_contract()
    if candidate_arm == "visibility" and fixed32_mode != "hydra27_fixed32":
        raise RuntimeError("FR13 qrow32 visibility gate requires Hydra27")
    candidate_so_size_raw = os.environ.get("FR13_FA2_QROW32_SO_SIZE", "")
    try:
        candidate_so_size = int(candidate_so_size_raw)
    except ValueError as error:
        raise RuntimeError("FR13 qrow32 live gate has no candidate SO size") from error
    fa2_head = os.environ.get("FR13_FA2_QROW32_FA2_HEAD", "")
    fa2_source_closure_sha256 = os.environ.get(
        "FR13_FA2_QROW32_SOURCE_CLOSURE_SHA256", ""
    )
    if (
        candidate_so_sha256 != candidate_contract["candidate_so_sha256"]
        or candidate_so_size != candidate_contract["candidate_so_size"]
        or fa2_head != candidate_contract["fa2_head"]
        or fa2_source_closure_sha256
        != candidate_contract["fa2_source_closure_sha256"]
    ):
        raise RuntimeError("FR13 qrow32 binary/source provenance drifted")
    if (
        os.environ.get("FR13_DRAFT_VOCAB_K", "") != "65536"
        or os.environ.get("FR13_DRAFT_VOCAB_ROOT", "") != "1"
    ):
        raise RuntimeError("FR13 qrow32 live gate requires K64/root1")
    source_commit = os.environ.get("FR13_FA2_QROW32_SOURCE_COMMIT", "")
    if (
        len(source_commit) != 40
        or any(char not in "0123456789abcdef" for char in source_commit)
    ):
        raise RuntimeError("FR13 qrow32 live gate has no source commit")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("FR13 qrow32 live gate ran inside CUDA capture")

    _FR13_FA2_QROW32_LIVE_AB_WIDTHS_ATTEMPTED.add(width)
    _FR13_FA2_QROW32_LIVE_AB_ATTEMPTED = True
    torch.cuda.synchronize()
    rows = _FR13_FA2_QROW32_LIVE_AB_ROWS
    real_rows = width * rows
    expected_q_start = list(range(0, real_rows + rows, rows))
    layer_records = []
    total_output_mismatches = 0
    total_lse_mismatches = 0
    total_poison_output_mismatches = 0
    total_poison_lse_mismatches = 0
    shadow_failures = []
    tree_bias_sha256 = set()
    shared_q_start = None
    shared_seq_lens = None

    import hashlib as _hashlib

    for layer_name in _FR13_FA2_QROW32_TARGET_LAYERS:
        bundle = graph[layer_name]
        if int(bundle.get("batch_size", -1)) != width:
            raise RuntimeError(
                "FR13 qrow32 live gate registered width disagrees with replay"
            )
        q_start = [int(x) for x in bundle["cu_seqlens_q"].cpu().tolist()]
        seq_lens = [int(x) for x in bundle["seqused_k"].cpu().tolist()]
        if q_start != expected_q_start:
            raise RuntimeError("FR13 qrow32 live query segments drifted")
        if (
            len(seq_lens) != width
            or any(length < 32 for length in seq_lens)
            or any(length > bundle["max_seqlen_k"] for length in seq_lens)
        ):
            raise RuntimeError("FR13 qrow32 live sequence metadata drifted")
        if shared_q_start is None:
            shared_q_start = q_start
            shared_seq_lens = seq_lens
        elif q_start != shared_q_start or seq_lens != shared_seq_lens:
            raise RuntimeError("FR13 qrow32 live metadata differs across layers")

        bias_bytes = _fr13_fa2_qrow32_raw_bytes(bundle["tree_bias"])
        bias_sha256 = _hashlib.sha256(bias_bytes).hexdigest()
        tree_bias_sha256.add(bias_sha256)
        stock_buf = torch.empty_like(bundle["query"])
        stock_out, stock_lse = _fr13_fa2_qrow32_live_ab_call(bundle, stock_buf)
        poison_summary = None
        if padded:
            # INCUMBENT: the NATIVE width-3 call, exactly as the runtime made
            # it. CANDIDATE: the same operands PADDED to canonical width 4
            # with an inert shadow in slot 3. The comparison is over the real
            # rows only, because the shadow rows do not exist in the native
            # call -- their contract is checked separately, below.
            padded_out, padded_lse = _fr13_fa2_qrow32_live_ab_padded_call(
                bundle, poison=False
            )
            torch.cuda.synchronize()
            clean_out = padded_out[:real_rows].clone()
            clean_lse = padded_lse[..., :real_rows].clone()
            shadow_failures.extend(
                layer_name + ":clean:" + failure
                for failure in _fr13_fa2_qrow32_live_ab_shadow_mismatches(
                    padded_out, padded_lse, real_rows
                )
            )
            # THE POISONED-SHADOW VARIANT. NaN Q rows and an impossible page
            # in the shadow block-table row. At seqused_k[3] == 0 the kernel
            # reads neither, so the real rows must be BIT-IDENTICAL to the
            # clean-shadow run -- a strictly sharper statement than "matches
            # stock", because it fails loudly if the early exit is not taken.
            poison_out, poison_lse = _fr13_fa2_qrow32_live_ab_padded_call(
                bundle, poison=True
            )
            torch.cuda.synchronize()
            shadow_failures.extend(
                layer_name + ":poisoned:" + failure
                for failure in _fr13_fa2_qrow32_live_ab_shadow_mismatches(
                    poison_out, poison_lse, real_rows
                )
            )
            poison_summary = {
                "output": _fr13_fa2_qrow32_byte_summary(
                    clean_out, poison_out[:real_rows]
                ),
                "lse": _fr13_fa2_qrow32_byte_summary(
                    clean_lse, poison_lse[..., :real_rows]
                ),
                "shadow_rows": [real_rows, _FR13_FA2_QROW32_LIVE_AB_CANONICAL_ROWS],
                "shadow_seqused_k": 0,
                "shadow_block_table_page": (
                    _FR13_FA2_QROW32_LIVE_AB_POISON_BLOCK_ID
                ),
                "shadow_query_fill": "nan",
            }
            total_poison_output_mismatches += int(
                poison_summary["output"]["raw_byte_mismatches"]
            )
            total_poison_lse_mismatches += int(
                poison_summary["lse"]["raw_byte_mismatches"]
            )
            candidate_out = clean_out
            candidate_lse = clean_lse
        else:
            candidate_buf = torch.empty_like(bundle["query"])
            candidate_out, candidate_lse = _fr13_fa2_qrow32_live_ab_call(
                bundle, candidate_buf, candidate=True
            )
            torch.cuda.synchronize()
        if stock_out.dtype != torch.bfloat16 or candidate_out.dtype != torch.bfloat16:
            raise RuntimeError("FR13 qrow32 live gate output is not BF16")
        if stock_lse.dtype != torch.float32 or candidate_lse.dtype != torch.float32:
            raise RuntimeError("FR13 qrow32 live gate LSE is not FP32")

        output_summary = _fr13_fa2_qrow32_byte_summary(stock_out, candidate_out)
        lse_summary = _fr13_fa2_qrow32_byte_summary(stock_lse, candidate_lse)
        slot_records = []
        for slot in range(width):
            begin = slot * rows
            end = begin + rows
            slot_records.append(
                {
                    "slot": slot,
                    "query_rows": [begin, end],
                    "output": _fr13_fa2_qrow32_byte_summary(
                        stock_out[begin:end], candidate_out[begin:end]
                    ),
                    "lse": _fr13_fa2_qrow32_byte_summary(
                        stock_lse[..., begin:end], candidate_lse[..., begin:end]
                    ),
                }
            )
        total_output_mismatches += int(output_summary["raw_byte_mismatches"])
        total_lse_mismatches += int(lse_summary["raw_byte_mismatches"])
        layer_records.append(
            {
                "layer_name": layer_name,
                "tree_bias_sha256": bias_sha256,
                "output": output_summary,
                "lse": lse_summary,
                "slots": slot_records,
                "poisoned_shadow": poison_summary,
            }
        )

    if len(tree_bias_sha256) != 1:
        raise RuntimeError("FR13 qrow32 physical32 mask differs across layers")
    passed = (
        total_output_mismatches == 0
        and total_lse_mismatches == 0
        and total_poison_output_mismatches == 0
        and total_poison_lse_mismatches == 0
        and not shadow_failures
    )
    record = {
        "schema": "fr13.fixed32.fa2_qrow32_live_paged_exact4_ab.v1",
        "status": "PASS" if passed else "FAIL",
        "suite": "SWE-Verified",
        "task_ids": list(task_ids),
        "subset_sha256": _FR13_FA2_QROW32_EXACT4_SUBSET_SHA256,
        "concurrency": width,
        "batch_size": width,
        "physical_rows_per_slot": 32,
        "total_query_rows": real_rows,
        # What the .SO actually saw, which at width 3 is not what the runtime
        # served. Disclosing both is the whole point of the padded arm.
        "padded_to_canonical_width": bool(padded),
        "canonical_width": _FR13_FA2_QROW32_LIVE_AB_CANONICAL_WIDTH,
        "canonical_query_rows": (
            _FR13_FA2_QROW32_LIVE_AB_CANONICAL_ROWS if padded else real_rows
        ),
        "shadow_slot": (
            _FR13_FA2_QROW32_LIVE_AB_CANONICAL_WIDTH - 1 if padded else None
        ),
        "shadow_seqused_k": 0 if padded else None,
        "shadow_block_table_page": (
            _FR13_FA2_QROW32_LIVE_AB_NULL_BLOCK_ID if padded else None
        ),
        "poisoned_shadow_arm": bool(padded),
        "poisoned_shadow_output_raw_byte_mismatches": (
            total_poison_output_mismatches if padded else None
        ),
        "poisoned_shadow_lse_raw_byte_mismatches": (
            total_poison_lse_mismatches if padded else None
        ),
        "shadow_contract_failures": sorted(shadow_failures),
        "fixed32_mode": fixed32_mode,
        "candidate_arm": candidate_arm,
        "selector_sentinel": int(candidate_contract["sentinel"]),
        "candidate_so_sha256": candidate_so_sha256,
        "candidate_so_size": candidate_so_size,
        "fa2_head": fa2_head,
        "fa2_source_closure_sha256": fa2_source_closure_sha256,
        "source_commit": source_commit,
        "draft_vocab_k": 65536,
        "draft_vocab_root": 1,
        "engine_pid": os.getpid(),
        "graph_id": int(graph_id),
        "runtime_mode": str(runtime_mode).upper(),
        "layer_count": len(layer_records),
        "target_layers": list(_FR13_FA2_QROW32_TARGET_LAYERS),
        "stock_calls": len(layer_records),
        "candidate_calls": len(layer_records),
        "operands": {
            "query_shape": [real_rows, 24, 256],
            "query_start_loc": shared_q_start,
            "seq_lens": shared_seq_lens,
            "suffix_start_mod64": [
                (length - 32) % 64 for length in shared_seq_lens
            ],
            "slot_coverage": list(range(width)),
            "key_cache_tail_shape": [1024, 4, 256],
            "tree_bias_shape": list(next(iter(graph.values()))["tree_bias"].shape),
            "tree_bias_sha256": next(iter(tree_bias_sha256)),
        },
        "output_raw_byte_mismatches": total_output_mismatches,
        "lse_raw_byte_mismatches": total_lse_mismatches,
        "layers": layer_records,
        "incumbent_dispatch": (
            "stock FA2 native width-%d geometry; no fallback" % width
            if padded
            else "stock FA2 exact geometry; no fallback"
        ),
        "candidate_dispatch": (
            candidate_contract["candidate_dispatch"] + "; padded to canonical "
            "width 4 with a zero-key shadow request in slot 3"
            if padded
            else candidate_contract["candidate_dispatch"]
        ),
        "served_return": "stock captured graph output unchanged",
        "fallback_allowed": False,
        "performance_measurement": False,
    }
    _fr13_fa2_qrow32_live_ab_write(record, batch_size=width)
    if not passed:
        raise RuntimeError(
            "FR13 qrow32 live paged exact4 byte A/B mismatch: "
            f"width={width} "
            f"output_bytes={total_output_mismatches} "
            f"lse_bytes={total_lse_mismatches} "
            f"poisoned_output_bytes={total_poison_output_mismatches} "
            f"poisoned_lse_bytes={total_poison_lse_mismatches} "
            f"shadow_failures={sorted(shadow_failures)!r}"
        )
    _FR13_FA2_QROW32_LIVE_AB_WIDTHS_PASSED.add(width)
    _FR13_FA2_QROW32_LIVE_AB_PASSED = True
    logger.warning(
        "[FR13_FA2_QROW32_LIVE_PAGED_AB] PASS mode=%s width=%d layers=16 "
        "slots=%d output_byte_mismatches=0 lse_byte_mismatches=0 "
        "poisoned_shadow_mismatches=%d stock_served=1",
        fixed32_mode,
        width,
        width,
        total_poison_output_mismatches + total_poison_lse_mismatches,
    )


'''


FIXED32_QUERY_TILE32_B1_SELECTOR_HELPERS = r'''# FR13_FA2_QROW32_B1_SELECTORS
_FR13_FA2_QROW32_B1_ARMS = {
    "nosplit": {
        "sentinel": 1179791668,
        "num_splits": 0,
        "split_scratch_allocation": "not used; num_splits=0",
        "candidate_dispatch": "qrow32 B1 nosplit exact geometry; no fallback",
    },
    "split2": {
        "sentinel": 1179791669,
        "num_splits": 2,
        "split_scratch_allocation": (
            "stock FA2 set_params_splitkv via num_splits=2"
        ),
        "candidate_dispatch": "qrow32 B1 split2 exact geometry; no fallback",
    },
    "visibility": {
        "sentinel": 1179791668,
        "num_splits": 0,
        "split_scratch_allocation": "not used; num_splits=0",
        "candidate_dispatch": (
            "qrow32 B1 fixed32 visibility exact geometry; no fallback"
        ),
    },
    "gqa_pair": {
        "sentinel": 1179791670,
        "num_splits": 0,
        "split_scratch_allocation": "not used; num_splits=0",
        "candidate_dispatch": (
            "qrow32 B1 GQA-pair exact geometry; no fallback"
        ),
    },
    # FR14 Tier-B. Same GQA-pair traits and the same per-split K-block order,
    # but the context walk is partitioned four ways across blockIdx.y and
    # re-reduced by FA2's combine kernel, so per-row output is NOT
    # bit-identical to gqa_pair and no byte gate can qualify it --
    # _fr13_fa2_qrow32_b1_require_same_reduction refuses the comparison
    # outright. Gate-only: absent from _FR13_FA2_QROW32_B1_PRODUCTION_ARMS.
    "gqa_pair_splitk": {
        "sentinel": 1179791671,
        "num_splits": 4,
        "split_scratch_allocation": (
            "stock FA2 set_params_splitkv via num_splits=4"
        ),
        "candidate_dispatch": (
            "qrow32 B1 GQA-pair split-K exact geometry; no fallback"
        ),
    },
}
_FR13_FA2_QROW32_B1_QROW16_REFERENCE_SENTINEL = 1179791667
_FR13_FA2_QROW32_B1_CANDIDATE_SHA256 = (
    "a9d8a6887b8b27b3a83af60bba7945eb66caff174ba710c2ee2aea92b8e7081a"
)
_FR13_FA2_QROW32_B1_CANDIDATE_SIZE = 300154616
_FR13_FA2_QROW32_B1_FA2_HEAD = "29210221863736a08f71a866459e368ad1ac4a95"
_FR13_FA2_QROW32_B1_SOURCE_CLOSURE_SHA256 = (
    "22b8c2016443a151bf50f62166f7cc3b9ce45137138d948b76fdfded74c395ff"
)
_FR13_FA2_QROW32_B1_VISIBILITY_CANDIDATE_SHA256 = (
    "c5ab32a6ae4e615f1e77a4997db5429152053c549e761fb11d90b33bb3959a79"
)
_FR13_FA2_QROW32_B1_VISIBILITY_CANDIDATE_SIZE = 300200192
_FR13_FA2_QROW32_B1_VISIBILITY_SOURCE_CLOSURE_SHA256 = (
    "a30eca031cd5067133e6278527787c5987635670930e5840ac983f66b088e4fc"
)
_FR13_FA2_QROW32_B1_GQA_PAIR_CANDIDATE_SHA256 = (
    "3560cdc0c1ebbe3d912858ea447b350edefc0d6749950d6353e5f763185da6ae"
)
_FR13_FA2_QROW32_B1_GQA_PAIR_CANDIDATE_SIZE = 299815552
_FR13_FA2_QROW32_B1_GQA_PAIR_SOURCE_CLOSURE_SHA256 = (
    "172b5e7131841ce45650bb8eea35f0b427ca660ce8f145bd39b55b00a336ebf4"
)
# FR14 split-K (tier-b). These did not exist in this blob at all until Arm S's
# 17th refusal: the launcher, the contract and the sidecar each carried the
# split-K pins, and the in-container identity resolver -- the last thing
# between a selected arm and a served token -- carried none of them, so it
# answered with split2's. Both SASS digests are here for the same reason they
# are in the bash pin case: this arm's .so sha is not rebuild-reproducible, so
# the digests are what attest that the KERNEL is the characterized one.
_FR13_FA2_QROW32_B1_SPLITK_CANDIDATE_SHA256 = (
    "28570f835ea72c99d03aab9fb03c494388bbb9c264ee4dc96eec047f50d7f857"
)
_FR13_FA2_QROW32_B1_SPLITK_CANDIDATE_SIZE = 300123792
_FR13_FA2_QROW32_B1_SPLITK_SOURCE_CLOSURE_SHA256 = (
    "4ed00909cef7ea83849f897018ea4f6a14119b8d160927af426938920c170878"
)
_FR13_FA2_QROW32_B1_SPLITK_SASS_DIGEST_SHA256 = (
    "3f24d70dce2ff70ad9209bad5af2a93cc39453df529cb298e4476cbfbfd80b9e"
)
_FR13_FA2_QROW32_B1_SPLITK_BASELINE_SASS_DIGEST_SHA256 = (
    "fa01f98840420b9c0177d06297aacabb0ed5e00c674511fdaa4aa618c3473470"
)
_FR13_FA2_QROW32_B1_TARGET_LAYERS = tuple(
    f"language_model.model.layers.{index}.self_attn.attn"
    for index in range(3, 64, 4)
)
_FR13_FA2_QROW32_B1_CANONICAL_TASK_IDS = (
    "astropy__astropy-12907",
    "astropy__astropy-13033",
    "astropy__astropy-13236",
    "astropy__astropy-13398",
)
_FR13_FA2_QROW32_B1_EXACT4_SUBSET_SHA256 = (
    "0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5"
)
# The arms that may serve REAL traffic. "visibility" and "split2" are gate-only
# instruments -- they exist to observe or to vary reduction topology, and
# neither was ever byte-qualified as a served dispatch -- so admitting them here
# would let a diagnostic build answer production requests.
_FR13_FA2_QROW32_B1_PRODUCTION_ARMS = ("nosplit", "gqa_pair")
_FR13_FA2_QROW32_B1_LIVE_GRAPHS = {}
_FR13_FA2_QROW32_B1_LIVE_ATTEMPTED = False
_FR13_FA2_QROW32_B1_PRODUCTION_GRAPHS = {}
_FR13_FA2_QROW32_B1_EAGER_STATE = {
    "layers": set(),
    "calls": 0,
    "emitted": False,
}


def _fr13_fa2_qrow32_b1_arm(env_name):
    arm = os.environ.get(env_name, "")
    if not arm:
        return None
    if env_name == "FR13_FA2_QROW32_B1_PRODUCTION_ARM":
        if arm not in _FR13_FA2_QROW32_B1_PRODUCTION_ARMS:
            raise RuntimeError(
                f"{env_name} must be empty or one of "
                f"{', '.join(_FR13_FA2_QROW32_B1_PRODUCTION_ARMS)}; got {arm!r}"
            )
    elif env_name == "FR13_FA2_QROW32_B1_LIVE_AB_ARM":
        if arm not in _FR13_FA2_QROW32_B1_ARMS:
            raise RuntimeError(
                f"{env_name} must be empty or one of "
                f"{', '.join(sorted(_FR13_FA2_QROW32_B1_ARMS))}; got {arm!r}"
            )
    else:
        raise RuntimeError(f"unknown FR13 qrow32 B1 arm selector: {env_name}")
    return arm


def _fr13_fa2_qrow32_b1_require_same_reduction(arm, reference_num_splits):
    """Keep the raw-byte gate on the incumbent reduction topology."""
    candidate_num_splits = int(_FR13_FA2_QROW32_B1_ARMS[arm]["num_splits"])
    reference_partitions = max(1, int(reference_num_splits))
    candidate_partitions = max(1, candidate_num_splits)
    if reference_partitions != candidate_partitions:
        raise RuntimeError(
            "FR13 qrow32 B1 raw-byte qualification requires identical "
            "reduction topology: "
            f"reference_partitions={reference_partitions} "
            f"candidate_partitions={candidate_partitions}"
        )


def _fr13_fa2_qrow32_b1_digest(env_name, label, *, length=64):
    value = os.environ.get(env_name, "")
    if len(value) != length or any(c not in "0123456789abcdef" for c in value):
        raise RuntimeError(f"FR13 qrow32 B1 {label} digest drifted")
    return value


# ------------------------------------------------------------------ TIER B
#
# Mark, FR14 pass 64: an arm that cannot be byte-identical to the incumbent BY
# CONSTRUCTION may serve on a Tier-B credential -- determinism proven, ULP
# characterised against pre-registered bounds, and no worse than the incumbent
# against a float64 reference. Live-A/B serving only; promoted-default remains
# a separate, later, Mark-gated decision.
#
# NOTHING BYTE-GATED BECOMES EASIER. _FR13_FA2_QROW32_B1_PRODUCTION_ARMS is
# unchanged and still refuses this arm; the raw-byte gate's
# _fr13_fa2_qrow32_b1_require_same_reduction is unchanged and still refuses to
# compare differing reduction topologies. What is added is a SECOND door that
# only a tier-b-marked arm can walk through, and only while carrying a
# credential bound to its exact binary, source closure, SASS digests, HEAD and
# patcher.
_FR13_FA2_QROW32_B1_TIER_B_ARMS = ("gqa_pair_splitk",)
_FR13_FA2_QROW32_B1_TIER_B_SCHEMA = "fr13.fixed32.fa2_tierb_qualification.v1"
_FR13_FA2_QROW32_B1_TIER_B_STATE = {}


def _fr13_fa2_qrow32_b1_tier_b_credential(arm, identity):
    """Load and check the Tier-B credential authorising THIS arm to serve.

    Container-side, so it validates by DIGEST and by BINDING rather than by
    re-deriving the bounds: the pinned image ships no git and the full
    re-evaluation already happened twice on the host -- once in the gate runner
    that wrote the credential, once in the launcher preflight that validated it
    through scripts/fr13_qrow32_b1_pass_sidecar.py. What has to be established
    here is narrower and, for the risk that matters, stricter: that the bytes
    on disk ARE the credential those validations were made against, and that it
    names THIS binary at THIS commit.
    """
    import hashlib as _hashlib
    import json as _json

    cached = _FR13_FA2_QROW32_B1_TIER_B_STATE.get(arm)
    if cached is not None:
        return cached
    path = os.environ.get("FR13_FA2_QROW32_B1_TIER_B_CREDENTIAL", "")
    if not path:
        raise RuntimeError(
            "FR13 qrow32 B1 tier-b serving requires "
            "FR13_FA2_QROW32_B1_TIER_B_CREDENTIAL"
        )
    expected_digest = _fr13_fa2_qrow32_b1_digest(
        "FR13_FA2_QROW32_B1_TIER_B_CREDENTIAL_SHA256", "tier-b credential"
    )
    raw = open(path, "rb").read()
    if _hashlib.sha256(raw).hexdigest() != expected_digest:
        raise RuntimeError("FR13 qrow32 B1 tier-b credential digest drifted")
    payload = _json.loads(raw.decode("ascii"))
    if payload.get("schema") != _FR13_FA2_QROW32_B1_TIER_B_SCHEMA:
        raise RuntimeError("FR13 qrow32 B1 tier-b credential schema drifted")
    if payload.get("tier") != "B" or payload.get("arm") != arm:
        raise RuntimeError("FR13 qrow32 B1 tier-b credential names another arm")
    candidate_digest, source_commit, patch_source_digest = identity
    bound = payload.get("identity")
    if not isinstance(bound, dict):
        raise RuntimeError("FR13 qrow32 B1 tier-b credential is unbound")
    config = _FR13_FA2_QROW32_B1_ARMS[arm]
    for field, expected in (
        ("arm", arm),
        ("so_sha256", candidate_digest),
        ("source_commit", source_commit),
        ("patch_source_sha256", patch_source_digest),
    ):
        if bound.get(field) != expected:
            raise RuntimeError(
                f"FR13 qrow32 B1 tier-b credential {field} does not bind this "
                "serve"
            )
    selector = payload.get("selector")
    if not isinstance(selector, dict) or (
        int(selector.get("sentinel", -1)) != int(config["sentinel"])
        or int(selector.get("num_splits", -1)) != int(config["num_splits"])
    ):
        raise RuntimeError(
            "FR13 qrow32 B1 tier-b credential selector does not match the arm"
        )
    determinism = payload.get("determinism")
    if not isinstance(determinism, dict) or (
        determinism.get("all_cases_bitwise_identical") is not True
        or determinism.get("cross_process_digests_identical") is not True
    ):
        raise RuntimeError(
            "FR13 qrow32 B1 tier-b credential does not carry a passed "
            "determinism gate"
        )
    evaluation = payload.get("bounds_evaluation")
    if not isinstance(evaluation, dict) or evaluation.get(
        "bounds_passed"
    ) is not True:
        raise RuntimeError(
            "FR13 qrow32 B1 tier-b credential did not clear its bounds"
        )
    strength = payload.get("probe_strength")
    if not isinstance(strength, dict) or strength.get(
        "probe_strength_passed"
    ) is not True:
        raise RuntimeError(
            "FR13 qrow32 B1 tier-b credential probe is below the "
            "pre-registered strength floor"
        )
    record = {
        "credential_sha256": expected_digest,
        "bounds_sha256": bound.get("bounds_sha256"),
        "path": path,
    }
    _FR13_FA2_QROW32_B1_TIER_B_STATE[arm] = record
    return record


def _fr13_fa2_qrow32_b1_tier_b_arm():
    """The tier-b arm authorised to SERVE, or None.

    Naming a tier-b arm as the live-A/B arm is not enough on its own: that is
    still the shadow-comparison route it always was. Serving its output
    requires FR13_FA2_QROW32_B1_TIER_B_SERVE=1 as well, so a run cannot start
    serving Tier-B numerics because someone reused a live-A/B invocation.
    """
    arm = os.environ.get("FR13_FA2_QROW32_B1_LIVE_AB_ARM", "")
    if not arm or arm not in _FR13_FA2_QROW32_B1_TIER_B_ARMS:
        return None
    serve = os.environ.get("FR13_FA2_QROW32_B1_TIER_B_SERVE", "0")
    if serve == "0":
        return None
    if serve != "1":
        raise RuntimeError(
            "FR13_FA2_QROW32_B1_TIER_B_SERVE must be exactly 0 or 1"
        )
    if arm in _FR13_FA2_QROW32_B1_PRODUCTION_ARMS:
        raise RuntimeError(
            "a tier-b arm must not also be a production arm: tier-b grants "
            "live-A/B serving, not promoted-default"
        )
    return arm


def _fr13_fa2_qrow32_b1_serving_arm():
    """(arm, tier) for the arm whose output actually reaches the model.

    Tier A is resolved FIRST and unchanged, so a launch that names no tier-b
    arm cannot reach one line of the tier-b path.
    """
    arm = _fr13_fa2_qrow32_b1_arm("FR13_FA2_QROW32_B1_PRODUCTION_ARM")
    if arm is not None:
        return arm, "A"
    arm = _fr13_fa2_qrow32_b1_tier_b_arm()
    if arm is not None:
        return arm, "B"
    return None, None


def _fr13_fa2_qrow32_b1_draft_vocab_identity():
    """The draft-vocabulary identity AS SERVED, for artifact emission.

    Emitters used to hardcode draft_vocab_root=1 / draft_vocab_k=65536 into the
    live gate JSON. Combined with a pass-sidecar validator that REQUIRED those
    same literals, a K0 gate produced a green credential describing a K64 serve:
    two hardcodings agreeing with each other while both disagreed with reality
    (fr14 2026-08-17 red-team). An artifact must report what ran.
    """
    return (
        int(os.environ.get("FR13_DRAFT_VOCAB_ROOT", "0")),
        int(os.environ.get("FR13_DRAFT_VOCAB_K", "0")),
        os.environ.get(
            "FR13_FA2_QROW32_B1_QUALIFICATION_PROFILE", "k64_root"
        ),
    )


def _fr13_fa2_qrow32_b1_require_draft_vocab_profile():
    """Kernel-side defence-in-depth on the draft-vocabulary IDENTITY.

    This is a workload-identity assertion, not a kernel-geometry constraint: it
    reads two environment variables and never inspects a tensor. The real shape
    guard is _fr13_fa2_qrow32_b1_geometry_mismatches(), which checks query rows,
    caches, cu_seqlens and the tree bias, and has nothing to do with the draft
    vocabulary. The qrow32 B1 kernel is therefore NOT K64-dependent; what this
    guard encodes is "the credential authorizing this selector was earned in
    workload shape X", enforced here as well as in the launcher so that
    bypassing the launcher cannot bypass it.

    It used to hard-code K64/root1, which is why it refused a K0 serve even
    after the launcher predicates were made profile-aware -- a third, deeper
    layer of the same binding (fr14 2026-08-17). It now honours the same
    qualification profile the launcher does, so a credential still cannot be
    used outside the shape it was earned in, in either direction.
    """
    profile = os.environ.get(
        "FR13_FA2_QROW32_B1_QUALIFICATION_PROFILE", "k64_root"
    )
    root = int(os.environ.get("FR13_DRAFT_VOCAB_ROOT", "0"))
    k = int(os.environ.get("FR13_DRAFT_VOCAB_K", "0"))
    if profile == "k64_root":
        if root != 1 or k != 65536:
            raise RuntimeError(
                "FR13 qrow32 B1 selectors under the k64_root profile require "
                "K64 ROOT=1"
            )
    elif profile == "full_vocab":
        if root != 0 or k != 0:
            raise RuntimeError(
                "FR13 qrow32 B1 selectors under the full_vocab profile require "
                "K0 ROOT=0"
            )
    else:
        raise RuntimeError(
            "FR13 qrow32 B1 qualification profile must be exactly k64_root or "
            f"full_vocab; got: {profile!r}"
        )


# THE IDENTITY TABLE. Every arm names its own pins; there is no default.
#
# This used to be an if/if/bare-return chain whose fall-through handed out
# split2's pins to anything it did not recognise. Arm S's fifth boot reached it
# with arm="gqa_pair_splitk" -- everything upstream having passed for the first
# time -- and was saved only by an accident: the environment declared split-K's
# sha, split2's pin did not match it, and the mismatch raised. Had the
# environment been permissive, or had the two binaries happened to share a
# size, the run would have SERVED split-K while ATTESTING split2. A fall-through
# in an identity resolver is not a convenience; it is a licence to attest the
# wrong artifact.
#
# So: a table, and an unknown arm REFUSES. Adding an arm to
# _FR13_FA2_QROW32_B1_ARMS without adding it here now fails loudly at selection
# instead of silently inheriting somebody else's identity.
_FR13_FA2_QROW32_B1_IDENTITIES = {
    "nosplit": {
        "candidate_sha256": _FR13_FA2_QROW32_B1_CANDIDATE_SHA256,
        "candidate_size": _FR13_FA2_QROW32_B1_CANDIDATE_SIZE,
        "fa2_head": _FR13_FA2_QROW32_B1_FA2_HEAD,
        "source_closure_sha256": _FR13_FA2_QROW32_B1_SOURCE_CLOSURE_SHA256,
    },
    "split2": {
        "candidate_sha256": _FR13_FA2_QROW32_B1_CANDIDATE_SHA256,
        "candidate_size": _FR13_FA2_QROW32_B1_CANDIDATE_SIZE,
        "fa2_head": _FR13_FA2_QROW32_B1_FA2_HEAD,
        "source_closure_sha256": _FR13_FA2_QROW32_B1_SOURCE_CLOSURE_SHA256,
    },
    "visibility": {
        "candidate_sha256": _FR13_FA2_QROW32_B1_VISIBILITY_CANDIDATE_SHA256,
        "candidate_size": _FR13_FA2_QROW32_B1_VISIBILITY_CANDIDATE_SIZE,
        "fa2_head": _FR13_FA2_QROW32_B1_FA2_HEAD,
        "source_closure_sha256": (
            _FR13_FA2_QROW32_B1_VISIBILITY_SOURCE_CLOSURE_SHA256
        ),
    },
    "gqa_pair": {
        "candidate_sha256": _FR13_FA2_QROW32_B1_GQA_PAIR_CANDIDATE_SHA256,
        "candidate_size": _FR13_FA2_QROW32_B1_GQA_PAIR_CANDIDATE_SIZE,
        "fa2_head": _FR13_FA2_QROW32_B1_FA2_HEAD,
        "source_closure_sha256": (
            _FR13_FA2_QROW32_B1_GQA_PAIR_SOURCE_CLOSURE_SHA256
        ),
    },
    "gqa_pair_splitk": {
        "candidate_sha256": _FR13_FA2_QROW32_B1_SPLITK_CANDIDATE_SHA256,
        "candidate_size": _FR13_FA2_QROW32_B1_SPLITK_CANDIDATE_SIZE,
        "fa2_head": _FR13_FA2_QROW32_B1_FA2_HEAD,
        "source_closure_sha256": (
            _FR13_FA2_QROW32_B1_SPLITK_SOURCE_CLOSURE_SHA256
        ),
        # Only this arm carries SASS digests, and only this arm needs them:
        # its .so sha is not rebuild-reproducible (two links, one size), so the
        # artifact hash alone cannot say the kernel reproduced.
        "sass_digest_sha256": _FR13_FA2_QROW32_B1_SPLITK_SASS_DIGEST_SHA256,
        "baseline_sass_digest_sha256": (
            _FR13_FA2_QROW32_B1_SPLITK_BASELINE_SASS_DIGEST_SHA256
        ),
    },
}


def _fr13_fa2_qrow32_b1_identity(arm=None):
    # arm=None is the historical no-arm call for the incumbent selector. It is
    # spelled explicitly rather than left to a default so that "no arm" and
    # "an arm nobody wrote a branch for" cannot be the same code path -- which
    # is exactly how the 17th site hid.
    resolved = "nosplit" if arm is None else arm
    identity = _FR13_FA2_QROW32_B1_IDENTITIES.get(resolved)
    if identity is None:
        raise RuntimeError(
            "FR13 qrow32 B1 has no pinned identity for arm "
            f"{arm!r}; every arm must name its own binary here -- there is no "
            "fall-through, because a fall-through attests the wrong artifact"
        )
    if not identity["candidate_sha256"] or not identity["candidate_size"]:
        raise RuntimeError(
            f"FR13 qrow32 B1 arm {resolved!r} binary is not pinned: fill its "
            "candidate sha256 and size from the build attestation before "
            "selecting this arm"
        )
    return dict(identity)


def _fr13_fa2_qrow32_b1_require_identity(arm=None):
    identity = _fr13_fa2_qrow32_b1_identity(arm)
    candidate_digest = _fr13_fa2_qrow32_b1_digest(
        "FR13_FA2_QROW32_B1_SO_SHA256", "candidate SO"
    )
    source_commit = _fr13_fa2_qrow32_b1_digest(
        "FR13_FA2_QROW32_B1_SOURCE_COMMIT", "source commit", length=40
    )
    patch_source = _fr13_fa2_qrow32_b1_digest(
        "FR13_FA2_QROW32_B1_PATCH_SOURCE_SHA256", "patch source"
    )
    if (
        candidate_digest != identity["candidate_sha256"]
        or int(os.environ.get("FR13_FA2_QROW32_B1_SO_SIZE", "0"))
        != identity["candidate_size"]
        or os.environ.get("FR13_FA2_QROW32_B1_FA2_HEAD", "")
        != identity["fa2_head"]
        or os.environ.get("FR13_FA2_QROW32_B1_SOURCE_CLOSURE_SHA256", "")
        != identity["source_closure_sha256"]
    ):
        raise RuntimeError("FR13 qrow32 B1 pinned identity drifted")
    # An arm whose identity carries SASS digests must present them too. The
    # bash pin case checks these on the host; this is the container-side half,
    # and it exists because the container-side half of the ARTIFACT check is
    # exactly what was missing when Arm S reached the 17th site.
    for field, env_name in (
        ("sass_digest_sha256", "FR13_FA2_QROW32_B1_SPLITK_SASS_DIGEST"),
        (
            "baseline_sass_digest_sha256",
            "FR13_FA2_QROW32_B1_SPLITK_BASELINE_SASS_DIGEST",
        ),
    ):
        expected = identity.get(field)
        if expected is None:
            continue
        if os.environ.get(env_name, "") != expected:
            raise RuntimeError(
                f"FR13 qrow32 B1 pinned {field} drifted for this arm"
            )
    return candidate_digest, source_commit, patch_source


def _fr13_fa2_qrow32_b1_require_exact4():
    task_ids = tuple(
        value
        for value in os.environ.get(
            "FR13_FA2_QROW32_B1_EXACT4_TASK_IDS", ""
        ).split(",")
        if value
    )
    subset = os.environ.get("FR13_FA2_QROW32_B1_EXACT4_SUBSET_SHA256", "")
    if (
        task_ids != _FR13_FA2_QROW32_B1_CANONICAL_TASK_IDS
        or subset != _FR13_FA2_QROW32_B1_EXACT4_SUBSET_SHA256
    ):
        raise RuntimeError("FR13 qrow32 B1 production exact4 identity drifted")
    return task_ids


def _fr13_fa2_qrow32_b1_geometry_mismatches(
    *,
    query,
    key_cache,
    value_cache,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    causal,
    window_size,
    block_table,
    softcap,
    num_splits,
    tree_bias,
):
    query_meta = (str(query.dtype), tuple(query.shape), tuple(query.stride()))
    key_meta = (
        str(key_cache.dtype), tuple(key_cache.shape), tuple(key_cache.stride())
    )
    value_meta = (
        str(value_cache.dtype),
        tuple(value_cache.shape),
        tuple(value_cache.stride()),
    )
    cu_seqlens_q_meta = (str(cu_seqlens_q.dtype), tuple(cu_seqlens_q.shape))
    seqused_k_meta = (str(seqused_k.dtype), tuple(seqused_k.shape))
    block_table_meta = (str(block_table.dtype), tuple(block_table.shape))
    tree_bias_meta = (
        str(tree_bias.dtype), tuple(tree_bias.shape), tuple(tree_bias.stride())
    )
    window_meta = (
        None
        if window_size is None
        else tuple(int(value) for value in window_size)
    )
    checks = (
        (
            "query(dtype,shape,stride)",
            query.dtype == torch.bfloat16
            and tuple(query.shape) == (32, 24, 256)
            and int(query.stride(-2)) == 256
            and int(query.stride(-1)) == 1,
            query_meta,
        ),
        (
            "key_cache(dtype,shape,stride)",
            key_cache.dtype == torch.bfloat16
            and tuple(key_cache.shape[1:]) == (1024, 4, 256)
            and tuple(key_cache.stride())
            == (2 * 1024 * 4 * 256, 4 * 256, 256, 1),
            key_meta,
        ),
        (
            "value_cache(dtype,shape,stride)",
            value_cache.dtype == torch.bfloat16
            and tuple(value_cache.shape) == tuple(key_cache.shape)
            and tuple(value_cache.stride()) == tuple(key_cache.stride()),
            value_meta,
        ),
        (
            "cu_seqlens_q(dtype,shape)",
            cu_seqlens_q.dtype == torch.int32
            and tuple(cu_seqlens_q.shape) == (2,),
            cu_seqlens_q_meta,
        ),
        (
            "seqused_k(dtype,shape)",
            seqused_k.dtype == torch.int32 and tuple(seqused_k.shape) == (1,),
            seqused_k_meta,
        ),
        (
            "block_table(dtype,shape)",
            block_table.dtype == torch.int32
            and block_table.ndim == 2
            and int(block_table.shape[0]) == 1,
            block_table_meta,
        ),
        (
            "tree_bias(dtype,shape,stride)",
            tree_bias.dtype == torch.float32
            and tuple(tree_bias.shape) in ((32, 32), (1, 32, 32))
            and int(tree_bias.stride(-1)) == 1,
            tree_bias_meta,
        ),
        ("max_seqlen_q", int(max_seqlen_q) == 32, int(max_seqlen_q)),
        ("max_seqlen_k", int(max_seqlen_k) > 0, int(max_seqlen_k)),
        ("causal", not bool(causal), bool(causal)),
        ("softcap", float(softcap) == 0.0, float(softcap)),
        ("num_splits", int(num_splits) in (0, 1), int(num_splits)),
        ("window_size", window_meta in (None, (-1, -1)), window_meta),
    )
    return tuple(
        f"{name}={actual!r}" for name, valid, actual in checks if not valid
    )


def _fr13_fa2_qrow32_b1_exact_geometry(**geometry):
    return not _fr13_fa2_qrow32_b1_geometry_mismatches(**geometry)


def _fr13_fa2_qrow32_b1_candidate_tree_bias(tree_bias, arm):
    """Retag the operand so the forked FA2 dispatch selects this arm's kernel.

    The tag IS the batch stride: flash_api gates on
    params.tree_bias_batch_stride == the arm's sentinel, so a mask laid out with
    that stride -- and only such a mask -- reaches the candidate kernel.

    At B1 this is a pure metadata retag (as_strided), NOT the B4 sibling's
    empty_strided+copy_. That divergence is deliberate and load-bearing twice
    over. Correctness: batch is 1, so stride(0) is never dereferenced and the
    view addresses exactly the same 1024 floats the untagged mask did -- whereas
    B4 must materialize because its sentinel stride really is used to index
    batches 1..3, which would read far out of bounds. Measurement: this path is
    about to be timed, and a copy would add 16 extra device-to-device kernels
    per step to the candidate arm alone, manufacturing an arm asymmetry in the
    very quantity under test. as_strided is also the exact construction the
    sealed byte gate qualified, so production serves the operand that was
    proven rather than a re-derived equivalent.
    """
    config = _FR13_FA2_QROW32_B1_ARMS[arm]
    base = tree_bias[0] if tree_bias.ndim == 3 else tree_bias
    if base.dtype != torch.float32 or tuple(base.shape) != (32, 32):
        raise RuntimeError("FR13 qrow32 B1 tree bias geometry drifted")
    if tuple(base.stride()) != (32, 1):
        raise RuntimeError("FR13 qrow32 B1 tree bias is not canonical contiguous")
    tagged = torch.as_strided(
        base,
        size=(1, 32, 32),
        stride=(config["sentinel"], 32, 1),
    )
    if int(tagged.stride(0)) != config["sentinel"]:
        raise RuntimeError("FR13 qrow32 B1 selector tag was not preserved")
    # Pin the zero-copy property the timing claim rests on: the tagged operand
    # must alias the incumbent's bytes, not a duplicate of them.
    if tagged.data_ptr() != base.data_ptr():
        raise RuntimeError("FR13 qrow32 B1 selector retag copied the operand")
    return tagged


def _fr13_fa2_qrow32_b1_reference_tree_bias(tree_bias):
    base = tree_bias[0] if tree_bias.ndim == 3 else tree_bias
    if (
        base.dtype != torch.float32
        or tuple(base.shape) != (32, 32)
        or tuple(base.stride()) != (32, 1)
    ):
        raise RuntimeError("FR13 qrow32 B1 Qrow16 reference geometry drifted")
    return torch.as_strided(
        base,
        size=(1, 32, 32),
        stride=(_FR13_FA2_QROW32_B1_QROW16_REFERENCE_SENTINEL, 32, 1),
    )


def _fr13_fa2_qrow32_b1_profile_capture_active():
    from vllm.model_executor.layers.mamba import gdn_linear_attn as _fr13_gdn

    profile_scope = getattr(
        _fr13_gdn, "_FR13_FIXED32_PROFILE_CAPTURE_SCOPE", None
    )
    if profile_scope is None:
        return False
    graph_id = (
        profile_scope.get("graph_id")
        if isinstance(profile_scope, dict)
        else None
    )
    expected_descriptor = {
        "runtime_mode": "FULL",
        "num_tokens": 32,
        "num_reqs": 1,
        "uniform": True,
        "has_lora": False,
        "num_active_loras": 0,
    }
    if (
        not isinstance(profile_scope, dict)
        or set(profile_scope) != {"descriptor", "graph_id", "completed"}
        or profile_scope.get("descriptor") != expected_descriptor
        or (
            graph_id is not None
            and (type(graph_id) is not int or graph_id <= 0)
        )
        or profile_scope.get("completed") is not False
        or getattr(
            _fr13_gdn, "_FR13_FIXED32_PROFILE_MEMORY_SCOPE", None
        ) is not True
        or getattr(
            _fr13_gdn, "_FR13_FIXED32_CAPTURE_CONTEXT", None
        ) is not None
    ):
        raise RuntimeError(
            "FR13 qrow32 B1 selector profile capture scope drifted"
        )
    return True


def _fr13_fa2_qrow32_b1_live_register(
    *, layer, flash_fn, query, key_cache, value_cache, cu_seqlens_q,
    max_seqlen_q, seqused_k, max_seqlen_k, softmax_scale, causal,
    window_size, block_table, softcap, num_splits, tree_bias,
):
    arm = _fr13_fa2_qrow32_b1_arm("FR13_FA2_QROW32_B1_LIVE_AB_ARM")
    if arm is None or _FR13_FA2_QROW32_B1_LIVE_ATTEMPTED:
        return tree_bias
    _fr13_fa2_qrow32_b1_require_draft_vocab_profile()
    _fr13_fa2_qrow32_b1_require_identity(arm)
    if _fr13_fa2_qrow32_b1_profile_capture_active():
        return _fr13_fa2_qrow32_b1_reference_tree_bias(tree_bias)
    geometry_mismatches = _fr13_fa2_qrow32_b1_geometry_mismatches(
        query=query, key_cache=key_cache, value_cache=value_cache,
        cu_seqlens_q=cu_seqlens_q, max_seqlen_q=max_seqlen_q,
        seqused_k=seqused_k, max_seqlen_k=max_seqlen_k, causal=causal,
        window_size=window_size, block_table=block_table, softcap=softcap,
        num_splits=num_splits, tree_bias=tree_bias,
    )
    if geometry_mismatches:
        query_rows = int(query.shape[0]) if query.ndim == 3 else -1
        if 0 < query_rows < 32 and int(max_seqlen_q) == query_rows:
            # The shadow gate waits for one exact physical32 event. Variable
            # speculative tails remain on the unmodified incumbent path.
            return tree_bias
        raise RuntimeError(
            "FR13 qrow32 B1 live gate geometry drifted: "
            + "; ".join(geometry_mismatches)
        )
    reference_tree_bias = _fr13_fa2_qrow32_b1_reference_tree_bias(tree_bias)
    if not (torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()):
        return reference_tree_bias
    from vllm.model_executor.layers.mamba import gdn_linear_attn as _fr13_gdn

    context = getattr(_fr13_gdn, "_FR13_FIXED32_CAPTURE_CONTEXT", None)
    if not isinstance(context, dict):
        return reference_tree_bias
    descriptor = context.get("descriptor")
    if not isinstance(descriptor, dict) or int(descriptor.get("num_reqs", -1)) != 1:
        return reference_tree_bias
    graph_id = int(context.get("graph_id", 0))
    if graph_id <= 0:
        raise RuntimeError("FR13 qrow32 B1 live gate graph identity drifted")
    layer_name = str(getattr(layer, "layer_name", ""))
    if layer_name not in _FR13_FA2_QROW32_B1_TARGET_LAYERS:
        raise RuntimeError("FR13 qrow32 B1 live gate reached a non-target layer")
    graph = _FR13_FA2_QROW32_B1_LIVE_GRAPHS.setdefault(graph_id, {})
    if layer_name in graph:
        raise RuntimeError("FR13 qrow32 B1 live target layer captured twice")
    graph[layer_name] = {
        "layer_name": layer_name, "flash_fn": flash_fn, "query": query,
        "key_cache": key_cache, "value_cache": value_cache,
        "cu_seqlens_q": cu_seqlens_q, "max_seqlen_q": int(max_seqlen_q),
        "seqused_k": seqused_k, "max_seqlen_k": int(max_seqlen_k),
        "softmax_scale": float(softmax_scale), "causal": bool(causal),
        "window_size": None if window_size is None else list(window_size),
        "block_table": block_table, "softcap": float(softcap),
        "num_splits": int(num_splits), "tree_bias": tree_bias,
    }
    return reference_tree_bias


def _fr13_fa2_qrow32_b1_live_call(bundle, out, *, arm=None):
    if arm is None:
        tree_bias = _fr13_fa2_qrow32_b1_reference_tree_bias(bundle["tree_bias"])
        num_splits = bundle["num_splits"]
    else:
        tree_bias = _fr13_fa2_qrow32_b1_candidate_tree_bias(
            bundle["tree_bias"], arm
        )
        num_splits = _FR13_FA2_QROW32_B1_ARMS[arm]["num_splits"]
    return bundle["flash_fn"](
        q=bundle["query"], k=bundle["key_cache"], v=bundle["value_cache"],
        out=out, cu_seqlens_q=bundle["cu_seqlens_q"],
        max_seqlen_q=bundle["max_seqlen_q"], seqused_k=bundle["seqused_k"],
        max_seqlen_k=bundle["max_seqlen_k"],
        softmax_scale=bundle["softmax_scale"], causal=bundle["causal"],
        alibi_slopes=None, window_size=bundle["window_size"],
        block_table=bundle["block_table"], softcap=bundle["softcap"],
        scheduler_metadata=None, fa_version=2, num_splits=num_splits,
        s_aux=None, tree_bias=tree_bias, return_softmax_lse=True,
    )


def _fr13_fa2_qrow32_b1_raw_bytes(tensor):
    return tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()


def _fr13_fa2_qrow32_b1_byte_summary(reference, candidate):
    import hashlib as _hashlib

    if reference.dtype != candidate.dtype or tuple(reference.shape) != tuple(candidate.shape):
        raise RuntimeError("FR13 qrow32 B1 comparison contract drifted")
    reference_raw = _fr13_fa2_qrow32_b1_raw_bytes(reference)
    candidate_raw = _fr13_fa2_qrow32_b1_raw_bytes(candidate)
    mismatches = abs(len(reference_raw) - len(candidate_raw)) + sum(
        left != right for left, right in zip(reference_raw, candidate_raw)
    )
    return {
        "dtype": str(reference.dtype), "shape": list(reference.shape),
        "bytes": len(reference_raw), "raw_byte_mismatches": mismatches,
        "reference_sha256": _hashlib.sha256(reference_raw).hexdigest(),
        "candidate_sha256": _hashlib.sha256(candidate_raw).hexdigest(),
    }


def _fr13_fa2_qrow32_b1_tierb_summary(reference, candidate):
    """What a Tier-B arm's live A/B records INSTEAD of byte equality.

    A byte comparison against an arm with a different reduction topology can
    only ever say "they differ", which is known before the run. This says HOW
    they differ, on the real served operands the offline probe cannot reach:
    the ULP distribution, the worst absolute disagreement expressed in bf16
    steps of the tensor's own maximum (so it is comparable with bound B3), and
    whether either arm produced a non-finite the other did not.
    """
    if reference.dtype != candidate.dtype or tuple(reference.shape) != tuple(
        candidate.shape
    ):
        raise RuntimeError("FR13 qrow32 B1 tier-b comparison contract drifted")
    ref = reference.detach().float()
    cand = candidate.detach().float()
    finite = torch.isfinite(ref) & torch.isfinite(cand)
    nonfinite_disagreements = int(
        (torch.isfinite(ref) != torch.isfinite(cand)).sum()
    )
    if reference.dtype == torch.bfloat16:
        bits_ref = reference.detach().contiguous().view(torch.int16).to(torch.int64)
        bits_cand = candidate.detach().contiguous().view(torch.int16).to(torch.int64)
        min_int = -(1 << 15)
    else:
        bits_ref = reference.detach().contiguous().view(torch.int32).to(torch.int64)
        bits_cand = candidate.detach().contiguous().view(torch.int32).to(torch.int64)
        min_int = -(1 << 31)
    key_ref = torch.where(bits_ref < 0, min_int - bits_ref, bits_ref)
    key_cand = torch.where(bits_cand < 0, min_int - bits_cand, bits_cand)
    ulp = (key_ref - key_cand).abs()[finite]
    delta = (ref - cand).abs()[finite]
    ref_max = float(ref.abs()[finite].max()) if int(finite.sum()) else 0.0
    max_abs = float(delta.max()) if delta.numel() else 0.0
    return {
        "elements": int(finite.numel()),
        "finite_elements": int(finite.sum()),
        "nonfinite_disagreements": nonfinite_disagreements,
        "ulp_le_2": int((ulp <= 2).sum()),
        "ulp_eq_0": int((ulp == 0).sum()),
        "max_abs_delta": max_abs,
        "reference_max_abs": ref_max,
        "max_abs_delta_in_bf16_eps_of_reference_max": (
            max_abs / (ref_max * (2.0 ** -8)) if ref_max > 0 else 0.0
        ),
    }


def _fr13_fa2_qrow32_b1_write(path_env, default_path, record):
    import json as _json
    from pathlib import Path as _Path

    path = _Path(os.environ.get(path_env, default_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        _json.dumps(record, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    temporary.replace(path)


def _fr13_fa2_qrow32_b1_live_replay(graph_id, runtime_mode, batch_size):
    global _FR13_FA2_QROW32_B1_LIVE_ATTEMPTED

    arm = _fr13_fa2_qrow32_b1_arm("FR13_FA2_QROW32_B1_LIVE_AB_ARM")
    if arm is None or _FR13_FA2_QROW32_B1_LIVE_ATTEMPTED:
        return
    if str(runtime_mode).upper() != "FULL" or int(batch_size) != 1:
        return
    from vllm.model_executor.layers.mamba import gdn_linear_attn as _fr13_gdn

    if not _fr13_gdn._fr13_fixed32_observed_event_active():
        return
    event = getattr(_fr13_gdn, "_FR13_FIXED32_OBSERVED_CURRENT", None)
    if not isinstance(event, dict) or int(event.get("batch_size", -1)) != 1:
        raise RuntimeError("FR13 qrow32 B1 live gate has no real B1 event")
    if os.environ.get("FR13_FA2_QROW32_B1_LIVE_AB_INSTANCE_ID", "") != (
        _FR13_FA2_QROW32_B1_CANONICAL_TASK_IDS[0]
    ):
        raise RuntimeError("FR13 qrow32 B1 live gate task identity drifted")
    if os.environ.get("FR13_FIXED32_MODE", "") not in (
        "tail6_fixed32", "hydra27_fixed32"
    ):
        raise RuntimeError("FR13 qrow32 B1 live topology drifted")
    _fr13_fa2_qrow32_b1_require_draft_vocab_profile()
    candidate_digest, source_commit, patch_source_digest = (
        _fr13_fa2_qrow32_b1_require_identity(arm)
    )
    graph = _FR13_FA2_QROW32_B1_LIVE_GRAPHS.get(int(graph_id))
    if not isinstance(graph, dict) or set(graph) != set(
        _FR13_FA2_QROW32_B1_TARGET_LAYERS
    ):
        raise RuntimeError("FR13 qrow32 B1 live gate did not retain all 16 layers")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("FR13 qrow32 B1 live gate ran inside capture")
    _FR13_FA2_QROW32_B1_LIVE_ATTEMPTED = True
    torch.cuda.synchronize()
    pending = []
    shared_seq_len = None
    tier_b = arm in _FR13_FA2_QROW32_B1_TIER_B_ARMS
    for layer_name in _FR13_FA2_QROW32_B1_TARGET_LAYERS:
        bundle = graph[layer_name]
        if not tier_b:
            # Unchanged for every byte-gated arm.
            _fr13_fa2_qrow32_b1_require_same_reduction(
                arm, bundle["num_splits"]
            )
        q_start = [int(x) for x in bundle["cu_seqlens_q"].cpu().tolist()]
        seq_lens = [int(x) for x in bundle["seqused_k"].cpu().tolist()]
        if q_start != [0, 32] or len(seq_lens) != 1 or seq_lens[0] < 32:
            raise RuntimeError("FR13 qrow32 B1 live sequence metadata drifted")
        if seq_lens[0] > bundle["max_seqlen_k"]:
            raise RuntimeError("FR13 qrow32 B1 live sequence exceeds max K")
        if shared_seq_len is None:
            shared_seq_len = seq_lens[0]
        elif shared_seq_len != seq_lens[0]:
            raise RuntimeError("FR13 qrow32 B1 live K length differs across layers")
        reference_out, reference_lse = _fr13_fa2_qrow32_b1_live_call(
            bundle, torch.empty_like(bundle["query"])
        )
        candidate_out, candidate_lse = _fr13_fa2_qrow32_b1_live_call(
            bundle, torch.empty_like(bundle["query"]), arm=arm
        )
        repeat_out = repeat_lse = None
        if tier_b:
            # THE hard gate, measured where it actually matters: on the real
            # served operands, in the real process, with fresh split
            # accumulators. The offline probe cannot reach these tensors.
            repeat_out, repeat_lse = _fr13_fa2_qrow32_b1_live_call(
                bundle, torch.empty_like(bundle["query"]), arm=arm
            )
        pending.append(
            (
                layer_name, reference_out, reference_lse,
                candidate_out, candidate_lse, repeat_out, repeat_lse,
            )
        )
    torch.cuda.synchronize()
    layers = []
    output_mismatches = 0
    lse_mismatches = 0
    tierb_layers = []
    tierb_determinism_mismatches = 0
    tierb_nonfinite = 0
    for (
        layer_name, reference_out, reference_lse, candidate_out, candidate_lse,
        repeat_out, repeat_lse,
    ) in pending:
        output = _fr13_fa2_qrow32_b1_byte_summary(reference_out, candidate_out)
        lse = _fr13_fa2_qrow32_b1_byte_summary(reference_lse, candidate_lse)
        output_mismatches += int(output["raw_byte_mismatches"])
        lse_mismatches += int(lse["raw_byte_mismatches"])
        layers.append({"layer_name": layer_name, "output": output, "lse": lse})
        if tier_b:
            repeat_output = _fr13_fa2_qrow32_b1_byte_summary(
                candidate_out, repeat_out
            )
            repeat_lse_summary = _fr13_fa2_qrow32_b1_byte_summary(
                candidate_lse, repeat_lse
            )
            repeats = int(repeat_output["raw_byte_mismatches"]) + int(
                repeat_lse_summary["raw_byte_mismatches"]
            )
            tierb_determinism_mismatches += repeats
            summary_out = _fr13_fa2_qrow32_b1_tierb_summary(
                reference_out, candidate_out
            )
            summary_lse = _fr13_fa2_qrow32_b1_tierb_summary(
                reference_lse, candidate_lse
            )
            tierb_nonfinite += int(
                summary_out["nonfinite_disagreements"]
            ) + int(summary_lse["nonfinite_disagreements"])
            tierb_layers.append({
                "layer_name": layer_name,
                "output": summary_out,
                "lse": summary_lse,
                "repeat_byte_mismatches": repeats,
            })
    tierb_record = None
    if tier_b:
        # A Tier-B arm CANNOT pass a byte comparison and is not asked to. Its
        # live verdict is the pair of things that would actually indict it:
        # the same inputs must give the same bits twice in this very process,
        # and neither arm may produce a non-finite the other did not. The
        # magnitude of the disagreement is recorded for the note and checked
        # against bound B3's scale, not re-gated here -- the bounds were
        # cleared offline, on a probe far wider than one served step.
        elements = sum(int(x["output"]["elements"]) for x in tierb_layers)
        ulp_le_2 = sum(int(x["output"]["ulp_le_2"]) for x in tierb_layers)
        ulp_eq_0 = sum(int(x["output"]["ulp_eq_0"]) for x in tierb_layers)
        tierb_record = {
            "layers": tierb_layers,
            "determinism_repeat_byte_mismatches": tierb_determinism_mismatches,
            "determinism_bitwise": tierb_determinism_mismatches == 0,
            "nonfinite_disagreements": tierb_nonfinite,
            "output_elements": elements,
            "output_ulp_le_2_fraction": (
                ulp_le_2 / elements if elements else 0.0
            ),
            "output_ulp_eq_0_fraction": (
                ulp_eq_0 / elements if elements else 0.0
            ),
            "output_max_abs_delta_in_bf16_eps_of_reference_max": max(
                (float(x["output"][
                    "max_abs_delta_in_bf16_eps_of_reference_max"
                ]) for x in tierb_layers), default=0.0
            ),
            "lse_max_abs_delta": max(
                (float(x["lse"]["max_abs_delta"]) for x in tierb_layers),
                default=0.0,
            ),
        }
        passed = (
            tierb_determinism_mismatches == 0 and tierb_nonfinite == 0
        )
    else:
        passed = output_mismatches == 0 and lse_mismatches == 0
    config = _FR13_FA2_QROW32_B1_ARMS[arm]
    identity = _fr13_fa2_qrow32_b1_identity(arm)
    record = {
        "schema": "fr13.fixed32.fa2_qrow32_b1_live_paged_ab.v2",
        "status": "PASS" if passed else "FAIL", "suite": "SWE-Verified",
        "instance_id": _FR13_FA2_QROW32_B1_CANONICAL_TASK_IDS[0],
        "concurrency": 1, "batch_size": 1, "physical_rows": 32,
        "draft_vocab_root": _fr13_fa2_qrow32_b1_draft_vocab_identity()[0],
        "draft_vocab_k": _fr13_fa2_qrow32_b1_draft_vocab_identity()[1],
        "qualification_profile": (
            _fr13_fa2_qrow32_b1_draft_vocab_identity()[2]
        ),
        "arm": arm, "selector_sentinel": config["sentinel"],
        "candidate_num_splits": config["num_splits"],
        "split_scratch_allocation": config["split_scratch_allocation"],
        "reference_selector_sentinel": (
            _FR13_FA2_QROW32_B1_QROW16_REFERENCE_SENTINEL
        ),
        "reference_dispatch": "qrow16 incumbent exact geometry; no fallback",
        "candidate_so_size": identity["candidate_size"],
        "candidate_so_sha256": candidate_digest,
        "fa2_head": identity["fa2_head"],
        "fa2_source_closure_sha256": identity["source_closure_sha256"],
        "source_commit": source_commit,
        "patch_source_sha256": patch_source_digest,
        "runtime_mode": "FULL", "graph_id": int(graph_id),
        "layer_count": len(layers), "target_layers": list(
            _FR13_FA2_QROW32_B1_TARGET_LAYERS
        ),
        "seq_len": shared_seq_len, "layers": layers,
        "output_raw_byte_mismatches": output_mismatches,
        "lse_raw_byte_mismatches": lse_mismatches,
        "candidate_dispatch": config["candidate_dispatch"],
        "tier": "B" if tier_b else "A",
        "tier_b_characterization": tierb_record,
        "tier_b_serving": (
            tier_b and _fr13_fa2_qrow32_b1_tier_b_arm() == arm
        ),
        "served_return": (
            "candidate output served (tier-b)"
            if tier_b and _fr13_fa2_qrow32_b1_tier_b_arm() == arm
            else "qrow16 captured graph output unchanged"
        ),
        "fallback_allowed": False, "performance_measurement": False,
    }
    _fr13_fa2_qrow32_b1_write(
        "FR13_FA2_QROW32_B1_LIVE_AB_JSON",
        "/logs/fr13_fa2_qrow32_b1_live_paged_ab.json", record,
    )
    if not passed:
        if tier_b:
            raise RuntimeError(
                "FR13 qrow32 B1 tier-b live gate failed: repeat_byte_"
                f"mismatches={tierb_determinism_mismatches} "
                f"nonfinite_disagreements={tierb_nonfinite}"
            )
        raise RuntimeError(
            "FR13 qrow32 B1 live byte mismatch: "
            f"output={output_mismatches} lse={lse_mismatches}"
        )


# The B1 candidate is qualified for exactly one operating point: the final
# fixed32 B1 FULL graph. Other tree-attention decodes the runtime is REQUIRED
# to execute keep the stock dispatch. The one that matters at boot is vLLM's
# memory-profiling warmup forward: with ENFORCE_EAGER=0 it runs before any
# capture exists, so it is neither capturing nor eager. Raising there killed
# engine-core init outright -- the legacy nosplit arm never hit it because it
# only ever ran eager-pinned. These are declared bypasses, not fallbacks: they
# are counted, reported in the engagement record, and cannot mask a missing
# engagement, because the capture-end hook still fails the run unless all 16
# target layers engaged the candidate in the FULL B1 graph.
_FR13_FA2_QROW32_B1_BYPASS_COUNTS = {
    "profile_capture": 0,
    "outside_capture": 0,
}


def _fr13_fa2_qrow32_b1_bypass(arm, tree_bias, num_splits, reason):
    """Serve a non-candidate operand at a non-qualified operating point."""
    if reason not in _FR13_FA2_QROW32_B1_BYPASS_COUNTS:
        raise RuntimeError("FR13 qrow32 B1 production bypass reason drifted")
    _FR13_FA2_QROW32_B1_BYPASS_COUNTS[reason] += 1
    return {
        "arm": arm,
        "candidate_served": False,
        "bypass_reason": reason,
        "profile_capture_bypass": reason == "profile_capture",
        "tree_bias": tree_bias,
        "num_splits": int(num_splits),
    }


def _fr13_fa2_qrow32_b1_production_begin(
    *, layer, query, key_cache, value_cache, cu_seqlens_q, max_seqlen_q,
    seqused_k, max_seqlen_k, causal, window_size, block_table, softcap,
    num_splits, tree_bias,
):
    arm, tier = _fr13_fa2_qrow32_b1_serving_arm()
    if arm is None:
        return None
    if os.environ.get("FR13_FA2_QROW32_B1_INTERNAL_ATTESTED") != "1":
        raise RuntimeError("FR13 qrow32 B1 production has no launcher attestation")
    _fr13_fa2_qrow32_b1_require_draft_vocab_profile()
    task_ids = _fr13_fa2_qrow32_b1_require_exact4()
    # Bind the identity of THIS arm's binary. Each production arm ships in its
    # own .so with its own source closure, so an arm-blind check would let the
    # GQA-pair selector run against the no-split binary (whose dispatch has no
    # GQA-pair gate at all, so the sentinel would be silently ignored and the
    # run would time the incumbent kernel while claiming the candidate).
    identity = _fr13_fa2_qrow32_b1_require_identity(arm)
    candidate_digest, source_commit, patch_source_digest = identity
    tier_b_credential = None
    if tier == "B":
        # The Tier-B credential is what authorises this arm's OUTPUT to reach
        # the model. It is checked before the first served token, not after.
        tier_b_credential = _fr13_fa2_qrow32_b1_tier_b_credential(arm, identity)
        pass_digest = None
    else:
        pass_digest = _fr13_fa2_qrow32_b1_digest(
            "FR13_FA2_QROW32_B1_PRODUCTION_PASS_SIDECAR_SHA256", "pass sidecar"
        )
    if _fr13_fa2_qrow32_b1_profile_capture_active():
        # Unchanged operand: the profile capture keeps serving the qrow16
        # incumbent geometry it always served. Only the accounting is new.
        return _fr13_fa2_qrow32_b1_bypass(
            arm,
            _fr13_fa2_qrow32_b1_reference_tree_bias(tree_bias),
            num_splits,
            "profile_capture",
        )
    capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    context = None
    if capturing:
        from vllm.model_executor.layers.mamba import gdn_linear_attn as _fr13_gdn

        context = getattr(_fr13_gdn, "_FR13_FIXED32_CAPTURE_CONTEXT", None)
        if not isinstance(context, dict):
            raise RuntimeError(
                "FR13 qrow32 B1 production has no final fixed32 capture context"
            )
        descriptor = context.get("descriptor")
        if not isinstance(descriptor, dict) or int(descriptor.get("num_reqs", -1)) != 1:
            raise RuntimeError("FR13 qrow32 B1 production is not final fixed32 B1")
    elif os.environ.get("ENFORCE_EAGER", "0") != "1":
        # vLLM's memory-profiling warmup forward runs at init, before any
        # graph exists, so it is neither capturing nor eager. It must serve
        # the UNTAGGED operand -- the candidate may only engage inside the
        # final FULL B1 graph -- and raising here killed engine-core init.
        return _fr13_fa2_qrow32_b1_bypass(
            arm, tree_bias, num_splits, "outside_capture"
        )
    geometry_mismatches = _fr13_fa2_qrow32_b1_geometry_mismatches(
        query=query, key_cache=key_cache, value_cache=value_cache,
        cu_seqlens_q=cu_seqlens_q, max_seqlen_q=max_seqlen_q,
        seqused_k=seqused_k, max_seqlen_k=max_seqlen_k, causal=causal,
        window_size=window_size, block_table=block_table, softcap=softcap,
        num_splits=num_splits, tree_bias=tree_bias,
    )
    if geometry_mismatches:
        raise RuntimeError(
            "FR13 qrow32 B1 production geometry drifted: "
            + "; ".join(geometry_mismatches)
        )
    if tier == "A":
        # Unchanged for every byte-qualified arm: a Tier-A serve still refuses
        # a reduction topology that differs from the reference's. Tier B is
        # exempt because differing topology is the whole point of it, and its
        # numerics are established by the credential instead -- not waived.
        _fr13_fa2_qrow32_b1_require_same_reduction(arm, num_splits)
    layer_name = str(getattr(layer, "layer_name", ""))
    if layer_name not in _FR13_FA2_QROW32_B1_TARGET_LAYERS:
        raise RuntimeError("FR13 qrow32 B1 production layer identity drifted")
    config = _FR13_FA2_QROW32_B1_ARMS[arm]
    return {
        "arm": arm, "tier": tier, "tier_b_credential": tier_b_credential,
        "candidate_served": True, "profile_capture_bypass": False,
        "tree_bias": _fr13_fa2_qrow32_b1_candidate_tree_bias(tree_bias, arm),
        "num_splits": config["num_splits"], "sentinel": config["sentinel"],
        "layer_name": layer_name, "capturing": capturing,
        "graph_id": int(context.get("graph_id", 0)) if capturing else 0,
        "candidate_so_sha256": candidate_digest,
        "source_commit": source_commit,
        "patch_source_sha256": patch_source_digest,
        "pass_sidecar_sha256": pass_digest, "task_ids": list(task_ids),
    }


def _fr13_fa2_qrow32_b1_production_end(selection, *, completed):
    arm, _tier = _fr13_fa2_qrow32_b1_serving_arm()
    if selection is None:
        if arm is not None:
            raise RuntimeError("FR13 qrow32 B1 production silently fell back")
        return
    if not completed:
        return
    reason = selection.get("bypass_reason")
    if reason is not None:
        if (
            reason not in _FR13_FA2_QROW32_B1_BYPASS_COUNTS
            or selection.get("candidate_served") is not False
            or selection.get("arm") != arm
        ):
            raise RuntimeError("FR13 qrow32 B1 production bypass drifted")
        # Sentinel-leak guard: a bypass must never have carried the retag,
        # or the candidate would have engaged at an unqualified point while
        # being accounted as stock.
        bypass_bias = selection.get("tree_bias")
        if (
            bypass_bias is not None
            and int(bypass_bias.stride(0))
            == _FR13_FA2_QROW32_B1_ARMS[arm]["sentinel"]
        ):
            raise RuntimeError(
                "FR13 qrow32 B1 production bypass carried the candidate sentinel"
            )
        return
    if selection.get("candidate_served") is not True or selection.get("arm") != arm:
        raise RuntimeError("FR13 qrow32 B1 production did not serve selected arm")
    config = _FR13_FA2_QROW32_B1_ARMS[arm]
    bias = selection["tree_bias"]
    if (
        int(selection["num_splits"]) != config["num_splits"]
        or int(bias.stride(0)) != config["sentinel"]
    ):
        raise RuntimeError("FR13 qrow32 B1 production selector was not preserved")
    if selection["capturing"]:
        graph_id = int(selection["graph_id"])
        if graph_id <= 0:
            raise RuntimeError("FR13 qrow32 B1 production graph identity drifted")
        graph = _FR13_FA2_QROW32_B1_PRODUCTION_GRAPHS.setdefault(
            graph_id, {"layers": set(), "arm": arm}
        )
        if graph["arm"] != arm or selection["layer_name"] in graph["layers"]:
            raise RuntimeError("FR13 qrow32 B1 production capture engagement drifted")
        graph["layers"].add(selection["layer_name"])
        return
    state = _FR13_FA2_QROW32_B1_EAGER_STATE
    state["calls"] = int(state["calls"]) + 1
    state["layers"].add(selection["layer_name"])
    if len(state["layers"]) == 16 and not state["emitted"]:
        record = _fr13_fa2_qrow32_b1_production_record(
            arm=arm, runtime_mode="EAGER", graph_id=0,
            graph_signature=None, layers=sorted(state["layers"]),
            calls=int(state["calls"]),
        )
        _fr13_fa2_qrow32_b1_write(
            "FR13_FA2_QROW32_B1_PRODUCTION_ENGAGEMENT_JSON",
            "/logs/fr13_fa2_qrow32_b1_production_engagement.json", record,
        )
        state["emitted"] = True


def _fr13_fa2_qrow32_b1_production_record(
    *, arm, runtime_mode, graph_id, graph_signature, layers, calls,
):
    config = _FR13_FA2_QROW32_B1_ARMS[arm]
    # Report the identity of the arm that actually ran, not the incumbent's.
    identity = _fr13_fa2_qrow32_b1_identity(arm)
    return {
        "schema": "fr13.fixed32.fa2_qrow32_b1_production_engagement.v2",
        "status": "ENGAGED", "runtime_mode": runtime_mode,
        "batch_size": 1, "physical_rows": 32, "arm": arm,
        "selector_sentinel": config["sentinel"],
        "num_splits": config["num_splits"],
        "split_scratch_allocation": config["split_scratch_allocation"],
        "graph_id": graph_id, "graph_signature": graph_signature,
        "layers": layers, "layer_count": len(layers), "calls_observed": calls,
        "candidate_so_sha256": os.environ["FR13_FA2_QROW32_B1_SO_SHA256"],
        "candidate_so_size": identity["candidate_size"],
        "fa2_head": identity["fa2_head"],
        "fa2_source_closure_sha256": identity["source_closure_sha256"],
        "source_commit": os.environ["FR13_FA2_QROW32_B1_SOURCE_COMMIT"],
        "patch_source_sha256": os.environ[
            "FR13_FA2_QROW32_B1_PATCH_SOURCE_SHA256"
        ],
        "pass_sidecar_sha256": os.environ[
            "FR13_FA2_QROW32_B1_PRODUCTION_PASS_SIDECAR_SHA256"
        ],
        "task_ids": list(_FR13_FA2_QROW32_B1_CANONICAL_TASK_IDS),
        "subset_sha256": _FR13_FA2_QROW32_B1_EXACT4_SUBSET_SHA256,
        "draft_vocab_root": 1, "draft_vocab_k": 65536,
        "candidate_served": True, "fallback_allowed": False,
        "candidate_scope": "final_fixed32_b1_full_graph_only",
        "bypass_counts": dict(sorted(_FR13_FA2_QROW32_B1_BYPASS_COUNTS.items())),
        "dispatch": config["candidate_dispatch"],
    }


def _fr13_fa2_qrow32_b1_production_capture_end(
    graph_id, graph_signature, runtime_mode, batch_size,
):
    arm, _tier = _fr13_fa2_qrow32_b1_serving_arm()
    if arm is None or graph_signature is None:
        return
    if str(runtime_mode).upper() != "FULL" or int(batch_size) != 1:
        raise RuntimeError("FR13 qrow32 B1 production captured outside FULL B1")
    graph = _FR13_FA2_QROW32_B1_PRODUCTION_GRAPHS.get(int(graph_id))
    layers = [] if not isinstance(graph, dict) else sorted(graph.get("layers", ()))
    if (
        len(layers) != 16
        or set(layers) != set(_FR13_FA2_QROW32_B1_TARGET_LAYERS)
        or graph.get("arm") != arm
    ):
        raise RuntimeError(
            "FR13 qrow32 B1 production did not capture all target tree layers"
        )
    record = _fr13_fa2_qrow32_b1_production_record(
        arm=arm, runtime_mode="FULL", graph_id=int(graph_id),
        graph_signature=str(graph_signature), layers=layers, calls=len(layers),
    )
    _fr13_fa2_qrow32_b1_write(
        "FR13_FA2_QROW32_B1_PRODUCTION_ENGAGEMENT_JSON",
        "/logs/fr13_fa2_qrow32_b1_production_engagement.json", record,
    )


'''


# The B4 GQA-pair production selector. Unlike the B4 live gate -- which always
# returns the stock captured output and exercises the candidate only in an
# offline replay -- this block injects the GQA-pair batch-stride sentinel on the
# SERVED decode call, so the forked FA2 C++ dispatch takes
# fr13_run_mha_fwd_fixed32_qrow32_gqa_pair for real exact4 traffic.
FIXED32_QUERY_TILE32_B4_PRODUCTION_HELPERS = r'''# FR13_FA2_QROW32_B4_PRODUCTION
_FR13_FA2_QROW32_B4_PRODUCTION_GRAPHS = {}
_FR13_FA2_QROW32_B4_EAGER_STATE = {
    "layers": set(),
    "calls": 0,
    "emitted": False,
}
_FR13_FA2_QROW32_B4_BATCH_STRIDE_SENTINEL = 131092
_FR13_FA2_QROW32_B4_ARMS = {
    "gqa_pair": {
        "sentinel": _FR13_FA2_QROW32_B4_BATCH_STRIDE_SENTINEL,
        "num_splits": 0,
        "candidate_dispatch": "qrow32 GQA-pair exact geometry; no fallback",
        "candidate_sha256": (
            "af9e9f24335db899468032f5b5a3eba100febe294932533cb9b87163ce2b3fdb"
        ),
        "candidate_size": 299813360,
        "fa2_head": "29210221863736a08f71a866459e368ad1ac4a95",
        "source_closure_sha256": (
            "9c3f9e751da7b783e9d07d8e40d5bc2234b99e719a1048668bd6c82244ed2d81"
        ),
    },
}
_FR13_FA2_QROW32_B4_TARGET_LAYERS = tuple(
    f"language_model.model.layers.{index}.self_attn.attn"
    for index in range(3, 64, 4)
)
_FR13_FA2_QROW32_B4_CANONICAL_TASK_IDS = (
    "astropy__astropy-12907",
    "astropy__astropy-13033",
    "astropy__astropy-13236",
    "astropy__astropy-13398",
)
_FR13_FA2_QROW32_B4_EXACT4_SUBSET_SHA256 = (
    "0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5"
)
# The 16-task pool, byte-pinned identically to fr13_floor_gate.EVIDENCE_SETS[16].
# exact4 is literally its first four entries, so this is the SAME traffic
# generator run against a deeper admission pool -- not a new workload.
_FR13_FA2_QROW32_B4_POOL16_TASK_IDS = _FR13_FA2_QROW32_B4_CANONICAL_TASK_IDS + (
    "astropy__astropy-13453",
    "astropy__astropy-13579",
    "astropy__astropy-13977",
    "astropy__astropy-14096",
    "astropy__astropy-14182",
    "astropy__astropy-14309",
    "astropy__astropy-14365",
    "astropy__astropy-14369",
    "astropy__astropy-14508",
    "astropy__astropy-14539",
    "astropy__astropy-14598",
    "astropy__astropy-14995",
)
_FR13_FA2_QROW32_B4_POOL16_SUBSET_SHA256 = (
    "47b0a3c9be49e2cb5f7e7217ae03c267a05359f269f3e3b038942f57d7dc0b5c"
)
# WHY TWO SETS ARE LEGAL AND A THIRD IS NOT.
#
# The dual raw-byte gate qualified a GEOMETRY, not a task list: the final fixed32
# B4 FULL graph at 4 slots x 32 physical rows = 128 query rows, 16 target layers,
# batch stride 0x20014.  A 16-task pool at 4 slots serves that IDENTICAL geometry
# -- MAX_NUM_SEQS is still 4, SWE_CONCURRENCY is still 4, total_query_rows is
# still 128.  What changes is only which tasks occupy the four slots and how
# often a finishing task is replaced rather than leaving the width to decay.
#
# So the binding is widened to exactly the two byte-pinned evidence sets the
# campaign already owns, and NOT to "any subset": an unpinned task list would
# let an arm serve traffic whose shape was never qualified, and the whole point
# of this predicate is that the candidate is served only where it is qualified.
_FR13_FA2_QROW32_B4_EVIDENCE_SETS = {
    4: (
        _FR13_FA2_QROW32_B4_CANONICAL_TASK_IDS,
        _FR13_FA2_QROW32_B4_EXACT4_SUBSET_SHA256,
    ),
    16: (
        _FR13_FA2_QROW32_B4_POOL16_TASK_IDS,
        _FR13_FA2_QROW32_B4_POOL16_SUBSET_SHA256,
    ),
}
# The candidate is qualified for exactly one operating point: the final fixed32
# B4 FULL graph. Every other tree-attention decode the runtime is REQUIRED to
# execute -- the memory-profile bootstrap graph, the mandatory FULL captures for
# batches 1..capacity-1, and any step routed piecewise/eager -- keeps the stock
# untagged dispatch. Those are declared bypasses, not fallbacks: they are
# counted, they are reported in the engagement record, and they cannot mask a
# missing B4 engagement because the capture-end hook still fails the run unless
# all 16 target layers engaged the candidate in the FULL B4 graph.
_FR13_FA2_QROW32_B4_BYPASS_COUNTS = {
    "profile_capture": 0,
    "non_b34_capture": 0,
    "outside_capture": 0,
}
# FR13_FA2_QROW32_B34_PADDED: the sealed .so serves ONE compile-time grid
# (3,4,4) and one canonical geometry (b == 4, total_q == 128). Width-3 FULL
# graphs are engaged by PADDING the call to that canonical geometry with a
# single inert shadow request in slot 3, never by widening the kernel. The
# shadow carries ZERO key rows, which is what makes it inert:
# compute_attn_1rowblock_splitkv takes the n_block_min >= n_block_max early
# exit (flash_fwd_kernel.h:759), writes zeros to its 32 O rows and +INF to its
# 32 LSE entries, and RETURNS BEFORE the block table is ever dereferenced
# (block_table is first formed at flash_fwd_kernel.h:872). Both of those
# writes land in private staging buffers that no consumer reads.
_FR13_FA2_QROW32_B34_WIDTHS = (3, 4)
_FR13_FA2_QROW32_B34_CANONICAL_WIDTH = 4
_FR13_FA2_QROW32_B34_ROWS = 32
_FR13_FA2_QROW32_B34_CANONICAL_ROWS = 128
# vllm.v1.attention.backends.utils.NULL_BLOCK_ID -- block 0 is popped out of
# the free queue by BlockPool.__init__ (vllm/v1/core/block_pool.py:173-177) and
# is never handed to a request, so it is the one page index that is always
# allocated and always safe to name. The shadow never reads it (seqused_k == 0)
# but the row is pinned to it rather than left stale, per fail-closed doctrine.
_FR13_FA2_QROW32_B34_NULL_BLOCK_ID = 0
# MARK'S RULING 2026-08-13. candidate_scope moves off the sealed
# final_fixed32_b4_full_graph_only token to this b3-inclusive one. It names
# what actually changed: the qualified operating points are now the FULL
# graphs at widths 3 and 4, width 3 reaching the sealed .so by padding to the
# canonical width-4 geometry with an inert shadow request. Nothing about the
# BINARY moved -- the .so sha256, its size, the six per-file source digests
# and the C++ 33-clause TORCH_CHECK are all unchanged, and the kernel is still
# only ever handed (b == 4, total_q == 128).
_FR13_FA2_QROW32_B34_CANDIDATE_SCOPE = "final_fixed32_b34_full_graph_only"
# The token the sealed +29.50 ms/step width-4 result was recorded under.
# Readers of banked artifacts must accept it; writers must not emit it.
_FR13_FA2_QROW32_B34_SEALED_B4_SCOPE = "final_fixed32_b4_full_graph_only"
_FR13_FA2_QROW32_B34_STAGING = {}
# CAVEAT 2 of the de-risk (verification_1.implementation_hazard_found): the
# staging MUST be allocated from a PRE-CAPTURE hook.
#
#   * Lazy allocate-on-first-use is UNREACHABLE. Outside capture the selector
#     takes the "outside_capture" bypass long before it reaches the padded
#     branch, so the first call that ever asks for staging is inside capture.
#   * An allocation raised inside capture comes from the graph's PRIVATE
#     memory pool. Its address is valid only while that graph owns the pool;
#     the moment another graph is captured into the same pool the staging
#     buffers silently alias somebody else's tensors. That is a corruption
#     with no exception attached to it.
#
# So allocation is done exactly once per (device, block_columns), from
# _fr13_fa2_qrow32_b34_precapture_staging, which the patcher injects into
# CUDAGraphWrapper.__call__ beside the fr10 fixed32 capture-lifecycle begin
# hook -- after torch.cuda.CUDAGraph() is constructed and BEFORE the
# torch.cuda.graph(...) context is entered. The selector never allocates: it
# uses _fr13_fa2_qrow32_b34_require_staging, which raises.
_FR13_FA2_QROW32_B34_PRECAPTURE = {
    "calls": 0,
    "allocations": [],
    "graphs": {},
}
# The widths this RUN is licensed to serve, resolved once from the credential.
_FR13_FA2_QROW32_B34_AUTHORISED = {}


def _fr13_fa2_qrow32_b34_authorised_widths():
    """The served widths the PASS SIDECAR authorises -- not the ones the code
    can do.

    _FR13_FA2_QROW32_B34_WIDTHS is a CAPABILITY: it says the padded path
    exists and what geometries it knows how to build. It is NOT a licence. The
    licence is the credential, and the credential authorises exactly the
    widths its dual raw-byte gate carries evidence for: the sealed width-4
    dual gate proves width 4 and nothing else, so a run holding it must serve
    width 4 and nothing else, however capable the code has become. Otherwise
    the widening turns padded width-3 traffic on for every timing run that
    reuses the banked width-4 credential, with the shadow doctrine never once
    byte-checked on that machine.

    So the widths are read from the credential BODY, whose bytes are pinned by
    the digest the launcher already validated on the host. A credential that
    predates the widening carries no `production_widths` and means (4,) --
    which is exactly the reading that keeps the banked lineage honest.
    """
    import hashlib as _hashlib
    import json as _json
    from pathlib import Path as _Path

    path = os.environ.get("FR13_FA2_QROW32_B4_PRODUCTION_PASS_SIDECAR", "")
    digest = _fr13_fa2_qrow32_b4_digest(
        "FR13_FA2_QROW32_B4_PRODUCTION_PASS_SIDECAR_SHA256", "pass sidecar"
    )
    cached = _FR13_FA2_QROW32_B34_AUTHORISED.get(digest)
    if cached is not None:
        return cached
    if not path:
        raise RuntimeError(
            "FR13 qrow32 B4 production has no pass sidecar path to resolve "
            "its authorised widths from"
        )
    raw = _Path(path).read_bytes()
    if _hashlib.sha256(raw).hexdigest() != digest:
        raise RuntimeError("FR13 qrow32 B4 pass sidecar bytes drifted")
    payload = _json.loads(raw)
    if not isinstance(payload, dict):
        raise RuntimeError("FR13 qrow32 B4 pass sidecar root is not an object")
    widths = payload.get(
        "production_widths", [_FR13_FA2_QROW32_B34_CANONICAL_WIDTH]
    )
    if (
        not isinstance(widths, list)
        or not widths
        or not all(isinstance(width, int) for width in widths)
        or _FR13_FA2_QROW32_B34_CANONICAL_WIDTH not in widths
        or any(width not in _FR13_FA2_QROW32_B34_WIDTHS for width in widths)
    ):
        raise RuntimeError(
            "FR13 qrow32 B4 pass sidecar production widths are not a "
            "qualified scope: " + repr(widths)
        )
    resolved = tuple(sorted(int(width) for width in widths))
    _FR13_FA2_QROW32_B34_AUTHORISED[digest] = resolved
    return resolved


def _fr13_fa2_qrow32_b4_bypass(arm, tree_bias, num_splits, reason):
    """Serve the stock untagged operand at a non-qualified operating point."""
    if reason not in _FR13_FA2_QROW32_B4_BYPASS_COUNTS:
        raise RuntimeError("FR13 qrow32 B4 production bypass reason drifted")
    _FR13_FA2_QROW32_B4_BYPASS_COUNTS[reason] += 1
    return {
        "arm": arm,
        "candidate_served": False,
        "bypass_reason": reason,
        "profile_capture_bypass": reason == "profile_capture",
        "tree_bias": tree_bias,
        "num_splits": int(num_splits),
    }


def _fr13_fa2_qrow32_b4_arm(env_name):
    arm = os.environ.get(env_name, "")
    if not arm:
        return None
    if env_name != "FR13_FA2_QROW32_B4_PRODUCTION_ARM":
        raise RuntimeError(f"unknown FR13 qrow32 B4 arm selector: {env_name}")
    if arm not in _FR13_FA2_QROW32_B4_ARMS:
        raise RuntimeError(f"{env_name} must be empty or gqa_pair; got {arm!r}")
    return arm


def _fr13_fa2_qrow32_b4_digest(env_name, label, *, length=64):
    value = os.environ.get(env_name, "")
    if len(value) != length or any(c not in "0123456789abcdef" for c in value):
        raise RuntimeError(f"FR13 qrow32 B4 {label} digest drifted")
    return value


def _fr13_fa2_qrow32_b4_require_k64():
    root = int(os.environ.get("FR13_DRAFT_VOCAB_ROOT", "0"))
    k = int(os.environ.get("FR13_DRAFT_VOCAB_K", "0"))
    if root != 1 or k != 65536:
        raise RuntimeError("FR13 qrow32 B4 production requires K64 ROOT=1")


def _fr13_fa2_qrow32_b4_require_topology():
    mode = os.environ.get("FR13_FIXED32_MODE", "")
    if mode not in ("tail6_fixed32", "hydra27_fixed32"):
        raise RuntimeError("FR13 qrow32 B4 production topology mode drifted")
    return mode


def _fr13_fa2_qrow32_b4_require_identity(arm):
    identity = _FR13_FA2_QROW32_B4_ARMS[arm]
    candidate_digest = _fr13_fa2_qrow32_b4_digest(
        "FR13_FA2_QROW32_SO_SHA256", "candidate SO"
    )
    source_commit = _fr13_fa2_qrow32_b4_digest(
        "FR13_FA2_QROW32_SOURCE_COMMIT", "source commit", length=40
    )
    patch_source = _fr13_fa2_qrow32_b4_digest(
        "FR13_FA2_QROW32_B4_PATCH_SOURCE_SHA256", "patch source"
    )
    if (
        candidate_digest != identity["candidate_sha256"]
        or int(os.environ.get("FR13_FA2_QROW32_SO_SIZE", "0"))
        != identity["candidate_size"]
        or os.environ.get("FR13_FA2_QROW32_FA2_HEAD", "") != identity["fa2_head"]
        or os.environ.get("FR13_FA2_QROW32_SOURCE_CLOSURE_SHA256", "")
        != identity["source_closure_sha256"]
    ):
        raise RuntimeError("FR13 qrow32 B4 pinned identity drifted")
    return candidate_digest, source_commit, patch_source


def _fr13_fa2_qrow32_b4_require_canonical_task_set():
    """Resolve the served evidence set, refusing anything not byte-pinned.

    Returns (task_ids, subset_sha256).  The pair must match one of the two
    campaign evidence sets ENTIRELY -- task list AND subset digest -- so a run
    cannot mix the exact4 digest with a 16-task list or vice versa.
    """
    task_ids = tuple(
        value
        for value in os.environ.get(
            "FR13_FA2_QROW32_B4_EXACT4_TASK_IDS", ""
        ).split(",")
        if value
    )
    subset = os.environ.get("FR13_FA2_QROW32_B4_EXACT4_SUBSET_SHA256", "")
    expected = _FR13_FA2_QROW32_B4_EVIDENCE_SETS.get(len(task_ids))
    if expected is None or (task_ids, subset) != expected:
        raise RuntimeError(
            "FR13 qrow32 B4 production canonical task-set identity drifted"
        )
    return task_ids, subset


def _fr13_fa2_qrow32_b4_geometry_mismatches(
    *,
    query,
    key_cache,
    value_cache,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    causal,
    window_size,
    block_table,
    softcap,
    num_splits,
    tree_bias,
    batch_size,
):
    """The exact B3/B4 INBOUND shape; nothing else.

    FR13_FA2_QROW32_B34_PADDED: this predicate qualifies the operands as the
    runtime hands them over, at width 3 or width 4. At width 4 it is the
    original exact4 predicate, clause for clause. At width 3 it qualifies the
    UNPADDED call; the padded operands that actually reach the .so are
    qualified separately and additionally by
    _fr13_fa2_qrow32_b34_staged_metadata_mismatches in the serving path, and
    by _fr13_fa2_qrow32_b34_shadow_mismatches -- the only place the shadow
    slot's VALUES are ever read -- in the pre-capture hook. Neither the
    33-clause C++ TORCH_CHECK nor this predicate reads a single seqused_k or
    block_table value, and neither does anything the padded branch runs INSIDE
    capture, where a device-to-host read would kill the graph.
    """
    if int(batch_size) not in _FR13_FA2_QROW32_B34_WIDTHS:
        return (f"batch_size={int(batch_size)!r}",)
    rows = _FR13_FA2_QROW32_B34_ROWS * int(batch_size)
    query_meta = (str(query.dtype), tuple(query.shape), tuple(query.stride()))
    key_meta = (
        str(key_cache.dtype), tuple(key_cache.shape), tuple(key_cache.stride())
    )
    value_meta = (
        str(value_cache.dtype),
        tuple(value_cache.shape),
        tuple(value_cache.stride()),
    )
    cu_seqlens_q_meta = (str(cu_seqlens_q.dtype), tuple(cu_seqlens_q.shape))
    seqused_k_meta = (str(seqused_k.dtype), tuple(seqused_k.shape))
    block_table_meta = (str(block_table.dtype), tuple(block_table.shape))
    tree_bias_meta = (
        str(tree_bias.dtype), tuple(tree_bias.shape), tuple(tree_bias.stride())
    )
    window_meta = (
        None
        if window_size is None
        else tuple(int(value) for value in window_size)
    )
    checks = (
        (
            "query(dtype,shape,stride)",
            query.dtype == torch.bfloat16
            and tuple(query.shape) == (rows, 24, 256)
            and int(query.stride(-2)) == 256
            and int(query.stride(-1)) == 1,
            query_meta,
        ),
        (
            "key_cache(dtype,shape,stride)",
            key_cache.dtype == torch.bfloat16
            and tuple(key_cache.shape[1:]) == (1024, 4, 256)
            and tuple(key_cache.stride())
            == (2 * 1024 * 4 * 256, 4 * 256, 256, 1),
            key_meta,
        ),
        (
            "value_cache(dtype,shape,stride)",
            value_cache.dtype == torch.bfloat16
            and tuple(value_cache.shape) == tuple(key_cache.shape)
            and tuple(value_cache.stride()) == tuple(key_cache.stride()),
            value_meta,
        ),
        (
            "cu_seqlens_q(dtype,shape)",
            cu_seqlens_q.dtype == torch.int32
            and tuple(cu_seqlens_q.shape) == (int(batch_size) + 1,),
            cu_seqlens_q_meta,
        ),
        (
            "seqused_k(dtype,shape)",
            seqused_k.dtype == torch.int32
            and tuple(seqused_k.shape) == (int(batch_size),),
            seqused_k_meta,
        ),
        (
            "block_table(dtype,shape)",
            block_table.dtype == torch.int32
            and block_table.ndim == 2
            and int(block_table.shape[0]) == int(batch_size),
            block_table_meta,
        ),
        (
            "tree_bias(dtype,shape,stride)",
            tree_bias.dtype == torch.float32
            and tuple(tree_bias.shape)
            in ((32, 32), (int(batch_size), 32, 32))
            and int(tree_bias.stride(-1)) == 1,
            tree_bias_meta,
        ),
        ("max_seqlen_q", int(max_seqlen_q) == 32, int(max_seqlen_q)),
        ("max_seqlen_k", int(max_seqlen_k) > 0, int(max_seqlen_k)),
        ("causal", not bool(causal), bool(causal)),
        ("softcap", float(softcap) == 0.0, float(softcap)),
        ("num_splits", int(num_splits) == 0, int(num_splits)),
        ("window_size", window_meta in (None, (-1, -1)), window_meta),
    )
    return tuple(
        f"{name}={actual!r}" for name, valid, actual in checks if not valid
    )


def _fr13_fa2_qrow32_b4_exact_geometry(**geometry):
    return not _fr13_fa2_qrow32_b4_geometry_mismatches(**geometry)


def _fr13_fa2_qrow32_b34_staging_key(device, block_columns):
    """One canonical registry key, so pre-capture and serve cannot disagree.

    torch.device("cuda") and torch.device("cuda:0") stringify differently but
    name the same allocator. Resolving the index here means a hook that saw
    the block table's device and a selector that saw the query's device
    produce the same key -- and a genuine cross-device drift still produces a
    different one.
    """
    resolved = device if isinstance(device, torch.device) else torch.device(device)
    if resolved.type != "cuda":
        raise RuntimeError(
            "FR13 qrow32 B34 staging is CUDA-only: " + repr(str(resolved))
        )
    index = resolved.index
    if index is None:
        index = int(torch.cuda.current_device())
    columns = int(block_columns)
    if columns <= 0:
        raise RuntimeError(
            "FR13 qrow32 B34 staging block columns are not positive: "
            + repr(columns)
        )
    return ("cuda:" + str(int(index)), columns)


def _fr13_fa2_qrow32_b34_staging(device, block_columns):
    """Persistent padded operands. Allocated ONCE, outside any graph pool.

    The b=3 FULL graph's query/output are NOT slices of a 128-row buffer: the
    attention layer allocates them per forward at exactly num_input_tokens
    rows (vllm/model_executor/layers/attention/attention.py:493-504,
    num_input_tokens == 96 for the width-3 decode graph). Rows [96:128) do not
    exist, so the padded call MUST run against private staging, and the
    canonical rows must be copied in and out.

    CALLED ONLY FROM THE PRE-CAPTURE HOOK. The serving path uses
    _fr13_fa2_qrow32_b34_require_staging, which raises instead of allocating.
    """
    key = _fr13_fa2_qrow32_b34_staging_key(device, block_columns)
    device = torch.device(key[0])
    block_columns = key[1]
    staged = _FR13_FA2_QROW32_B34_STAGING.get(key)
    if staged is not None:
        return staged
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "FR13 qrow32 B34 staging must be allocated before capture"
        )
    canonical = _FR13_FA2_QROW32_B34_CANONICAL_ROWS
    width = _FR13_FA2_QROW32_B34_CANONICAL_WIDTH
    rows = _FR13_FA2_QROW32_B34_ROWS
    cu_seqlens_q = torch.arange(
        0, canonical + rows, rows, dtype=torch.int32, device=device
    )
    if tuple(cu_seqlens_q.shape) != (width + 1,):
        raise RuntimeError("FR13 qrow32 B34 staged cu_seqlens_q shape drifted")
    staged = {
        "query": torch.zeros(
            (canonical, 24, 256), dtype=torch.bfloat16, device=device
        ),
        "out": torch.zeros(
            (canonical, 24, 256), dtype=torch.bfloat16, device=device
        ),
        "cu_seqlens_q": cu_seqlens_q,
        # seqused_k[3] == 0 is written once here and NEVER written again: the
        # per-step copy below touches [0:3] only. Zero key rows is what makes
        # the shadow take the kernel's early exit.
        "seqused_k": torch.zeros(
            (width,), dtype=torch.int32, device=device
        ),
        "block_table": torch.full(
            (width, int(block_columns)),
            _FR13_FA2_QROW32_B34_NULL_BLOCK_ID,
            dtype=torch.int32,
            device=device,
        ),
    }
    _FR13_FA2_QROW32_B34_STAGING[key] = staged
    return staged


def _fr13_fa2_qrow32_b34_require_staging(device, block_columns):
    """Look the staging up; NEVER allocate. The serving path's only door.

    If this raises, the pre-capture hook did not run for this operating point
    -- which means the padded call would otherwise have allocated from the
    capturing graph's private pool. Fail loud, on the capture, at startup,
    rather than serve a buffer that will alias another graph's memory.
    """
    key = _fr13_fa2_qrow32_b34_staging_key(device, block_columns)
    staged = _FR13_FA2_QROW32_B34_STAGING.get(key)
    if staged is None:
        raise RuntimeError(
            "FR13 qrow32 B34 staging was not allocated before capture for "
            + repr(key)
            + "; the pre-capture hook did not run (allocated keys: "
            + repr(sorted(_FR13_FA2_QROW32_B34_STAGING))
            + ", precapture calls: "
            + repr(int(_FR13_FA2_QROW32_B34_PRECAPTURE["calls"]))
            + ")"
        )
    return staged


def _fr13_fa2_qrow32_b34_precapture_block_table():
    """The live decode block table for a target tree layer, pre-capture.

    CUDAGraphWrapper.__call__ runs INSIDE set_forward_context, and a FULL
    capture always builds attention metadata first
    (gpu_model_runner._dummy_run: `if force_attention or
    cudagraph_runtime_mode == CUDAGraphMode.FULL:` -> _build_attention_metadata),
    so the block table the padded call will be asked to mirror is already
    resolvable at capture-begin. Reading it here -- rather than deriving
    max_model_len // block_size -- means the staged block table is the same
    width as the real one BY CONSTRUCTION, not by a re-derivation that could
    drift.
    """
    from vllm.forward_context import get_forward_context

    context = get_forward_context()
    metadata = getattr(context, "attn_metadata", None)
    if isinstance(metadata, list):
        # DP micro-batching hands over one dict per ubatch; the block table is
        # the same object in each, so the first is representative.
        metadata = metadata[0] if metadata else None
    if not isinstance(metadata, dict):
        raise RuntimeError(
            "FR13 qrow32 B34 pre-capture staging has no per-layer attention "
            "metadata: " + repr(type(metadata).__name__)
        )
    for layer_name in _FR13_FA2_QROW32_B4_TARGET_LAYERS:
        per_layer = metadata.get(layer_name)
        if per_layer is None:
            continue
        block_table = getattr(per_layer, "block_table", None)
        if block_table is None or block_table.ndim != 2:
            raise RuntimeError(
                "FR13 qrow32 B34 pre-capture staging saw a non-2D block table"
            )
        return block_table
    raise RuntimeError(
        "FR13 qrow32 B34 pre-capture staging found no target tree layer in "
        "the forward context attention metadata"
    )


def _fr13_fa2_qrow32_b34_precapture_staging(graph_id, runtime_mode, num_reqs):
    """CAVEAT 2: allocate the padded staging BEFORE the capture begins.

    Injected into CUDAGraphWrapper.__call__ immediately after
    `cudagraph = torch.cuda.CUDAGraph()` and before the `torch.cuda.graph(...)`
    context is entered -- the same lifecycle point the fr10 patcher uses for
    _fr13_fixed32_capture_begin, but strictly ahead of it, so the allocation
    can never land in the graph's private pool.

    Returns the staging dict it made available, or None when this operating
    point does not need one. It is a no-op unless the B4 production arm is
    armed, the graph is FULL, and its width is a qualified width that is not
    already the canonical width (only width 3 is padded; width 4 IS the
    canonical geometry and runs against the caller's own tensors).
    """
    arm = _fr13_fa2_qrow32_b4_arm("FR13_FA2_QROW32_B4_PRODUCTION_ARM")
    if arm is None:
        return None
    if str(runtime_mode).upper() != "FULL":
        return None
    width = int(num_reqs)
    if (
        # AUTHORISED, not merely qualified: a run whose credential licenses
        # width 4 only must not even allocate the padded staging, because the
        # selector will bypass width 3 and the 3.0 MiB would be a silent,
        # unexplained charge against the KV cache.
        width not in _fr13_fa2_qrow32_b34_authorised_widths()
        or width == _FR13_FA2_QROW32_B34_CANONICAL_WIDTH
    ):
        return None
    if _fr13_fa2_qrow32_b4_profile_capture_active():
        # The memory-profile bootstrap graph is not a qualified operating
        # point (the selector bypasses it), and allocating 3.0 MiB inside the
        # profile scope would bias the KV-cache sizing decision.
        return None
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "FR13 qrow32 B34 pre-capture staging hook ran INSIDE capture; "
            "the injection point moved and the allocation would come from "
            "the graph's private pool"
        )
    block_table = _fr13_fa2_qrow32_b34_precapture_block_table()
    key = _fr13_fa2_qrow32_b34_staging_key(
        block_table.device, int(block_table.shape[1])
    )
    fresh = key not in _FR13_FA2_QROW32_B34_STAGING
    staged = _fr13_fa2_qrow32_b34_staging(key[0], key[1])
    mismatches = _fr13_fa2_qrow32_b34_shadow_mismatches(staged, width)
    if mismatches:
        raise RuntimeError(
            "FR13 qrow32 B34 pre-capture staging is malformed: "
            + "; ".join(mismatches)
        )
    _FR13_FA2_QROW32_B34_PRECAPTURE["calls"] += 1
    _FR13_FA2_QROW32_B34_PRECAPTURE["graphs"][int(graph_id)] = {
        "batch_size": width,
        "key": list(key),
    }
    if fresh:
        _FR13_FA2_QROW32_B34_PRECAPTURE["allocations"].append(
            {
                "device": key[0],
                "block_columns": key[1],
                "batch_size": width,
                "graph_id": int(graph_id),
                # 2 x (128,24,256) bf16 = 3,145,728 B, shared by all 16 target
                # layers and by every padded width. The de-risk budgets 3.0 MiB.
                "staging_bytes": (
                    staged["query"].numel() * staged["query"].element_size()
                    + staged["out"].numel() * staged["out"].element_size()
                ),
            }
        )
    return staged


def _fr13_fa2_qrow32_b34_staged_metadata_mismatches(staged, batch_size):
    """Qualify the OUTBOUND padded operands WITHOUT reading one device value.

    CAPTURE-SAFE, and it is the only one of the two staging predicates the
    serving path may call. Every clause is tensor METADATA -- dtype, shape,
    stride, contiguity -- plus the pre-capture proof record deposited by
    _fr13_fa2_qrow32_b34_shadow_mismatches. Tensor.item() and Tensor.tolist()
    on a CUDA tensor lower to cudaMemcpyAsync D2H followed by
    cudaStreamSynchronize; cudaStreamSynchronize on a CAPTURING stream returns
    cudaErrorStreamCaptureUnsupported and invalidates the capture. This
    campaign has already lost boots 2-5 to exactly that class of call
    (fr10_phase4_patch_vllm_tree_gdn.py: a single async_copy_ready_event
    .synchronize() was the sole capture-killer), and the padded branch runs
    ONLY inside capture, so a value read here is unconditionally fatal.

    The shadow VALUES are not skipped, they are proven earlier: the
    pre-capture hook runs the value predicate outside capture and records what
    it proved, and slot `shadow` of seqused_k/block_table is never written
    again -- the per-step stage-in copies are bounded to [:batch_size].
    """
    width = _FR13_FA2_QROW32_B34_CANONICAL_WIDTH
    canonical = _FR13_FA2_QROW32_B34_CANONICAL_ROWS
    shadow = int(batch_size)
    proof = staged.get("precapture_proof")
    proof = proof.get(shadow) if isinstance(proof, dict) else None
    checks = (
        (
            "shadow_slot",
            shadow == width - 1,
            shadow,
        ),
        (
            "staged_query(dtype,shape,stride)",
            staged["query"].dtype == torch.bfloat16
            and tuple(staged["query"].shape) == (canonical, 24, 256)
            and staged["query"].is_contiguous(),
            (
                str(staged["query"].dtype),
                tuple(staged["query"].shape),
                tuple(staged["query"].stride()),
            ),
        ),
        (
            "staged_out(dtype,shape,stride)",
            staged["out"].dtype == torch.bfloat16
            and tuple(staged["out"].shape) == (canonical, 24, 256)
            and staged["out"].is_contiguous(),
            (
                str(staged["out"].dtype),
                tuple(staged["out"].shape),
                tuple(staged["out"].stride()),
            ),
        ),
        (
            "staged_cu_seqlens_q(dtype,shape)",
            staged["cu_seqlens_q"].dtype == torch.int32
            and tuple(staged["cu_seqlens_q"].shape) == (width + 1,),
            (
                str(staged["cu_seqlens_q"].dtype),
                tuple(staged["cu_seqlens_q"].shape),
            ),
        ),
        (
            "staged_seqused_k(dtype,shape)",
            staged["seqused_k"].dtype == torch.int32
            and tuple(staged["seqused_k"].shape) == (width,),
            (
                str(staged["seqused_k"].dtype),
                tuple(staged["seqused_k"].shape),
            ),
        ),
        (
            "staged_block_table(dtype,shape)",
            staged["block_table"].dtype == torch.int32
            and staged["block_table"].ndim == 2
            and int(staged["block_table"].shape[0]) == width,
            (
                str(staged["block_table"].dtype),
                tuple(staged["block_table"].shape),
            ),
        ),
        (
            # The shadow VALUES, carried forward from the pre-capture proof
            # rather than re-read. A missing or malformed record means the
            # hook did not qualify this width, and the padded call must not
            # run.
            "shadow_precapture_proof",
            proof
            == {
                "cu_seqlens_q": tuple(
                    range(
                        0,
                        canonical + _FR13_FA2_QROW32_B34_ROWS,
                        _FR13_FA2_QROW32_B34_ROWS,
                    )
                ),
                "shadow_seqused_k": 0,
                "shadow_block_table_page": (
                    _FR13_FA2_QROW32_B34_NULL_BLOCK_ID
                ),
                "shadow_block_table_row_all_null": True,
            },
            proof,
        ),
    )
    return tuple(
        f"{name}={actual!r}" for name, valid, actual in checks if not valid
    )


def _fr13_fa2_qrow32_b34_shadow_mismatches(staged, batch_size):
    """Qualify the OUTBOUND padded operands, shadow VALUES included.

    PRE-CAPTURE ONLY, and it refuses to run otherwise. It reads device values
    (.tolist()/.item()), which is capture-illegal -- see the capture-safe twin
    _fr13_fa2_qrow32_b34_staged_metadata_mismatches, which is what the serving
    path calls. On success this deposits what it proved into
    staged["precapture_proof"][shadow], so the in-capture predicate can
    require the proof without repeating the reads.
    """
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "FR13 qrow32 B34 shadow VALUE check ran inside CUDA capture; "
            ".item()/.tolist() on a capturing stream is capture-illegal"
        )
    metadata = _fr13_fa2_qrow32_b34_staged_metadata_mismatches_shape_only(
        staged, batch_size
    )
    if metadata:
        return metadata
    rows = _FR13_FA2_QROW32_B34_ROWS
    canonical = _FR13_FA2_QROW32_B34_CANONICAL_ROWS
    shadow = int(batch_size)
    expected_cu = tuple(range(0, canonical + rows, rows))
    # Read each device value EXACTLY ONCE; the tuple below is built eagerly.
    actual_cu = tuple(int(v) for v in staged["cu_seqlens_q"].tolist())
    actual_seqused = int(staged["seqused_k"][shadow].item())
    shadow_row_all_null = bool(
        (
            staged["block_table"][shadow]
            == _FR13_FA2_QROW32_B34_NULL_BLOCK_ID
        ).all().item()
    )
    actual_page = int(staged["block_table"][shadow][0].item())
    checks = (
        ("staged_cu_seqlens_q", actual_cu == expected_cu, actual_cu),
        ("shadow_seqused_k", actual_seqused == 0, actual_seqused),
        (
            "shadow_block_table_row",
            shadow_row_all_null
            and actual_page == _FR13_FA2_QROW32_B34_NULL_BLOCK_ID,
            actual_page,
        ),
    )
    mismatches = tuple(
        f"{name}={actual!r}" for name, valid, actual in checks if not valid
    )
    if not mismatches:
        staged.setdefault("precapture_proof", {})[shadow] = {
            "cu_seqlens_q": actual_cu,
            "shadow_seqused_k": actual_seqused,
            "shadow_block_table_page": actual_page,
            "shadow_block_table_row_all_null": shadow_row_all_null,
        }
    return mismatches


def _fr13_fa2_qrow32_b34_staged_metadata_mismatches_shape_only(
    staged, batch_size
):
    """The metadata predicate minus its proof clause, for the prover itself.

    The pre-capture value check must qualify shapes and dtypes BEFORE it
    indexes the shadow slot, but it cannot require the proof record it is
    about to write. Same clauses, one dropped.
    """
    return tuple(
        mismatch
        for mismatch in _fr13_fa2_qrow32_b34_staged_metadata_mismatches(
            staged, batch_size
        )
        if not mismatch.startswith("shadow_precapture_proof=")
    )


def _fr13_fa2_qrow32_b4_candidate_tree_bias(tree_bias, arm):
    """Byte-for-byte the tagged operand the dual raw-byte gate qualified.

    The tag is the batch stride itself: the forked FA2 flash_api dispatch is
    gated on params.tree_bias_batch_stride == 0x20014, so a mask laid out with
    that stride -- and only such a mask -- selects the GQA-pair kernel.
    """
    sentinel = int(_FR13_FA2_QROW32_B4_ARMS[arm]["sentinel"])
    canonical = _FR13_FA2_QROW32_B34_CANONICAL_WIDTH
    if tree_bias.dtype != torch.float32:
        raise RuntimeError("FR13 qrow32 B4 tree bias is not FP32")
    # The accepted set MUST match what the geometry predicate admits: it
    # qualifies (32,32) and (batch_size,32,32) at every qualified width, so a
    # per-slot width-3 mask reaches here and rejecting it would kill the
    # width-3 FULL capture on a shape this arm declared legal. Mirrors the
    # gate-side twin _fr13_fa2_qrow32_candidate_tree_bias clause for clause.
    if tuple(tree_bias.shape) not in ((32, 32),) + tuple(
        (int(width), 32, 32) for width in _FR13_FA2_QROW32_B34_WIDTHS
    ):
        raise RuntimeError("FR13 qrow32 B4 tree bias shape drifted")
    if int(tree_bias.stride(-1)) != 1:
        raise RuntimeError("FR13 qrow32 B4 tree bias columns are not contiguous")
    if tree_bias.ndim == 2:
        source = tree_bias.unsqueeze(0).expand(canonical, -1, -1)
    elif int(tree_bias.shape[0]) == canonical:
        source = tree_bias
    else:
        # Always four planes, because the tagged batch stride IS the dispatch
        # key and the C++ side checks tree_bias.size(0) == batch_size == 4.
        # The shadow's plane is filled from plane 0: the shadow never reads it
        # (seqused_k == 0 exits before the mask is touched), and filling it
        # deterministically rather than leaving it uninitialised is the
        # fail-closed choice.
        source = torch.cat(
            (
                tree_bias,
                tree_bias[:1].expand(
                    canonical - int(tree_bias.shape[0]), -1, -1
                ),
            ),
            dim=0,
        )
    tagged = torch.empty_strided(
        (4, 32, 32),
        (sentinel, 32, 1),
        dtype=tree_bias.dtype,
        device=tree_bias.device,
    )
    tagged.copy_(source)
    if tuple(tagged.stride()) != (sentinel, 32, 1):
        raise RuntimeError("FR13 qrow32 B4 selector stride was not preserved")
    return tagged


def _fr13_fa2_qrow32_b4_profile_capture_active():
    from vllm.model_executor.layers.mamba import gdn_linear_attn as _fr13_gdn

    profile_scope = getattr(
        _fr13_gdn, "_FR13_FIXED32_PROFILE_CAPTURE_SCOPE", None
    )
    if profile_scope is None:
        return False
    graph_id = (
        profile_scope.get("graph_id")
        if isinstance(profile_scope, dict)
        else None
    )
    if (
        not isinstance(profile_scope, dict)
        or set(profile_scope) != {"descriptor", "graph_id", "completed"}
        or not isinstance(profile_scope.get("descriptor"), dict)
        or (
            graph_id is not None
            and (type(graph_id) is not int or graph_id <= 0)
        )
        or profile_scope.get("completed") is not False
        or getattr(
            _fr13_gdn, "_FR13_FIXED32_PROFILE_MEMORY_SCOPE", None
        ) is not True
        or getattr(
            _fr13_gdn, "_FR13_FIXED32_CAPTURE_CONTEXT", None
        ) is not None
    ):
        raise RuntimeError(
            "FR13 qrow32 B4 selector profile capture scope drifted"
        )
    return True


def _fr13_fa2_qrow32_b4_write_path(path, record):
    import json as _json

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        _json.dumps(record, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    temporary.replace(path)


def _fr13_fa2_qrow32_b4_write(path_env, default_path, record):
    from pathlib import Path as _Path

    _fr13_fa2_qrow32_b4_write_path(
        _Path(os.environ.get(path_env, default_path)), record
    )


_FR13_FA2_QROW32_B34_ENGAGEMENT_ENV = (
    "FR13_FA2_QROW32_B4_PRODUCTION_ENGAGEMENT_JSON"
)
_FR13_FA2_QROW32_B34_ENGAGEMENT_DEFAULT = (
    "/logs/fr13_fa2_qrow32_b4_production_engagement.json"
)


def _fr13_fa2_qrow32_b34_engagement_path(batch_size):
    """One record per engaged width, never one overwriting the other.

    The CANONICAL width keeps the original variable and the original default
    path byte for byte: the sealed width-4 timing lineage reads exactly that
    file, and widening the scope must not move it. A padded width is written
    beside it with a `_b<width>` infix DERIVED from the same variable, so the
    launcher contract gains no new environment name and a b=3 capture can
    never clobber the b=4 engagement record.
    """
    from pathlib import Path as _Path

    base = _Path(
        os.environ.get(
            _FR13_FA2_QROW32_B34_ENGAGEMENT_ENV,
            _FR13_FA2_QROW32_B34_ENGAGEMENT_DEFAULT,
        )
    )
    width = int(batch_size)
    if width == _FR13_FA2_QROW32_B34_CANONICAL_WIDTH:
        return base
    if width not in _FR13_FA2_QROW32_B34_WIDTHS:
        raise RuntimeError(
            "FR13 qrow32 B34 engagement width is not qualified: " + repr(width)
        )
    return base.with_name(base.stem + "_b" + str(width) + base.suffix)


def _fr13_fa2_qrow32_b34_write_engagement(batch_size, record):
    _fr13_fa2_qrow32_b4_write_path(
        _fr13_fa2_qrow32_b34_engagement_path(batch_size), record
    )


def _fr13_fa2_qrow32_b4_production_begin(
    *, layer, query, key_cache, value_cache, cu_seqlens_q, max_seqlen_q,
    seqused_k, max_seqlen_k, causal, window_size, block_table, softcap,
    num_splits, tree_bias,
):
    arm = _fr13_fa2_qrow32_b4_arm("FR13_FA2_QROW32_B4_PRODUCTION_ARM")
    if arm is None:
        return None
    if os.environ.get("FR13_FA2_QROW32_B4_INTERNAL_ATTESTED") != "1":
        raise RuntimeError("FR13 qrow32 B4 production has no launcher attestation")
    _fr13_fa2_qrow32_b4_require_k64()
    fixed32_mode = _fr13_fa2_qrow32_b4_require_topology()
    task_ids, _subset_sha256 = _fr13_fa2_qrow32_b4_require_canonical_task_set()
    candidate_digest, source_commit, patch_source_digest = (
        _fr13_fa2_qrow32_b4_require_identity(arm)
    )
    pass_digest = _fr13_fa2_qrow32_b4_digest(
        "FR13_FA2_QROW32_B4_PRODUCTION_PASS_SIDECAR_SHA256", "pass sidecar"
    )
    dual_gate_digest = _fr13_fa2_qrow32_b4_digest(
        "FR13_FA2_QROW32_B4_DUAL_GATE_SHA256", "dual gate"
    )
    if _fr13_fa2_qrow32_b4_profile_capture_active():
        # Memory-profile bootstrap graphs are not exact4 traffic. Serving the
        # untagged mask there keeps the stock dispatch and emits no engagement.
        return _fr13_fa2_qrow32_b4_bypass(
            arm, tree_bias, num_splits, "profile_capture"
        )
    capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    context = None
    if capturing:
        from vllm.model_executor.layers.mamba import gdn_linear_attn as _fr13_gdn

        context = getattr(_fr13_gdn, "_FR13_FIXED32_CAPTURE_CONTEXT", None)
        if not isinstance(context, dict):
            raise RuntimeError(
                "FR13 qrow32 B4 production has no final fixed32 capture context"
            )
        descriptor = context.get("descriptor")
        if not isinstance(descriptor, dict):
            raise RuntimeError(
                "FR13 qrow32 B4 production capture descriptor drifted"
            )
        capture_num_reqs = int(descriptor.get("num_reqs", -1))
        if capture_num_reqs not in (1, 2, 3, 4):
            raise RuntimeError("FR13 qrow32 B4 production capture batch drifted")
        if capture_num_reqs not in _fr13_fa2_qrow32_b34_authorised_widths():
            # The fixed32 runtime MANDATES a FULL graph for every batch in
            # 1..capacity (fr10 freeze check), and every one of those captures
            # runs all 16 tree layers. Width 4 is always an authorised
            # operating point; width 3 is one only when the credential carries
            # the width-3 padded byte evidence, so a run holding the sealed
            # width-4 dual gate bypasses width 3 here exactly as it bypasses
            # widths 1 and 2. Widths 1 and 2 are excluded on ECONOMICS, not on
            # safety: stock already fits one wave there (24b <= 48 CTAs), so
            # the paired kernel cannot pay. The capture-end hook still fails
            # the run if an authorised graph does not engage the candidate on
            # all 16 target layers.
            return _fr13_fa2_qrow32_b4_bypass(
                arm, tree_bias, num_splits, "non_b34_capture"
            )
    elif os.environ.get("ENFORCE_EAGER", "0") != "1":
        # A step routed piecewise (a mixed prefill+decode step is routine at
        # concurrency 4) reaches this decode eagerly. The stock dispatch is
        # byte-identical there, so bypassing is correct; raising would kill the
        # server on ordinary traffic.
        return _fr13_fa2_qrow32_b4_bypass(
            arm, tree_bias, num_splits, "outside_capture"
        )
    batch_size = int(cu_seqlens_q.shape[0]) - 1
    geometry_mismatches = _fr13_fa2_qrow32_b4_geometry_mismatches(
        query=query, key_cache=key_cache, value_cache=value_cache,
        cu_seqlens_q=cu_seqlens_q, max_seqlen_q=max_seqlen_q,
        seqused_k=seqused_k, max_seqlen_k=max_seqlen_k, causal=causal,
        window_size=window_size, block_table=block_table, softcap=softcap,
        num_splits=num_splits, tree_bias=tree_bias, batch_size=batch_size,
    )
    if geometry_mismatches:
        raise RuntimeError(
            "FR13 qrow32 B4 production geometry drifted: "
            + "; ".join(geometry_mismatches)
        )
    if capturing and batch_size != capture_num_reqs:
        raise RuntimeError(
            "FR13 qrow32 B34 capture width disagrees with the served call"
        )
    staged = None
    if batch_size != _FR13_FA2_QROW32_B34_CANONICAL_WIDTH:
        if not capturing:
            # Unreachable by construction (the outside-capture bypass fires
            # first), and stated anyway: the padded branch must never enqueue
            # its stage-in/copy-back outside the graph, because then they
            # would not be graph NODES and the replay would serve stale rows.
            raise RuntimeError(
                "FR13 qrow32 B34 padded branch reached outside CUDA capture"
            )
        # NEVER allocate here -- see _FR13_FA2_QROW32_B34_PRECAPTURE. This is
        # a lookup that raises if the pre-capture hook did not run.
        staged = _fr13_fa2_qrow32_b34_require_staging(
            query.device, int(block_table.shape[1])
        )
        # METADATA ONLY. This runs inside CUDA capture, where .item()/
        # .tolist() are capture-illegal; the shadow VALUES were proven by the
        # pre-capture hook and are carried in staged["precapture_proof"],
        # which the clause below requires.
        staged_mismatches = _fr13_fa2_qrow32_b34_staged_metadata_mismatches(
            staged, batch_size
        )
        if staged_mismatches:
            raise RuntimeError(
                "FR13 qrow32 B34 staged operands drifted: "
                + "; ".join(staged_mismatches)
            )
    layer_name = str(getattr(layer, "layer_name", ""))
    if layer_name not in _FR13_FA2_QROW32_B4_TARGET_LAYERS:
        raise RuntimeError("FR13 qrow32 B4 production layer identity drifted")
    config = _FR13_FA2_QROW32_B4_ARMS[arm]
    return {
        "arm": arm, "candidate_served": True, "profile_capture_bypass": False,
        "tree_bias": _fr13_fa2_qrow32_b4_candidate_tree_bias(tree_bias, arm),
        "num_splits": config["num_splits"], "sentinel": config["sentinel"],
        "batch_size": batch_size, "staged": staged,
        "layer_name": layer_name, "capturing": capturing,
        "graph_id": int(context.get("graph_id", 0)) if capturing else 0,
        "fixed32_mode": fixed32_mode,
        "candidate_so_sha256": candidate_digest,
        "source_commit": source_commit,
        "patch_source_sha256": patch_source_digest,
        "pass_sidecar_sha256": pass_digest,
        "dual_gate_sha256": dual_gate_digest,
        "task_ids": list(task_ids),
    }


def _fr13_fa2_qrow32_b4_production_end(selection, *, completed):
    arm = _fr13_fa2_qrow32_b4_arm("FR13_FA2_QROW32_B4_PRODUCTION_ARM")
    if selection is None:
        if arm is not None:
            raise RuntimeError("FR13 qrow32 B4 production silently fell back")
        return
    if not completed:
        return
    reason = selection.get("bypass_reason")
    if reason is not None:
        if (
            reason not in _FR13_FA2_QROW32_B4_BYPASS_COUNTS
            or selection.get("candidate_served") is not False
            or selection.get("arm") != arm
        ):
            raise RuntimeError("FR13 qrow32 B4 production bypass drifted")
        return
    if selection.get("candidate_served") is not True or selection.get("arm") != arm:
        raise RuntimeError("FR13 qrow32 B4 production did not serve selected arm")
    config = _FR13_FA2_QROW32_B4_ARMS[arm]
    bias = selection["tree_bias"]
    if (
        int(selection["num_splits"]) != config["num_splits"]
        or int(bias.stride(0)) != config["sentinel"]
    ):
        raise RuntimeError("FR13 qrow32 B4 production selector was not preserved")
    if selection["capturing"]:
        graph_id = int(selection["graph_id"])
        if graph_id <= 0:
            raise RuntimeError("FR13 qrow32 B4 production graph identity drifted")
        graph = _FR13_FA2_QROW32_B4_PRODUCTION_GRAPHS.setdefault(
            graph_id,
            {
                "layers": set(),
                "arm": arm,
                "staged_layers": set(),
                "batch_size": int(selection.get("batch_size", 0)),
            },
        )
        if (
            graph["arm"] != arm
            or selection["layer_name"] in graph["layers"]
            or int(graph.get("batch_size", 0))
            != int(selection.get("batch_size", 0))
        ):
            raise RuntimeError("FR13 qrow32 B4 production capture engagement drifted")
        graph["layers"].add(selection["layer_name"])
        if selection.get("staged") is not None:
            # The stage-in and copy-back were enqueued on the capturing stream
            # inside this layer's forward, so they ARE nodes of this graph.
            # Recording it per layer is what lets the capture-end hook prove
            # that no target layer served the padded width eagerly.
            graph["staged_layers"].add(selection["layer_name"])
        return
    state = _FR13_FA2_QROW32_B4_EAGER_STATE
    state["calls"] = int(state["calls"]) + 1
    state["layers"].add(selection["layer_name"])
    if len(state["layers"]) == 16 and not state["emitted"]:
        record = _fr13_fa2_qrow32_b4_production_record(
            arm=arm, runtime_mode="EAGER", graph_id=0,
            graph_signature=None, layers=sorted(state["layers"]),
            calls=int(state["calls"]),
            batch_size=int(selection.get("batch_size", 0)),
            # ENFORCE_EAGER has no capture, so a padded width cannot get this
            # far -- the padded branch refuses to run outside capture. Derive
            # the flag from the width anyway rather than hard-coding a value
            # that is only correct because of a guard somewhere else.
            padded=(
                int(selection.get("batch_size", 0))
                != _FR13_FA2_QROW32_B34_CANONICAL_WIDTH
            ),
        )
        _fr13_fa2_qrow32_b34_write_engagement(
            int(selection.get("batch_size", 0)), record
        )
        state["emitted"] = True


def _fr13_fa2_qrow32_b4_production_record(
    *, arm, runtime_mode, graph_id, graph_signature, layers, calls,
    batch_size, padded,
):
    config = _FR13_FA2_QROW32_B4_ARMS[arm]
    task_ids, subset_sha256 = _fr13_fa2_qrow32_b4_require_canonical_task_set()
    width = int(batch_size)
    authorised = _fr13_fa2_qrow32_b34_authorised_widths()
    if width not in authorised:
        raise RuntimeError(
            "FR13 qrow32 B34 engagement record width is not authorised: "
            + repr(width)
        )
    if bool(padded) != (width != _FR13_FA2_QROW32_B34_CANONICAL_WIDTH):
        raise RuntimeError(
            "FR13 qrow32 B34 engagement record padding disagrees with width"
        )
    return {
        "schema": "fr13.fixed32.fa2_qrow32_b4_production_engagement.v1",
        "status": "ENGAGED", "runtime_mode": runtime_mode,
        # batch_size is the width the RUNTIME served. total_query_rows is the
        # width the .SO saw: at width 3 they differ by exactly the 32 shadow
        # rows, and saying so in the record is the whole disclosure.
        "batch_size": width, "concurrency": width,
        "physical_rows_per_slot": 32,
        "total_query_rows": width * _FR13_FA2_QROW32_B34_ROWS,
        "padded_to_canonical_width": bool(padded),
        "canonical_width": _FR13_FA2_QROW32_B34_CANONICAL_WIDTH,
        "canonical_query_rows": _FR13_FA2_QROW32_B34_CANONICAL_ROWS,
        "shadow_slot": (
            _FR13_FA2_QROW32_B34_CANONICAL_WIDTH - 1 if padded else None
        ),
        "shadow_seqused_k": 0 if padded else None,
        "shadow_block_table_page": (
            _FR13_FA2_QROW32_B34_NULL_BLOCK_ID if padded else None
        ),
        "staging_precapture_allocations": [
            dict(entry)
            for entry in _FR13_FA2_QROW32_B34_PRECAPTURE["allocations"]
        ],
        "arm": arm,
        "fixed32_mode": os.environ["FR13_FIXED32_MODE"],
        "selector_sentinel": config["sentinel"],
        "num_splits": config["num_splits"],
        "graph_id": graph_id, "graph_signature": graph_signature,
        "layers": layers, "layer_count": len(layers), "calls_observed": calls,
        "candidate_so_sha256": os.environ["FR13_FA2_QROW32_SO_SHA256"],
        "candidate_so_size": config["candidate_size"],
        "fa2_head": config["fa2_head"],
        "fa2_source_closure_sha256": config["source_closure_sha256"],
        "source_commit": os.environ["FR13_FA2_QROW32_SOURCE_COMMIT"],
        "patch_source_sha256": os.environ[
            "FR13_FA2_QROW32_B4_PATCH_SOURCE_SHA256"
        ],
        "pass_sidecar_sha256": os.environ[
            "FR13_FA2_QROW32_B4_PRODUCTION_PASS_SIDECAR_SHA256"
        ],
        "dual_gate_sha256": os.environ["FR13_FA2_QROW32_B4_DUAL_GATE_SHA256"],
        "task_ids": list(task_ids),
        "subset_sha256": subset_sha256,
        "task_count": len(task_ids),
        "draft_vocab_root": 1, "draft_vocab_k": 65536,
        "candidate_served": True, "fallback_allowed": False,
        # MARK'S RULING 2026-08-13: the scope token MAY move off
        # final_fixed32_b4_full_graph_only to the b3-inclusive token. The .so,
        # its six source digests and the C++ 33-clause TORCH_CHECK are
        # UNCHANGED -- the widening is python-side only, and the kernel still
        # only ever sees the canonical (b == 4, total_q == 128) geometry.
        #
        # The token reports what this run was AUTHORISED to serve, not what
        # the code can do: a run holding the sealed width-4 credential serves
        # width 4 only and says so, so the reducer that keys its treated set
        # off candidate_scope_widths cannot be told the wrong story.
        "candidate_scope": (
            _FR13_FA2_QROW32_B34_CANDIDATE_SCOPE
            if len(authorised) > 1
            else _FR13_FA2_QROW32_B34_SEALED_B4_SCOPE
        ),
        "candidate_scope_widths": list(authorised),
        "bypass_counts": dict(sorted(_FR13_FA2_QROW32_B4_BYPASS_COUNTS.items())),
        "dispatch": config["candidate_dispatch"],
    }


def _fr13_fa2_qrow32_b4_production_capture_end(
    graph_id, graph_signature, runtime_mode, batch_size,
):
    arm = _fr13_fa2_qrow32_b4_arm("FR13_FA2_QROW32_B4_PRODUCTION_ARM")
    if arm is None or graph_signature is None:
        return
    if (
        str(runtime_mode).upper() != "FULL"
        or int(batch_size) not in _fr13_fa2_qrow32_b34_authorised_widths()
    ):
        # The runtime captures a FULL graph for every batch in 1..capacity and
        # signs all of them; only the B3 and B4 graphs are qualified operating
        # points. Such a graph is not a failure -- but it must not have engaged
        # the candidate, or the sentinel leaked outside its qualification.
        if _FR13_FA2_QROW32_B4_PRODUCTION_GRAPHS.get(int(graph_id)) is not None:
            raise RuntimeError(
                "FR13 qrow32 B4 production engaged outside FULL B4"
            )
        return
    graph = _FR13_FA2_QROW32_B4_PRODUCTION_GRAPHS.get(int(graph_id))
    layers = [] if not isinstance(graph, dict) else sorted(graph.get("layers", ()))
    if (
        len(layers) != 16
        or set(layers) != set(_FR13_FA2_QROW32_B4_TARGET_LAYERS)
        or graph.get("arm") != arm
    ):
        raise RuntimeError(
            "FR13 qrow32 B4 production did not capture all target tree layers"
        )
    width = int(batch_size)
    padded = width != _FR13_FA2_QROW32_B34_CANONICAL_WIDTH
    if int(graph.get("batch_size", 0)) != width:
        raise RuntimeError(
            "FR13 qrow32 B34 capture width disagrees with the served width"
        )
    staged_layers = set(graph.get("staged_layers", ()))
    if padded:
        # CAVEAT 1/2 closure. Every target layer of a padded graph must have
        # gone through the staging branch, which means its stage-in and
        # copy-back were enqueued while this graph was capturing and are
        # therefore graph NODES -- not eager copies that would leave the
        # replay serving stale rows. A pre-capture allocation must also exist.
        if staged_layers != set(_FR13_FA2_QROW32_B4_TARGET_LAYERS):
            raise RuntimeError(
                "FR13 qrow32 B34 padded graph did not stage every target "
                "layer: " + repr(sorted(
                    set(_FR13_FA2_QROW32_B4_TARGET_LAYERS) - staged_layers
                ))
            )
        precapture = _FR13_FA2_QROW32_B34_PRECAPTURE["graphs"].get(int(graph_id))
        if (
            not isinstance(precapture, dict)
            or int(precapture.get("batch_size", -1)) != width
            or not _FR13_FA2_QROW32_B34_PRECAPTURE["allocations"]
        ):
            raise RuntimeError(
                "FR13 qrow32 B34 padded graph has no pre-capture staging "
                "record; the allocation hook did not run for this graph"
            )
    elif staged_layers:
        raise RuntimeError(
            "FR13 qrow32 B34 canonical-width graph used padded staging"
        )
    record = _fr13_fa2_qrow32_b4_production_record(
        arm=arm, runtime_mode="FULL", graph_id=int(graph_id),
        graph_signature=str(graph_signature), layers=layers, calls=len(layers),
        batch_size=width, padded=padded,
    )
    _fr13_fa2_qrow32_b34_write_engagement(width, record)


'''


FIXED32_QUERY_TILE16_PRODUCTION_HELPERS = r'''# FR13_FA2_QROW16_PRODUCTION
_FR13_FA2_QROW16_PRODUCTION_GRAPHS = {}
_FR13_FA2_QROW16_EAGER_STATE = {
    "layers": set(),
    "calls": 0,
    "emitted": False,
}
_FR13_FA2_QROW16_BATCH_STRIDE_SENTINEL = 1179791667
_FR13_FA2_QROW16_PRODUCTION_ROWS = 32


def _fr13_fa2_qrow16_candidate_tree_bias(tree_bias):
    """Tag an exact B1 bias using its semantically inert batch stride."""
    if tree_bias.dtype != torch.float32:
        raise RuntimeError("FR13 qrow16 tree bias is not FP32")
    if tuple(tree_bias.shape) not in ((32, 32), (1, 32, 32)):
        raise RuntimeError("FR13 qrow16 tree bias shape drifted")
    base = tree_bias[0] if tree_bias.ndim == 3 else tree_bias
    if int(base.stride(-1)) != 1:
        raise RuntimeError("FR13 qrow16 tree bias columns are not contiguous")
    return torch.as_strided(
        base,
        size=(1, 32, 32),
        stride=(
            _FR13_FA2_QROW16_BATCH_STRIDE_SENTINEL,
            int(base.stride(-2)),
            1,
        ),
    )


def _fr13_fa2_qrow16_production_begin(
    *,
    layer,
    query,
    key_cache,
    value_cache,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    causal,
    window_size,
    block_table,
    num_splits,
    tree_bias,
):
    """Select qrow16 on the attested exact B1 graph or SFWD eager stack."""
    if os.environ.get("FR13_FA2_QROW16_PRODUCTION", "0") != "1":
        return None
    if os.environ.get("FR13_FA2_QROW16_INTERNAL_PRODUCTION_ATTESTED") != "1":
        raise RuntimeError("FR13 qrow16 production has no launcher attestation")
    capturing = (
        torch.cuda.is_available()
        and torch.cuda.is_current_stream_capturing()
    )
    eager_state_fusion = os.environ.get(
        "FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION", "0"
    ) == "1"
    eager_conv_postprep_gate = os.environ.get(
        "FR13_FIXED32_SFWD_CONV_POSTPREP_BYTE_AB", "0"
    ) == "1"
    if not capturing and eager_state_fusion and eager_conv_postprep_gate:
        raise RuntimeError("FR13 qrow16 eager SFWD routes overlap")
    eager_sfwd_stack = (
        not capturing
        and (eager_state_fusion or eager_conv_postprep_gate)
        and os.environ.get("ENFORCE_EAGER", "0") == "1"
        and os.environ.get("FR13_DRAFT_VOCAB_ROOT", "0") == "1"
        and os.environ.get("FR13_DRAFT_VOCAB_K", "") == "65536"
    )
    if not capturing and not eager_sfwd_stack:
        return None
    if not capturing and 1 < int(query.shape[0]) < _FR13_FA2_QROW16_PRODUCTION_ROWS:
        # The tree metadata builder splits with
        # decode_threshold=tree_attn_bias.shape[0], so the *final chunk of a
        # chunked prefill* is classified as a decode whenever its length lands
        # in (1, 32]. Those segments carry the tree bias and reach this arm
        # with fewer than 32 query rows even though they are not the attested
        # fixed32 B1 spec-decode step (no drafts are scheduled). Decline so the
        # reference bias serves them, exactly as before qrow16 production was
        # engaged. A real spec step is always 32 rows, so anything that claims
        # the tree shape still gets the exact check below and fails loud.
        return None
    context = None
    if capturing:
        from vllm.model_executor.layers.mamba import (
            gdn_linear_attn as _fr13_qrow_gdn,
        )

        context = getattr(
            _fr13_qrow_gdn, "_FR13_FIXED32_CAPTURE_CONTEXT", None
        )
        if not isinstance(context, dict):
            # Throwaway memory-profile graphs deliberately have no final context.
            return None
        descriptor = context.get("descriptor")
        if (
            not isinstance(descriptor, dict)
            or int(descriptor.get("num_reqs", -1)) != 1
        ):
            raise RuntimeError("FR13 qrow16 production is not final fixed32 B1")
    sidecar_digest = os.environ.get(
        "FR13_FA2_QROW16_PRODUCTION_PASS_SIDECAR_SHA256", ""
    )
    candidate_digest = os.environ.get("FR13_FA2_QROW16_SO_SHA256", "")
    if (
        len(sidecar_digest) != 64
        or len(candidate_digest) != 64
        or any(char not in "0123456789abcdef" for char in sidecar_digest)
        or any(char not in "0123456789abcdef" for char in candidate_digest)
    ):
        raise RuntimeError("FR13 qrow16 production attestation digest drifted")
    exact = (
        query.dtype == torch.bfloat16
        and tuple(query.shape) == (32, 24, 256)
        and key_cache.dtype == torch.bfloat16
        and value_cache.dtype == torch.bfloat16
        and tuple(key_cache.shape[1:]) == (1024, 4, 256)
        and tuple(value_cache.shape) == tuple(key_cache.shape)
        and cu_seqlens_q.dtype == torch.int32
        and tuple(cu_seqlens_q.shape) == (2,)
        and seqused_k.dtype == torch.int32
        and tuple(seqused_k.shape) == (1,)
        and block_table.dtype == torch.int32
        and block_table.ndim == 2
        and int(block_table.shape[0]) == 1
        and tree_bias.dtype == torch.float32
        and tuple(tree_bias.shape) in ((32, 32), (1, 32, 32))
        and int(max_seqlen_q) == 32
        and int(max_seqlen_k) > 0
        and not bool(causal)
        and int(num_splits) in (0, 1)
    )
    if not exact:
        raise RuntimeError("FR13 qrow16 production geometry drifted")
    if window_size is not None and tuple(int(x) for x in window_size) != (-1, -1):
        raise RuntimeError("FR13 qrow16 production requires a full window")
    layer_name = str(getattr(layer, "layer_name", ""))
    if not layer_name:
        raise RuntimeError("FR13 qrow16 production layer identity drifted")
    if capturing:
        graph_id = int(context.get("graph_id", 0))
        if graph_id <= 0:
            raise RuntimeError("FR13 qrow16 production graph identity drifted")
        graph = _FR13_FA2_QROW16_PRODUCTION_GRAPHS.setdefault(
            graph_id, {"layers": set()}
        )
        if layer_name in graph["layers"]:
            raise RuntimeError(
                "FR13 qrow16 production layer executed twice in capture"
            )
        graph["layers"].add(layer_name)
    else:
        state = _FR13_FA2_QROW16_EAGER_STATE
        state["calls"] = int(state["calls"]) + 1
        state["layers"].add(layer_name)
        if len(state["layers"]) == 16 and not bool(state["emitted"]):
            import json as _json
            from pathlib import Path as _Path

            record = {
                "schema": (
                    "fr13.fixed32.fa2_qrow16_eager_production_engagement.v1"
                ),
                "status": "ENGAGED",
                "runtime_mode": "EAGER",
                "batch_size": 1,
                "layers": sorted(state["layers"]),
                "layer_count": 16,
                "calls_observed": int(state["calls"]),
                "candidate_so_sha256": os.environ[
                    "FR13_FA2_QROW16_SO_SHA256"
                ],
                "pass_sidecar_sha256": os.environ[
                    "FR13_FA2_QROW16_PRODUCTION_PASS_SIDECAR_SHA256"
                ],
                "sfwd_state_fusion_production": eager_state_fusion,
                "sfwd_conv_postprep_byte_ab": eager_conv_postprep_gate,
                "draft_vocab_root": 1,
                "draft_vocab_k": 65536,
                "dispatch": "qrow16 exact geometry; no fallback",
            }
            path = _Path("/logs/fr13_fa2_qrow16_production_capture.json")
            temporary = path.with_name(path.name + ".tmp")
            temporary.write_text(
                _json.dumps(
                    record, ensure_ascii=True, indent=2, sort_keys=True
                )
                + "\n",
                encoding="ascii",
            )
            temporary.replace(path)
            state["emitted"] = True
    return _fr13_fa2_qrow16_candidate_tree_bias(tree_bias)


def _fr13_fa2_qrow16_production_end(candidate_tree_bias):
    if candidate_tree_bias is None:
        return
    if (
        candidate_tree_bias.ndim != 3
        or int(candidate_tree_bias.shape[0]) != 1
        or int(candidate_tree_bias.stride(0))
        != _FR13_FA2_QROW16_BATCH_STRIDE_SENTINEL
    ):
        raise RuntimeError("FR13 qrow16 production selector was not preserved")


def _fr13_fa2_qrow16_production_capture_end(
    graph_id,
    graph_signature,
    runtime_mode,
    batch_size,
):
    if os.environ.get("FR13_FA2_QROW16_PRODUCTION", "0") != "1":
        return
    if graph_signature is None:
        # Throwaway profile graphs do not publish fixed32 graph signatures.
        return
    if str(runtime_mode).upper() != "FULL" or int(batch_size) != 1:
        raise RuntimeError("FR13 qrow16 production captured outside FULL B1")
    graph = _FR13_FA2_QROW16_PRODUCTION_GRAPHS.get(int(graph_id))
    layers = [] if not isinstance(graph, dict) else sorted(graph.get("layers", ()))
    if len(layers) != 16:
        raise RuntimeError(
            "FR13 qrow16 production did not capture all target tree layers: "
            + repr(layers)
        )
    import json as _json
    from pathlib import Path as _Path

    record = {
        "schema": "fr13.fixed32.fa2_qrow16_production_capture.v1",
        "status": "ENGAGED",
        "graph_id": int(graph_id),
        "graph_signature": str(graph_signature),
        "runtime_mode": "FULL",
        "batch_size": 1,
        "layers": layers,
        "layer_count": 16,
        "candidate_so_sha256": os.environ["FR13_FA2_QROW16_SO_SHA256"],
        "pass_sidecar_sha256": os.environ[
            "FR13_FA2_QROW16_PRODUCTION_PASS_SIDECAR_SHA256"
        ],
        "dispatch": "qrow16 exact geometry; no fallback",
    }
    path = _Path("/logs/fr13_fa2_qrow16_production_capture.json")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        _json.dumps(record, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    temporary.replace(path)


'''


_DFWD_UNIFIED_BM8_FALLBACK = """                unified_attention(
                    q=query[:num_decode_tokens],
                    k=key_cache,
                    v=value_cache,
                    out=output[:num_decode_tokens],
                    cu_seqlens_q=decode_meta.query_start_loc,
                    max_seqlen_q=decode_meta.max_query_len,
                    seqused_k=decode_meta.seq_lens,
                    max_seqlen_k=decode_meta.max_seq_len,
                    softmax_scale=self.scale,
                    causal=True,
                    alibi_slopes=self.alibi_slopes,
                    qq_bias=decode_meta.tree_attn_bias,
                    window_size=self.sliding_window,
                    block_table=decode_meta.block_table,
                    softcap=self.logits_soft_cap,
                    q_descale=None,  # Not supported
                    k_descale=layer._k_scale.expand(descale_shape),
                    v_descale=layer._v_scale.expand(descale_shape),
                )
"""

_DFWD_UNIFIED_BM8_GUARDED_FALLBACK = """                _fr13_dfwd_unified_bm8_production_call(
                    layer=layer,
                    q=query[:num_decode_tokens],
                    k=key_cache,
                    v=value_cache,
                    out=output[:num_decode_tokens],
                    cu_seqlens_q=decode_meta.query_start_loc,
                    max_seqlen_q=decode_meta.max_query_len,
                    seqused_k=decode_meta.seq_lens,
                    max_seqlen_k=decode_meta.max_seq_len,
                    softmax_scale=self.scale,
                    causal=True,
                    alibi_slopes=self.alibi_slopes,
                    qq_bias=decode_meta.tree_attn_bias,
                    window_size=self.sliding_window,
                    block_table=decode_meta.block_table,
                    softcap=self.logits_soft_cap,
                    q_descale=None,  # Not supported
                    k_descale=layer._k_scale.expand(descale_shape),
                    v_descale=layer._v_scale.expand(descale_shape),
                )
"""


def _patch_dfwd_unified_bm8_production_call(text: str) -> tuple[str, bool]:
    sentinel = "# FR13_DFWD_UNIFIED_BM8_PRODUCTION_CALL"
    if sentinel not in text:
        raise RuntimeError("BM8 production helper was not installed before FA2")
    try:
        tree_impl = text.split("class TreeAttentionImpl", 1)[1]
    except IndexError as exc:
        raise RuntimeError("BM8 production TreeAttentionImpl is missing") from exc
    fa2_route = (
        'if os.environ.get("FR13_FA2_TREE_BIAS", "0") == "1" '
        "and use_tree_bias:"
    )
    if fa2_route not in tree_impl or "flash_attn_varlen_func(" not in tree_impl:
        raise RuntimeError("BM8 production target FA2 decode route is missing")
    guarded_calls = tree_impl.count(
        "_fr13_dfwd_unified_bm8_production_call("
    )
    if guarded_calls:
        if guarded_calls != 1:
            raise RuntimeError("BM8 production guarded fallback is not unique")
        if _DFWD_UNIFIED_BM8_FALLBACK in tree_impl:
            raise RuntimeError("BM8 production stock fallback was retained")
        return text, False
    if text.count(_DFWD_UNIFIED_BM8_FALLBACK) != 1:
        raise RuntimeError("BM8 production unified fallback is not unique")
    return (
        text.replace(
            _DFWD_UNIFIED_BM8_FALLBACK,
            _DFWD_UNIFIED_BM8_GUARDED_FALLBACK,
            1,
        ),
        True,
    )


def _patch_tree_attn(
    path: Path,
    *,
    fixed32_query_tile16_live_ab: bool = False,
    fixed32_query_tile32_live_ab: bool = False,
    fixed32_query_tile32_b1_live_ab: bool = False,
    fixed32_query_tile32_b1_production: bool = False,
    fixed32_query_tile32_b4_production: bool = False,
    fixed32_query_tile16_production: bool = False,
    dfwd_unified_bm8_production: bool = False,
) -> bool:
    # Every private FA2 decode selector rewrites the SAME
    # `if not _fr13_reordered:` decode call. Two of them installed together
    # would silently produce a first-one-wins call site, so the invariant is
    # enforced here -- at the only place that edits tree_attn.py -- rather than
    # left to the argparse front door, which is not the only caller.
    _private_selectors = {
        "fixed32_query_tile16_live_ab": fixed32_query_tile16_live_ab,
        "fixed32_query_tile32_live_ab": fixed32_query_tile32_live_ab,
        "fixed32_query_tile32_b1_live_ab": fixed32_query_tile32_b1_live_ab,
        "fixed32_query_tile32_b1_production": fixed32_query_tile32_b1_production,
        "fixed32_query_tile32_b4_production": fixed32_query_tile32_b4_production,
        "fixed32_query_tile16_production": fixed32_query_tile16_production,
    }
    _active_selectors = sorted(
        name for name, value in _private_selectors.items() if value
    )
    if len(_active_selectors) > 1:
        raise ValueError(
            "tree_attn private FA2 decode selectors are mutually exclusive: "
            + ", ".join(_active_selectors)
        )
    text = path.read_text()
    changed = False
    old_filter = (
        '        if want and want != "*" and not layer_name.startswith(want):\n'
        "            return\n"
    )
    new_filter = (
        '        if want and want != "*" and layer_name and not layer_name.startswith(want):\n'
        "            return\n"
    )
    if old_filter in text:
        text = text.replace(old_filter, new_filter)
        changed = True
    if "import os\n" not in text:
        text = text.replace("import ast\n", "import ast\nimport os\n", 1)
        changed = True
    text, did = _insert_once(
        text,
        "from vllm.v1.attention.ops.triton_unified_attention import unified_attention\n",
        "from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func\n",
        "tree_attn fa_utils import",
    )
    changed = changed or did
    # FR13_FA2_PREFILL_NATIVE needs the same helpers native FlashAttentionImpl
    # uses to pick the FA version / batch-invariant num_splits.  Mirror native's
    # imports from vllm.v1.attention.backends.fa_utils (get_flash_attn_version)
    # and vllm.envs.
    text, did = _insert_once(
        text,
        "from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func\n",
        "from vllm.v1.attention.backends.fa_utils import get_flash_attn_version\n",
        "tree_attn get_flash_attn_version import",
    )
    changed = changed or did
    # FR13_FA2_SPINE_REORDER: dense-suffix hybrid (context paged + suffix dense
    # permuted spine-first + merge_attn_states).  merge_attn_states is the shipped
    # vLLM cascade merge; hybrid_reorder_decode is our CPU-tested pure helper.
    text, did = _insert_once(
        text,
        "from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func\n",
        "from vllm.v1.attention.ops.merge_attn_states import merge_attn_states\n",
        "tree_attn merge_attn_states import",
    )
    changed = changed or did
    text, did = _insert_once(
        text,
        "from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func\n",
        "from lumo_flywheel_serving.fr13_fa2_spine_reorder import hybrid_reorder_decode\n",
        "tree_attn spine-reorder helper import",
    )
    changed = changed or did
    # FR13_SLOT_REORDER (edits 2+5/5): decode-call bias column permutation +
    # causal-redundancy flag. Companions of the runner-side slot permutation
    # (fr10_phase4_patch_vllm_tree_gdn.py _patch_gpu_model_runner_slot_reorder)
    # and the committer dst-flat fix (fr10_gdn_tree_kernel.py dst_pi). Design +
    # hazard log: FR13_CAT6_CAT8_ACCEPT_INVESTIGATION.md (v2).
    sr_helpers = '''# FR13_SLOT_REORDER (edits 2+5/5): verify-bias column permutation + causal flag.
_FR13_SR_PI = None
_FR13_SR_BIAS_CACHE = {}
_FR13_SR_CAUSAL = None


def _fr13_sr_pi_list():
    # pi = [spine depth-order, then branches BFS] over [root]+sorted(choices);
    # SAME algorithm text as the runner-side permute (both parse SPEC_CONFIG and
    # log pi at boot -- the gate script cross-asserts the two log lines).
    global _FR13_SR_PI
    if _FR13_SR_PI is None:
        import ast as _ast
        import json as _json
        import os as _os
        _cfg = _os.environ.get("SPEC_CONFIG")
        _tree = _json.loads(_cfg).get("speculative_token_tree") if _cfg else None
        if not _tree:
            raise RuntimeError("FR13_SLOT_REORDER: no speculative_token_tree")
        _ch = sorted(_ast.literal_eval(_tree), key=lambda _p: (len(_p), _p))
        _pi = (
            [0]
            + [_i + 1 for _i, _c in enumerate(_ch) if all(int(_x) == 0 for _x in _c)]
            + [_i + 1 for _i, _c in enumerate(_ch) if not all(int(_x) == 0 for _x in _c)]
        )
        assert _pi[0] == 0 and sorted(_pi) == list(range(len(_ch) + 1)), _pi
        _FR13_SR_PI = _pi
        logger.info(
            "FR13_SLOT_REORDER ENGAGED (tree_attn bias): tree_n=%d pi=%s",
            len(_pi), _pi,
        )
    return _FR13_SR_PI


def _fr13_sr_bias_perm(bias):
    # Column-permute the VERIFY tree bias (KEY axis only; query rows stay BFS)
    # to match the spine-first slot layout: new col k <- node pi[k]. Cache keyed
    # by (id, data_ptr) => one gather per boot, stable output tensor (safe under
    # cudagraph capture). SHAPE GUARD: only the full [tree_n, tree_n] verify
    # bias is permuted -- build_for_drafting slices (rows/cols < tree_n, draft
    # path must stay BFS) pass through untouched.
    _pi = _fr13_sr_pi_list()
    _n = len(_pi)
    if bias.shape[-1] != _n or bias.shape[-2] != _n:
        return bias
    _key = (id(bias), bias.data_ptr())
    _hit = _FR13_SR_BIAS_CACHE.get(_key)
    if _hit is None:
        _pit = torch.tensor(_pi, dtype=torch.long, device=bias.device)
        _hit = bias.index_select(-1, _pit).contiguous()
        _FR13_SR_BIAS_CACHE[_key] = _hit
    return _hit


def _fr13_sr_causal_flag():
    # causal for the decode tree-verify flash call. False when FR13_SLOT_REORDER
    # =1 or FR13_TREE_CAUSAL_OFF=1 (stage gate). PROVABLY REDUNDANT here: all
    # context cols precede all tree rows (causal never fires on context), and in
    # the BFS suffix the ancestry bias is already -inf at every col the causal
    # mask would hit (anc(m) is a subset of {0..m}) => False is byte-identical
    # today, and it unlocks the permuted layout (branch self-cols sit beyond the
    # old diagonal). Env read once (static per boot).
    global _FR13_SR_CAUSAL
    if _FR13_SR_CAUSAL is None:
        import os as _os
        _FR13_SR_CAUSAL = not (
            _os.environ.get("FR13_SLOT_REORDER", "0") == "1"
            or _os.environ.get("FR13_TREE_CAUSAL_OFF", "0") == "1"
        )
        if not _FR13_SR_CAUSAL:
            logger.info(
                "FR13_SLOT_REORDER/TREE_CAUSAL_OFF: decode tree causal=False"
            )
    return _FR13_SR_CAUSAL


'''
    text, did = _insert_once(
        text,
        "def _get_depth_counts(",
        sr_helpers,
        "tree_attn slot-reorder helpers",
    )
    changed = changed or did
    if fixed32_query_tile16_live_ab:
        text, did = _insert_once(
            text,
            "def _get_depth_counts(",
            FIXED32_QUERY_TILE16_LIVE_AB_HELPERS,
            "qrow16 live paged A/B helpers",
        )
        changed = changed or did
    if fixed32_query_tile32_live_ab:
        text, did = _insert_once(
            text,
            "def _get_depth_counts(",
            FIXED32_QUERY_TILE32_LIVE_AB_HELPERS,
            "qrow32 live paged exact4 A/B helpers",
        )
        changed = changed or did
    if fixed32_query_tile32_b1_live_ab or fixed32_query_tile32_b1_production:
        text, did = _insert_once(
            text,
            "def _get_depth_counts(",
            FIXED32_QUERY_TILE32_B1_SELECTOR_HELPERS,
            "qrow32 B1 live and production selector helpers",
        )
        changed = changed or did
    if fixed32_query_tile32_b4_production:
        text, did = _insert_once(
            text,
            "def _get_depth_counts(",
            FIXED32_QUERY_TILE32_B4_PRODUCTION_HELPERS,
            "qrow32 B4 GQA-pair production selector helpers",
        )
        changed = changed or did
    if fixed32_query_tile16_production:
        text, did = _insert_once(
            text,
            "def _get_depth_counts(",
            FIXED32_QUERY_TILE16_PRODUCTION_HELPERS,
            "qrow16 attested production helpers",
        )
        changed = changed or did
    if "import vllm.envs as envs\n" not in text:
        text = text.replace(
            "from vllm.v1.attention.ops.triton_unified_attention import unified_attention\n",
            "from vllm.v1.attention.ops.triton_unified_attention import unified_attention\nimport vllm.envs as envs\n",
            1,
        )
        changed = True
    # FR13_FA2_PREFILL_NATIVE: route the TREE_ATTN *prefill* branch through the
    # exact same flash_attn_varlen_func(...) call native FlashAttentionImpl
    # prefill makes (NO tree bias), so prefill is byte-identical to native
    # FLASH_ATTN.  Flag-gated, default OFF.  Decode path is untouched.
    prefill_anchor = (
        "        if prefill_meta := attn_metadata.prefill_metadata:\n"
        "            unified_attention(\n"
        "                q=query[num_decode_tokens:num_actual_tokens],\n"
        "                k=key_cache,\n"
        "                v=value_cache,\n"
        "                out=output[num_decode_tokens:num_actual_tokens],\n"
        "                cu_seqlens_q=prefill_meta.query_start_loc,\n"
        "                max_seqlen_q=prefill_meta.max_query_len,\n"
        "                seqused_k=prefill_meta.seq_lens,\n"
        "                max_seqlen_k=prefill_meta.max_seq_len,\n"
        "                softmax_scale=self.scale,\n"
        "                causal=True,\n"
        "                alibi_slopes=self.alibi_slopes,\n"
        "                window_size=self.sliding_window,\n"
        "                block_table=prefill_meta.block_table,\n"
        "                softcap=self.logits_soft_cap,\n"
        "                q_descale=None,  # Not supported\n"
        "                k_descale=layer._k_scale.expand(descale_shape),\n"
        "                v_descale=layer._v_scale.expand(descale_shape),\n"
        "            )\n"
    )
    prefill_replacement = (
        "        if prefill_meta := attn_metadata.prefill_metadata:\n"
        '            if os.environ.get("FR13_FA2_PREFILL_NATIVE", "0") == "1":\n'
        "                # FR13_FA2_PREFILL_NATIVE: mirror native FlashAttentionImpl\n"
        "                # prefill (flash_attn.py: the non-cascade else branch)\n"
        "                # call-for-call, with NO tree bias.  descale_shape uses\n"
        "                # the per-prefill cu_seqlens and self.num_kv_heads exactly\n"
        "                # like native (flash_attn.py descale_shape line).\n"
        "                prefill_descale_shape = (\n"
        "                    prefill_meta.query_start_loc.shape[0] - 1,\n"
        "                    self.num_kv_heads,\n"
        "                )\n"
        "                prefill_sliding_window_size = (\n"
        "                    list(self.sliding_window)\n"
        "                    if self.sliding_window is not None\n"
        "                    else None\n"
        "                )\n"
        "                flash_attn_varlen_func(\n"
        "                    q=query[num_decode_tokens:num_actual_tokens],\n"
        "                    k=key_cache,\n"
        "                    v=value_cache,\n"
        "                    out=output[num_decode_tokens:num_actual_tokens],\n"
        "                    cu_seqlens_q=prefill_meta.query_start_loc,\n"
        "                    max_seqlen_q=prefill_meta.max_query_len,\n"
        "                    seqused_k=prefill_meta.seq_lens,\n"
        "                    max_seqlen_k=prefill_meta.max_seq_len,\n"
        "                    softmax_scale=self.scale,\n"
        "                    causal=True,\n"
        "                    alibi_slopes=self.alibi_slopes,\n"
        "                    window_size=prefill_sliding_window_size,\n"
        "                    block_table=prefill_meta.block_table,\n"
        "                    softcap=self.logits_soft_cap,\n"
        "                    scheduler_metadata=None,\n"
        "                    fa_version=get_flash_attn_version(\n"
        "                        requires_alibi=self.alibi_slopes is not None,\n"
        "                        head_size=self.head_size,\n"
        "                    ),\n"
        "                    q_descale=layer._q_scale.expand(prefill_descale_shape),\n"
        "                    k_descale=layer._k_scale.expand(prefill_descale_shape),\n"
        "                    v_descale=layer._v_scale.expand(prefill_descale_shape),\n"
        "                    num_splits=1 if envs.VLLM_BATCH_INVARIANT else 0,\n"
        "                    s_aux=None,\n"
        "                )\n"
        "            else:\n"
        "                unified_attention(\n"
        "                    q=query[num_decode_tokens:num_actual_tokens],\n"
        "                    k=key_cache,\n"
        "                    v=value_cache,\n"
        "                    out=output[num_decode_tokens:num_actual_tokens],\n"
        "                    cu_seqlens_q=prefill_meta.query_start_loc,\n"
        "                    max_seqlen_q=prefill_meta.max_query_len,\n"
        "                    seqused_k=prefill_meta.seq_lens,\n"
        "                    max_seqlen_k=prefill_meta.max_seq_len,\n"
        "                    softmax_scale=self.scale,\n"
        "                    causal=True,\n"
        "                    alibi_slopes=self.alibi_slopes,\n"
        "                    window_size=self.sliding_window,\n"
        "                    block_table=prefill_meta.block_table,\n"
        "                    softcap=self.logits_soft_cap,\n"
        "                    q_descale=None,  # Not supported\n"
        "                    k_descale=layer._k_scale.expand(descale_shape),\n"
        "                    v_descale=layer._v_scale.expand(descale_shape),\n"
        "                )\n"
    )
    if prefill_anchor in text:
        text = text.replace(prefill_anchor, prefill_replacement, 1)
        changed = True
    elif 'os.environ.get("FR13_FA2_PREFILL_NATIVE", "0") == "1"' not in text:
        raise RuntimeError(
            "FR13_FA2_PREFILL_NATIVE: tree_attn prefill anchor not found "
            "(unified_attention prefill branch already mutated?)"
        )
    anchor = """        if decode_meta := attn_metadata.decode_metadata:\n            unified_attention(\n                q=query[:num_decode_tokens],\n                k=key_cache,\n                v=value_cache,\n                out=output[:num_decode_tokens],\n                cu_seqlens_q=decode_meta.query_start_loc,\n                max_seqlen_q=decode_meta.max_query_len,\n                seqused_k=decode_meta.seq_lens,\n                max_seqlen_k=decode_meta.max_seq_len,\n                softmax_scale=self.scale,\n                causal=True,\n                alibi_slopes=self.alibi_slopes,\n                qq_bias=decode_meta.tree_attn_bias,\n                window_size=self.sliding_window,\n                block_table=decode_meta.block_table,\n                softcap=self.logits_soft_cap,\n                q_descale=None,  # Not supported\n                k_descale=layer._k_scale.expand(descale_shape),\n                v_descale=layer._v_scale.expand(descale_shape),\n            )\n"""
    replacement = """        if decode_meta := attn_metadata.decode_metadata:\n            tree_bias = decode_meta.tree_attn_bias\n            use_tree_bias = tree_bias is not None and tree_bias.numel() > 0\n            if use_tree_bias and os.environ.get("FR13_SLOT_REORDER", "0") == "1":\n                # FR13_SLOT_REORDER (edit 2/5): key-axis column permutation to\n                # match the spine-first slot layout (rows stay BFS). Cached\n                # gather; draft-slice shapes pass through untouched.\n                tree_bias = _fr13_sr_bias_perm(tree_bias)\n            if os.environ.get("FR13_FA2_TREE_BIAS", "0") == "1" and use_tree_bias:\n                if self.alibi_slopes is not None:\n                    raise NotImplementedError("FR13_FA2_TREE_BIAS does not support ALiBi")\n                sliding_window_size = (\n                    list(self.sliding_window)\n                    if self.sliding_window is not None\n                    else None\n                )\n                _fr13_reordered = False\n                if os.environ.get("FR13_FA2_SPINE_REORDER", "0") in ("1", "2", "3"):\n                    _fr13_reordered = hybrid_reorder_decode(\n                        query=query[:num_decode_tokens],\n                        key=key[:num_decode_tokens],\n                        value=value[:num_decode_tokens],\n                        key_cache=key_cache,\n                        value_cache=value_cache,\n                        output=output[:num_decode_tokens],\n                        cu_seqlens_q=decode_meta.query_start_loc,\n                        seq_lens=decode_meta.seq_lens,\n                        max_query_len=decode_meta.max_query_len,\n                        max_seq_len=decode_meta.max_seq_len,\n                        block_table=decode_meta.block_table,\n                        tree_bias=tree_bias,\n                        scale=self.scale,\n                        softcap=self.logits_soft_cap,\n                        sliding_window_size=sliding_window_size,\n                        flash_fn=flash_attn_varlen_func,\n                        merge_fn=merge_attn_states,\n                        fa_version=2,\n                        split_only=(os.environ.get("FR13_FA2_SPINE_REORDER", "0") in ("2", "3")),\n                        self_check=(os.environ.get("FR13_FA2_SPINE_REORDER", "0") == "3"),\n                    )\n                if not _fr13_reordered:\n                    flash_attn_varlen_func(\n                        q=query[:num_decode_tokens],\n                        k=key_cache,\n                        v=value_cache,\n                        out=output[:num_decode_tokens],\n                        cu_seqlens_q=decode_meta.query_start_loc,\n                        max_seqlen_q=decode_meta.max_query_len,\n                        seqused_k=decode_meta.seq_lens,\n                        max_seqlen_k=decode_meta.max_seq_len,\n                        softmax_scale=self.scale,\n                        causal=_fr13_sr_causal_flag(),\n                        alibi_slopes=None,\n                        window_size=sliding_window_size,\n                        block_table=decode_meta.block_table,\n                        softcap=self.logits_soft_cap,\n                        fa_version=2,\n                        tree_bias=tree_bias,\n                    )\n            else:\n                unified_attention(\n                    q=query[:num_decode_tokens],\n                    k=key_cache,\n                    v=value_cache,\n                    out=output[:num_decode_tokens],\n                    cu_seqlens_q=decode_meta.query_start_loc,\n                    max_seqlen_q=decode_meta.max_query_len,\n                    seqused_k=decode_meta.seq_lens,\n                    max_seqlen_k=decode_meta.max_seq_len,\n                    softmax_scale=self.scale,\n                    causal=True,\n                    alibi_slopes=self.alibi_slopes,\n                    qq_bias=decode_meta.tree_attn_bias,\n                    window_size=self.sliding_window,\n                    block_table=decode_meta.block_table,\n                    softcap=self.logits_soft_cap,\n                    q_descale=None,  # Not supported\n                    k_descale=layer._k_scale.expand(descale_shape),\n                    v_descale=layer._v_scale.expand(descale_shape),\n                )\n"""
    if anchor in text:
        text = text.replace(anchor, replacement, 1)
        did = True
    elif (
        'if os.environ.get("FR13_FA2_TREE_BIAS", "0") == "1":' in text
        and "use_tree_bias = tree_bias is not None and tree_bias.numel() > 0" not in text
    ):
        text = text.replace(
            '        if decode_meta := attn_metadata.decode_metadata:\n'
            '            if os.environ.get("FR13_FA2_TREE_BIAS", "0") == "1":\n',
            '        if decode_meta := attn_metadata.decode_metadata:\n'
            '            tree_bias = decode_meta.tree_attn_bias\n'
            '            use_tree_bias = tree_bias is not None and tree_bias.numel() > 0\n'
            '            if os.environ.get("FR13_FA2_TREE_BIAS", "0") == "1" and use_tree_bias:\n',
            1,
        )
        text = text.replace("                    tree_bias=decode_meta.tree_attn_bias,\n", "                    tree_bias=tree_bias,\n", 1)
        did = True
    elif (
        "flash_attn_varlen_func(\n" in text
        and 'globals().get("_fr13_tree_attn_op_capture")' not in text
    ):
        text = text.replace(
            "                    tree_bias=tree_bias,\n                )\n            else:\n",
            """                    tree_bias=tree_bias,
                )
            else:
""",
            1,
        )
        did = True
    else:
        did = False
    changed = changed or did
    # FR13 swapped-mode hardening: gate use_tree_bias on max_query_len > 1.
    # FA2 swapped mode (seqlenq_ngroups_swapped, requires max_seqlen_q == 1)
    # reassigns max_seqlen_q before set_params_tree_bias, so a hypothetical
    # all-1-token decode segment with a non-empty bias would mis-address the
    # bias; route it to the unified_attention fallback instead.  Deployed tree
    # decode has max_query_len == tree_len > 1, so behavior there is unchanged.
    old_gate = (
        "            use_tree_bias = tree_bias is not None and tree_bias.numel() > 0\n"
    )
    new_gate = (
        "            use_tree_bias = (\n"
        "                tree_bias is not None\n"
        "                and tree_bias.numel() > 0\n"
        "                and decode_meta.max_query_len > 1\n"
        "            )\n"
    )
    if old_gate in text:
        text = text.replace(old_gate, new_gate, 1)
        changed = True
    # FR13_BI_TREE_ATTN EDIT 2: under VLLM_BATCH_INVARIANT force non-split
    # dispatch on the tree-bias decode call, mirroring native FLASH_ATTN's
    # batch-invariant num_splits expression (provably inert for tree shapes:
    # max_seqlen_q = tree_len > 1 keeps num_splits at 0 anyway; this closes
    # the hypothetical all-1-token split-kv gap).  Evaluates to the previous
    # default 0 when VLLM_BATCH_INVARIANT is unset.
    old_call_tail = (
        "                    fa_version=2,\n"
        "                    tree_bias=tree_bias,\n"
    )
    new_call_tail = (
        "                    fa_version=2,\n"
        "                    num_splits=1 if envs.VLLM_BATCH_INVARIANT else 0,\n"
        "                    tree_bias=tree_bias,\n"
    )
    if old_call_tail in text:
        text = text.replace(old_call_tail, new_call_tail, 1)
        changed = True
    if fixed32_query_tile16_live_ab and (
        "_fr13_fa2_qrow16_live_ab_register(\n" not in text.split(
            "class TreeAttentionImpl", 1
        )[-1]
    ):
        live_call_anchor = (
            "                if not _fr13_reordered:\n"
            "                    flash_attn_varlen_func(\n"
        )
        live_call_replacement = """                if not _fr13_reordered:
                    _fr13_fa2_qrow16_live_ab_register(
                        layer=layer,
                        flash_fn=flash_attn_varlen_func,
                        query=query[:num_decode_tokens],
                        key_cache=key_cache,
                        value_cache=value_cache,
                        cu_seqlens_q=decode_meta.query_start_loc,
                        max_seqlen_q=decode_meta.max_query_len,
                        seqused_k=decode_meta.seq_lens,
                        max_seqlen_k=decode_meta.max_seq_len,
                        softmax_scale=self.scale,
                        causal=_fr13_sr_causal_flag(),
                        window_size=sliding_window_size,
                        block_table=decode_meta.block_table,
                        softcap=self.logits_soft_cap,
                        num_splits=1 if envs.VLLM_BATCH_INVARIANT else 0,
                        tree_bias=tree_bias,
                    )
                    flash_attn_varlen_func(
"""
        if text.count(live_call_anchor) != 1:
            raise RuntimeError(
                "qrow16 live paged A/B decode-call anchor is not unique"
            )
        text = text.replace(live_call_anchor, live_call_replacement, 1)
        changed = True
    if fixed32_query_tile32_live_ab and (
        "_fr13_fa2_qrow32_live_ab_register(\n" not in text.split(
            "class TreeAttentionImpl", 1
        )[-1]
    ):
        live_call_anchor = (
            "                if not _fr13_reordered:\n"
            "                    flash_attn_varlen_func(\n"
        )
        live_call_replacement = """                if not _fr13_reordered:
                    _fr13_fa2_qrow32_live_ab_register(
                        layer=layer,
                        flash_fn=flash_attn_varlen_func,
                        query=query[:num_decode_tokens],
                        key_cache=key_cache,
                        value_cache=value_cache,
                        cu_seqlens_q=decode_meta.query_start_loc,
                        max_seqlen_q=decode_meta.max_query_len,
                        seqused_k=decode_meta.seq_lens,
                        max_seqlen_k=decode_meta.max_seq_len,
                        softmax_scale=self.scale,
                        causal=_fr13_sr_causal_flag(),
                        window_size=sliding_window_size,
                        block_table=decode_meta.block_table,
                        softcap=self.logits_soft_cap,
                        num_splits=1 if envs.VLLM_BATCH_INVARIANT else 0,
                        tree_bias=tree_bias,
                    )
                    flash_attn_varlen_func(
"""
        if text.count(live_call_anchor) != 1:
            raise RuntimeError(
                "qrow32 live paged exact4 A/B decode-call anchor is not unique"
            )
        text = text.replace(live_call_anchor, live_call_replacement, 1)
        changed = True
    if fixed32_query_tile32_b1_live_ab and (
        "_fr13_fa2_qrow32_b1_live_register(\n" not in text.split(
            "class TreeAttentionImpl", 1
        )[-1]
    ):
        live_call_anchor = (
            "                if not _fr13_reordered:\n"
            "                    flash_attn_varlen_func(\n"
        )
        live_call_replacement = """                if not _fr13_reordered:
                    tree_bias = _fr13_fa2_qrow32_b1_live_register(
                        layer=layer,
                        flash_fn=flash_attn_varlen_func,
                        query=query[:num_decode_tokens],
                        key_cache=key_cache,
                        value_cache=value_cache,
                        cu_seqlens_q=decode_meta.query_start_loc,
                        max_seqlen_q=decode_meta.max_query_len,
                        seqused_k=decode_meta.seq_lens,
                        max_seqlen_k=decode_meta.max_seq_len,
                        softmax_scale=self.scale,
                        causal=_fr13_sr_causal_flag(),
                        window_size=sliding_window_size,
                        block_table=decode_meta.block_table,
                        softcap=self.logits_soft_cap,
                        num_splits=1 if envs.VLLM_BATCH_INVARIANT else 0,
                        tree_bias=tree_bias,
                    )
                    flash_attn_varlen_func(
"""
        if text.count(live_call_anchor) != 1:
            raise RuntimeError(
                "qrow32 B1 live paged A/B decode-call anchor is not unique"
            )
        text = text.replace(live_call_anchor, live_call_replacement, 1)
        changed = True
    if fixed32_query_tile32_b1_production and (
        "_fr13_fa2_qrow32_b1_production_begin(\n" not in text.split(
            "class TreeAttentionImpl", 1
        )[-1]
    ):
        production_call = """                    flash_attn_varlen_func(
                        q=query[:num_decode_tokens],
                        k=key_cache,
                        v=value_cache,
                        out=output[:num_decode_tokens],
                        cu_seqlens_q=decode_meta.query_start_loc,
                        max_seqlen_q=decode_meta.max_query_len,
                        seqused_k=decode_meta.seq_lens,
                        max_seqlen_k=decode_meta.max_seq_len,
                        softmax_scale=self.scale,
                        causal=_fr13_sr_causal_flag(),
                        alibi_slopes=None,
                        window_size=sliding_window_size,
                        block_table=decode_meta.block_table,
                        softcap=self.logits_soft_cap,
                        fa_version=2,
                        tree_bias=tree_bias,
                    )
"""
        production_replacement = """                    _fr13_qrow32_b1_selection = _fr13_fa2_qrow32_b1_production_begin(
                        layer=layer,
                        query=query[:num_decode_tokens],
                        key_cache=key_cache,
                        value_cache=value_cache,
                        cu_seqlens_q=decode_meta.query_start_loc,
                        max_seqlen_q=decode_meta.max_query_len,
                        seqused_k=decode_meta.seq_lens,
                        max_seqlen_k=decode_meta.max_seq_len,
                        causal=_fr13_sr_causal_flag(),
                        window_size=sliding_window_size,
                        block_table=decode_meta.block_table,
                        softcap=self.logits_soft_cap,
                        num_splits=1 if envs.VLLM_BATCH_INVARIANT else 0,
                        tree_bias=tree_bias,
                    )
                    try:
                        flash_attn_varlen_func(
                            q=query[:num_decode_tokens],
                            k=key_cache,
                            v=value_cache,
                            out=output[:num_decode_tokens],
                            cu_seqlens_q=decode_meta.query_start_loc,
                            max_seqlen_q=decode_meta.max_query_len,
                            seqused_k=decode_meta.seq_lens,
                            max_seqlen_k=decode_meta.max_seq_len,
                            softmax_scale=self.scale,
                            causal=_fr13_sr_causal_flag(),
                            alibi_slopes=None,
                            window_size=sliding_window_size,
                            block_table=decode_meta.block_table,
                            softcap=self.logits_soft_cap,
                            fa_version=2,
                            num_splits=(
                                _fr13_qrow32_b1_selection["num_splits"]
                                if _fr13_qrow32_b1_selection is not None
                                else (1 if envs.VLLM_BATCH_INVARIANT else 0)
                            ),
                            tree_bias=(
                                _fr13_qrow32_b1_selection["tree_bias"]
                                if _fr13_qrow32_b1_selection is not None
                                else tree_bias
                            ),
                        )
                    except BaseException:
                        _fr13_fa2_qrow32_b1_production_end(
                            _fr13_qrow32_b1_selection, completed=False
                        )
                        raise
                    else:
                        _fr13_fa2_qrow32_b1_production_end(
                            _fr13_qrow32_b1_selection, completed=True
                        )
"""
        if text.count(production_call) != 1:
            raise RuntimeError(
                "qrow32 B1 attested production decode call is not unique"
            )
        text = text.replace(production_call, production_replacement, 1)
        changed = True
    if fixed32_query_tile32_b4_production and (
        "_fr13_fa2_qrow32_b4_production_begin(\n" not in text.split(
            "class TreeAttentionImpl", 1
        )[-1]
    ):
        # The served decode call itself: the selection's tagged tree_bias goes
        # straight into flash_attn_varlen_func(out=output[...]), so the forked
        # FA2 dispatch takes the GQA-pair kernel for real exact4 traffic.
        production_call = """                    flash_attn_varlen_func(
                        q=query[:num_decode_tokens],
                        k=key_cache,
                        v=value_cache,
                        out=output[:num_decode_tokens],
                        cu_seqlens_q=decode_meta.query_start_loc,
                        max_seqlen_q=decode_meta.max_query_len,
                        seqused_k=decode_meta.seq_lens,
                        max_seqlen_k=decode_meta.max_seq_len,
                        softmax_scale=self.scale,
                        causal=_fr13_sr_causal_flag(),
                        alibi_slopes=None,
                        window_size=sliding_window_size,
                        block_table=decode_meta.block_table,
                        softcap=self.logits_soft_cap,
                        fa_version=2,
                        tree_bias=tree_bias,
                    )
"""
        production_replacement = """                    _fr13_qrow32_b4_selection = _fr13_fa2_qrow32_b4_production_begin(
                        layer=layer,
                        query=query[:num_decode_tokens],
                        key_cache=key_cache,
                        value_cache=value_cache,
                        cu_seqlens_q=decode_meta.query_start_loc,
                        max_seqlen_q=decode_meta.max_query_len,
                        seqused_k=decode_meta.seq_lens,
                        max_seqlen_k=decode_meta.max_seq_len,
                        causal=_fr13_sr_causal_flag(),
                        window_size=sliding_window_size,
                        block_table=decode_meta.block_table,
                        softcap=self.logits_soft_cap,
                        num_splits=1 if envs.VLLM_BATCH_INVARIANT else 0,
                        tree_bias=tree_bias,
                    )
                    _fr13_qrow32_b34_staged = (
                        None
                        if _fr13_qrow32_b4_selection is None
                        else _fr13_qrow32_b4_selection.get("staged")
                    )
                    try:
                        if _fr13_qrow32_b34_staged is None:
                            flash_attn_varlen_func(
                                q=query[:num_decode_tokens],
                                k=key_cache,
                                v=value_cache,
                                out=output[:num_decode_tokens],
                                cu_seqlens_q=decode_meta.query_start_loc,
                                max_seqlen_q=decode_meta.max_query_len,
                                seqused_k=decode_meta.seq_lens,
                                max_seqlen_k=decode_meta.max_seq_len,
                                softmax_scale=self.scale,
                                causal=_fr13_sr_causal_flag(),
                                alibi_slopes=None,
                                window_size=sliding_window_size,
                                block_table=decode_meta.block_table,
                                softcap=self.logits_soft_cap,
                                fa_version=2,
                                num_splits=(
                                    _fr13_qrow32_b4_selection["num_splits"]
                                    if _fr13_qrow32_b4_selection is not None
                                    else (1 if envs.VLLM_BATCH_INVARIANT else 0)
                                ),
                                tree_bias=(
                                    _fr13_qrow32_b4_selection["tree_bias"]
                                    if _fr13_qrow32_b4_selection is not None
                                    else tree_bias
                                ),
                            )
                        else:
                            # FR13_FA2_QROW32_B34_PADDED: present the width-3
                            # call as the canonical width-4 geometry. Rows
                            # [96:128) of q and out live ONLY in the staging
                            # buffers, which nothing downstream reads; the real
                            # rows are copied back into output[:96] before the
                            # gate multiply and o_proj consume them.
                            _fr13_qrow32_b34_staged["query"][
                                :num_decode_tokens
                            ].copy_(query[:num_decode_tokens])
                            _fr13_qrow32_b34_staged["seqused_k"][
                                : decode_meta.seq_lens.shape[0]
                            ].copy_(decode_meta.seq_lens)
                            _fr13_qrow32_b34_staged["block_table"][
                                : decode_meta.block_table.shape[0]
                            ].copy_(decode_meta.block_table)
                            flash_attn_varlen_func(
                                q=_fr13_qrow32_b34_staged["query"],
                                k=key_cache,
                                v=value_cache,
                                out=_fr13_qrow32_b34_staged["out"],
                                cu_seqlens_q=_fr13_qrow32_b34_staged[
                                    "cu_seqlens_q"
                                ],
                                max_seqlen_q=decode_meta.max_query_len,
                                seqused_k=_fr13_qrow32_b34_staged["seqused_k"],
                                max_seqlen_k=decode_meta.max_seq_len,
                                softmax_scale=self.scale,
                                causal=_fr13_sr_causal_flag(),
                                alibi_slopes=None,
                                window_size=sliding_window_size,
                                block_table=_fr13_qrow32_b34_staged[
                                    "block_table"
                                ],
                                softcap=self.logits_soft_cap,
                                fa_version=2,
                                num_splits=_fr13_qrow32_b4_selection[
                                    "num_splits"
                                ],
                                tree_bias=_fr13_qrow32_b4_selection[
                                    "tree_bias"
                                ],
                            )
                            output[:num_decode_tokens].copy_(
                                _fr13_qrow32_b34_staged["out"][
                                    :num_decode_tokens
                                ]
                            )
                    except BaseException:
                        _fr13_fa2_qrow32_b4_production_end(
                            _fr13_qrow32_b4_selection, completed=False
                        )
                        raise
                    else:
                        _fr13_fa2_qrow32_b4_production_end(
                            _fr13_qrow32_b4_selection, completed=True
                        )
"""
        if text.count(production_call) != 1:
            raise RuntimeError(
                "qrow32 B4 attested production decode call is not unique"
            )
        text = text.replace(production_call, production_replacement, 1)
        changed = True
    if fixed32_query_tile16_production and (
        "_fr13_fa2_qrow16_production_begin(\n" not in text.split(
            "class TreeAttentionImpl", 1
        )[-1]
    ):
        production_call = """                    flash_attn_varlen_func(
                        q=query[:num_decode_tokens],
                        k=key_cache,
                        v=value_cache,
                        out=output[:num_decode_tokens],
                        cu_seqlens_q=decode_meta.query_start_loc,
                        max_seqlen_q=decode_meta.max_query_len,
                        seqused_k=decode_meta.seq_lens,
                        max_seqlen_k=decode_meta.max_seq_len,
                        softmax_scale=self.scale,
                        causal=_fr13_sr_causal_flag(),
                        alibi_slopes=None,
                        window_size=sliding_window_size,
                        block_table=decode_meta.block_table,
                        softcap=self.logits_soft_cap,
                        fa_version=2,
                        tree_bias=tree_bias,
                    )
"""
        production_replacement = """                    _fr13_qrow16_production_bias = _fr13_fa2_qrow16_production_begin(
                        layer=layer,
                        query=query[:num_decode_tokens],
                        key_cache=key_cache,
                        value_cache=value_cache,
                        cu_seqlens_q=decode_meta.query_start_loc,
                        max_seqlen_q=decode_meta.max_query_len,
                        seqused_k=decode_meta.seq_lens,
                        max_seqlen_k=decode_meta.max_seq_len,
                        causal=_fr13_sr_causal_flag(),
                        window_size=sliding_window_size,
                        block_table=decode_meta.block_table,
                        num_splits=1 if envs.VLLM_BATCH_INVARIANT else 0,
                        tree_bias=tree_bias,
                    )
                    try:
                        flash_attn_varlen_func(
                            q=query[:num_decode_tokens],
                            k=key_cache,
                            v=value_cache,
                            out=output[:num_decode_tokens],
                            cu_seqlens_q=decode_meta.query_start_loc,
                            max_seqlen_q=decode_meta.max_query_len,
                            seqused_k=decode_meta.seq_lens,
                            max_seqlen_k=decode_meta.max_seq_len,
                            softmax_scale=self.scale,
                            causal=_fr13_sr_causal_flag(),
                            alibi_slopes=None,
                            window_size=sliding_window_size,
                            block_table=decode_meta.block_table,
                            softcap=self.logits_soft_cap,
                            fa_version=2,
                            tree_bias=(
                                _fr13_qrow16_production_bias
                                if _fr13_qrow16_production_bias is not None
                                else tree_bias
                            ),
                        )
                    finally:
                        _fr13_fa2_qrow16_production_end(
                            _fr13_qrow16_production_bias
                        )
"""
        if text.count(production_call) != 1:
            raise RuntimeError(
                "qrow16 attested production decode call is not unique"
            )
        text = text.replace(production_call, production_replacement, 1)
        changed = True
    if dfwd_unified_bm8_production:
        text, did = _patch_dfwd_unified_bm8_production_call(text)
        changed = changed or did
    if changed:
        path.write_text(text)
        py_compile.compile(path, doraise=True)
    return changed


def _patch_flash_attn_backend(path: Path) -> bool:
    text = path.read_text()
    changed = False
    old_filter = (
        '        if want and want != "*" and not layer_name.startswith(want):\n'
        "            return\n"
    )
    new_filter = (
        '        if want and want != "*" and layer_name and not layer_name.startswith(want):\n'
        "            return\n"
    )
    if old_filter in text:
        text = text.replace(old_filter, new_filter)
        changed = True
    if "FR13_FLASH_ATTN_OP_CAPTURE_TREE_ONLY" not in text:
        text = text.replace(
            '        layer_name = str(getattr(layer, "layer_name", ""))\n',
            '''        tree_bias = getattr(attn_metadata, "tree_attn_bias", None)
        if (
            os.environ.get("FR13_FLASH_ATTN_OP_CAPTURE_TREE_ONLY", "0") == "1"
            and (tree_bias is None or tree_bias.numel() == 0)
        ):
            return
        layer_name = str(getattr(layer, "layer_name", ""))
''',
            1,
        )
        text = text.replace(
            '            "used_blocks": used_blocks,\n',
            '''            "used_blocks": used_blocks,
            "tree_attn_bias": (
                tree_bias.detach().to(torch.float32).cpu()
                if tree_bias is not None
                else None
            ),
''',
            1,
        )
        changed = True

    old = """                flash_attn_varlen_func(
                    q=query[:num_actual_tokens],
                    k=key_cache,
                    v=value_cache,
                    out=output[:num_actual_tokens],
                    cu_seqlens_q=cu_seqlens_q,
                    max_seqlen_q=max_seqlen_q,
                    seqused_k=seqused_k,
                    max_seqlen_k=max_seqlen_k,
                    softmax_scale=self.scale,
                    causal=attn_metadata.causal,
                    alibi_slopes=self.alibi_slopes,
                    window_size=sliding_window_size,
                    block_table=block_table,
                    softcap=self.logits_soft_cap,
                    scheduler_metadata=scheduler_metadata,
                    fa_version=self.vllm_flash_attn_version,
                    q_descale=q_descale,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    num_splits=attn_metadata.max_num_splits,
                    s_aux=self.sinks,
                )
"""
    new = """                tree_bias = getattr(attn_metadata, "tree_attn_bias", None)
                use_tree_bias = (
                    os.environ.get("FR13_FA2_TREE_BIAS", "0") == "1"
                    and tree_bias is not None
                    and tree_bias.numel() > 0
                )
                if use_tree_bias and self.alibi_slopes is not None:
                    raise NotImplementedError("FR13_FA2_TREE_BIAS does not support ALiBi")
                flash_attn_varlen_func(
                    q=query[:num_actual_tokens],
                    k=key_cache,
                    v=value_cache,
                    out=output[:num_actual_tokens],
                    cu_seqlens_q=cu_seqlens_q,
                    max_seqlen_q=max_seqlen_q,
                    seqused_k=seqused_k,
                    max_seqlen_k=max_seqlen_k,
                    softmax_scale=self.scale,
                    causal=attn_metadata.causal,
                    alibi_slopes=None if use_tree_bias else self.alibi_slopes,
                    window_size=sliding_window_size,
                    block_table=block_table,
                    softcap=self.logits_soft_cap,
                    scheduler_metadata=scheduler_metadata,
                    fa_version=self.vllm_flash_attn_version,
                    q_descale=q_descale,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    num_splits=attn_metadata.max_num_splits,
                    s_aux=self.sinks,
                    tree_bias=tree_bias if use_tree_bias else None,
                )
"""
    text, did = _replace_once(text, old, new, "flash_attn decode tree_bias")
    changed = changed or did
    if changed:
        path.write_text(text)
        py_compile.compile(path, doraise=True)
    return changed


def _patch_batch_invariant_guard(path: Path) -> bool:
    """FR13 Method-A (FR13_BI_TREE_ATTN): flag-gated BI allowlist for TREE_ATTN.

    On-disk edit of the installed vllm batch_invariant.py, anchored on the
    decode_invariant_backends block inside override_envs_for_invariance().
    The injected code is runtime-gated on env FR13_BI_TREE_ATTN=1 and
    double-gated on FR13_FA2_TREE_BIAS=1 + FR13_FA2_PREFILL_NATIVE=1 (raises
    a loud RuntimeError otherwise: the BI justification covers only the
    forked-FA2 tree decode + native-FA2 prefill paths).  Inert by default:
    the env gate defaults to "0" AND override_envs_for_invariance() is only
    ever called when VLLM_BATCH_INVARIANT is enabled.
    """
    text = path.read_text()
    if "FR13_BI_TREE_ATTN" in text:
        return False
    anchor = (
        "    decode_invariant_backends = [\n"
        "        AttentionBackendEnum.FLASH_ATTN,  # best supported backend\n"
        "        AttentionBackendEnum.TRITON_ATTN,\n"
        "    ]\n"
    )
    guard = anchor + (
        "    # FR13_BI_TREE_ATTN: Method-A flag-gated allowlist relaxation.\n"
        "    # Appending to decode_invariant_backends (before supported_backends\n"
        "    # is built from it) also allowlists TREE_ATTN as supported and\n"
        "    # suppresses the prefill-vs-decode warning: deployed prefill AND\n"
        "    # decode run the same forked FA2 varlen binary with non-split\n"
        "    # dispatch (num_splits forced under BI).\n"
        '    if os.environ.get("FR13_BI_TREE_ATTN", "0") == "1":\n'
        "        if not (\n"
        '            os.environ.get("FR13_FA2_TREE_BIAS", "0") == "1"\n'
        '            and os.environ.get("FR13_FA2_PREFILL_NATIVE", "0") == "1"\n'
        "        ):\n"
        "            raise RuntimeError(\n"
        '                "FR13_BI_TREE_ATTN=1 requires FR13_FA2_TREE_BIAS=1 and "\n'
        '                "FR13_FA2_PREFILL_NATIVE=1: BI justification covers only "\n'
        '                "the forked-FA2 tree decode + native-FA2 prefill paths"\n'
        "            )\n"
        "        decode_invariant_backends.append(AttentionBackendEnum.TREE_ATTN)\n"
    )
    if anchor not in text:
        raise RuntimeError(
            "anchor not found for batch_invariant decode_invariant_backends block"
        )
    text = text.replace(anchor, guard, 1)
    path.write_text(text)
    py_compile.compile(path, doraise=True)
    return True


def _patch_cuda_graph_qrow16_live_ab(path: Path) -> bool:
    text = path.read_text()
    sentinel = "# FR13_FA2_QROW16_LIVE_PAGED_AB_REPLAY"
    if sentinel in text:
        return False
    anchor = "        entry.cudagraph.replay()\n"
    if text.count(anchor) != 1:
        raise RuntimeError("qrow16 live paged A/B replay anchor is not unique")
    replacement = anchor + f'''        {sentinel}: the stock graph has now
        # produced the first real event's retained query. Diagnostic recalls
        # are ordered after it and never replace entry.output.
        from vllm.v1.attention.backends.tree_attn import (
            _fr13_fa2_qrow16_live_ab_replay,
        )
        _fr13_fa2_qrow16_live_ab_replay(
            id(entry.cudagraph),
            self.runtime_mode.name,
            entry.batch_descriptor.num_reqs,
        )
'''
    path.write_text(text.replace(anchor, replacement, 1))
    py_compile.compile(path, doraise=True)
    return True


def _patch_cuda_graph_qrow32_live_ab(path: Path) -> bool:
    text = path.read_text()
    sentinel = "# FR13_FA2_QROW32_LIVE_PAGED_AB_REPLAY"
    if sentinel in text:
        return False
    anchor = "        entry.cudagraph.replay()\n"
    if text.count(anchor) != 1:
        raise RuntimeError("qrow32 live paged A/B replay anchor is not unique")
    replacement = anchor + f'''        {sentinel}: the stock FULL graph has
        # produced the first real exact4 event's retained queries. All-layer
        # diagnostic recalls are ordered after it and never replace entry.output.
        from vllm.v1.attention.backends.tree_attn import (
            _fr13_fa2_qrow32_live_ab_replay,
        )
        _fr13_fa2_qrow32_live_ab_replay(
            id(entry.cudagraph),
            self.runtime_mode.name,
            entry.batch_descriptor.num_reqs,
        )
'''
    path.write_text(text.replace(anchor, replacement, 1))
    py_compile.compile(path, doraise=True)
    return True


def _patch_cuda_graph_qrow32_b1_live_ab(path: Path) -> bool:
    text = path.read_text()
    sentinel = "# FR13_FA2_QROW32_B1_LIVE_PAGED_AB_REPLAY"
    if sentinel in text:
        return False
    anchor = "        entry.cudagraph.replay()\n"
    if text.count(anchor) != 1:
        raise RuntimeError("qrow32 B1 live paged A/B replay anchor is not unique")
    replacement = anchor + f'''        {sentinel}: the stock graph has produced
        # the first real B1 event. Diagnostic recalls never replace entry.output.
        from vllm.v1.attention.backends.tree_attn import (
            _fr13_fa2_qrow32_b1_live_replay,
        )
        _fr13_fa2_qrow32_b1_live_replay(
            id(entry.cudagraph),
            self.runtime_mode.name,
            entry.batch_descriptor.num_reqs,
        )
'''
    path.write_text(text.replace(anchor, replacement, 1))
    py_compile.compile(path, doraise=True)
    return True


def _patch_cuda_graph_qrow16_production(path: Path) -> bool:
    text = path.read_text()
    sentinel = "# FR13_FA2_QROW16_PRODUCTION_CAPTURE_END"
    if sentinel in text:
        return False
    anchor = "            entry.cudagraph = cudagraph\n"
    if text.count(anchor) != 1:
        raise RuntimeError("qrow16 production capture-end anchor is not unique")
    replacement = anchor + f'''            {sentinel}: fail unless every exact
            # target tree layer captured qrow16 in the attested final B1 graph.
            from vllm.v1.attention.backends.tree_attn import (
                _fr13_fa2_qrow16_production_capture_end,
            )
            _fr13_fa2_qrow16_production_capture_end(
                id(entry.cudagraph),
                getattr(entry, "_fr13_fixed32_graph_signature", None),
                self.runtime_mode.name,
                entry.batch_descriptor.num_reqs,
            )
'''
    path.write_text(text.replace(anchor, replacement, 1))
    py_compile.compile(path, doraise=True)
    return True


def _patch_cuda_graph_qrow32_b1_production(path: Path) -> bool:
    text = path.read_text()
    sentinel = "# FR13_FA2_QROW32_B1_PRODUCTION_CAPTURE_END"
    if sentinel in text:
        return False
    anchor = "            entry.cudagraph = cudagraph\n"
    if text.count(anchor) != 1:
        raise RuntimeError("qrow32 B1 production capture-end anchor is not unique")
    replacement = anchor + f'''            {sentinel}: fail unless every exact
            # target tree layer captured the selected candidate in final B1.
            from vllm.v1.attention.backends.tree_attn import (
                _fr13_fa2_qrow32_b1_production_capture_end,
            )
            _fr13_fa2_qrow32_b1_production_capture_end(
                id(entry.cudagraph),
                getattr(entry, "_fr13_fixed32_graph_signature", None),
                self.runtime_mode.name,
                entry.batch_descriptor.num_reqs,
            )
'''
    path.write_text(text.replace(anchor, replacement, 1))
    py_compile.compile(path, doraise=True)
    return True


def _patch_cuda_graph_qrow32_b34_precapture_staging(path: Path) -> bool:
    """CAVEAT 2: inject the pre-capture staging allocation hook.

    The anchor is the CUDAGraph construction itself, and the hook is inserted
    IMMEDIATELY AFTER it -- which is:

      * after `validate_cudagraph_capturing_enabled()` and the input-address
        snapshot, so the wrapper has already decided that it is going to
        capture this batch descriptor, and
      * before the `with ExitStack()` / `with torch.cuda.graph(...)` block, so
        the stream is NOT capturing yet and the 3.0 MiB of staging comes from
        the ordinary caching allocator rather than the graph's private pool,
        and
      * strictly ahead of the fr10 patcher's `_fr13_fixed32_capture_begin`
        injection, which anchors on the same line
        (scripts/fr10_phase4_patch_vllm_tree_gdn.py:39965-39986) -- exactly
        where the de-risk puts it ("the allocation belongs immediately before
        that", verification_1.implementation_hazard_found.required_fix).

    The hook takes runtime_mode and num_reqs straight from the wrapper's own
    scope, so it does not depend on the fr10 capture context existing yet and
    the two patchers may be applied in either order.
    """
    text = path.read_text()
    sentinel = "# FR13_FA2_QROW32_B34_PRECAPTURE_STAGING"
    if sentinel in text:
        return False
    # ORDERING IS LOAD-BEARING AND IS THEREFORE ASSERTED, NOT ASSUMED. The fr10
    # patcher's capture-begin anchor spans three lines --
    # `cudagraph = torch.cuda.CUDAGraph()`, a blank line, and
    # `with ExitStack() as stack:` -- so an insertion between them would break
    # it. The launcher already runs fr10 first
    # (scripts/fr13_launch_forked_fa2_tree_server.sh:6024 then :6033); this
    # check turns "already ran" from a coincidence into a precondition. It is
    # also the honest statement of the dependency: the B4/B3 production
    # selector cannot serve without the fixed32 capture context that hook
    # installs.
    if "_fr13_fixed32_capture_begin(" not in text:
        raise RuntimeError(
            "qrow32 B34 pre-capture staging requires the fr10 fixed32 "
            "capture-begin hook to be installed first"
        )
    anchor = "            cudagraph = torch.cuda.CUDAGraph()\n"
    if text.count(anchor) != 1:
        raise RuntimeError(
            "qrow32 B34 pre-capture staging anchor is not unique"
        )
    replacement = anchor + f'''            {sentinel}: allocate the padded
            # b=3 staging buffers BEFORE capture begins. Lazy allocation is
            # unreachable and in-capture allocation would come from this
            # graph's private pool; the serving path therefore only ever
            # looks the staging up, and raises if it is absent.
            if self.runtime_mode.name == "FULL":
                from vllm.v1.attention.backends.tree_attn import (
                    _fr13_fa2_qrow32_b34_precapture_staging,
                )
                _fr13_fa2_qrow32_b34_precapture_staging(
                    id(cudagraph),
                    self.runtime_mode.name,
                    entry.batch_descriptor.num_reqs,
                )
'''
    path.write_text(text.replace(anchor, replacement, 1))
    py_compile.compile(path, doraise=True)
    return True


def _patch_cuda_graph_qrow32_b4_production(path: Path) -> bool:
    text = path.read_text()
    sentinel = "# FR13_FA2_QROW32_B4_PRODUCTION_CAPTURE_END"
    if sentinel in text:
        return False
    anchor = "            entry.cudagraph = cudagraph\n"
    if text.count(anchor) != 1:
        raise RuntimeError("qrow32 B4 production capture-end anchor is not unique")
    replacement = anchor + f'''            {sentinel}: fail unless every exact
            # target tree layer captured the GQA-pair candidate in final B4.
            from vllm.v1.attention.backends.tree_attn import (
                _fr13_fa2_qrow32_b4_production_capture_end,
            )
            _fr13_fa2_qrow32_b4_production_capture_end(
                id(entry.cudagraph),
                getattr(entry, "_fr13_fixed32_graph_signature", None),
                self.runtime_mode.name,
                entry.batch_descriptor.num_reqs,
            )
'''
    path.write_text(text.replace(anchor, replacement, 1))
    py_compile.compile(path, doraise=True)
    return True


def patch_installed_vllm(
    site_packages: Path,
    *,
    fixed32_query_tile16_live_ab: bool = False,
    fixed32_query_tile32_live_ab: bool = False,
    fixed32_query_tile32_b1_live_ab: bool = False,
    fixed32_query_tile32_b1_production: bool = False,
    fixed32_query_tile32_b4_production: bool = False,
    fixed32_query_tile16_production: bool = False,
    dfwd_unified_bm8_production: bool = False,
) -> dict[str, bool]:
    result = {
        "flash_attn_interface.py": _patch_flash_attn_interface(
            site_packages / "vllm/vllm_flash_attn/flash_attn_interface.py"
        ),
        "tree_attn.py": _patch_tree_attn(
            site_packages / "vllm/v1/attention/backends/tree_attn.py",
            fixed32_query_tile16_live_ab=fixed32_query_tile16_live_ab,
            fixed32_query_tile32_live_ab=fixed32_query_tile32_live_ab,
            fixed32_query_tile32_b1_live_ab=fixed32_query_tile32_b1_live_ab,
            fixed32_query_tile32_b1_production=(
                fixed32_query_tile32_b1_production
            ),
            fixed32_query_tile32_b4_production=(
                fixed32_query_tile32_b4_production
            ),
            fixed32_query_tile16_production=fixed32_query_tile16_production,
            dfwd_unified_bm8_production=dfwd_unified_bm8_production,
        ),
        "flash_attn.py": _patch_flash_attn_backend(
            site_packages / "vllm/v1/attention/backends/flash_attn.py"
        ),
        "batch_invariant.py": _patch_batch_invariant_guard(
            site_packages / "vllm/model_executor/layers/batch_invariant.py"
        ),
    }
    if fixed32_query_tile16_live_ab:
        result["cuda_graph.py"] = _patch_cuda_graph_qrow16_live_ab(
            site_packages / "vllm/compilation/cuda_graph.py"
        )
    elif fixed32_query_tile32_live_ab:
        result["cuda_graph.py"] = _patch_cuda_graph_qrow32_live_ab(
            site_packages / "vllm/compilation/cuda_graph.py"
        )
    elif fixed32_query_tile32_b1_live_ab:
        result["cuda_graph.py"] = _patch_cuda_graph_qrow32_b1_live_ab(
            site_packages / "vllm/compilation/cuda_graph.py"
        )
    elif fixed32_query_tile32_b1_production:
        result["cuda_graph.py"] = _patch_cuda_graph_qrow32_b1_production(
            site_packages / "vllm/compilation/cuda_graph.py"
        )
    elif fixed32_query_tile32_b4_production:
        cuda_graph_path = site_packages / "vllm/compilation/cuda_graph.py"
        # Order matters only for readability: both edits are idempotent and
        # anchor on distinct lines. The staging hook MUST be present whenever
        # the B4/B3 production selector is, because the selector refuses to
        # allocate and a width-3 FULL capture would otherwise fail closed.
        result["cuda_graph.py"] = _patch_cuda_graph_qrow32_b34_precapture_staging(
            cuda_graph_path
        )
        result["cuda_graph.py"] = (
            _patch_cuda_graph_qrow32_b4_production(cuda_graph_path)
            or result["cuda_graph.py"]
        )
    elif fixed32_query_tile16_production:
        result["cuda_graph.py"] = _patch_cuda_graph_qrow16_production(
            site_packages / "vllm/compilation/cuda_graph.py"
        )
    return result


def find_fa2_source(value: str | None) -> Path:
    candidates = []
    if value:
        candidates.append(Path(value))
    env_value = os.environ.get("FR13_FA2_SRC_DIR")
    if env_value:
        candidates.append(Path(env_value))
    candidates.extend(DEFAULT_FA2_CANDIDATES)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("could not locate vllm-flash-attn-src")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fa2-src", type=Path)
    parser.add_argument(
        "--site-packages",
        type=Path,
        default=Path("/usr/local/lib/python3.12/dist-packages"),
    )
    parser.add_argument("--skip-source", action="store_true")
    parser.add_argument("--skip-python", action="store_true")
    parser.add_argument(
        "--tree-bias-tile-earlyout",
        action="store_true",
        help="build the exact tree-bias K-tile overlap early-out candidate",
    )
    parser.add_argument(
        "--fixed32-query-tile16",
        action="store_true",
        help="build the fixed32 B1 tree-only FA2 16-row query-tile candidate",
    )
    parser.add_argument(
        "--fixed32-query-tile16-static-strides",
        action="store_true",
        help="fix canonical K/V strides in the private qrow16 candidate",
    )
    parser.add_argument(
        "--fixed32-tree-visibility-mask",
        action="store_true",
        help=(
            "replace dense tree-bias reads with the exact physical32 "
            "self-plus-ancestor bit masks in a private qrow32 kernel"
        ),
    )
    parser.add_argument(
        "--fixed32-query-tile32",
        action="store_true",
        help="build the gate-only fixed32 B4 FA2 32-row query-tile candidate",
    )
    parser.add_argument(
        "--fixed32-query-gqa-pair32",
        action="store_true",
        help="build the gate-only B4 FA2 two-query-head GQA-pair candidate",
    )
    parser.add_argument(
        "--fixed32-query-tile32-b1",
        action="store_true",
        help="build the gate-only fixed32 B1 FA2 32-row query-tile candidate",
    )
    parser.add_argument(
        "--fixed32-query-gqa-pair32-b1",
        action="store_true",
        help="build the gate-only B1 FA2 two-query-head GQA-pair candidate",
    )
    parser.add_argument(
        "--fixed32-query-gqa-pair32-splitk-b1",
        action="store_true",
        help=(
            "build the gate-only B1 FA2 GQA-pair SPLIT-K candidate (Tier-B: "
            "changes per-row accumulation order by design)"
        ),
    )
    parser.add_argument(
        "--fixed32-query-tile16-live-ab",
        action="store_true",
        help="install the one-shot live paged B1 stock/qrow16 byte gate",
    )
    parser.add_argument(
        "--fixed32-query-tile32-live-ab",
        action="store_true",
        help="install the all-layer live paged exact4 B4 stock/qrow32 byte gate",
    )
    parser.add_argument(
        "--fixed32-query-tile32-b1-live-ab",
        action="store_true",
        help="install the all-layer real-B1 qrow32-vs-Qrow16 byte gate",
    )
    parser.add_argument(
        "--fixed32-query-tile32-b1-production",
        action="store_true",
        help="install the attested B1 no-split production selector",
    )
    parser.add_argument(
        "--fixed32-query-tile32-b4-production",
        action="store_true",
        help="install the attested exact4 B4 qrow32 GQA-pair production selector",
    )
    parser.add_argument(
        "--fixed32-query-tile16-production",
        action="store_true",
        help="install the attested exact-B1 qrow16 production selector",
    )
    parser.add_argument(
        "--dfwd-unified-bm8-production",
        action="store_true",
        help="install the attested exact-B1 BM8 drafter fallback selector",
    )
    args = parser.parse_args()
    if (
        args.fixed32_query_tile16_live_ab
        and args.fixed32_query_tile16_production
    ):
        parser.error("qrow16 live A/B and production selectors are mutually exclusive")
    private_selectors = sum(
        bool(value)
        for value in (
            args.fixed32_query_tile16_live_ab,
            args.fixed32_query_tile32_live_ab,
            args.fixed32_query_tile32_b1_live_ab,
            args.fixed32_query_tile32_b1_production,
            args.fixed32_query_tile32_b4_production,
            args.fixed32_query_tile16_production,
        )
    )
    if private_selectors > 1:
        parser.error("qrow16/qrow32 private selectors are mutually exclusive")
    if args.fixed32_query_tile32 and not args.tree_bias_tile_earlyout:
        parser.error(
            "--fixed32-query-tile32 requires --tree-bias-tile-earlyout in "
            "the same source-build invocation"
        )
    if args.fixed32_query_tile32_b1 and not args.tree_bias_tile_earlyout:
        parser.error(
            "--fixed32-query-tile32-b1 requires --tree-bias-tile-earlyout in "
            "the same source-build invocation"
        )
    if args.fixed32_query_gqa_pair32 and not args.tree_bias_tile_earlyout:
        parser.error(
            "--fixed32-query-gqa-pair32 requires --tree-bias-tile-earlyout "
            "in the same source-build invocation"
        )
    if args.fixed32_query_gqa_pair32_b1 and not args.tree_bias_tile_earlyout:
        parser.error(
            "--fixed32-query-gqa-pair32-b1 requires --tree-bias-tile-earlyout "
            "in the same source-build invocation"
        )
    if (
        args.fixed32_query_gqa_pair32_splitk_b1
        and not args.tree_bias_tile_earlyout
    ):
        parser.error(
            "--fixed32-query-gqa-pair32-splitk-b1 requires "
            "--tree-bias-tile-earlyout in the same source-build invocation"
        )
    qrow32_source_builds = sum(
        bool(value)
        for value in (
            args.fixed32_query_tile32,
            args.fixed32_query_gqa_pair32,
            args.fixed32_query_tile32_b1,
            args.fixed32_query_gqa_pair32_b1,
            args.fixed32_query_gqa_pair32_splitk_b1,
        )
    )
    if qrow32_source_builds > 1:
        parser.error("qrow32 source candidates are mutually exclusive")
    if args.fixed32_tree_visibility_mask and not qrow32_source_builds:
        parser.error(
            "--fixed32-tree-visibility-mask requires a private qrow32 "
            "source candidate"
        )
    if (
        args.fixed32_query_tile16_static_strides
        and not args.fixed32_query_tile16
    ):
        parser.error(
            "--fixed32-query-tile16-static-strides requires "
            "--fixed32-query-tile16"
        )
    if (
        args.fixed32_query_tile32_live_ab
        and not args.skip_source
        and not args.fixed32_query_tile32
    ):
        parser.error(
            "a combined qrow32 source/live patch requires "
            "--fixed32-query-tile32"
        )
    if (
        (args.fixed32_query_tile32_b1_live_ab
         or args.fixed32_query_tile32_b1_production)
        and not args.skip_source
        and not (
            args.fixed32_query_tile32_b1 or args.fixed32_query_gqa_pair32_b1
        )
    ):
        parser.error(
            "a combined qrow32 B1 source/selector patch requires "
            "--fixed32-query-tile32-b1 or --fixed32-query-gqa-pair32-b1"
        )
    if (
        args.fixed32_query_tile32_b1_production
        and args.fixed32_query_gqa_pair32_b1
    ):
        # The B1 production selector only issues the attested no-split
        # credential; the GQA-pair arm is byte-gate-only until it earns one.
        parser.error(
            "the B1 production selector requires --fixed32-query-tile32-b1"
        )
    if (
        args.fixed32_query_tile32_b4_production
        and not args.skip_source
        and not args.fixed32_query_gqa_pair32
    ):
        parser.error(
            "a combined qrow32 B4 source/production patch requires "
            "--fixed32-query-gqa-pair32"
        )

    payload: dict[str, object] = {
        "tree_bias_tile_earlyout": args.tree_bias_tile_earlyout,
        "fixed32_query_tile16": args.fixed32_query_tile16,
        "fixed32_query_tile16_static_strides": (
            args.fixed32_query_tile16_static_strides
        ),
        "fixed32_tree_visibility_mask": args.fixed32_tree_visibility_mask,
        "fixed32_query_tile32": args.fixed32_query_tile32,
        "fixed32_query_gqa_pair32": args.fixed32_query_gqa_pair32,
        "fixed32_query_tile32_b1": args.fixed32_query_tile32_b1,
        "fixed32_query_gqa_pair32_b1": args.fixed32_query_gqa_pair32_b1,
        "fixed32_query_gqa_pair32_splitk_b1": (
            args.fixed32_query_gqa_pair32_splitk_b1
        ),
        "fixed32_query_tile16_live_ab": args.fixed32_query_tile16_live_ab,
        "fixed32_query_tile32_live_ab": args.fixed32_query_tile32_live_ab,
        "fixed32_query_tile32_b1_live_ab": (
            args.fixed32_query_tile32_b1_live_ab
        ),
        "fixed32_query_tile32_b1_production": (
            args.fixed32_query_tile32_b1_production
        ),
        "fixed32_query_tile32_b4_production": (
            args.fixed32_query_tile32_b4_production
        ),
        "fixed32_query_tile16_production": args.fixed32_query_tile16_production,
        "dfwd_unified_bm8_production": args.dfwd_unified_bm8_production,
    }
    if not args.skip_source:
        fa2_src = find_fa2_source(str(args.fa2_src) if args.fa2_src else None)
        payload["fa2_src"] = str(fa2_src)
        payload["source"] = patch_fa2_source(
            fa2_src,
            tree_bias_tile_earlyout=args.tree_bias_tile_earlyout,
            fixed32_query_tile16=args.fixed32_query_tile16,
            fixed32_query_tile16_static_strides=(
                args.fixed32_query_tile16_static_strides
            ),
            fixed32_query_tile32=args.fixed32_query_tile32,
            fixed32_query_gqa_pair32=args.fixed32_query_gqa_pair32,
            fixed32_query_tile32_b1=args.fixed32_query_tile32_b1,
            fixed32_query_gqa_pair32_b1=args.fixed32_query_gqa_pair32_b1,
            fixed32_query_gqa_pair32_splitk_b1=(
                args.fixed32_query_gqa_pair32_splitk_b1
            ),
            fixed32_tree_visibility_mask=(
                args.fixed32_tree_visibility_mask
            ),
        )
    if not args.skip_python:
        payload["python"] = patch_installed_vllm(
            args.site_packages,
            fixed32_query_tile16_live_ab=args.fixed32_query_tile16_live_ab,
            fixed32_query_tile32_live_ab=args.fixed32_query_tile32_live_ab,
            fixed32_query_tile32_b1_live_ab=(
                args.fixed32_query_tile32_b1_live_ab
            ),
            fixed32_query_tile32_b1_production=(
                args.fixed32_query_tile32_b1_production
            ),
            fixed32_query_tile32_b4_production=(
                args.fixed32_query_tile32_b4_production
            ),
            fixed32_query_tile16_production=args.fixed32_query_tile16_production,
            dfwd_unified_bm8_production=args.dfwd_unified_bm8_production,
        )
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
