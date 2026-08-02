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

The source-only ``--fixed32-query-tile32`` build adds the corresponding B4
specialization: one 32-row, two-warp CTA per batch/head. It has the same
96-CTA layer grid and complete ordered K loop as stock BM64 while avoiding the
32 query rows outside each physical32 batch slot. No runtime selector is
installed until the private binary passes resource and real-byte gates.
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
    const int bias_col_begin = context_len + params.tree_bias_k_offset;
    const int bias_col_end = bias_col_begin + params.tree_bias_cols;
    const int block_col_begin = n_block * Kernel_traits::kBlockN;
    const int block_col_end = block_col_begin + Kernel_traits::kBlockN;
    if (block_col_end <= bias_col_begin || block_col_begin >= bias_col_end) {
        return;
    }
'''


def _tree_bias_helper(tile_earlyout: bool) -> str:
    overlap_guard = TREE_BIAS_TILE_OVERLAP_GUARD if tile_earlyout else ""
    return r'''
// FR13_FA2_TREE_BIAS: add a dense query-suffix ancestry bias after QK.
template <typename Kernel_traits, typename Engine, typename Layout, typename Params, typename BlockInfoT>
__forceinline__ __device__ void apply_tree_bias(Tensor<Engine, Layout> &tensor_,
                                                const Params &params,
                                                const BlockInfoT &binfo,
                                                const int bidb,
                                                const int n_block,
                                                const int row_idx_offset,
                                                const int warp_row_stride) {
    if (params.tree_bias_ptr == nullptr) { return; }
    static_assert(Layout::rank == 3, "Only support 3D Tensor");
    static_assert(decltype(size<0>(tensor_))::value == 4, "First dimension must be 4");
    Tensor tensor = make_tensor(tensor_.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(tensor_.layout()));
    const float *tree_bias = reinterpret_cast<const float *>(params.tree_bias_ptr)
        + bidb * params.tree_bias_batch_stride;
    const int context_len = binfo.actual_seqlen_k - binfo.actual_seqlen_q;
''' + overlap_guard + r'''    const int lane_id = threadIdx.x % 32;
    const int col_idx_offset = n_block * Kernel_traits::kBlockN + (lane_id % 4) * 2;
    #pragma unroll
    for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
        const int row_idx_base = row_idx_offset + mi * warp_row_stride;
        #pragma unroll
        for (int i = 0; i < size<0, 0>(tensor); ++i) {
            const int q_rel = row_idx_base + i * 8 - params.tree_bias_q_offset;
            if (q_rel >= 0 && q_rel < params.tree_bias_rows) {
                #pragma unroll
                for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                    const int col_idx_base = col_idx_offset + nj * 8;
                    #pragma unroll
                    for (int j = 0; j < size<1, 0>(tensor); ++j) {
                        const int k_rel = col_idx_base + j - context_len - params.tree_bias_k_offset;
                        if (k_rel >= 0 && k_rel < params.tree_bias_cols) {
                            const float bias = tree_bias[
                                q_rel * params.tree_bias_row_stride
                                + k_rel * params.tree_bias_col_stride
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


TREE_BIAS_HELPER = _tree_bias_helper(tile_earlyout=False)


STOCK_FIXED32_QUERY_INSTANTIATION = r'''template void run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 256, false>(Flash_fwd_params &params, cudaStream_t stream);'''


FIXED32_QUERY_TILE16_BATCH_STRIDE_SENTINEL = 0x46523133


FIXED32_QUERY_TILE16_TRANSLATION_UNIT = r'''// FR13 fixed32 B1 qrow16 internal kernel.
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

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
    using TreeKernelTraits = Flash_fwd_kernel_traits<
        256, kTreeBlockM, kTreeBlockN, kTreeWarps,
        false, false, cutlass::bfloat16_t>;
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
    auto kernel = &flash_fwd_splitkv_kernel<
        TreeKernelTraits,
        false,  // Is_causal
        false,  // Is_local
        false,  // Has_alibi
        false,  // Is_even_MN: paged varlen Q has cu_seqlens_q
        true,   // Is_even_K: d == kHeadDim == 256
        false,  // Is_softcap
        false,  // Split
        false   // Append_KV
    >;
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


FIXED32_QUERY_TILE32_TRANSLATION_UNIT = r'''// FR13 fixed32 B4 qrow32 source candidate.
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

// Source/resource-gate entry point only. No production selector references
// this hidden function until the exact4 live paged raw-byte gate passes.
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
    using TreeKernelTraits = Flash_fwd_kernel_traits<
        256, kTreeBlockM, kTreeBlockN, kTreeWarps,
        false, false, cutlass::bfloat16_t>;
    static_assert(TreeKernelTraits::kNThreads == 64);
    static_assert(TreeKernelTraits::kGmemThreadsPerRow == 8);
    static_assert(TreeKernelTraits::kGmemRowsPerThread == 8);
    static_assert(1024 % TreeKernelTraits::kGmemRowsPerThread == 0);
    constexpr size_t smem_size = TreeKernelTraits::kSmemSize;
    const int num_m_block =
        (params.seqlen_q + TreeKernelTraits::kBlockM - 1)
        / TreeKernelTraits::kBlockM;
    dim3 grid(num_m_block, params.b, params.h);
    auto kernel = &flash_fwd_splitkv_kernel<
        TreeKernelTraits,
        false,  // Is_causal
        false,  // Is_local
        false,  // Has_alibi
        false,  // Is_even_MN: paged varlen Q has cu_seqlens_q
        true,   // Is_even_K: d == kHeadDim == 256
        false,  // Is_softcap
        false,  // Split
        false   // Append_KV
    >;
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


def _patch_flash_fwd_kernel(path: Path, *, tile_earlyout: bool = False) -> bool:
    text = path.read_text()
    changed = False
    tree_bias_helper = _tree_bias_helper(tile_earlyout)
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


def _patch_fixed32_query_translation_unit(
    path: Path,
    *,
    fixed32_query_tile16: bool = False,
) -> bool:
    if not fixed32_query_tile16:
        return False
    stock_text = path.read_text()
    if STOCK_FIXED32_QUERY_INSTANTIATION not in stock_text:
        raise RuntimeError("stock fixed32 FA2 explicit instantiation drifted")
    if "FR13_FA2_FIXED32_QUERY_TILE16" in stock_text:
        raise RuntimeError("qrow16 must not share the stock instantiation TU")
    qrow_path = path.with_name("flash_fwd_fr13_qrow16_hdim256_bf16_sm80.cu")
    if qrow_path.exists():
        if qrow_path.read_text() != FIXED32_QUERY_TILE16_TRANSLATION_UNIT:
            raise RuntimeError("existing qrow16 translation unit drifted")
        return False
    qrow_path.write_text(FIXED32_QUERY_TILE16_TRANSLATION_UNIT)
    return True


def _patch_fixed32_query_tile32_translation_unit(
    path: Path,
    *,
    fixed32_query_tile32: bool = False,
) -> bool:
    if not fixed32_query_tile32:
        return False
    stock_text = path.read_text()
    if STOCK_FIXED32_QUERY_INSTANTIATION not in stock_text:
        raise RuntimeError("stock fixed32 FA2 explicit instantiation drifted")
    if "FR13_FA2_FIXED32_QUERY_TILE32" in stock_text:
        raise RuntimeError("qrow32 must not share the stock instantiation TU")
    qrow_path = path.with_name("flash_fwd_fr13_qrow32_hdim256_bf16_sm80.cu")
    if qrow_path.exists():
        if qrow_path.read_text() != FIXED32_QUERY_TILE32_TRANSLATION_UNIT:
            raise RuntimeError("existing qrow32 translation unit drifted")
        return False
    qrow_path.write_text(FIXED32_QUERY_TILE32_TRANSLATION_UNIT)
    return True


def _patch_flash_api_cpp(
    path: Path,
    *,
    fixed32_query_tile16: bool = False,
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
        text, did = _replace_once(
            text,
            STOCK_RUN_MHA_FWD,
            FIXED32_QUERY_TILE16_API_DISPATCH,
            "fixed32 FA2 query tile16 hidden API dispatch",
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
    fixed32_query_tile32: bool = False,
) -> dict[str, bool]:
    files = {
        "flash.h": fa2_src / "csrc/flash_attn/src/flash.h",
        "flash_fwd_kernel.h": fa2_src / "csrc/flash_attn/src/flash_fwd_kernel.h",
        "flash_fwd_split_hdim256_bf16_sm80.cu": fa2_src
        / "csrc/flash_attn/src/flash_fwd_split_hdim256_bf16_sm80.cu",
        "flash_api.cpp": fa2_src / "csrc/flash_attn/flash_api.cpp",
        "flash_api_torch_lib.cpp": fa2_src / "csrc/flash_attn/flash_api_torch_lib.cpp",
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing FA2 source files: " + ", ".join(missing))
    return {
        "flash.h": _patch_flash_h(files["flash.h"]),
        "flash_fwd_kernel.h": _patch_flash_fwd_kernel(
            files["flash_fwd_kernel.h"],
            tile_earlyout=tree_bias_tile_earlyout,
        ),
        "flash_fwd_fr13_qrow16_hdim256_bf16_sm80.cu": _patch_fixed32_query_translation_unit(
            files["flash_fwd_split_hdim256_bf16_sm80.cu"],
            fixed32_query_tile16=fixed32_query_tile16,
        ),
        "flash_fwd_fr13_qrow32_hdim256_bf16_sm80.cu": _patch_fixed32_query_tile32_translation_unit(
            files["flash_fwd_split_hdim256_bf16_sm80.cu"],
            fixed32_query_tile32=fixed32_query_tile32,
        ),
        "flash_api.cpp": _patch_flash_api_cpp(
            files["flash_api.cpp"],
            fixed32_query_tile16=fixed32_query_tile16,
        ),
        "flash_api_torch_lib.cpp": _patch_torch_lib(files["flash_api_torch_lib.cpp"]),
    }


def _patch_flash_attn_interface(path: Path) -> bool:
    text = path.read_text()
    changed = False
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


FIXED32_QUERY_TILE16_PRODUCTION_HELPERS = r'''# FR13_FA2_QROW16_PRODUCTION
_FR13_FA2_QROW16_PRODUCTION_GRAPHS = {}
_FR13_FA2_QROW16_EAGER_STATE = {
    "layers": set(),
    "calls": 0,
    "emitted": False,
}
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
    eager_sfwd_stack = (
        not capturing
        and os.environ.get(
            "FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION", "0"
        ) == "1"
        and os.environ.get("ENFORCE_EAGER", "0") == "1"
        and os.environ.get("FR13_DRAFT_VOCAB_ROOT", "0") == "1"
        and os.environ.get("FR13_DRAFT_VOCAB_K", "") == "65536"
    )
    if not capturing and not eager_sfwd_stack:
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
                "sfwd_state_fusion_production": True,
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
    fixed32_query_tile16_production: bool = False,
    dfwd_unified_bm8_production: bool = False,
) -> bool:
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


def patch_installed_vllm(
    site_packages: Path,
    *,
    fixed32_query_tile16_live_ab: bool = False,
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
        "--fixed32-query-tile32",
        action="store_true",
        help="build the source-only fixed32 B4 FA2 32-row query-tile candidate",
    )
    parser.add_argument(
        "--fixed32-query-tile16-live-ab",
        action="store_true",
        help="install the one-shot live paged B1 stock/qrow16 byte gate",
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

    payload: dict[str, object] = {
        "tree_bias_tile_earlyout": args.tree_bias_tile_earlyout,
        "fixed32_query_tile16": args.fixed32_query_tile16,
        "fixed32_query_tile32": args.fixed32_query_tile32,
        "fixed32_query_tile16_live_ab": args.fixed32_query_tile16_live_ab,
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
            fixed32_query_tile32=args.fixed32_query_tile32,
        )
    if not args.skip_python:
        payload["python"] = patch_installed_vllm(
            args.site_packages,
            fixed32_query_tile16_live_ab=args.fixed32_query_tile16_live_ab,
            fixed32_query_tile16_production=args.fixed32_query_tile16_production,
            dfwd_unified_bm8_production=args.dfwd_unified_bm8_production,
        )
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
