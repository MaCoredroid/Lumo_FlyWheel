#!/usr/bin/env python3
"""Patch vLLM FA2 with an FR13 tree-bias varlen forward op.

The patch intentionally leaves the stock ``varlen_fwd`` op unchanged and adds
``varlen_fwd_tree_bias``.  The new op carries a dense ancestry-bias matrix into
FA2 and adds it to score tiles after QK and before masking/softmax.
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


TREE_BIAS_HELPER = r'''
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
    const int lane_id = threadIdx.x % 32;
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


def _patch_flash_fwd_kernel(path: Path) -> bool:
    text = path.read_text()
    changed = False
    helper_marker = "// FR13_FA2_TREE_BIAS: add a dense query-suffix ancestry bias after QK."
    helper_end_marker = "template<typename Kernel_traits, bool Is_dropout"
    if helper_marker in text:
        start = text.index(helper_marker)
        end = text.index(helper_end_marker, start)
        if text[start:end] != TREE_BIAS_HELPER.lstrip():
            text = text[:start] + TREE_BIAS_HELPER.lstrip() + text[end:]
            changed = True
    else:
        text, did = _insert_once(
            text,
            helper_end_marker,
            TREE_BIAS_HELPER,
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


def _patch_flash_api_cpp(path: Path) -> bool:
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


def patch_fa2_source(fa2_src: Path) -> dict[str, bool]:
    files = {
        "flash.h": fa2_src / "csrc/flash_attn/src/flash.h",
        "flash_fwd_kernel.h": fa2_src / "csrc/flash_attn/src/flash_fwd_kernel.h",
        "flash_api.cpp": fa2_src / "csrc/flash_attn/flash_api.cpp",
        "flash_api_torch_lib.cpp": fa2_src / "csrc/flash_attn/flash_api_torch_lib.cpp",
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing FA2 source files: " + ", ".join(missing))
    return {
        "flash.h": _patch_flash_h(files["flash.h"]),
        "flash_fwd_kernel.h": _patch_flash_fwd_kernel(files["flash_fwd_kernel.h"]),
        "flash_api.cpp": _patch_flash_api_cpp(files["flash_api.cpp"]),
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


def _patch_tree_attn(path: Path) -> bool:
    text = path.read_text()
    changed = False
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
    anchor = """        if decode_meta := attn_metadata.decode_metadata:\n            unified_attention(\n                q=query[:num_decode_tokens],\n                k=key_cache,\n                v=value_cache,\n                out=output[:num_decode_tokens],\n                cu_seqlens_q=decode_meta.query_start_loc,\n                max_seqlen_q=decode_meta.max_query_len,\n                seqused_k=decode_meta.seq_lens,\n                max_seqlen_k=decode_meta.max_seq_len,\n                softmax_scale=self.scale,\n                causal=True,\n                alibi_slopes=self.alibi_slopes,\n                qq_bias=decode_meta.tree_attn_bias,\n                window_size=self.sliding_window,\n                block_table=decode_meta.block_table,\n                softcap=self.logits_soft_cap,\n                q_descale=None,  # Not supported\n                k_descale=layer._k_scale.expand(descale_shape),\n                v_descale=layer._v_scale.expand(descale_shape),\n            )\n"""
    replacement = """        if decode_meta := attn_metadata.decode_metadata:\n            tree_bias = decode_meta.tree_attn_bias\n            use_tree_bias = tree_bias is not None and tree_bias.numel() > 0\n            if os.environ.get("FR13_FA2_TREE_BIAS", "0") == "1" and use_tree_bias:\n                if self.alibi_slopes is not None:\n                    raise NotImplementedError("FR13_FA2_TREE_BIAS does not support ALiBi")\n                sliding_window_size = (\n                    list(self.sliding_window)\n                    if self.sliding_window is not None\n                    else None\n                )\n                flash_attn_varlen_func(\n                    q=query[:num_decode_tokens],\n                    k=key_cache,\n                    v=value_cache,\n                    out=output[:num_decode_tokens],\n                    cu_seqlens_q=decode_meta.query_start_loc,\n                    max_seqlen_q=decode_meta.max_query_len,\n                    seqused_k=decode_meta.seq_lens,\n                    max_seqlen_k=decode_meta.max_seq_len,\n                    softmax_scale=self.scale,\n                    causal=True,\n                    alibi_slopes=None,\n                    window_size=sliding_window_size,\n                    block_table=decode_meta.block_table,\n                    softcap=self.logits_soft_cap,\n                    fa_version=2,\n                    tree_bias=tree_bias,\n                )\n            else:\n                unified_attention(\n                    q=query[:num_decode_tokens],\n                    k=key_cache,\n                    v=value_cache,\n                    out=output[:num_decode_tokens],\n                    cu_seqlens_q=decode_meta.query_start_loc,\n                    max_seqlen_q=decode_meta.max_query_len,\n                    seqused_k=decode_meta.seq_lens,\n                    max_seqlen_k=decode_meta.max_seq_len,\n                    softmax_scale=self.scale,\n                    causal=True,\n                    alibi_slopes=self.alibi_slopes,\n                    qq_bias=decode_meta.tree_attn_bias,\n                    window_size=self.sliding_window,\n                    block_table=decode_meta.block_table,\n                    softcap=self.logits_soft_cap,\n                    q_descale=None,  # Not supported\n                    k_descale=layer._k_scale.expand(descale_shape),\n                    v_descale=layer._v_scale.expand(descale_shape),\n                )\n"""
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
    else:
        did = False
    changed = changed or did
    if changed:
        path.write_text(text)
        py_compile.compile(path, doraise=True)
    return changed


def patch_installed_vllm(site_packages: Path) -> dict[str, bool]:
    return {
        "flash_attn_interface.py": _patch_flash_attn_interface(
            site_packages / "vllm/vllm_flash_attn/flash_attn_interface.py"
        ),
        "tree_attn.py": _patch_tree_attn(
            site_packages / "vllm/v1/attention/backends/tree_attn.py"
        ),
    }


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
    args = parser.parse_args()

    payload: dict[str, object] = {}
    if not args.skip_source:
        fa2_src = find_fa2_source(str(args.fa2_src) if args.fa2_src else None)
        payload["fa2_src"] = str(fa2_src)
        payload["source"] = patch_fa2_source(fa2_src)
    if not args.skip_python:
        payload["python"] = patch_installed_vllm(args.site_packages)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
