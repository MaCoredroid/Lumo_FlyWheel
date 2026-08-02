"""Codegen-only fixed32 GDN single-launch root-loop candidate.

This module is intentionally absent from the serving launchers. It preserves
the audited kernel's node helper and exact fixed32 descriptors while replacing
only the five-way static root expansion with an ordered runtime loop.
"""

from __future__ import annotations

import triton
import triton.language as tl

from .fr10_gdn_tree_kernel import (
    _FR13_FIXED32_PARENT as _FR13_FIXED32_PARENT,
    _fr13_fixed32_gdn_single_launch_contract as _fr13_fixed32_gdn_single_launch_contract,
    _subtree_decompose as _subtree_decompose,
    _tree_gdn_fixed32_single_launch_node as _tree_gdn_fixed32_single_launch_node,
    _tree_gdn_path_kernel as _tree_gdn_path_kernel,
    _tree_gdn_path_kernel_fixed32_batch as _tree_gdn_path_kernel_fixed32_batch,
)


CANDIDATE = "fixed32_gdn_single_launch_root_loop_v1"


@triton.jit
def _tree_gdn_kernel_fixed32_single_launch_root_loop(
    q,
    k,
    v,
    g,
    beta,
    raw_a,
    raw_b,
    A_log,
    dt_bias,
    h0,
    h0_indices,
    h0_num_accepted_tokens,
    invocation_counter,
    root_nodes,
    branch_nodes,
    branch_lengths,
    group_path_indices,
    group_path_counts,
    out,
    ring_k,
    ring_v,
    ring_a,
    ring_b,
    flags_ptr,
    N_ACTUAL: tl.constexpr,
    NUM_KH: tl.constexpr,
    NUM_VH: tl.constexpr,
    DIM_K: tl.constexpr,
    DIM_V: tl.constexpr,
    BLOCK_V: tl.constexpr,
    OUTPUT_SCALE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    H0_IS_BANK: tl.constexpr,
    H0_INDEX_ROW: tl.constexpr,
    H0_INDEX_BATCH_STRIDE: tl.constexpr,
    H0_BATCH_INDEX: tl.constexpr,
    H0_ACCEPTED_BATCH_STRIDE: tl.constexpr,
    H0_BANK_STRIDE: tl.constexpr,
    H0_USE_ACCEPTED_COLUMN: tl.constexpr,
    RAW_GATING: tl.constexpr,
    COUNT_INVOCATION: tl.constexpr,
    SCAN_ALIGN: tl.constexpr,
    ROOT_STEPS: tl.constexpr,
    MAX_PATH_LEN: tl.constexpr,
    MAX_GROUP_PATHS: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    RING_EXPORT: tl.constexpr = False,
    FLAGS_EXPORT: tl.constexpr = False,
    FLAGS_ROWS: tl.constexpr = 0,
):
    """Run the exact depth-first recurrence with an ordered root loop."""
    pid_vh = tl.program_id(0)
    pid_v = tl.program_id(1)
    pid_batch = tl.program_id(2)
    if COUNT_INVOCATION:
        tl.atomic_add(
            invocation_counter,
            1,
            sem="relaxed",
            mask=(pid_vh == 0) & (pid_v == 0),
        )
    head_group = NUM_VH // NUM_KH
    pid_kh = pid_vh // head_group
    offs_k = tl.arange(0, DIM_K)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = offs_v < DIM_V

    h0_base = h0
    if H0_IS_BANK:
        h0_column = 0
        if H0_USE_ACCEPTED_COLUMN:
            accepted_index = H0_BATCH_INDEX + pid_batch * H0_ACCEPTED_BATCH_STRIDE
            h0_column = tl.maximum(
                tl.load(h0_num_accepted_tokens + accepted_index).to(tl.int64) - 1,
                0,
            )
        h0_index_row = H0_INDEX_ROW + pid_batch * H0_INDEX_BATCH_STRIDE
        h0_index = tl.load(h0_indices + h0_index_row + h0_column)
        h0_base = h0 + h0_index * H0_BANK_STRIDE
    root_state = tl.load(
        h0_base + (pid_vh * DIM_V + offs_v[:, None]) * DIM_K + offs_k[None, :],
        mask=v_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    b_a_log = tl.load(A_log + pid_vh).to(tl.float32)
    b_dt_bias = tl.load(dt_bias + pid_vh).to(tl.float32)

    for root_index in tl.range(0, ROOT_STEPS):
        root_node = tl.load(root_nodes + root_index)
        root_state = _tree_gdn_fixed32_single_launch_node(
            root_state,
            q,
            k,
            v,
            g,
            beta,
            raw_a,
            raw_b,
            b_a_log,
            b_dt_bias,
            out,
            ring_k,
            ring_v,
            ring_a,
            ring_b,
            pid_batch,
            pid_vh,
            pid_v,
            pid_kh,
            offs_k,
            offs_v,
            v_mask,
            root_node,
            N_ACTUAL=N_ACTUAL,
            NUM_KH=NUM_KH,
            NUM_VH=NUM_VH,
            DIM_K=DIM_K,
            DIM_V=DIM_V,
            OUTPUT_SCALE=OUTPUT_SCALE,
            USE_QK_L2NORM_IN_KERNEL=USE_QK_L2NORM_IN_KERNEL,
            RAW_GATING=RAW_GATING,
            SCAN_ALIGN=SCAN_ALIGN,
            RING_EXPORT=RING_EXPORT,
        )
        group_path_count = tl.load(group_path_counts + root_index)
        for member in tl.static_range(0, MAX_GROUP_PATHS):
            member_ok = member < group_path_count
            path_index = tl.load(
                group_path_indices + root_index * MAX_GROUP_PATHS + member,
                mask=member_ok,
                other=0,
            )
            path_len = tl.load(
                branch_lengths + path_index,
                mask=member_ok,
                other=0,
            )
            branch_state = root_state
            for path_offset in tl.range(0, path_len):
                branch_node = tl.load(
                    branch_nodes + path_index * MAX_PATH_LEN + path_offset
                )
                branch_state = _tree_gdn_fixed32_single_launch_node(
                    branch_state,
                    q,
                    k,
                    v,
                    g,
                    beta,
                    raw_a,
                    raw_b,
                    b_a_log,
                    b_dt_bias,
                    out,
                    ring_k,
                    ring_v,
                    ring_a,
                    ring_b,
                    pid_batch,
                    pid_vh,
                    pid_v,
                    pid_kh,
                    offs_k,
                    offs_v,
                    v_mask,
                    branch_node,
                    N_ACTUAL=N_ACTUAL,
                    NUM_KH=NUM_KH,
                    NUM_VH=NUM_VH,
                    DIM_K=DIM_K,
                    DIM_V=DIM_V,
                    OUTPUT_SCALE=OUTPUT_SCALE,
                    USE_QK_L2NORM_IN_KERNEL=USE_QK_L2NORM_IN_KERNEL,
                    RAW_GATING=RAW_GATING,
                    SCAN_ALIGN=SCAN_ALIGN,
                    RING_EXPORT=RING_EXPORT,
                )
    if FLAGS_EXPORT:
        flag_writer = (pid_vh == 0) & (pid_v == 0) & (pid_batch == 0)
        tl.store(flags_ptr + 0, 1, mask=flag_writer)
        tl.store(flags_ptr + 1, FLAGS_ROWS, mask=flag_writer)


# The existing audit's variant inventory looks up this compatibility name.
_tree_gdn_kernel_fixed32_single_launch = (
    _tree_gdn_kernel_fixed32_single_launch_root_loop
)
