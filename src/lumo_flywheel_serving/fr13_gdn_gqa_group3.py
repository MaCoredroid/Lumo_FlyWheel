"""Fixed32 GDN single-launch candidate grouped by shared key head.

This module is deliberately not wired into serving.  It contains the next
kernel candidate and its closed physical-work contract so SM121a resource,
byte, and real-task gates can qualify it without changing the served arm.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


CANDIDATE = "fixed32_gdn_single_launch_gqa_group3_v1"
FIXED32_MODES = frozenset(("tail6_fixed32", "hydra27_fixed32"))
PHYSICAL_ROWS = 32
NUM_K_HEADS = 16
NUM_V_HEADS = 48
HEAD_GROUP = 3
DIM_K = 128
DIM_V = 128
BLOCK_V = 8
GDN_LAYERS = 48
BF16_BYTES = 2

_ROOT_STEPS = 5
_MAX_PATH_LEN = 7
_MAX_GROUP_PATHS = 3
_ROOT_NODES = (0, 1, 4, 9, 14)
_BRANCH_NODES = (
    (19, 24, 26, 28, 29, 30, 31),
    (2, 7, 12, 17, 22, -1, -1),
    (3, 8, 13, 18, 23, 25, 27),
    (5, -1, -1, -1, -1, -1, -1),
    (6, -1, -1, -1, -1, -1, -1),
    (10, -1, -1, -1, -1, -1, -1),
    (11, -1, -1, -1, -1, -1, -1),
    (15, -1, -1, -1, -1, -1, -1),
    (16, -1, -1, -1, -1, -1, -1),
    (20, -1, -1, -1, -1, -1, -1),
    (21, -1, -1, -1, -1, -1, -1),
)
_BRANCH_LENGTHS = (7, 5, 7, 1, 1, 1, 1, 1, 1, 1, 1)
_GROUP_PATH_INDICES = (
    (1, 2, -1),
    (3, 4, -1),
    (5, 6, -1),
    (7, 8, -1),
    (0, 9, 10),
)
_PRESCALED_GROUP_PATH_BASES = (
    (7, 14, 0),
    (21, 28, 0),
    (35, 42, 0),
    (49, 56, 0),
    (0, 63, 70),
)
_GROUP_PATH_COUNTS = (2, 2, 2, 2, 3)
_PRESCALED_BRANCH_LENGTHS = tuple(
    length if offset % _MAX_PATH_LEN == 0 else 0
    for length in _BRANCH_LENGTHS
    for offset in range(_MAX_PATH_LEN)
)


def fixed32_gdn_gqa_group3_contract(
    batch_size: int,
    *,
    mode: str,
    physical_rows: int = PHYSICAL_ROWS,
    num_k_heads: int = NUM_K_HEADS,
    num_v_heads: int = NUM_V_HEADS,
    dim_k: int = DIM_K,
    dim_v: int = DIM_V,
    block_v: int = BLOCK_V,
    layers: int = GDN_LAYERS,
) -> dict[str, object]:
    """Validate the exact padded tree and expose candidate work removal."""
    batch = int(batch_size)
    rows = int(physical_rows)
    kh = int(num_k_heads)
    vh = int(num_v_heads)
    dk = int(dim_k)
    dv = int(dim_v)
    bv = int(block_v)
    layer_count = int(layers)
    if batch not in (1, 4):
        raise ValueError("GQA-group3 qualification is restricted to B1 or B4")
    if mode not in FIXED32_MODES:
        raise ValueError("GQA-group3 requires Tail23 or Hydra27 fixed32 mode")
    observed = (rows, kh, vh, dk, dv, bv, layer_count)
    expected = (
        PHYSICAL_ROWS,
        NUM_K_HEADS,
        NUM_V_HEADS,
        DIM_K,
        DIM_V,
        BLOCK_V,
        GDN_LAYERS,
    )
    if observed != expected:
        raise ValueError(
            "GQA-group3 exact physical32 geometry drift: "
            f"observed={observed!r} expected={expected!r}"
        )
    if vh != kh * HEAD_GROUP or dv % bv:
        raise ValueError("GQA-group3 head ratio or value tiling drift")

    value_tiles = dv // bv
    reference_ctas_per_layer = batch * vh * value_tiles
    candidate_ctas_per_layer = batch * kh * value_tiles
    ctas_removed_per_layer = reference_ctas_per_layer - candidate_ctas_per_layer
    qk_bytes_per_node_cta = 2 * dk * BF16_BYTES
    qk_bytes_removed_per_event = (
        ctas_removed_per_layer
        * rows
        * qk_bytes_per_node_cta
        * layer_count
    )
    qk_norm_reductions_removed_per_event = (
        ctas_removed_per_layer * rows * 2 * layer_count
    )
    return {
        "candidate": CANDIDATE,
        "mode": mode,
        "batch_size": batch,
        "physical_rows_per_request": rows,
        "logical_tree_limit": rows,
        "fixed_work_for_any_logical_tree_lte": rows,
        "num_k_heads": kh,
        "num_v_heads": vh,
        "value_heads_per_key_head": HEAD_GROUP,
        "dim_k": dk,
        "dim_v": dv,
        "block_v": bv,
        "gdn_layers": layer_count,
        "physical_launches_per_layer": 1,
        "reference_ctas_per_layer": reference_ctas_per_layer,
        "candidate_ctas_per_layer": candidate_ctas_per_layer,
        "ctas_removed_per_layer": ctas_removed_per_layer,
        "ctas_removed_per_event": ctas_removed_per_layer * layer_count,
        "qk_bytes_removed_per_event": qk_bytes_removed_per_event,
        "qk_norm_reductions_removed_per_event": (
            qk_norm_reductions_removed_per_event
        ),
        "qk_norm_lane_terms_removed_per_event": (
            qk_norm_reductions_removed_per_event * dk
        ),
        "node_updates_per_request_layer": rows,
        "state_export_writes": 0,
        "state_parent_reads": 0,
        "candidate_default_off": True,
    }


@triton.jit
def _fr13_fixed32_gdn_gqa_group3_value_head_node(
    state_i,
    b_q,
    b_k,
    v,
    g,
    beta,
    raw_a,
    raw_b,
    b_a_log,
    b_dt_bias,
    out,
    ring_v,
    ring_a,
    ring_b,
    ring_gate,
    pid_batch,
    pid_vh,
    pid_v,
    offs_v,
    node,
    N_ACTUAL: tl.constexpr,
    NUM_VH: tl.constexpr,
    DIM_V: tl.constexpr,
    OUTPUT_SCALE: tl.constexpr,
    RAW_GATING: tl.constexpr,
    SCAN_ALIGN: tl.constexpr,
    RING_EXPORT: tl.constexpr,
    GATE_EXPORT: tl.constexpr,
    DECAY_EXPORT: tl.constexpr,
):
    """Run one sibling recurrence after its key head's q/k is prepared."""
    prior_state = state_i
    n_ok = (node >= 0) & (node < N_ACTUAL)
    node_c = tl.maximum(node, 0)
    global_node = pid_batch * N_ACTUAL + node_c
    value_offsets = (global_node * NUM_VH + pid_vh) * DIM_V + offs_v
    v_mask = offs_v < DIM_V
    b_v_raw = tl.load(
        v + value_offsets,
        mask=n_ok & v_mask,
        other=0.0,
    )
    b_v = b_v_raw.to(tl.float32)
    if RING_EXPORT:
        tl.store(
            ring_v + value_offsets,
            b_v_raw,
            mask=n_ok & v_mask,
        )

    head_offset = global_node * NUM_VH + pid_vh
    b_b = tl.load(beta + head_offset, mask=n_ok, other=0.0).to(tl.float32)
    b_g = tl.load(g + head_offset, mask=n_ok, other=0.0).to(tl.float32)
    b_raw_a = b_g
    b_raw_b = b_b
    if RAW_GATING:
        b_raw_a_in = tl.load(raw_a + head_offset, mask=n_ok, other=0.0)
        b_raw_b_in = tl.load(raw_b + head_offset, mask=n_ok, other=0.0)
        b_raw_a = b_raw_a_in.to(tl.float32)
        b_raw_b = b_raw_b_in.to(tl.float32)
        if RING_EXPORT:
            tl.store(
                ring_a + head_offset,
                b_raw_a_in,
                mask=n_ok & (pid_v == 0),
            )
            tl.store(
                ring_b + head_offset,
                b_raw_b_in,
                mask=n_ok & (pid_v == 0),
            )

    b_decay = tl.exp(b_g)
    if GATE_EXPORT:
        x = b_raw_a + b_dt_bias
        softplus_x = tl.where(
            x <= 20.0,
            tl.log(1.0 + tl.exp(x)),
            x,
        )
        b_g = -tl.exp(b_a_log) * softplus_x
        b_b = tl.sigmoid(b_raw_b.to(tl.float32))
        b_decay = tl.exp(b_g)
        gate_offset = head_offset * 2
        tl.store(
            ring_gate + gate_offset,
            b_decay if DECAY_EXPORT else b_g,
            mask=n_ok & (pid_v == 0),
        )
        tl.store(
            ring_gate + gate_offset + 1,
            b_b,
            mask=n_ok & (pid_v == 0),
        )

    if not GATE_EXPORT and RAW_GATING:
        x = b_raw_a + b_dt_bias
        softplus_x = tl.where(
            x <= 20.0,
            tl.log(1.0 + tl.exp(x)),
            x,
        )
        b_g = -tl.exp(b_a_log) * softplus_x
        if SCAN_ALIGN:
            b_b = tl.sigmoid(b_raw_b.to(tl.float32)).to(tl.bfloat16).to(
                tl.float32
            )
        else:
            b_b = tl.sigmoid(b_raw_b.to(tl.float32))

    b_q = b_q * OUTPUT_SCALE
    if DECAY_EXPORT:
        state_i *= b_decay
    else:
        state_i *= tl.exp(b_g)
    b_v -= tl.sum(state_i * b_k[None, :], axis=1)
    b_v *= b_b
    state_i += b_v[:, None] * b_k[None, :]
    out_i = tl.sum(state_i * b_q[None, :], axis=1)
    if SCAN_ALIGN:
        state_i = state_i.to(tl.bfloat16).to(tl.float32)

    tl.store(out + value_offsets, out_i, mask=n_ok & v_mask)
    return tl.where(n_ok, state_i, prior_state)


@triton.jit
def _fr13_fixed32_gdn_gqa_group3_node(
    state_0,
    state_1,
    state_2,
    q,
    k,
    v,
    g,
    beta,
    raw_a,
    raw_b,
    b_a_log_0,
    b_a_log_1,
    b_a_log_2,
    b_dt_bias_0,
    b_dt_bias_1,
    b_dt_bias_2,
    out,
    ring_k,
    ring_v,
    ring_a,
    ring_b,
    ring_k_norm,
    ring_gate,
    pid_batch,
    pid_kh,
    pid_v,
    offs_k,
    offs_v,
    node,
    N_ACTUAL: tl.constexpr,
    NUM_KH: tl.constexpr,
    NUM_VH: tl.constexpr,
    HEAD_GROUP: tl.constexpr,
    DIM_K: tl.constexpr,
    DIM_V: tl.constexpr,
    OUTPUT_SCALE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    RAW_GATING: tl.constexpr,
    SCAN_ALIGN: tl.constexpr,
    RING_EXPORT: tl.constexpr,
    K_NORM_EXPORT: tl.constexpr,
    GATE_EXPORT: tl.constexpr,
    DECAY_EXPORT: tl.constexpr,
):
    """Run three value heads after loading and normalizing shared q/k once."""
    n_ok = (node >= 0) & (node < N_ACTUAL)
    node_c = tl.maximum(node, 0)
    global_node = pid_batch * N_ACTUAL + node_c
    b_q = tl.load(
        q + (global_node * NUM_KH + pid_kh) * DIM_K + offs_k,
        mask=n_ok,
        other=0.0,
    ).to(tl.float32)
    b_k_raw = tl.load(
        k + (global_node * NUM_KH + pid_kh) * DIM_K + offs_k,
        mask=n_ok,
        other=0.0,
    )
    b_k = b_k_raw.to(tl.float32)
    if RING_EXPORT:
        tl.store(
            ring_k + (global_node * NUM_KH + pid_kh) * DIM_K + offs_k,
            b_k_raw,
            mask=n_ok & (pid_v == 0),
        )
    if K_NORM_EXPORT:
        b_q = b_q * tl.rsqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k_inv_norm = tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)
        b_k = b_k * b_k_inv_norm
        tl.store(
            ring_k_norm + global_node * NUM_KH + pid_kh,
            b_k_inv_norm,
            mask=n_ok & (pid_v == 0),
        )
    if not K_NORM_EXPORT and USE_QK_L2NORM_IN_KERNEL:
        if SCAN_ALIGN:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        else:
            b_q = b_q * tl.rsqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)

    pid_vh_0 = pid_kh * HEAD_GROUP
    pid_vh_1 = pid_vh_0 + 1
    pid_vh_2 = pid_vh_0 + 2
    state_0 = _fr13_fixed32_gdn_gqa_group3_value_head_node(
        state_0,
        b_q,
        b_k,
        v,
        g,
        beta,
        raw_a,
        raw_b,
        b_a_log_0,
        b_dt_bias_0,
        out,
        ring_v,
        ring_a,
        ring_b,
        ring_gate,
        pid_batch,
        pid_vh_0,
        pid_v,
        offs_v,
        node,
        N_ACTUAL=N_ACTUAL,
        NUM_VH=NUM_VH,
        DIM_V=DIM_V,
        OUTPUT_SCALE=OUTPUT_SCALE,
        RAW_GATING=RAW_GATING,
        SCAN_ALIGN=SCAN_ALIGN and not K_NORM_EXPORT and not GATE_EXPORT,
        RING_EXPORT=RING_EXPORT,
        GATE_EXPORT=GATE_EXPORT,
        DECAY_EXPORT=DECAY_EXPORT,
    )
    state_1 = _fr13_fixed32_gdn_gqa_group3_value_head_node(
        state_1,
        b_q,
        b_k,
        v,
        g,
        beta,
        raw_a,
        raw_b,
        b_a_log_1,
        b_dt_bias_1,
        out,
        ring_v,
        ring_a,
        ring_b,
        ring_gate,
        pid_batch,
        pid_vh_1,
        pid_v,
        offs_v,
        node,
        N_ACTUAL=N_ACTUAL,
        NUM_VH=NUM_VH,
        DIM_V=DIM_V,
        OUTPUT_SCALE=OUTPUT_SCALE,
        RAW_GATING=RAW_GATING,
        SCAN_ALIGN=SCAN_ALIGN and not K_NORM_EXPORT and not GATE_EXPORT,
        RING_EXPORT=RING_EXPORT,
        GATE_EXPORT=GATE_EXPORT,
        DECAY_EXPORT=DECAY_EXPORT,
    )
    state_2 = _fr13_fixed32_gdn_gqa_group3_value_head_node(
        state_2,
        b_q,
        b_k,
        v,
        g,
        beta,
        raw_a,
        raw_b,
        b_a_log_2,
        b_dt_bias_2,
        out,
        ring_v,
        ring_a,
        ring_b,
        ring_gate,
        pid_batch,
        pid_vh_2,
        pid_v,
        offs_v,
        node,
        N_ACTUAL=N_ACTUAL,
        NUM_VH=NUM_VH,
        DIM_V=DIM_V,
        OUTPUT_SCALE=OUTPUT_SCALE,
        RAW_GATING=RAW_GATING,
        SCAN_ALIGN=SCAN_ALIGN and not K_NORM_EXPORT and not GATE_EXPORT,
        RING_EXPORT=RING_EXPORT,
        GATE_EXPORT=GATE_EXPORT,
        DECAY_EXPORT=DECAY_EXPORT,
    )
    return state_0, state_1, state_2


@triton.jit
def _fr13_fixed32_gdn_gqa_group3_single_launch_kernel(
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
    ring_k_norm,
    ring_gate,
    N_ACTUAL: tl.constexpr,
    NUM_KH: tl.constexpr,
    NUM_VH: tl.constexpr,
    HEAD_GROUP: tl.constexpr,
    DIM_K: tl.constexpr,
    DIM_V: tl.constexpr,
    BLOCK_V: tl.constexpr,
    OUTPUT_SCALE: tl.constexpr,
    H0_IS_BANK: tl.constexpr,
    H0_INDEX_ROW: tl.constexpr,
    H0_INDEX_BATCH_STRIDE: tl.constexpr,
    H0_BATCH_INDEX: tl.constexpr,
    H0_ACCEPTED_BATCH_STRIDE: tl.constexpr,
    H0_BANK_STRIDE: tl.constexpr,
    H0_USE_ACCEPTED_COLUMN: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    RAW_GATING: tl.constexpr,
    COUNT_INVOCATION: tl.constexpr,
    SCAN_ALIGN: tl.constexpr,
    ROOT_STEPS: tl.constexpr,
    MAX_PATH_LEN: tl.constexpr,
    MAX_GROUP_PATHS: tl.constexpr,
    PRESCALED_PATH_BASE: tl.constexpr,
    RING_EXPORT: tl.constexpr,
    K_NORM_EXPORT: tl.constexpr,
    GATE_EXPORT: tl.constexpr,
    DECAY_EXPORT: tl.constexpr,
    FLAGS_EXPORT: tl.constexpr,
    FLAGS_ROWS: tl.constexpr,
):
    """Map one CTA to the three value heads sharing a fixed32 key head."""
    pid_kh = tl.program_id(0)
    pid_v = tl.program_id(1)
    pid_batch = tl.program_id(2)
    if COUNT_INVOCATION:
        tl.atomic_add(
            invocation_counter,
            1,
            sem="relaxed",
            mask=(pid_kh == 0) & (pid_v == 0),
        )

    pid_vh_0 = pid_kh * HEAD_GROUP
    pid_vh_1 = pid_vh_0 + 1
    pid_vh_2 = pid_vh_0 + 2
    offs_k = tl.arange(0, DIM_K)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = offs_v < DIM_V

    h0_base = h0
    if H0_IS_BANK:
        h0_column = 0
        if H0_USE_ACCEPTED_COLUMN:
            accepted_index = (
                H0_BATCH_INDEX + pid_batch * H0_ACCEPTED_BATCH_STRIDE
            )
            h0_column = tl.maximum(
                tl.load(h0_num_accepted_tokens + accepted_index).to(tl.int64)
                - 1,
                0,
            )
        h0_index_row = H0_INDEX_ROW + pid_batch * H0_INDEX_BATCH_STRIDE
        h0_index = tl.load(h0_indices + h0_index_row + h0_column)
        h0_base = h0 + h0_index * H0_BANK_STRIDE
    state_offsets_0 = (
        (pid_vh_0 * DIM_V + offs_v[:, None]) * DIM_K + offs_k[None, :]
    )
    state_offsets_1 = (
        (pid_vh_1 * DIM_V + offs_v[:, None]) * DIM_K + offs_k[None, :]
    )
    state_offsets_2 = (
        (pid_vh_2 * DIM_V + offs_v[:, None]) * DIM_K + offs_k[None, :]
    )
    root_state_0 = tl.load(
        h0_base + state_offsets_0,
        mask=v_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    root_state_1 = tl.load(
        h0_base + state_offsets_1,
        mask=v_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    root_state_2 = tl.load(
        h0_base + state_offsets_2,
        mask=v_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    b_a_log_0 = tl.load(A_log + pid_vh_0).to(tl.float32)
    b_a_log_1 = tl.load(A_log + pid_vh_1).to(tl.float32)
    b_a_log_2 = tl.load(A_log + pid_vh_2).to(tl.float32)
    b_dt_bias_0 = tl.load(dt_bias + pid_vh_0).to(tl.float32)
    b_dt_bias_1 = tl.load(dt_bias + pid_vh_1).to(tl.float32)
    b_dt_bias_2 = tl.load(dt_bias + pid_vh_2).to(tl.float32)

    for root_index in tl.range(0, ROOT_STEPS):
        root_node = tl.load(root_nodes + root_index)
        root_state_0, root_state_1, root_state_2 = (
            _fr13_fixed32_gdn_gqa_group3_node(
                root_state_0,
                root_state_1,
                root_state_2,
                q,
                k,
                v,
                g,
                beta,
                raw_a,
                raw_b,
                b_a_log_0,
                b_a_log_1,
                b_a_log_2,
                b_dt_bias_0,
                b_dt_bias_1,
                b_dt_bias_2,
                out,
                ring_k,
                ring_v,
                ring_a,
                ring_b,
                ring_k_norm,
                ring_gate,
                pid_batch,
                pid_kh,
                pid_v,
                offs_k,
                offs_v,
                root_node,
                N_ACTUAL=N_ACTUAL,
                NUM_KH=NUM_KH,
                NUM_VH=NUM_VH,
                HEAD_GROUP=HEAD_GROUP,
                DIM_K=DIM_K,
                DIM_V=DIM_V,
                OUTPUT_SCALE=OUTPUT_SCALE,
                USE_QK_L2NORM_IN_KERNEL=USE_QK_L2NORM_IN_KERNEL,
                RAW_GATING=RAW_GATING,
                SCAN_ALIGN=SCAN_ALIGN,
                RING_EXPORT=RING_EXPORT,
                K_NORM_EXPORT=K_NORM_EXPORT,
                GATE_EXPORT=GATE_EXPORT,
                DECAY_EXPORT=DECAY_EXPORT,
            )
        )
        group_path_count = tl.load(group_path_counts + root_index)
        for member in tl.static_range(0, MAX_GROUP_PATHS):
            member_ok = member < group_path_count
            path_index = tl.load(
                group_path_indices
                + root_index * MAX_GROUP_PATHS
                + member,
                mask=member_ok,
                other=0,
            )
            path_base = (
                path_index
                if PRESCALED_PATH_BASE
                else path_index * MAX_PATH_LEN
            )
            path_len = tl.load(
                branch_lengths
                + (path_base if PRESCALED_PATH_BASE else path_index),
                mask=member_ok,
                other=0,
            )
            branch_state_0 = root_state_0
            branch_state_1 = root_state_1
            branch_state_2 = root_state_2
            for path_offset in tl.range(0, path_len):
                branch_node = tl.load(branch_nodes + path_base + path_offset)
                branch_state_0, branch_state_1, branch_state_2 = (
                    _fr13_fixed32_gdn_gqa_group3_node(
                        branch_state_0,
                        branch_state_1,
                        branch_state_2,
                        q,
                        k,
                        v,
                        g,
                        beta,
                        raw_a,
                        raw_b,
                        b_a_log_0,
                        b_a_log_1,
                        b_a_log_2,
                        b_dt_bias_0,
                        b_dt_bias_1,
                        b_dt_bias_2,
                        out,
                        ring_k,
                        ring_v,
                        ring_a,
                        ring_b,
                        ring_k_norm,
                        ring_gate,
                        pid_batch,
                        pid_kh,
                        pid_v,
                        offs_k,
                        offs_v,
                        branch_node,
                        N_ACTUAL=N_ACTUAL,
                        NUM_KH=NUM_KH,
                        NUM_VH=NUM_VH,
                        HEAD_GROUP=HEAD_GROUP,
                        DIM_K=DIM_K,
                        DIM_V=DIM_V,
                        OUTPUT_SCALE=OUTPUT_SCALE,
                        USE_QK_L2NORM_IN_KERNEL=USE_QK_L2NORM_IN_KERNEL,
                        RAW_GATING=RAW_GATING,
                        SCAN_ALIGN=SCAN_ALIGN,
                        RING_EXPORT=RING_EXPORT,
                        K_NORM_EXPORT=K_NORM_EXPORT,
                        GATE_EXPORT=GATE_EXPORT,
                        DECAY_EXPORT=DECAY_EXPORT,
                    )
                )

    if FLAGS_EXPORT:
        writer = (pid_kh == 0) & (pid_v == 0) & (pid_batch == 0)
        tl.store(flags_ptr + 0, 1, mask=writer)
        tl.store(flags_ptr + 1, FLAGS_ROWS, mask=writer)


def _canonical_contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = []
    stride = 1
    for size in reversed(shape):
        strides.append(stride)
        stride *= int(size)
    return tuple(reversed(strides))


def _require_exact_int(name: str, value: object, *, minimum: int = 0) -> int:
    if type(value) is not int or int(value) < minimum:
        raise ValueError(
            f"GQA-group3 {name} must be an integer >= {minimum}"
        )
    return int(value)


def _require_exact_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> None:
    observed_shape = tuple(int(size) for size in tensor.shape)
    expected_strides = _canonical_contiguous_strides(shape)
    observed_strides = tuple(int(stride) for stride in tensor.stride())
    if tensor.device != device:
        raise ValueError(f"GQA-group3 {name} must share q.device")
    if tensor.dtype != dtype:
        raise ValueError(
            f"GQA-group3 {name} dtype drift: {tensor.dtype} != {dtype}"
        )
    if observed_shape != shape:
        raise ValueError(
            f"GQA-group3 {name} shape drift: "
            f"{observed_shape!r} != {shape!r}"
        )
    if not tensor.is_contiguous() or observed_strides != expected_strides:
        raise ValueError(
            f"GQA-group3 {name} must have canonical contiguous strides "
            f"{expected_strides!r}, got {observed_strides!r}"
        )


def _flattened_int_values(tensor: torch.Tensor) -> tuple[int, ...]:
    return tuple(
        int(value)
        for value in tensor.detach()
        .reshape(-1)
        .to(device="cpu", dtype=torch.int64)
        .tolist()
    )


def _tensor_interval(tensor: torch.Tensor) -> tuple[int, int]:
    start = int(tensor.data_ptr())
    return start, start + int(tensor.numel()) * int(tensor.element_size())


def _require_disjoint_writes(
    read_tensors: dict[str, torch.Tensor],
    write_tensors: dict[str, torch.Tensor],
) -> None:
    prior = list(read_tensors.items())
    for write_name, write_tensor in write_tensors.items():
        write_start, write_end = _tensor_interval(write_tensor)
        for other_name, other_tensor in prior:
            other_start, other_end = _tensor_interval(other_tensor)
            if write_start < other_end and other_start < write_end:
                raise ValueError(
                    "GQA-group3 writable tensor storage overlap: "
                    f"{write_name}/{other_name}"
                )
        prior.append((write_name, write_tensor))


def _validate_fixed32_gdn_gqa_group3_launch(
    tensors: dict[str, torch.Tensor],
    *,
    batch_size: int,
    output_scale: float,
    h0_is_bank: bool,
    h0_index_row: int,
    h0_index_batch_stride: int,
    h0_batch_index: int,
    h0_accepted_batch_stride: int,
    h0_bank_stride: int,
    h0_use_accepted_column: bool,
    use_qk_l2norm_in_kernel: bool,
    raw_gating: bool,
    count_invocation: bool,
    scan_align: bool,
    root_steps: int,
    max_path_len: int,
    max_group_paths: int,
    prescaled_path_base: bool,
    ring_export: bool,
    k_norm_export: bool,
    gate_export: bool,
    decay_export: bool,
    flags_export: bool,
    flags_rows: int,
) -> None:
    expected_names = {
        "q",
        "k",
        "v",
        "g",
        "beta",
        "raw_a",
        "raw_b",
        "A_log",
        "dt_bias",
        "h0",
        "h0_indices",
        "h0_num_accepted_tokens",
        "invocation_counter",
        "root_nodes",
        "branch_nodes",
        "branch_lengths",
        "group_path_indices",
        "group_path_counts",
        "out",
        "ring_k",
        "ring_v",
        "ring_a",
        "ring_b",
        "flags",
        "ring_k_norm",
        "ring_gate",
    }
    if set(tensors) != expected_names:
        raise ValueError("GQA-group3 operand binding drift")
    if any(not torch.is_tensor(tensor) for tensor in tensors.values()):
        raise TypeError("GQA-group3 candidate operands must all be tensors")

    q = tensors["q"]
    if q.device.type != "cuda":
        raise ValueError("GQA-group3 source candidate requires CUDA operands")
    device = q.device
    if any(tensor.device != device for tensor in tensors.values()):
        raise ValueError("GQA-group3 candidate operands must share one device")

    if type(batch_size) is not int or batch_size not in (1, 4):
        raise ValueError("GQA-group3 launch batch_size must be exactly 1 or 4")
    bool_parameters = {
        "h0_is_bank": h0_is_bank,
        "h0_use_accepted_column": h0_use_accepted_column,
        "use_qk_l2norm_in_kernel": use_qk_l2norm_in_kernel,
        "raw_gating": raw_gating,
        "count_invocation": count_invocation,
        "scan_align": scan_align,
        "prescaled_path_base": prescaled_path_base,
        "ring_export": ring_export,
        "k_norm_export": k_norm_export,
        "gate_export": gate_export,
        "decay_export": decay_export,
        "flags_export": flags_export,
    }
    invalid_bool = next(
        (name for name, value in bool_parameters.items() if type(value) is not bool),
        None,
    )
    if invalid_bool is not None:
        raise TypeError(f"GQA-group3 {invalid_bool} must be bool")
    try:
        scale = float(output_scale)
    except (TypeError, ValueError) as error:
        raise TypeError("GQA-group3 output_scale must be finite") from error
    if not math.isfinite(scale):
        raise ValueError("GQA-group3 output_scale must be finite")

    root_steps = _require_exact_int("root_steps", root_steps)
    max_path_len = _require_exact_int("max_path_len", max_path_len)
    max_group_paths = _require_exact_int(
        "max_group_paths", max_group_paths
    )
    if (
        root_steps,
        max_path_len,
        max_group_paths,
    ) != (_ROOT_STEPS, _MAX_PATH_LEN, _MAX_GROUP_PATHS):
        raise ValueError("GQA-group3 exact fixed32 schedule extent drift")

    rows = batch_size * PHYSICAL_ROWS
    exact_tensors = (
        ("q", torch.bfloat16, (rows, NUM_K_HEADS, DIM_K)),
        ("k", torch.bfloat16, (rows, NUM_K_HEADS, DIM_K)),
        ("v", torch.bfloat16, (rows, NUM_V_HEADS, DIM_V)),
        ("g", torch.float32, (rows, NUM_V_HEADS)),
        ("beta", torch.float32, (rows, NUM_V_HEADS)),
        ("raw_a", torch.bfloat16, (rows, NUM_V_HEADS)),
        ("raw_b", torch.bfloat16, (rows, NUM_V_HEADS)),
        ("A_log", torch.float32, (NUM_V_HEADS,)),
        ("dt_bias", torch.float32, (NUM_V_HEADS,)),
        ("root_nodes", torch.int32, (1, _ROOT_STEPS)),
        ("branch_nodes", torch.int32, (len(_BRANCH_NODES), _MAX_PATH_LEN)),
        ("group_path_indices", torch.int32, (_ROOT_STEPS, _MAX_GROUP_PATHS)),
        ("group_path_counts", torch.int32, (_ROOT_STEPS,)),
        ("out", torch.bfloat16, (rows, NUM_V_HEADS, DIM_V)),
    )
    for name, dtype, shape in exact_tensors:
        _require_exact_tensor(
            name,
            tensors[name],
            device=device,
            dtype=dtype,
            shape=tuple(int(size) for size in shape),
        )

    branch_length_shape = (
        len(_PRESCALED_BRANCH_LENGTHS),
    ) if prescaled_path_base else (len(_BRANCH_LENGTHS),)
    _require_exact_tensor(
        "branch_lengths",
        tensors["branch_lengths"],
        device=device,
        dtype=torch.int32,
        shape=branch_length_shape,
    )

    h0 = tensors["h0"]
    state_shape = (NUM_V_HEADS, DIM_V, DIM_K)
    if h0_is_bank:
        if h0.ndim != 4 or tuple(h0.shape[1:]) != state_shape:
            raise ValueError(
                "GQA-group3 h0 bank shape must be [bank, 48, 128, 128]"
            )
        if int(h0.shape[0]) < 1:
            raise ValueError("GQA-group3 h0 bank must not be empty")
        _require_exact_tensor(
            "h0",
            h0,
            device=device,
            dtype=torch.float32,
            shape=tuple(int(size) for size in h0.shape),
        )
        h0_indices = tensors["h0_indices"]
        if h0_indices.ndim != 2 or int(h0_indices.shape[1]) < 1:
            raise ValueError(
                "GQA-group3 h0_indices must be rank-2 with columns"
            )
        _require_exact_tensor(
            "h0_indices",
            h0_indices,
            device=device,
            dtype=torch.int64,
            shape=(batch_size, int(h0_indices.shape[1])),
        )
        _require_exact_tensor(
            "h0_num_accepted_tokens",
            tensors["h0_num_accepted_tokens"],
            device=device,
            dtype=torch.int32,
            shape=(batch_size,),
        )
    else:
        _require_exact_tensor(
            "h0",
            h0,
            device=device,
            dtype=torch.float32,
            shape=state_shape,
        )

    descriptor_expectations = (
        ("root_nodes", _ROOT_NODES),
        ("branch_nodes", tuple(value for row in _BRANCH_NODES for value in row)),
        (
            "branch_lengths",
            _PRESCALED_BRANCH_LENGTHS
            if prescaled_path_base
            else _BRANCH_LENGTHS,
        ),
        (
            "group_path_indices",
            tuple(
                value
                for row in (
                    _PRESCALED_GROUP_PATH_BASES
                    if prescaled_path_base
                    else _GROUP_PATH_INDICES
                )
                for value in row
            ),
        ),
        ("group_path_counts", _GROUP_PATH_COUNTS),
    )
    for name, expected in descriptor_expectations:
        if _flattened_int_values(tensors[name]) != tuple(expected):
            raise ValueError(f"GQA-group3 exact {name} descriptor drift")

    h0_index_row = _require_exact_int("h0_index_row", h0_index_row)
    h0_index_batch_stride = _require_exact_int(
        "h0_index_batch_stride", h0_index_batch_stride
    )
    h0_batch_index = _require_exact_int("h0_batch_index", h0_batch_index)
    h0_accepted_batch_stride = _require_exact_int(
        "h0_accepted_batch_stride", h0_accepted_batch_stride
    )
    h0_bank_stride = _require_exact_int("h0_bank_stride", h0_bank_stride)
    h0_indices = tensors["h0_indices"]
    accepted_counts = tensors["h0_num_accepted_tokens"]
    if h0_is_bank:
        expected_index_stride = int(h0_indices.stride(0))
        expected_accepted_stride = int(accepted_counts.stride(0))
        if h0_index_batch_stride != expected_index_stride:
            raise ValueError("GQA-group3 h0 index batch stride drift")
        if h0_accepted_batch_stride != expected_accepted_stride:
            raise ValueError("GQA-group3 accepted-count batch stride drift")
        if h0_bank_stride != int(h0.stride(0)):
            raise ValueError("GQA-group3 h0 bank stride drift")
        accepted_values = _flattened_int_values(accepted_counts)
        h0_index_values = _flattened_int_values(h0_indices)
        for request in range(batch_size):
            accepted_address = (
                h0_batch_index + request * h0_accepted_batch_stride
            )
            if accepted_address >= len(accepted_values):
                raise ValueError("GQA-group3 accepted-count pointer domain drift")
            accepted_count = accepted_values[accepted_address]
            if accepted_count < 0:
                raise ValueError("GQA-group3 accepted counts must be nonnegative")
            column = max(accepted_count - 1, 0) if h0_use_accepted_column else 0
            index_address = (
                h0_index_row
                + request * h0_index_batch_stride
                + column
            )
            row_end = (
                h0_index_row
                + request * h0_index_batch_stride
                + int(h0_indices.shape[1])
            )
            if index_address < 0 or index_address >= min(row_end, len(h0_index_values)):
                raise ValueError("GQA-group3 h0 index pointer domain drift")
            bank_index = h0_index_values[index_address]
            if bank_index < 0 or bank_index >= int(h0.shape[0]):
                raise ValueError("GQA-group3 h0 bank index domain drift")
    elif (
        h0_index_row != 0
        or h0_index_batch_stride != 0
        or h0_batch_index != 0
        or h0_accepted_batch_stride != 0
        or h0_bank_stride != 0
        or h0_use_accepted_column
    ):
        raise ValueError("GQA-group3 non-bank h0 requires neutral bank metadata")

    if ring_export and not raw_gating:
        raise ValueError("GQA-group3 ring export requires raw gating")
    if k_norm_export and (
        not ring_export or not use_qk_l2norm_in_kernel or scan_align
    ):
        raise ValueError(
            "GQA-group3 K-norm export requires ring export, kernel L2 norm, "
            "and scan_align=False"
        )
    if gate_export and (not k_norm_export or not raw_gating):
        raise ValueError(
            "GQA-group3 gate export requires K-norm export and raw gating"
        )
    if decay_export and not gate_export:
        raise ValueError("GQA-group3 decay export requires gate export")
    if count_invocation:
        _require_exact_tensor(
            "invocation_counter",
            tensors["invocation_counter"],
            device=device,
            dtype=torch.int32,
            shape=(),
        )

    if ring_export:
        for name, dtype, shape in (
            ("ring_k", torch.bfloat16, (rows, NUM_K_HEADS, DIM_K)),
            ("ring_v", torch.bfloat16, (rows, NUM_V_HEADS, DIM_V)),
            ("ring_a", torch.bfloat16, (rows, NUM_V_HEADS)),
            ("ring_b", torch.bfloat16, (rows, NUM_V_HEADS)),
        ):
            _require_exact_tensor(
                name,
                tensors[name],
                device=device,
                dtype=dtype,
                shape=shape,
            )
    if k_norm_export:
        _require_exact_tensor(
            "ring_k_norm",
            tensors["ring_k_norm"],
            device=device,
            dtype=torch.float32,
            shape=(rows, NUM_K_HEADS),
        )
    if gate_export:
        _require_exact_tensor(
            "ring_gate",
            tensors["ring_gate"],
            device=device,
            dtype=torch.float32,
            shape=(rows, NUM_V_HEADS, 2),
        )
    if flags_export:
        if _require_exact_int("flags_rows", flags_rows) != batch_size:
            raise ValueError("GQA-group3 flags_rows must equal batch_size")
        _require_exact_tensor(
            "flags",
            tensors["flags"],
            device=device,
            dtype=torch.int32,
            shape=(2,),
        )
    elif flags_rows != 0:
        raise ValueError("GQA-group3 flags_rows must be zero when export is off")

    reads = {
        name: tensor
        for name, tensor in tensors.items()
        if name
        not in {
            "out",
            "invocation_counter",
            "ring_k",
            "ring_v",
            "ring_a",
            "ring_b",
            "flags",
            "ring_k_norm",
            "ring_gate",
        }
    }
    writes = {"out": tensors["out"]}
    if count_invocation:
        writes["invocation_counter"] = tensors["invocation_counter"]
    if ring_export:
        writes.update(
            (name, tensors[name])
            for name in ("ring_k", "ring_v", "ring_a", "ring_b")
        )
    if k_norm_export:
        writes["ring_k_norm"] = tensors["ring_k_norm"]
    if gate_export:
        writes["ring_gate"] = tensors["ring_gate"]
    if flags_export:
        writes["flags"] = tensors["flags"]
    _require_disjoint_writes(reads, writes)


def launch_fixed32_gdn_gqa_group3_source_candidate(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    raw_a: torch.Tensor,
    raw_b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    h0: torch.Tensor,
    h0_indices: torch.Tensor,
    h0_num_accepted_tokens: torch.Tensor,
    invocation_counter: torch.Tensor,
    root_nodes: torch.Tensor,
    branch_nodes: torch.Tensor,
    branch_lengths: torch.Tensor,
    group_path_indices: torch.Tensor,
    group_path_counts: torch.Tensor,
    out: torch.Tensor,
    ring_k: torch.Tensor,
    ring_v: torch.Tensor,
    ring_a: torch.Tensor,
    ring_b: torch.Tensor,
    flags: torch.Tensor,
    ring_k_norm: torch.Tensor,
    ring_gate: torch.Tensor,
    batch_size: int,
    mode: str,
    output_scale: float,
    h0_is_bank: bool,
    h0_index_row: int,
    h0_index_batch_stride: int,
    h0_batch_index: int,
    h0_accepted_batch_stride: int,
    h0_bank_stride: int,
    h0_use_accepted_column: bool,
    use_qk_l2norm_in_kernel: bool,
    raw_gating: bool,
    count_invocation: bool,
    scan_align: bool,
    root_steps: int,
    max_path_len: int,
    max_group_paths: int,
    prescaled_path_base: bool,
    ring_export: bool,
    k_norm_export: bool,
    gate_export: bool,
    decay_export: bool,
    flags_export: bool,
    flags_rows: int,
) -> dict[str, object]:
    """Launch the unserved candidate after explicit caller-side qualification."""
    contract = fixed32_gdn_gqa_group3_contract(batch_size, mode=mode)
    tensors = {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "raw_a": raw_a,
        "raw_b": raw_b,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "h0": h0,
        "h0_indices": h0_indices,
        "h0_num_accepted_tokens": h0_num_accepted_tokens,
        "invocation_counter": invocation_counter,
        "root_nodes": root_nodes,
        "branch_nodes": branch_nodes,
        "branch_lengths": branch_lengths,
        "group_path_indices": group_path_indices,
        "group_path_counts": group_path_counts,
        "out": out,
        "ring_k": ring_k,
        "ring_v": ring_v,
        "ring_a": ring_a,
        "ring_b": ring_b,
        "flags": flags,
        "ring_k_norm": ring_k_norm,
        "ring_gate": ring_gate,
    }
    _validate_fixed32_gdn_gqa_group3_launch(
        tensors,
        batch_size=batch_size,
        output_scale=output_scale,
        h0_is_bank=h0_is_bank,
        h0_index_row=h0_index_row,
        h0_index_batch_stride=h0_index_batch_stride,
        h0_batch_index=h0_batch_index,
        h0_accepted_batch_stride=h0_accepted_batch_stride,
        h0_bank_stride=h0_bank_stride,
        h0_use_accepted_column=h0_use_accepted_column,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        raw_gating=raw_gating,
        count_invocation=count_invocation,
        scan_align=scan_align,
        root_steps=root_steps,
        max_path_len=max_path_len,
        max_group_paths=max_group_paths,
        prescaled_path_base=prescaled_path_base,
        ring_export=ring_export,
        k_norm_export=k_norm_export,
        gate_export=gate_export,
        decay_export=decay_export,
        flags_export=flags_export,
        flags_rows=flags_rows,
    )

    grid = (NUM_K_HEADS, DIM_V // BLOCK_V, int(batch_size))
    _fr13_fixed32_gdn_gqa_group3_single_launch_kernel[grid](
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
        flags,
        ring_k_norm,
        ring_gate,
        N_ACTUAL=PHYSICAL_ROWS,
        NUM_KH=NUM_K_HEADS,
        NUM_VH=NUM_V_HEADS,
        HEAD_GROUP=HEAD_GROUP,
        DIM_K=DIM_K,
        DIM_V=DIM_V,
        BLOCK_V=BLOCK_V,
        OUTPUT_SCALE=float(output_scale),
        H0_IS_BANK=bool(h0_is_bank),
        H0_INDEX_ROW=int(h0_index_row),
        H0_INDEX_BATCH_STRIDE=int(h0_index_batch_stride),
        H0_BATCH_INDEX=int(h0_batch_index),
        H0_ACCEPTED_BATCH_STRIDE=int(h0_accepted_batch_stride),
        H0_BANK_STRIDE=int(h0_bank_stride),
        H0_USE_ACCEPTED_COLUMN=bool(h0_use_accepted_column),
        USE_QK_L2NORM_IN_KERNEL=bool(use_qk_l2norm_in_kernel),
        RAW_GATING=bool(raw_gating),
        COUNT_INVOCATION=bool(count_invocation),
        SCAN_ALIGN=bool(scan_align),
        ROOT_STEPS=int(root_steps),
        MAX_PATH_LEN=int(max_path_len),
        MAX_GROUP_PATHS=int(max_group_paths),
        PRESCALED_PATH_BASE=bool(prescaled_path_base),
        RING_EXPORT=bool(ring_export),
        K_NORM_EXPORT=bool(k_norm_export),
        GATE_EXPORT=bool(gate_export),
        DECAY_EXPORT=bool(decay_export),
        FLAGS_EXPORT=bool(flags_export),
        FLAGS_ROWS=int(flags_rows),
        num_warps=8,
    )
    return contract
