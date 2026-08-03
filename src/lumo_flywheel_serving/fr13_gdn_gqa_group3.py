"""Fixed32 GDN single-launch candidate grouped by shared key head.

The runtime imports this module only for the default-off ``gqa_group3`` live
gate.  The incumbent remains served while the authenticated post-replay
comparator qualifies this kernel on the captured graph's persistent operands.
"""

from __future__ import annotations

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
    maxnreg: int | None = None,
) -> dict[str, object]:
    """Launch the unserved candidate after explicit caller-side qualification."""
    contract = fixed32_gdn_gqa_group3_contract(batch_size, mode=mode)
    tensors = (
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
    )
    if any(not torch.is_tensor(tensor) for tensor in tensors):
        raise TypeError("GQA-group3 candidate operands must all be tensors")
    if any(tensor.device.type != "cuda" for tensor in tensors):
        raise ValueError("GQA-group3 source candidate requires CUDA operands")
    if int(q.shape[0]) != int(batch_size) * PHYSICAL_ROWS:
        raise ValueError("GQA-group3 q rows must equal B*physical32")
    if int(q.shape[1]) != NUM_K_HEADS or int(q.shape[2]) != DIM_K:
        raise ValueError("GQA-group3 q geometry drift")
    if int(v.shape[1]) != NUM_V_HEADS or int(v.shape[2]) != DIM_V:
        raise ValueError("GQA-group3 v geometry drift")
    if decay_export and not gate_export:
        raise ValueError("GQA-group3 decay export requires gate export")
    if maxnreg is not None and int(maxnreg) != 128:
        raise ValueError("GQA-group3 maxnreg must be unset or exactly 128")

    grid = (NUM_K_HEADS, DIM_V // BLOCK_V, int(batch_size))
    launch_options = {"num_warps": 8}
    if maxnreg is not None:
        launch_options["maxnreg"] = int(maxnreg)
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
        **launch_options,
    )
    return contract
