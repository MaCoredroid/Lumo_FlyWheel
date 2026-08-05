"""Generated fixed32 SFWD tree-conv/post-prep fusion kernel.

Regenerate with ``scripts/fr13_generate_sfwd_conv_postprep_fusion_kernel.py``.
Do not edit the unrolled producer body by hand.
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _fr13_store_fixed32_conv_outputs(
    query,
    key,
    value_spec,
    value_tree,
    conv_tap,
    activated,
    pid_b,
    node: tl.constexpr,
    offs_c,
    N: tl.constexpr,
    C: tl.constexpr,
    Q_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    STORE_CONV_TAP: tl.constexpr,
):
    """Publish the exact BF16 boundary without a full conv intermediate."""
    row = pid_b * N + node
    activated_bf16 = activated.to(tl.bfloat16)
    q_mask = offs_c < Q_DIM
    k_mask = (offs_c >= Q_DIM) & (offs_c < 2 * Q_DIM)
    v_mask = offs_c >= 2 * Q_DIM
    tl.store(
        query + row * Q_DIM + offs_c,
        activated_bf16,
        mask=q_mask,
    )
    tl.store(
        key + row * Q_DIM + (offs_c - Q_DIM),
        activated_bf16,
        mask=k_mask,
    )
    value_offset = row * V_DIM + (offs_c - 2 * Q_DIM)
    tl.store(value_spec + value_offset, activated_bf16, mask=v_mask)
    tl.store(value_tree + value_offset, activated_bf16, mask=v_mask)
    if STORE_CONV_TAP:
        tl.store(conv_tap + row * C + offs_c, activated_bf16)


@triton.jit
def _fr13_fixed32_sfwd_conv_postprep_fusion_kernel(
    x,
    conv_state,
    spec_state_indices,
    sticky_guard_ok,
    conv_weights,
    bias,
    a,
    b,
    A_log,
    dt_bias,
    query,
    key,
    value_spec,
    value_tree,
    g,
    beta,
    source_stage,
    conv_tap,
    CONV_STRIDE_ROW: tl.constexpr,
    BANK_ROWS: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    WIDTH: tl.constexpr,
    STATE_LEN: tl.constexpr,
    SOURCE_ROWS: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    STORE_CONV_TAP: tl.constexpr,
    CAPTURE_GUARD: tl.constexpr,
    X_STRIDE_ROW: tl.constexpr,
    BLOCK_C: tl.constexpr,
    GATE_BLOCK: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    EMBED_GATE_CTA: tl.constexpr,
):
    """Fuse one fixed32 layer's conv, recurrence outputs, and post-prep."""
    pid_b = tl.program_id(0)
    pid_task = tl.program_id(1)
    channel_tasks: tl.constexpr = C // BLOCK_C
    Q_DIM: tl.constexpr = H * K
    V_DIM: tl.constexpr = HV * V
    GATE_ROWS: tl.constexpr = 2 * BLOCK_C // GATE_BLOCK
    GATE_TASKS: tl.constexpr = N // GATE_ROWS
    if EMBED_GATE_CTA:
        pid_c = pid_task
        offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        x_batch = x + pid_b * N * X_STRIDE_ROW
        stage_batch = source_stage + pid_b * SOURCE_ROWS * C

        bank_row_raw = tl.load(spec_state_indices + pid_b * N).to(tl.int64)
        bank_row_ok = (bank_row_raw >= 0) & (bank_row_raw < BANK_ROWS)
        bank_row = tl.maximum(0, tl.minimum(bank_row_raw, BANK_ROWS - 1))
        if CAPTURE_GUARD:
            # Valid replays perform no store. The first invalid replay makes the
            # committer's existing async assertion fail permanently, while the
            # clamped row prevents an out-of-bounds read before that assertion.
            tl.atomic_xchg(sticky_guard_ok, 0, mask=~bank_row_ok)
        prior_base = conv_state + bank_row * CONV_STRIDE_ROW + offs_c
        prior_0 = tl.load(prior_base)
        prior_1 = tl.load(prior_base + C)
        prior_2 = tl.load(prior_base + 2 * C)

        weight_channels = conv_weights + offs_c * WIDTH
        weight_pair_01 = tl.load(weight_channels.to(tl.pointer_type(tl.uint32)))
        weight_pair_23 = tl.load(
            (weight_channels + 2).to(tl.pointer_type(tl.uint32))
        )
        weight_0 = weight_pair_01.to(tl.uint16).to(tl.bfloat16, bitcast=True)
        weight_1 = (weight_pair_01 >> 16).to(tl.uint16).to(
            tl.bfloat16, bitcast=True
        )
        weight_2 = weight_pair_23.to(tl.uint16).to(tl.bfloat16, bitcast=True)
        weight_3 = (weight_pair_23 >> 16).to(tl.uint16).to(
            tl.bfloat16, bitcast=True
        )

        bias_value = tl.zeros((BLOCK_C,), dtype=tl.float32)
        if HAS_BIAS:
            bias_value = tl.load(bias + offs_c).to(tl.float32)

        # Exact load-once order keeps the peak 5 frontier while adjacent nodes
        # expose two independent activation chains before either store.
        x_18 = tl.load(x_batch + 18 * X_STRIDE_ROW + offs_c)
        x_23 = tl.load(x_batch + 23 * X_STRIDE_ROW + offs_c)
        x_25 = tl.load(x_batch + 25 * X_STRIDE_ROW + offs_c)
        x_27 = tl.load(x_batch + 27 * X_STRIDE_ROW + offs_c)
        product_0 = (x_18 * weight_0).to(tl.bfloat16).to(tl.float32)
        acc = bias_value + product_0
        product_1 = (x_23 * weight_1).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_1
        product_2 = (x_25 * weight_2).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_2
        product_3 = (x_27 * weight_3).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_3
        acc_27 = acc
        x_13 = tl.load(x_batch + 13 * X_STRIDE_ROW + offs_c)
        product_0 = (x_13 * weight_0).to(tl.bfloat16).to(tl.float32)
        acc = bias_value + product_0
        product_1 = (x_18 * weight_1).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_1
        product_2 = (x_23 * weight_2).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_2
        product_3 = (x_25 * weight_3).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_3
        activated_27 = acc_27 / (1.0 + tl.exp(0.0 - acc_27))
        activated_25 = acc / (1.0 + tl.exp(0.0 - acc))
        _fr13_store_fixed32_conv_outputs(
            query,
            key,
            value_spec,
            value_tree,
            conv_tap,
            activated_27,
            pid_b,
            27,
            offs_c,
            N,
            C,
            Q_DIM,
            V_DIM,
            STORE_CONV_TAP,
        )
        tl.store(stage_batch + ((WIDTH - 1) + 27) * C + offs_c, x_27)
        _fr13_store_fixed32_conv_outputs(
            query,
            key,
            value_spec,
            value_tree,
            conv_tap,
            activated_25,
            pid_b,
            25,
            offs_c,
            N,
            C,
            Q_DIM,
            V_DIM,
            STORE_CONV_TAP,
        )
        tl.store(stage_batch + ((WIDTH - 1) + 25) * C + offs_c, x_25)
        x_8 = tl.load(x_batch + 8 * X_STRIDE_ROW + offs_c)
        product_0 = (x_8 * weight_0).to(tl.bfloat16).to(tl.float32)
        acc = bias_value + product_0
        product_1 = (x_13 * weight_1).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_1
        product_2 = (x_18 * weight_2).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_2
        product_3 = (x_23 * weight_3).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_3
        acc_23 = acc
        x_3 = tl.load(x_batch + 3 * X_STRIDE_ROW + offs_c)
        product_0 = (x_3 * weight_0).to(tl.bfloat16).to(tl.float32)
        acc = bias_value + product_0
        product_1 = (x_8 * weight_1).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_1
        product_2 = (x_13 * weight_2).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_2
        product_3 = (x_18 * weight_3).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_3
        activated_23 = acc_23 / (1.0 + tl.exp(0.0 - acc_23))
        activated_18 = acc / (1.0 + tl.exp(0.0 - acc))
        _fr13_store_fixed32_conv_outputs(
            query,
            key,
            value_spec,
            value_tree,
            conv_tap,
            activated_23,
            pid_b,
            23,
            offs_c,
            N,
            C,
            Q_DIM,
            V_DIM,
            STORE_CONV_TAP,
        )
        tl.store(stage_batch + ((WIDTH - 1) + 23) * C + offs_c, x_23)
        _fr13_store_fixed32_conv_outputs(
            query,
            key,
            value_spec,
            value_tree,
            conv_tap,
            activated_18,
            pid_b,
            18,
            offs_c,
            N,
            C,
            Q_DIM,
            V_DIM,
            STORE_CONV_TAP,
        )
        tl.store(stage_batch + ((WIDTH - 1) + 18) * C + offs_c, x_18)
        x_0 = tl.load(x_batch + 0 * X_STRIDE_ROW + offs_c)
        product_0 = (x_0 * weight_0).to(tl.bfloat16).to(tl.float32)
        acc = bias_value + product_0
        product_1 = (x_3 * weight_1).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_1
        product_2 = (x_8 * weight_2).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_2
        product_3 = (x_13 * weight_3).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_3
        acc_13 = acc
        product_0 = (prior_2 * weight_0).to(tl.bfloat16).to(tl.float32)
        acc = bias_value + product_0
        product_1 = (x_0 * weight_1).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_1
        product_2 = (x_3 * weight_2).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_2
        product_3 = (x_8 * weight_3).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_3
        activated_13 = acc_13 / (1.0 + tl.exp(0.0 - acc_13))
        activated_8 = acc / (1.0 + tl.exp(0.0 - acc))
        _fr13_store_fixed32_conv_outputs(
            query,
            key,
            value_spec,
            value_tree,
            conv_tap,
            activated_13,
            pid_b,
            13,
            offs_c,
            N,
            C,
            Q_DIM,
            V_DIM,
            STORE_CONV_TAP,
        )
        tl.store(stage_batch + ((WIDTH - 1) + 13) * C + offs_c, x_13)
        _fr13_store_fixed32_conv_outputs(
            query,
            key,
            value_spec,
            value_tree,
            conv_tap,
            activated_8,
            pid_b,
            8,
            offs_c,
            N,
            C,
            Q_DIM,
            V_DIM,
            STORE_CONV_TAP,
        )
        tl.store(stage_batch + ((WIDTH - 1) + 8) * C + offs_c, x_8)
        product_0 = (prior_1 * weight_0).to(tl.bfloat16).to(tl.float32)
        acc = bias_value + product_0
        product_1 = (prior_2 * weight_1).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_1
        product_2 = (x_0 * weight_2).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_2
        product_3 = (x_3 * weight_3).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_3
        acc_3 = acc
        product_0 = (prior_0 * weight_0).to(tl.bfloat16).to(tl.float32)
        acc = bias_value + product_0
        product_1 = (prior_1 * weight_1).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_1
        product_2 = (prior_2 * weight_2).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_2
        product_3 = (x_0 * weight_3).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_3
        activated_3 = acc_3 / (1.0 + tl.exp(0.0 - acc_3))
        activated_0 = acc / (1.0 + tl.exp(0.0 - acc))
        _fr13_store_fixed32_conv_outputs(
            query,
            key,
            value_spec,
            value_tree,
            conv_tap,
            activated_3,
            pid_b,
            3,
            offs_c,
            N,
            C,
            Q_DIM,
            V_DIM,
            STORE_CONV_TAP,
        )
        tl.store(stage_batch + ((WIDTH - 1) + 3) * C + offs_c, x_3)
        _fr13_store_fixed32_conv_outputs(
            query,
            key,
            value_spec,
            value_tree,
            conv_tap,
            activated_0,
            pid_b,
            0,
            offs_c,
            N,
            C,
            Q_DIM,
            V_DIM,
            STORE_CONV_TAP,
        )
        tl.store(stage_batch + ((WIDTH - 1) + 0) * C + offs_c, x_0)
        x_2 = tl.load(x_batch + 2 * X_STRIDE_ROW + offs_c)
        product_0 = (prior_1 * weight_0).to(tl.bfloat16).to(tl.float32)
        acc = bias_value + product_0
        product_1 = (prior_2 * weight_1).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_1
        product_2 = (x_0 * weight_2).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_2
        product_3 = (x_2 * weight_3).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_3
        acc_2 = acc
        x_7 = tl.load(x_batch + 7 * X_STRIDE_ROW + offs_c)
        product_0 = (prior_2 * weight_0).to(tl.bfloat16).to(tl.float32)
        acc = bias_value + product_0
        product_1 = (x_0 * weight_1).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_1
        product_2 = (x_2 * weight_2).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_2
        product_3 = (x_7 * weight_3).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_3
        activated_2 = acc_2 / (1.0 + tl.exp(0.0 - acc_2))
        activated_7 = acc / (1.0 + tl.exp(0.0 - acc))
        _fr13_store_fixed32_conv_outputs(
            query,
            key,
            value_spec,
            value_tree,
            conv_tap,
            activated_2,
            pid_b,
            2,
            offs_c,
            N,
            C,
            Q_DIM,
            V_DIM,
            STORE_CONV_TAP,
        )
        tl.store(stage_batch + ((WIDTH - 1) + 2) * C + offs_c, x_2)
        _fr13_store_fixed32_conv_outputs(
            query,
            key,
            value_spec,
            value_tree,
            conv_tap,
            activated_7,
            pid_b,
            7,
            offs_c,
            N,
            C,
            Q_DIM,
            V_DIM,
            STORE_CONV_TAP,
        )
        tl.store(stage_batch + ((WIDTH - 1) + 7) * C + offs_c, x_7)
        x_12 = tl.load(x_batch + 12 * X_STRIDE_ROW + offs_c)
        product_0 = (x_0 * weight_0).to(tl.bfloat16).to(tl.float32)
        acc = bias_value + product_0
        product_1 = (x_2 * weight_1).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_1
        product_2 = (x_7 * weight_2).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_2
        product_3 = (x_12 * weight_3).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_3
        acc_12 = acc
        x_17 = tl.load(x_batch + 17 * X_STRIDE_ROW + offs_c)
        product_0 = (x_2 * weight_0).to(tl.bfloat16).to(tl.float32)
        acc = bias_value + product_0
        product_1 = (x_7 * weight_1).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_1
        product_2 = (x_12 * weight_2).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_2
        product_3 = (x_17 * weight_3).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_3
        activated_12 = acc_12 / (1.0 + tl.exp(0.0 - acc_12))
        activated_17 = acc / (1.0 + tl.exp(0.0 - acc))
        _fr13_store_fixed32_conv_outputs(
            query,
            key,
            value_spec,
            value_tree,
            conv_tap,
            activated_12,
            pid_b,
            12,
            offs_c,
            N,
            C,
            Q_DIM,
            V_DIM,
            STORE_CONV_TAP,
        )
        tl.store(stage_batch + ((WIDTH - 1) + 12) * C + offs_c, x_12)
        _fr13_store_fixed32_conv_outputs(
            query,
            key,
            value_spec,
            value_tree,
            conv_tap,
            activated_17,
            pid_b,
            17,
            offs_c,
            N,
            C,
            Q_DIM,
            V_DIM,
            STORE_CONV_TAP,
        )
        tl.store(stage_batch + ((WIDTH - 1) + 17) * C + offs_c, x_17)
        x_22 = tl.load(x_batch + 22 * X_STRIDE_ROW + offs_c)
        product_0 = (x_7 * weight_0).to(tl.bfloat16).to(tl.float32)
        acc = bias_value + product_0
        product_1 = (x_12 * weight_1).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_1
        product_2 = (x_17 * weight_2).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_2
        product_3 = (x_22 * weight_3).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_3
        acc_22 = acc
        x_1 = tl.load(x_batch + 1 * X_STRIDE_ROW + offs_c)
        product_0 = (prior_1 * weight_0).to(tl.bfloat16).to(tl.float32)
        acc = bias_value + product_0
        product_1 = (prior_2 * weight_1).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_1
        product_2 = (x_0 * weight_2).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_2
        product_3 = (x_1 * weight_3).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_3
        activated_22 = acc_22 / (1.0 + tl.exp(0.0 - acc_22))
        activated_1 = acc / (1.0 + tl.exp(0.0 - acc))
        _fr13_store_fixed32_conv_outputs(
            query,
            key,
            value_spec,
            value_tree,
            conv_tap,
            activated_22,
            pid_b,
            22,
            offs_c,
            N,
            C,
            Q_DIM,
            V_DIM,
            STORE_CONV_TAP,
        )
        tl.store(stage_batch + ((WIDTH - 1) + 22) * C + offs_c, x_22)
        _fr13_store_fixed32_conv_outputs(
            query,
            key,
            value_spec,
            value_tree,
            conv_tap,
            activated_1,
            pid_b,
            1,
            offs_c,
            N,
            C,
            Q_DIM,
            V_DIM,
            STORE_CONV_TAP,
        )
        tl.store(stage_batch + ((WIDTH - 1) + 1) * C + offs_c, x_1)
        x_5 = tl.load(x_batch + 5 * X_STRIDE_ROW + offs_c)
        product_0 = (prior_2 * weight_0).to(tl.bfloat16).to(tl.float32)
        acc = bias_value + product_0
        product_1 = (x_0 * weight_1).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_1
        product_2 = (x_1 * weight_2).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_2
        product_3 = (x_5 * weight_3).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_3
        acc_5 = acc
        x_6 = tl.load(x_batch + 6 * X_STRIDE_ROW + offs_c)
        product_0 = (prior_2 * weight_0).to(tl.bfloat16).to(tl.float32)
        acc = bias_value + product_0
        product_1 = (x_0 * weight_1).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_1
        product_2 = (x_1 * weight_2).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_2
        product_3 = (x_6 * weight_3).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_3
        activated_5 = acc_5 / (1.0 + tl.exp(0.0 - acc_5))
        activated_6 = acc / (1.0 + tl.exp(0.0 - acc))
        _fr13_store_fixed32_conv_outputs(
            query,
            key,
            value_spec,
            value_tree,
            conv_tap,
            activated_5,
            pid_b,
            5,
            offs_c,
            N,
            C,
            Q_DIM,
            V_DIM,
            STORE_CONV_TAP,
        )
        tl.store(stage_batch + ((WIDTH - 1) + 5) * C + offs_c, x_5)
        _fr13_store_fixed32_conv_outputs(
            query,
            key,
            value_spec,
            value_tree,
            conv_tap,
            activated_6,
            pid_b,
            6,
            offs_c,
            N,
            C,
            Q_DIM,
            V_DIM,
            STORE_CONV_TAP,
        )
        tl.store(stage_batch + ((WIDTH - 1) + 6) * C + offs_c, x_6)
        x_4 = tl.load(x_batch + 4 * X_STRIDE_ROW + offs_c)
        product_0 = (prior_2 * weight_0).to(tl.bfloat16).to(tl.float32)
        acc = bias_value + product_0
        product_1 = (x_0 * weight_1).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_1
        product_2 = (x_1 * weight_2).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_2
        product_3 = (x_4 * weight_3).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_3
        acc_4 = acc
        x_10 = tl.load(x_batch + 10 * X_STRIDE_ROW + offs_c)
        product_0 = (x_0 * weight_0).to(tl.bfloat16).to(tl.float32)
        acc = bias_value + product_0
        product_1 = (x_1 * weight_1).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_1
        product_2 = (x_4 * weight_2).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_2
        product_3 = (x_10 * weight_3).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_3
        activated_4 = acc_4 / (1.0 + tl.exp(0.0 - acc_4))
        activated_10 = acc / (1.0 + tl.exp(0.0 - acc))
        _fr13_store_fixed32_conv_outputs(
            query,
            key,
            value_spec,
            value_tree,
            conv_tap,
            activated_4,
            pid_b,
            4,
            offs_c,
            N,
            C,
            Q_DIM,
            V_DIM,
            STORE_CONV_TAP,
        )
        tl.store(stage_batch + ((WIDTH - 1) + 4) * C + offs_c, x_4)
        _fr13_store_fixed32_conv_outputs(
            query,
            key,
            value_spec,
            value_tree,
            conv_tap,
            activated_10,
            pid_b,
            10,
            offs_c,
            N,
            C,
            Q_DIM,
            V_DIM,
            STORE_CONV_TAP,
        )
        tl.store(stage_batch + ((WIDTH - 1) + 10) * C + offs_c, x_10)
        x_11 = tl.load(x_batch + 11 * X_STRIDE_ROW + offs_c)
        product_0 = (x_0 * weight_0).to(tl.bfloat16).to(tl.float32)
        acc = bias_value + product_0
        product_1 = (x_1 * weight_1).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_1
        product_2 = (x_4 * weight_2).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_2
        product_3 = (x_11 * weight_3).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_3
        acc_11 = acc
        x_9 = tl.load(x_batch + 9 * X_STRIDE_ROW + offs_c)
        product_0 = (x_0 * weight_0).to(tl.bfloat16).to(tl.float32)
        acc = bias_value + product_0
        product_1 = (x_1 * weight_1).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_1
        product_2 = (x_4 * weight_2).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_2
        product_3 = (x_9 * weight_3).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_3
        activated_11 = acc_11 / (1.0 + tl.exp(0.0 - acc_11))
        activated_9 = acc / (1.0 + tl.exp(0.0 - acc))
        _fr13_store_fixed32_conv_outputs(
            query,
            key,
            value_spec,
            value_tree,
            conv_tap,
            activated_11,
            pid_b,
            11,
            offs_c,
            N,
            C,
            Q_DIM,
            V_DIM,
            STORE_CONV_TAP,
        )
        tl.store(stage_batch + ((WIDTH - 1) + 11) * C + offs_c, x_11)
        _fr13_store_fixed32_conv_outputs(
            query,
            key,
            value_spec,
            value_tree,
            conv_tap,
            activated_9,
            pid_b,
            9,
            offs_c,
            N,
            C,
            Q_DIM,
            V_DIM,
            STORE_CONV_TAP,
        )
        tl.store(stage_batch + ((WIDTH - 1) + 9) * C + offs_c, x_9)
        x_15 = tl.load(x_batch + 15 * X_STRIDE_ROW + offs_c)
        product_0 = (x_1 * weight_0).to(tl.bfloat16).to(tl.float32)
        acc = bias_value + product_0
        product_1 = (x_4 * weight_1).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_1
        product_2 = (x_9 * weight_2).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_2
        product_3 = (x_15 * weight_3).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_3
        acc_15 = acc
        x_16 = tl.load(x_batch + 16 * X_STRIDE_ROW + offs_c)
        product_0 = (x_1 * weight_0).to(tl.bfloat16).to(tl.float32)
        acc = bias_value + product_0
        product_1 = (x_4 * weight_1).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_1
        product_2 = (x_9 * weight_2).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_2
        product_3 = (x_16 * weight_3).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_3
        activated_15 = acc_15 / (1.0 + tl.exp(0.0 - acc_15))
        activated_16 = acc / (1.0 + tl.exp(0.0 - acc))
        _fr13_store_fixed32_conv_outputs(
            query,
            key,
            value_spec,
            value_tree,
            conv_tap,
            activated_15,
            pid_b,
            15,
            offs_c,
            N,
            C,
            Q_DIM,
            V_DIM,
            STORE_CONV_TAP,
        )
        tl.store(stage_batch + ((WIDTH - 1) + 15) * C + offs_c, x_15)
        _fr13_store_fixed32_conv_outputs(
            query,
            key,
            value_spec,
            value_tree,
            conv_tap,
            activated_16,
            pid_b,
            16,
            offs_c,
            N,
            C,
            Q_DIM,
            V_DIM,
            STORE_CONV_TAP,
        )
        tl.store(stage_batch + ((WIDTH - 1) + 16) * C + offs_c, x_16)
        x_14 = tl.load(x_batch + 14 * X_STRIDE_ROW + offs_c)
        product_0 = (x_1 * weight_0).to(tl.bfloat16).to(tl.float32)
        acc = bias_value + product_0
        product_1 = (x_4 * weight_1).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_1
        product_2 = (x_9 * weight_2).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_2
        product_3 = (x_14 * weight_3).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_3
        acc_14 = acc
        x_20 = tl.load(x_batch + 20 * X_STRIDE_ROW + offs_c)
        product_0 = (x_4 * weight_0).to(tl.bfloat16).to(tl.float32)
        acc = bias_value + product_0
        product_1 = (x_9 * weight_1).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_1
        product_2 = (x_14 * weight_2).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_2
        product_3 = (x_20 * weight_3).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_3
        activated_14 = acc_14 / (1.0 + tl.exp(0.0 - acc_14))
        activated_20 = acc / (1.0 + tl.exp(0.0 - acc))
        _fr13_store_fixed32_conv_outputs(
            query,
            key,
            value_spec,
            value_tree,
            conv_tap,
            activated_14,
            pid_b,
            14,
            offs_c,
            N,
            C,
            Q_DIM,
            V_DIM,
            STORE_CONV_TAP,
        )
        tl.store(stage_batch + ((WIDTH - 1) + 14) * C + offs_c, x_14)
        _fr13_store_fixed32_conv_outputs(
            query,
            key,
            value_spec,
            value_tree,
            conv_tap,
            activated_20,
            pid_b,
            20,
            offs_c,
            N,
            C,
            Q_DIM,
            V_DIM,
            STORE_CONV_TAP,
        )
        tl.store(stage_batch + ((WIDTH - 1) + 20) * C + offs_c, x_20)
        x_21 = tl.load(x_batch + 21 * X_STRIDE_ROW + offs_c)
        product_0 = (x_4 * weight_0).to(tl.bfloat16).to(tl.float32)
        acc = bias_value + product_0
        product_1 = (x_9 * weight_1).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_1
        product_2 = (x_14 * weight_2).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_2
        product_3 = (x_21 * weight_3).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_3
        acc_21 = acc
        x_19 = tl.load(x_batch + 19 * X_STRIDE_ROW + offs_c)
        product_0 = (x_4 * weight_0).to(tl.bfloat16).to(tl.float32)
        acc = bias_value + product_0
        product_1 = (x_9 * weight_1).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_1
        product_2 = (x_14 * weight_2).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_2
        product_3 = (x_19 * weight_3).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_3
        activated_21 = acc_21 / (1.0 + tl.exp(0.0 - acc_21))
        activated_19 = acc / (1.0 + tl.exp(0.0 - acc))
        _fr13_store_fixed32_conv_outputs(
            query,
            key,
            value_spec,
            value_tree,
            conv_tap,
            activated_21,
            pid_b,
            21,
            offs_c,
            N,
            C,
            Q_DIM,
            V_DIM,
            STORE_CONV_TAP,
        )
        tl.store(stage_batch + ((WIDTH - 1) + 21) * C + offs_c, x_21)
        _fr13_store_fixed32_conv_outputs(
            query,
            key,
            value_spec,
            value_tree,
            conv_tap,
            activated_19,
            pid_b,
            19,
            offs_c,
            N,
            C,
            Q_DIM,
            V_DIM,
            STORE_CONV_TAP,
        )
        tl.store(stage_batch + ((WIDTH - 1) + 19) * C + offs_c, x_19)
        x_24 = tl.load(x_batch + 24 * X_STRIDE_ROW + offs_c)
        product_0 = (x_9 * weight_0).to(tl.bfloat16).to(tl.float32)
        acc = bias_value + product_0
        product_1 = (x_14 * weight_1).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_1
        product_2 = (x_19 * weight_2).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_2
        product_3 = (x_24 * weight_3).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_3
        acc_24 = acc
        x_26 = tl.load(x_batch + 26 * X_STRIDE_ROW + offs_c)
        product_0 = (x_14 * weight_0).to(tl.bfloat16).to(tl.float32)
        acc = bias_value + product_0
        product_1 = (x_19 * weight_1).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_1
        product_2 = (x_24 * weight_2).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_2
        product_3 = (x_26 * weight_3).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_3
        activated_24 = acc_24 / (1.0 + tl.exp(0.0 - acc_24))
        activated_26 = acc / (1.0 + tl.exp(0.0 - acc))
        _fr13_store_fixed32_conv_outputs(
            query,
            key,
            value_spec,
            value_tree,
            conv_tap,
            activated_24,
            pid_b,
            24,
            offs_c,
            N,
            C,
            Q_DIM,
            V_DIM,
            STORE_CONV_TAP,
        )
        tl.store(stage_batch + ((WIDTH - 1) + 24) * C + offs_c, x_24)
        _fr13_store_fixed32_conv_outputs(
            query,
            key,
            value_spec,
            value_tree,
            conv_tap,
            activated_26,
            pid_b,
            26,
            offs_c,
            N,
            C,
            Q_DIM,
            V_DIM,
            STORE_CONV_TAP,
        )
        tl.store(stage_batch + ((WIDTH - 1) + 26) * C + offs_c, x_26)
        x_28 = tl.load(x_batch + 28 * X_STRIDE_ROW + offs_c)
        product_0 = (x_19 * weight_0).to(tl.bfloat16).to(tl.float32)
        acc = bias_value + product_0
        product_1 = (x_24 * weight_1).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_1
        product_2 = (x_26 * weight_2).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_2
        product_3 = (x_28 * weight_3).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_3
        acc_28 = acc
        x_29 = tl.load(x_batch + 29 * X_STRIDE_ROW + offs_c)
        product_0 = (x_24 * weight_0).to(tl.bfloat16).to(tl.float32)
        acc = bias_value + product_0
        product_1 = (x_26 * weight_1).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_1
        product_2 = (x_28 * weight_2).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_2
        product_3 = (x_29 * weight_3).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_3
        activated_28 = acc_28 / (1.0 + tl.exp(0.0 - acc_28))
        activated_29 = acc / (1.0 + tl.exp(0.0 - acc))
        _fr13_store_fixed32_conv_outputs(
            query,
            key,
            value_spec,
            value_tree,
            conv_tap,
            activated_28,
            pid_b,
            28,
            offs_c,
            N,
            C,
            Q_DIM,
            V_DIM,
            STORE_CONV_TAP,
        )
        tl.store(stage_batch + ((WIDTH - 1) + 28) * C + offs_c, x_28)
        _fr13_store_fixed32_conv_outputs(
            query,
            key,
            value_spec,
            value_tree,
            conv_tap,
            activated_29,
            pid_b,
            29,
            offs_c,
            N,
            C,
            Q_DIM,
            V_DIM,
            STORE_CONV_TAP,
        )
        tl.store(stage_batch + ((WIDTH - 1) + 29) * C + offs_c, x_29)
        x_30 = tl.load(x_batch + 30 * X_STRIDE_ROW + offs_c)
        product_0 = (x_26 * weight_0).to(tl.bfloat16).to(tl.float32)
        acc = bias_value + product_0
        product_1 = (x_28 * weight_1).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_1
        product_2 = (x_29 * weight_2).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_2
        product_3 = (x_30 * weight_3).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_3
        acc_30 = acc
        x_31 = tl.load(x_batch + 31 * X_STRIDE_ROW + offs_c)
        product_0 = (x_28 * weight_0).to(tl.bfloat16).to(tl.float32)
        acc = bias_value + product_0
        product_1 = (x_29 * weight_1).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_1
        product_2 = (x_30 * weight_2).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_2
        product_3 = (x_31 * weight_3).to(tl.bfloat16).to(tl.float32)
        acc = acc + product_3
        activated_30 = acc_30 / (1.0 + tl.exp(0.0 - acc_30))
        activated_31 = acc / (1.0 + tl.exp(0.0 - acc))
        _fr13_store_fixed32_conv_outputs(
            query,
            key,
            value_spec,
            value_tree,
            conv_tap,
            activated_30,
            pid_b,
            30,
            offs_c,
            N,
            C,
            Q_DIM,
            V_DIM,
            STORE_CONV_TAP,
        )
        tl.store(stage_batch + ((WIDTH - 1) + 30) * C + offs_c, x_30)
        _fr13_store_fixed32_conv_outputs(
            query,
            key,
            value_spec,
            value_tree,
            conv_tap,
            activated_31,
            pid_b,
            31,
            offs_c,
            N,
            C,
            Q_DIM,
            V_DIM,
            STORE_CONV_TAP,
        )
        tl.store(stage_batch + ((WIDTH - 1) + 31) * C + offs_c, x_31)
        tl.store(stage_batch + offs_c, prior_0)
        tl.store(stage_batch + C + offs_c, prior_1)
        tl.store(stage_batch + 2 * C + offs_c, prior_2)
        tl.store(stage_batch + (SOURCE_ROWS - 1) * C + offs_c, 0.0)
        # The 40-program schedule appends the four unchanged gate tiles to
        # its first four channel programs.
        if pid_task < GATE_TASKS:
            pid_n_base = pid_task * GATE_ROWS
            offs_n = pid_n_base + tl.arange(0, GATE_ROWS)[:, None]
            offs_h_1d = tl.arange(0, GATE_BLOCK)
            h_mask = offs_h_1d < HV
            offs_h = offs_h_1d[None, :]
            gate_mask = (offs_n < N) & (offs_h < HV)
            row = pid_b * N + offs_n
            if pid_n_base < N:
                a_value = tl.load(
                    a + row * HV + offs_h,
                    mask=gate_mask,
                    other=0.0,
                ).to(tl.float32)
                b_value = tl.load(
                    b + row * HV + offs_h,
                    mask=gate_mask,
                    other=0.0,
                ).to(tl.float32)
                A_log_value = tl.load(
                    A_log + offs_h_1d,
                    mask=h_mask,
                    other=0.0,
                ).to(tl.float32)[None, :]
                dt_bias_value = tl.load(
                    dt_bias + offs_h_1d,
                    mask=h_mask,
                    other=0.0,
                ).to(tl.float32)[None, :]
                gate_input = a_value + dt_bias_value
                softplus = tl.where(
                    gate_input > 0,
                    gate_input + tl.log(1.0 + tl.exp(-gate_input)),
                    tl.log(1.0 + tl.exp(gate_input)),
                )
                softplus = tl.where(
                    gate_input <= SOFTPLUS_THRESHOLD,
                    softplus,
                    gate_input,
                )
                g_value = -tl.exp(A_log_value) * softplus
                beta_value = tl.sigmoid(b_value)
                tl.store(g + row * HV + offs_h, g_value, mask=gate_mask)
                tl.store(beta + row * HV + offs_h, beta_value, mask=gate_mask)
    else:
        if pid_task < channel_tasks:
            pid_c = pid_task
            offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
            x_batch = x + pid_b * N * X_STRIDE_ROW
            stage_batch = source_stage + pid_b * SOURCE_ROWS * C

            bank_row_raw = tl.load(spec_state_indices + pid_b * N).to(tl.int64)
            bank_row_ok = (bank_row_raw >= 0) & (bank_row_raw < BANK_ROWS)
            bank_row = tl.maximum(0, tl.minimum(bank_row_raw, BANK_ROWS - 1))
            if CAPTURE_GUARD:
                # Valid replays perform no store. The first invalid replay makes the
                # committer's existing async assertion fail permanently, while the
                # clamped row prevents an out-of-bounds read before that assertion.
                tl.atomic_xchg(sticky_guard_ok, 0, mask=~bank_row_ok)
            prior_base = conv_state + bank_row * CONV_STRIDE_ROW + offs_c
            prior_0 = tl.load(prior_base)
            prior_1 = tl.load(prior_base + C)
            prior_2 = tl.load(prior_base + 2 * C)

            weight_channels = conv_weights + offs_c * WIDTH
            weight_pair_01 = tl.load(weight_channels.to(tl.pointer_type(tl.uint32)))
            weight_pair_23 = tl.load(
                (weight_channels + 2).to(tl.pointer_type(tl.uint32))
            )
            weight_0 = weight_pair_01.to(tl.uint16).to(tl.bfloat16, bitcast=True)
            weight_1 = (weight_pair_01 >> 16).to(tl.uint16).to(
                tl.bfloat16, bitcast=True
            )
            weight_2 = weight_pair_23.to(tl.uint16).to(tl.bfloat16, bitcast=True)
            weight_3 = (weight_pair_23 >> 16).to(tl.uint16).to(
                tl.bfloat16, bitcast=True
            )

            bias_value = tl.zeros((BLOCK_C,), dtype=tl.float32)
            if HAS_BIAS:
                bias_value = tl.load(bias + offs_c).to(tl.float32)

            # Exact load-once order keeps the peak 5 frontier while adjacent nodes
            # expose two independent activation chains before either store.
            x_18 = tl.load(x_batch + 18 * X_STRIDE_ROW + offs_c)
            x_23 = tl.load(x_batch + 23 * X_STRIDE_ROW + offs_c)
            x_25 = tl.load(x_batch + 25 * X_STRIDE_ROW + offs_c)
            x_27 = tl.load(x_batch + 27 * X_STRIDE_ROW + offs_c)
            product_0 = (x_18 * weight_0).to(tl.bfloat16).to(tl.float32)
            acc = bias_value + product_0
            product_1 = (x_23 * weight_1).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_1
            product_2 = (x_25 * weight_2).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_2
            product_3 = (x_27 * weight_3).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_3
            acc_27 = acc
            x_13 = tl.load(x_batch + 13 * X_STRIDE_ROW + offs_c)
            product_0 = (x_13 * weight_0).to(tl.bfloat16).to(tl.float32)
            acc = bias_value + product_0
            product_1 = (x_18 * weight_1).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_1
            product_2 = (x_23 * weight_2).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_2
            product_3 = (x_25 * weight_3).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_3
            activated_27 = acc_27 / (1.0 + tl.exp(0.0 - acc_27))
            activated_25 = acc / (1.0 + tl.exp(0.0 - acc))
            _fr13_store_fixed32_conv_outputs(
                query,
                key,
                value_spec,
                value_tree,
                conv_tap,
                activated_27,
                pid_b,
                27,
                offs_c,
                N,
                C,
                Q_DIM,
                V_DIM,
                STORE_CONV_TAP,
            )
            tl.store(stage_batch + ((WIDTH - 1) + 27) * C + offs_c, x_27)
            _fr13_store_fixed32_conv_outputs(
                query,
                key,
                value_spec,
                value_tree,
                conv_tap,
                activated_25,
                pid_b,
                25,
                offs_c,
                N,
                C,
                Q_DIM,
                V_DIM,
                STORE_CONV_TAP,
            )
            tl.store(stage_batch + ((WIDTH - 1) + 25) * C + offs_c, x_25)
            x_8 = tl.load(x_batch + 8 * X_STRIDE_ROW + offs_c)
            product_0 = (x_8 * weight_0).to(tl.bfloat16).to(tl.float32)
            acc = bias_value + product_0
            product_1 = (x_13 * weight_1).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_1
            product_2 = (x_18 * weight_2).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_2
            product_3 = (x_23 * weight_3).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_3
            acc_23 = acc
            x_3 = tl.load(x_batch + 3 * X_STRIDE_ROW + offs_c)
            product_0 = (x_3 * weight_0).to(tl.bfloat16).to(tl.float32)
            acc = bias_value + product_0
            product_1 = (x_8 * weight_1).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_1
            product_2 = (x_13 * weight_2).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_2
            product_3 = (x_18 * weight_3).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_3
            activated_23 = acc_23 / (1.0 + tl.exp(0.0 - acc_23))
            activated_18 = acc / (1.0 + tl.exp(0.0 - acc))
            _fr13_store_fixed32_conv_outputs(
                query,
                key,
                value_spec,
                value_tree,
                conv_tap,
                activated_23,
                pid_b,
                23,
                offs_c,
                N,
                C,
                Q_DIM,
                V_DIM,
                STORE_CONV_TAP,
            )
            tl.store(stage_batch + ((WIDTH - 1) + 23) * C + offs_c, x_23)
            _fr13_store_fixed32_conv_outputs(
                query,
                key,
                value_spec,
                value_tree,
                conv_tap,
                activated_18,
                pid_b,
                18,
                offs_c,
                N,
                C,
                Q_DIM,
                V_DIM,
                STORE_CONV_TAP,
            )
            tl.store(stage_batch + ((WIDTH - 1) + 18) * C + offs_c, x_18)
            x_0 = tl.load(x_batch + 0 * X_STRIDE_ROW + offs_c)
            product_0 = (x_0 * weight_0).to(tl.bfloat16).to(tl.float32)
            acc = bias_value + product_0
            product_1 = (x_3 * weight_1).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_1
            product_2 = (x_8 * weight_2).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_2
            product_3 = (x_13 * weight_3).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_3
            acc_13 = acc
            product_0 = (prior_2 * weight_0).to(tl.bfloat16).to(tl.float32)
            acc = bias_value + product_0
            product_1 = (x_0 * weight_1).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_1
            product_2 = (x_3 * weight_2).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_2
            product_3 = (x_8 * weight_3).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_3
            activated_13 = acc_13 / (1.0 + tl.exp(0.0 - acc_13))
            activated_8 = acc / (1.0 + tl.exp(0.0 - acc))
            _fr13_store_fixed32_conv_outputs(
                query,
                key,
                value_spec,
                value_tree,
                conv_tap,
                activated_13,
                pid_b,
                13,
                offs_c,
                N,
                C,
                Q_DIM,
                V_DIM,
                STORE_CONV_TAP,
            )
            tl.store(stage_batch + ((WIDTH - 1) + 13) * C + offs_c, x_13)
            _fr13_store_fixed32_conv_outputs(
                query,
                key,
                value_spec,
                value_tree,
                conv_tap,
                activated_8,
                pid_b,
                8,
                offs_c,
                N,
                C,
                Q_DIM,
                V_DIM,
                STORE_CONV_TAP,
            )
            tl.store(stage_batch + ((WIDTH - 1) + 8) * C + offs_c, x_8)
            product_0 = (prior_1 * weight_0).to(tl.bfloat16).to(tl.float32)
            acc = bias_value + product_0
            product_1 = (prior_2 * weight_1).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_1
            product_2 = (x_0 * weight_2).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_2
            product_3 = (x_3 * weight_3).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_3
            acc_3 = acc
            product_0 = (prior_0 * weight_0).to(tl.bfloat16).to(tl.float32)
            acc = bias_value + product_0
            product_1 = (prior_1 * weight_1).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_1
            product_2 = (prior_2 * weight_2).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_2
            product_3 = (x_0 * weight_3).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_3
            activated_3 = acc_3 / (1.0 + tl.exp(0.0 - acc_3))
            activated_0 = acc / (1.0 + tl.exp(0.0 - acc))
            _fr13_store_fixed32_conv_outputs(
                query,
                key,
                value_spec,
                value_tree,
                conv_tap,
                activated_3,
                pid_b,
                3,
                offs_c,
                N,
                C,
                Q_DIM,
                V_DIM,
                STORE_CONV_TAP,
            )
            tl.store(stage_batch + ((WIDTH - 1) + 3) * C + offs_c, x_3)
            _fr13_store_fixed32_conv_outputs(
                query,
                key,
                value_spec,
                value_tree,
                conv_tap,
                activated_0,
                pid_b,
                0,
                offs_c,
                N,
                C,
                Q_DIM,
                V_DIM,
                STORE_CONV_TAP,
            )
            tl.store(stage_batch + ((WIDTH - 1) + 0) * C + offs_c, x_0)
            x_2 = tl.load(x_batch + 2 * X_STRIDE_ROW + offs_c)
            product_0 = (prior_1 * weight_0).to(tl.bfloat16).to(tl.float32)
            acc = bias_value + product_0
            product_1 = (prior_2 * weight_1).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_1
            product_2 = (x_0 * weight_2).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_2
            product_3 = (x_2 * weight_3).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_3
            acc_2 = acc
            x_7 = tl.load(x_batch + 7 * X_STRIDE_ROW + offs_c)
            product_0 = (prior_2 * weight_0).to(tl.bfloat16).to(tl.float32)
            acc = bias_value + product_0
            product_1 = (x_0 * weight_1).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_1
            product_2 = (x_2 * weight_2).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_2
            product_3 = (x_7 * weight_3).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_3
            activated_2 = acc_2 / (1.0 + tl.exp(0.0 - acc_2))
            activated_7 = acc / (1.0 + tl.exp(0.0 - acc))
            _fr13_store_fixed32_conv_outputs(
                query,
                key,
                value_spec,
                value_tree,
                conv_tap,
                activated_2,
                pid_b,
                2,
                offs_c,
                N,
                C,
                Q_DIM,
                V_DIM,
                STORE_CONV_TAP,
            )
            tl.store(stage_batch + ((WIDTH - 1) + 2) * C + offs_c, x_2)
            _fr13_store_fixed32_conv_outputs(
                query,
                key,
                value_spec,
                value_tree,
                conv_tap,
                activated_7,
                pid_b,
                7,
                offs_c,
                N,
                C,
                Q_DIM,
                V_DIM,
                STORE_CONV_TAP,
            )
            tl.store(stage_batch + ((WIDTH - 1) + 7) * C + offs_c, x_7)
            x_12 = tl.load(x_batch + 12 * X_STRIDE_ROW + offs_c)
            product_0 = (x_0 * weight_0).to(tl.bfloat16).to(tl.float32)
            acc = bias_value + product_0
            product_1 = (x_2 * weight_1).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_1
            product_2 = (x_7 * weight_2).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_2
            product_3 = (x_12 * weight_3).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_3
            acc_12 = acc
            x_17 = tl.load(x_batch + 17 * X_STRIDE_ROW + offs_c)
            product_0 = (x_2 * weight_0).to(tl.bfloat16).to(tl.float32)
            acc = bias_value + product_0
            product_1 = (x_7 * weight_1).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_1
            product_2 = (x_12 * weight_2).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_2
            product_3 = (x_17 * weight_3).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_3
            activated_12 = acc_12 / (1.0 + tl.exp(0.0 - acc_12))
            activated_17 = acc / (1.0 + tl.exp(0.0 - acc))
            _fr13_store_fixed32_conv_outputs(
                query,
                key,
                value_spec,
                value_tree,
                conv_tap,
                activated_12,
                pid_b,
                12,
                offs_c,
                N,
                C,
                Q_DIM,
                V_DIM,
                STORE_CONV_TAP,
            )
            tl.store(stage_batch + ((WIDTH - 1) + 12) * C + offs_c, x_12)
            _fr13_store_fixed32_conv_outputs(
                query,
                key,
                value_spec,
                value_tree,
                conv_tap,
                activated_17,
                pid_b,
                17,
                offs_c,
                N,
                C,
                Q_DIM,
                V_DIM,
                STORE_CONV_TAP,
            )
            tl.store(stage_batch + ((WIDTH - 1) + 17) * C + offs_c, x_17)
            x_22 = tl.load(x_batch + 22 * X_STRIDE_ROW + offs_c)
            product_0 = (x_7 * weight_0).to(tl.bfloat16).to(tl.float32)
            acc = bias_value + product_0
            product_1 = (x_12 * weight_1).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_1
            product_2 = (x_17 * weight_2).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_2
            product_3 = (x_22 * weight_3).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_3
            acc_22 = acc
            x_1 = tl.load(x_batch + 1 * X_STRIDE_ROW + offs_c)
            product_0 = (prior_1 * weight_0).to(tl.bfloat16).to(tl.float32)
            acc = bias_value + product_0
            product_1 = (prior_2 * weight_1).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_1
            product_2 = (x_0 * weight_2).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_2
            product_3 = (x_1 * weight_3).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_3
            activated_22 = acc_22 / (1.0 + tl.exp(0.0 - acc_22))
            activated_1 = acc / (1.0 + tl.exp(0.0 - acc))
            _fr13_store_fixed32_conv_outputs(
                query,
                key,
                value_spec,
                value_tree,
                conv_tap,
                activated_22,
                pid_b,
                22,
                offs_c,
                N,
                C,
                Q_DIM,
                V_DIM,
                STORE_CONV_TAP,
            )
            tl.store(stage_batch + ((WIDTH - 1) + 22) * C + offs_c, x_22)
            _fr13_store_fixed32_conv_outputs(
                query,
                key,
                value_spec,
                value_tree,
                conv_tap,
                activated_1,
                pid_b,
                1,
                offs_c,
                N,
                C,
                Q_DIM,
                V_DIM,
                STORE_CONV_TAP,
            )
            tl.store(stage_batch + ((WIDTH - 1) + 1) * C + offs_c, x_1)
            x_5 = tl.load(x_batch + 5 * X_STRIDE_ROW + offs_c)
            product_0 = (prior_2 * weight_0).to(tl.bfloat16).to(tl.float32)
            acc = bias_value + product_0
            product_1 = (x_0 * weight_1).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_1
            product_2 = (x_1 * weight_2).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_2
            product_3 = (x_5 * weight_3).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_3
            acc_5 = acc
            x_6 = tl.load(x_batch + 6 * X_STRIDE_ROW + offs_c)
            product_0 = (prior_2 * weight_0).to(tl.bfloat16).to(tl.float32)
            acc = bias_value + product_0
            product_1 = (x_0 * weight_1).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_1
            product_2 = (x_1 * weight_2).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_2
            product_3 = (x_6 * weight_3).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_3
            activated_5 = acc_5 / (1.0 + tl.exp(0.0 - acc_5))
            activated_6 = acc / (1.0 + tl.exp(0.0 - acc))
            _fr13_store_fixed32_conv_outputs(
                query,
                key,
                value_spec,
                value_tree,
                conv_tap,
                activated_5,
                pid_b,
                5,
                offs_c,
                N,
                C,
                Q_DIM,
                V_DIM,
                STORE_CONV_TAP,
            )
            tl.store(stage_batch + ((WIDTH - 1) + 5) * C + offs_c, x_5)
            _fr13_store_fixed32_conv_outputs(
                query,
                key,
                value_spec,
                value_tree,
                conv_tap,
                activated_6,
                pid_b,
                6,
                offs_c,
                N,
                C,
                Q_DIM,
                V_DIM,
                STORE_CONV_TAP,
            )
            tl.store(stage_batch + ((WIDTH - 1) + 6) * C + offs_c, x_6)
            x_4 = tl.load(x_batch + 4 * X_STRIDE_ROW + offs_c)
            product_0 = (prior_2 * weight_0).to(tl.bfloat16).to(tl.float32)
            acc = bias_value + product_0
            product_1 = (x_0 * weight_1).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_1
            product_2 = (x_1 * weight_2).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_2
            product_3 = (x_4 * weight_3).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_3
            acc_4 = acc
            x_10 = tl.load(x_batch + 10 * X_STRIDE_ROW + offs_c)
            product_0 = (x_0 * weight_0).to(tl.bfloat16).to(tl.float32)
            acc = bias_value + product_0
            product_1 = (x_1 * weight_1).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_1
            product_2 = (x_4 * weight_2).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_2
            product_3 = (x_10 * weight_3).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_3
            activated_4 = acc_4 / (1.0 + tl.exp(0.0 - acc_4))
            activated_10 = acc / (1.0 + tl.exp(0.0 - acc))
            _fr13_store_fixed32_conv_outputs(
                query,
                key,
                value_spec,
                value_tree,
                conv_tap,
                activated_4,
                pid_b,
                4,
                offs_c,
                N,
                C,
                Q_DIM,
                V_DIM,
                STORE_CONV_TAP,
            )
            tl.store(stage_batch + ((WIDTH - 1) + 4) * C + offs_c, x_4)
            _fr13_store_fixed32_conv_outputs(
                query,
                key,
                value_spec,
                value_tree,
                conv_tap,
                activated_10,
                pid_b,
                10,
                offs_c,
                N,
                C,
                Q_DIM,
                V_DIM,
                STORE_CONV_TAP,
            )
            tl.store(stage_batch + ((WIDTH - 1) + 10) * C + offs_c, x_10)
            x_11 = tl.load(x_batch + 11 * X_STRIDE_ROW + offs_c)
            product_0 = (x_0 * weight_0).to(tl.bfloat16).to(tl.float32)
            acc = bias_value + product_0
            product_1 = (x_1 * weight_1).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_1
            product_2 = (x_4 * weight_2).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_2
            product_3 = (x_11 * weight_3).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_3
            acc_11 = acc
            x_9 = tl.load(x_batch + 9 * X_STRIDE_ROW + offs_c)
            product_0 = (x_0 * weight_0).to(tl.bfloat16).to(tl.float32)
            acc = bias_value + product_0
            product_1 = (x_1 * weight_1).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_1
            product_2 = (x_4 * weight_2).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_2
            product_3 = (x_9 * weight_3).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_3
            activated_11 = acc_11 / (1.0 + tl.exp(0.0 - acc_11))
            activated_9 = acc / (1.0 + tl.exp(0.0 - acc))
            _fr13_store_fixed32_conv_outputs(
                query,
                key,
                value_spec,
                value_tree,
                conv_tap,
                activated_11,
                pid_b,
                11,
                offs_c,
                N,
                C,
                Q_DIM,
                V_DIM,
                STORE_CONV_TAP,
            )
            tl.store(stage_batch + ((WIDTH - 1) + 11) * C + offs_c, x_11)
            _fr13_store_fixed32_conv_outputs(
                query,
                key,
                value_spec,
                value_tree,
                conv_tap,
                activated_9,
                pid_b,
                9,
                offs_c,
                N,
                C,
                Q_DIM,
                V_DIM,
                STORE_CONV_TAP,
            )
            tl.store(stage_batch + ((WIDTH - 1) + 9) * C + offs_c, x_9)
            x_15 = tl.load(x_batch + 15 * X_STRIDE_ROW + offs_c)
            product_0 = (x_1 * weight_0).to(tl.bfloat16).to(tl.float32)
            acc = bias_value + product_0
            product_1 = (x_4 * weight_1).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_1
            product_2 = (x_9 * weight_2).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_2
            product_3 = (x_15 * weight_3).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_3
            acc_15 = acc
            x_16 = tl.load(x_batch + 16 * X_STRIDE_ROW + offs_c)
            product_0 = (x_1 * weight_0).to(tl.bfloat16).to(tl.float32)
            acc = bias_value + product_0
            product_1 = (x_4 * weight_1).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_1
            product_2 = (x_9 * weight_2).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_2
            product_3 = (x_16 * weight_3).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_3
            activated_15 = acc_15 / (1.0 + tl.exp(0.0 - acc_15))
            activated_16 = acc / (1.0 + tl.exp(0.0 - acc))
            _fr13_store_fixed32_conv_outputs(
                query,
                key,
                value_spec,
                value_tree,
                conv_tap,
                activated_15,
                pid_b,
                15,
                offs_c,
                N,
                C,
                Q_DIM,
                V_DIM,
                STORE_CONV_TAP,
            )
            tl.store(stage_batch + ((WIDTH - 1) + 15) * C + offs_c, x_15)
            _fr13_store_fixed32_conv_outputs(
                query,
                key,
                value_spec,
                value_tree,
                conv_tap,
                activated_16,
                pid_b,
                16,
                offs_c,
                N,
                C,
                Q_DIM,
                V_DIM,
                STORE_CONV_TAP,
            )
            tl.store(stage_batch + ((WIDTH - 1) + 16) * C + offs_c, x_16)
            x_14 = tl.load(x_batch + 14 * X_STRIDE_ROW + offs_c)
            product_0 = (x_1 * weight_0).to(tl.bfloat16).to(tl.float32)
            acc = bias_value + product_0
            product_1 = (x_4 * weight_1).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_1
            product_2 = (x_9 * weight_2).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_2
            product_3 = (x_14 * weight_3).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_3
            acc_14 = acc
            x_20 = tl.load(x_batch + 20 * X_STRIDE_ROW + offs_c)
            product_0 = (x_4 * weight_0).to(tl.bfloat16).to(tl.float32)
            acc = bias_value + product_0
            product_1 = (x_9 * weight_1).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_1
            product_2 = (x_14 * weight_2).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_2
            product_3 = (x_20 * weight_3).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_3
            activated_14 = acc_14 / (1.0 + tl.exp(0.0 - acc_14))
            activated_20 = acc / (1.0 + tl.exp(0.0 - acc))
            _fr13_store_fixed32_conv_outputs(
                query,
                key,
                value_spec,
                value_tree,
                conv_tap,
                activated_14,
                pid_b,
                14,
                offs_c,
                N,
                C,
                Q_DIM,
                V_DIM,
                STORE_CONV_TAP,
            )
            tl.store(stage_batch + ((WIDTH - 1) + 14) * C + offs_c, x_14)
            _fr13_store_fixed32_conv_outputs(
                query,
                key,
                value_spec,
                value_tree,
                conv_tap,
                activated_20,
                pid_b,
                20,
                offs_c,
                N,
                C,
                Q_DIM,
                V_DIM,
                STORE_CONV_TAP,
            )
            tl.store(stage_batch + ((WIDTH - 1) + 20) * C + offs_c, x_20)
            x_21 = tl.load(x_batch + 21 * X_STRIDE_ROW + offs_c)
            product_0 = (x_4 * weight_0).to(tl.bfloat16).to(tl.float32)
            acc = bias_value + product_0
            product_1 = (x_9 * weight_1).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_1
            product_2 = (x_14 * weight_2).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_2
            product_3 = (x_21 * weight_3).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_3
            acc_21 = acc
            x_19 = tl.load(x_batch + 19 * X_STRIDE_ROW + offs_c)
            product_0 = (x_4 * weight_0).to(tl.bfloat16).to(tl.float32)
            acc = bias_value + product_0
            product_1 = (x_9 * weight_1).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_1
            product_2 = (x_14 * weight_2).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_2
            product_3 = (x_19 * weight_3).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_3
            activated_21 = acc_21 / (1.0 + tl.exp(0.0 - acc_21))
            activated_19 = acc / (1.0 + tl.exp(0.0 - acc))
            _fr13_store_fixed32_conv_outputs(
                query,
                key,
                value_spec,
                value_tree,
                conv_tap,
                activated_21,
                pid_b,
                21,
                offs_c,
                N,
                C,
                Q_DIM,
                V_DIM,
                STORE_CONV_TAP,
            )
            tl.store(stage_batch + ((WIDTH - 1) + 21) * C + offs_c, x_21)
            _fr13_store_fixed32_conv_outputs(
                query,
                key,
                value_spec,
                value_tree,
                conv_tap,
                activated_19,
                pid_b,
                19,
                offs_c,
                N,
                C,
                Q_DIM,
                V_DIM,
                STORE_CONV_TAP,
            )
            tl.store(stage_batch + ((WIDTH - 1) + 19) * C + offs_c, x_19)
            x_24 = tl.load(x_batch + 24 * X_STRIDE_ROW + offs_c)
            product_0 = (x_9 * weight_0).to(tl.bfloat16).to(tl.float32)
            acc = bias_value + product_0
            product_1 = (x_14 * weight_1).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_1
            product_2 = (x_19 * weight_2).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_2
            product_3 = (x_24 * weight_3).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_3
            acc_24 = acc
            x_26 = tl.load(x_batch + 26 * X_STRIDE_ROW + offs_c)
            product_0 = (x_14 * weight_0).to(tl.bfloat16).to(tl.float32)
            acc = bias_value + product_0
            product_1 = (x_19 * weight_1).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_1
            product_2 = (x_24 * weight_2).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_2
            product_3 = (x_26 * weight_3).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_3
            activated_24 = acc_24 / (1.0 + tl.exp(0.0 - acc_24))
            activated_26 = acc / (1.0 + tl.exp(0.0 - acc))
            _fr13_store_fixed32_conv_outputs(
                query,
                key,
                value_spec,
                value_tree,
                conv_tap,
                activated_24,
                pid_b,
                24,
                offs_c,
                N,
                C,
                Q_DIM,
                V_DIM,
                STORE_CONV_TAP,
            )
            tl.store(stage_batch + ((WIDTH - 1) + 24) * C + offs_c, x_24)
            _fr13_store_fixed32_conv_outputs(
                query,
                key,
                value_spec,
                value_tree,
                conv_tap,
                activated_26,
                pid_b,
                26,
                offs_c,
                N,
                C,
                Q_DIM,
                V_DIM,
                STORE_CONV_TAP,
            )
            tl.store(stage_batch + ((WIDTH - 1) + 26) * C + offs_c, x_26)
            x_28 = tl.load(x_batch + 28 * X_STRIDE_ROW + offs_c)
            product_0 = (x_19 * weight_0).to(tl.bfloat16).to(tl.float32)
            acc = bias_value + product_0
            product_1 = (x_24 * weight_1).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_1
            product_2 = (x_26 * weight_2).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_2
            product_3 = (x_28 * weight_3).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_3
            acc_28 = acc
            x_29 = tl.load(x_batch + 29 * X_STRIDE_ROW + offs_c)
            product_0 = (x_24 * weight_0).to(tl.bfloat16).to(tl.float32)
            acc = bias_value + product_0
            product_1 = (x_26 * weight_1).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_1
            product_2 = (x_28 * weight_2).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_2
            product_3 = (x_29 * weight_3).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_3
            activated_28 = acc_28 / (1.0 + tl.exp(0.0 - acc_28))
            activated_29 = acc / (1.0 + tl.exp(0.0 - acc))
            _fr13_store_fixed32_conv_outputs(
                query,
                key,
                value_spec,
                value_tree,
                conv_tap,
                activated_28,
                pid_b,
                28,
                offs_c,
                N,
                C,
                Q_DIM,
                V_DIM,
                STORE_CONV_TAP,
            )
            tl.store(stage_batch + ((WIDTH - 1) + 28) * C + offs_c, x_28)
            _fr13_store_fixed32_conv_outputs(
                query,
                key,
                value_spec,
                value_tree,
                conv_tap,
                activated_29,
                pid_b,
                29,
                offs_c,
                N,
                C,
                Q_DIM,
                V_DIM,
                STORE_CONV_TAP,
            )
            tl.store(stage_batch + ((WIDTH - 1) + 29) * C + offs_c, x_29)
            x_30 = tl.load(x_batch + 30 * X_STRIDE_ROW + offs_c)
            product_0 = (x_26 * weight_0).to(tl.bfloat16).to(tl.float32)
            acc = bias_value + product_0
            product_1 = (x_28 * weight_1).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_1
            product_2 = (x_29 * weight_2).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_2
            product_3 = (x_30 * weight_3).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_3
            acc_30 = acc
            x_31 = tl.load(x_batch + 31 * X_STRIDE_ROW + offs_c)
            product_0 = (x_28 * weight_0).to(tl.bfloat16).to(tl.float32)
            acc = bias_value + product_0
            product_1 = (x_29 * weight_1).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_1
            product_2 = (x_30 * weight_2).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_2
            product_3 = (x_31 * weight_3).to(tl.bfloat16).to(tl.float32)
            acc = acc + product_3
            activated_30 = acc_30 / (1.0 + tl.exp(0.0 - acc_30))
            activated_31 = acc / (1.0 + tl.exp(0.0 - acc))
            _fr13_store_fixed32_conv_outputs(
                query,
                key,
                value_spec,
                value_tree,
                conv_tap,
                activated_30,
                pid_b,
                30,
                offs_c,
                N,
                C,
                Q_DIM,
                V_DIM,
                STORE_CONV_TAP,
            )
            tl.store(stage_batch + ((WIDTH - 1) + 30) * C + offs_c, x_30)
            _fr13_store_fixed32_conv_outputs(
                query,
                key,
                value_spec,
                value_tree,
                conv_tap,
                activated_31,
                pid_b,
                31,
                offs_c,
                N,
                C,
                Q_DIM,
                V_DIM,
                STORE_CONV_TAP,
            )
            tl.store(stage_batch + ((WIDTH - 1) + 31) * C + offs_c, x_31)
            tl.store(stage_batch + offs_c, prior_0)
            tl.store(stage_batch + C + offs_c, prior_1)
            tl.store(stage_batch + 2 * C + offs_c, prior_2)
            tl.store(stage_batch + (SOURCE_ROWS - 1) * C + offs_c, 0.0)
        else:
            pid_n_base = (pid_task - channel_tasks) * GATE_ROWS
            offs_n = pid_n_base + tl.arange(0, GATE_ROWS)[:, None]
            offs_h_1d = tl.arange(0, GATE_BLOCK)
            h_mask = offs_h_1d < HV
            offs_h = offs_h_1d[None, :]
            gate_mask = (offs_n < N) & (offs_h < HV)
            row = pid_b * N + offs_n
            if pid_n_base < N:
                a_value = tl.load(
                    a + row * HV + offs_h,
                    mask=gate_mask,
                    other=0.0,
                ).to(tl.float32)
                b_value = tl.load(
                    b + row * HV + offs_h,
                    mask=gate_mask,
                    other=0.0,
                ).to(tl.float32)
                A_log_value = tl.load(
                    A_log + offs_h_1d,
                    mask=h_mask,
                    other=0.0,
                ).to(tl.float32)[None, :]
                dt_bias_value = tl.load(
                    dt_bias + offs_h_1d,
                    mask=h_mask,
                    other=0.0,
                ).to(tl.float32)[None, :]
                gate_input = a_value + dt_bias_value
                softplus = tl.where(
                    gate_input > 0,
                    gate_input + tl.log(1.0 + tl.exp(-gate_input)),
                    tl.log(1.0 + tl.exp(gate_input)),
                )
                softplus = tl.where(
                    gate_input <= SOFTPLUS_THRESHOLD,
                    softplus,
                    gate_input,
                )
                g_value = -tl.exp(A_log_value) * softplus
                beta_value = tl.sigmoid(b_value)
                tl.store(g + row * HV + offs_h, g_value, mask=gate_mask)
                tl.store(beta + row * HV + offs_h, beta_value, mask=gate_mask)
