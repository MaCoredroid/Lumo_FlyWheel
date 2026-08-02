"""Offline-only fixed32 SFWD prior-reuse kernel with derived topology."""

from __future__ import annotations

import triton
import triton.language as tl


FIXED32_PARENT = (
    -1,
    0,
    0,
    0,
    1,
    1,
    1,
    2,
    3,
    4,
    4,
    4,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    14,
    14,
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

FIXED32_ROWS = 32
CHANNELS = 10240
SOURCE_ROWS = 36
CONV_WIDTH = 4
X_ROW_STRIDE = 16384
SIGNED_INT32_MAX = (1 << 31) - 1


def fixed32_i32_address_contract(
    batch_size: int,
    *,
    x_stride_row: int,
) -> dict[str, int]:
    """Prove the fixed padded/dense offsets remain signed-int32-safe."""
    batch = int(batch_size)
    stride = int(x_stride_row)
    if batch not in (1, 2, 3, 4):
        raise ValueError("descriptorless SFWD addressing requires B1-B4")
    if stride != X_ROW_STRIDE:
        raise ValueError("descriptorless SFWD requires the fixed padded x stride")
    maxima = {
        "x": (batch * FIXED32_ROWS - 1) * stride + CHANNELS - 1,
        "out": batch * FIXED32_ROWS * CHANNELS - 1,
        "source_stage": batch * SOURCE_ROWS * CHANNELS - 1,
    }
    if max(maxima.values()) > SIGNED_INT32_MAX:
        raise ValueError("descriptorless SFWD dense-buffer offset would overflow")
    return maxima


def fixed32_specialized_layout_contract(
    batch_size: int,
    *,
    x_shape: tuple[int, ...],
    x_stride: tuple[int, ...],
    out_shape: tuple[int, ...],
    out_stride: tuple[int, ...],
    source_stage_shape: tuple[int, ...],
    source_stage_stride: tuple[int, ...],
    conv_weights_shape: tuple[int, ...],
    conv_weights_stride: tuple[int, ...],
) -> dict[str, object]:
    """Validate every fixed layout term removed from the kernel signature."""
    batch = int(batch_size)
    if batch not in (1, 2, 3, 4):
        raise ValueError("descriptorless fixed-layout SFWD requires B1-B4")
    rows = batch * FIXED32_ROWS
    source_rows = batch * SOURCE_ROWS
    expected = {
        "x_shape": (rows, CHANNELS),
        "x_stride": (X_ROW_STRIDE, 1),
        "out_shape": (rows, CHANNELS),
        "out_stride": (CHANNELS, 1),
        "source_stage_shape": (source_rows, CHANNELS),
        "source_stage_stride": (CHANNELS, 1),
        "conv_weights_shape": (CHANNELS, CONV_WIDTH),
        "conv_weights_stride": (CONV_WIDTH, 1),
    }
    observed = {
        "x_shape": tuple(int(value) for value in x_shape),
        "x_stride": tuple(int(value) for value in x_stride),
        "out_shape": tuple(int(value) for value in out_shape),
        "out_stride": tuple(int(value) for value in out_stride),
        "source_stage_shape": tuple(int(value) for value in source_stage_shape),
        "source_stage_stride": tuple(int(value) for value in source_stage_stride),
        "conv_weights_shape": tuple(int(value) for value in conv_weights_shape),
        "conv_weights_stride": tuple(int(value) for value in conv_weights_stride),
    }
    failures = tuple(
        name for name, value in observed.items() if value != expected[name]
    )
    if failures:
        raise ValueError(
            "descriptorless fixed-layout SFWD operand drift: "
            + ",".join(failures)
        )
    return {
        "batch_size": batch,
        "layouts": expected,
        "maximum_offsets": fixed32_i32_address_contract(
            batch, x_stride_row=X_ROW_STRIDE
        ),
    }


def fixed32_derived_parent_q(node: int) -> int:
    """Derive parent(node) + 1 from the fixed topology without a table."""
    index = int(node)
    if index < 0 or index >= FIXED32_ROWS:
        raise ValueError("descriptorless SFWD node must be in [0, 32)")
    if index < 7:
        return (index + 2) // 3
    if index < 25:
        middle = index - 4
        penalty = max(((middle + 2) % 5) - 2, 0)
        return middle - penalty
    if index < 29:
        return index - 1
    return index


def fixed32_descriptorless_sources() -> tuple[tuple[int, int, int], ...]:
    """Return the three derived source rows for every fixed node."""
    rows: list[tuple[int, int, int]] = []
    for node in range(FIXED32_ROWS):
        q1 = fixed32_derived_parent_q(node)
        q2 = fixed32_derived_parent_q(max(q1 - 1, 0))
        q3 = fixed32_derived_parent_q(max(q2 - 1, 0))
        source_2 = q1 + 2
        source_1 = 1 if q1 == 0 else q2 + 2
        source_0 = 0 if q1 == 0 else 1 if q2 == 0 else q3 + 2
        rows.append((source_0, source_1, source_2))
    return tuple(rows)


@triton.jit
def _fr13_fixed32_sfwd_prior_reuse_descriptorless_kernel(
    x,
    conv_state,
    spec_state_indices,
    conv_weights,
    bias,
    out,
    source_stage,
    conv_stride_row,
    conv_stride_c,
    conv_stride_l,
    ssi_stride_b,
    ssi_stride_s,
    B: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    WIDTH: tl.constexpr,
    STATE_LEN: tl.constexpr,
    SOURCE_ROWS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    X_STRIDE_ROW: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Fuse fixed32 convolution and state motion with derived fixed topology."""
    pid_row_group = tl.program_id(0)
    pid_c = tl.program_id(1)
    row_groups = N // ROWS_PER_PROGRAM
    pid_b = pid_row_group // row_groups
    pid_n_group = pid_row_group - pid_b * row_groups
    pid_n_base = pid_n_group * ROWS_PER_PROGRAM
    offs_n = pid_n_base + tl.arange(0, ROWS_PER_PROGRAM)[:, None]
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
    x_batch = x + pid_b * N * X_STRIDE_ROW
    out_batch = out + pid_b * N * C
    weight_channels = conv_weights + offs_c * WIDTH

    bank_row = tl.load(
        spec_state_indices + pid_b * ssi_stride_b + 0 * ssi_stride_s
    ).to(tl.int64)
    stage_offset = pid_b * SOURCE_ROWS * C
    prior_0 = tl.load(
        conv_state
        + bank_row * conv_stride_row
        + offs_c * conv_stride_c
        + 0 * conv_stride_l
    )
    prior_1 = tl.load(
        conv_state
        + bank_row * conv_stride_row
        + offs_c * conv_stride_c
        + 1 * conv_stride_l
    )
    prior_2 = tl.load(
        conv_state
        + bank_row * conv_stride_row
        + offs_c * conv_stride_c
        + 2 * conv_stride_l
    )

    q1_small = (offs_n + 2) // 3
    q1_mid_raw = offs_n - 4
    q1_mid_penalty = tl.maximum(((q1_mid_raw + 2) % 5) - 2, 0)
    q1_mid = q1_mid_raw - q1_mid_penalty
    q1 = tl.where(
        offs_n < 7,
        q1_small,
        tl.where(
            offs_n < 25,
            q1_mid,
            tl.where(offs_n < 29, offs_n - 1, offs_n),
        ),
    )

    q2_index = tl.maximum(q1 - 1, 0)
    q2_small = (q2_index + 2) // 3
    q2_mid_raw = q2_index - 4
    q2_mid_penalty = tl.maximum(((q2_mid_raw + 2) % 5) - 2, 0)
    q2_mid = q2_mid_raw - q2_mid_penalty
    q2 = tl.where(
        q2_index < 7,
        q2_small,
        tl.where(
            q2_index < 25,
            q2_mid,
            tl.where(q2_index < 29, q2_index - 1, q2_index),
        ),
    )

    q3_index = tl.maximum(q2 - 1, 0)
    q3_small = (q3_index + 2) // 3
    q3_mid_raw = q3_index - 4
    q3_mid_penalty = tl.maximum(((q3_mid_raw + 2) % 5) - 2, 0)
    q3_mid = q3_mid_raw - q3_mid_penalty
    q3 = tl.where(
        q3_index < 7,
        q3_small,
        tl.where(
            q3_index < 25,
            q3_mid,
            tl.where(q3_index < 29, q3_index - 1, q3_index),
        ),
    )

    source_2 = q1 + 2
    source_1 = tl.where(q1 == 0, 1, q2 + 2)
    source_0 = tl.where(q1 == 0, 0, tl.where(q2 == 0, 1, q3 + 2))

    acc = tl.zeros((ROWS_PER_PROGRAM, BLOCK_C), dtype=tl.float32)
    if HAS_BIAS:
        acc = tl.load(bias + offs_c).to(tl.float32)
    for tap in tl.static_range(0, WIDTH - 1):
        source_row = tl.where(
            tap == 0,
            source_0,
            tl.where(tap == 1, source_1, source_2),
        )
        from_prior = source_row < (WIDTH - 1)
        prior_value = tl.where(
            source_row == 0,
            prior_0,
            tl.where(source_row == 1, prior_1, prior_2),
        )
        x_node = source_row - (WIDTH - 1)
        x_value = tl.load(
            x_batch + x_node * X_STRIDE_ROW + offs_c,
            mask=(~from_prior) & (x_node >= 0) & (x_node < N),
            other=0.0,
        )
        value = tl.where(from_prior, prior_value, x_value).to(tl.bfloat16)
        weight = tl.load(weight_channels + tap).to(tl.bfloat16)
        product = (value * weight).to(tl.bfloat16).to(tl.float32)
        acc = acc + product

    current_x = tl.load(
        x_batch + offs_n * X_STRIDE_ROW + offs_c
    )
    current_weight = tl.load(weight_channels + (WIDTH - 1)).to(tl.bfloat16)
    current_product = (
        current_x * current_weight
    ).to(tl.bfloat16).to(tl.float32)
    acc = acc + current_product

    activated = acc / (1.0 + tl.exp(0.0 - acc))
    tl.store(out_batch + offs_n * C + offs_c, activated)

    tl.store(
        source_stage
        + stage_offset
        + ((WIDTH - 1) + offs_n) * C
        + offs_c,
        current_x,
    )
    source_edge_writer = pid_n_base == 0
    tl.store(
        source_stage + stage_offset + offs_c,
        prior_0,
        mask=source_edge_writer,
    )
    tl.store(
        source_stage + stage_offset + C + offs_c,
        prior_1,
        mask=source_edge_writer,
    )
    tl.store(
        source_stage + stage_offset + 2 * C + offs_c,
        prior_2,
        mask=source_edge_writer,
    )
    tl.store(
        source_stage
        + stage_offset
        + (SOURCE_ROWS - 1) * C
        + offs_c,
        0.0,
        mask=source_edge_writer,
    )
