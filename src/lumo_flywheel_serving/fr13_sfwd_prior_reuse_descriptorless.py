"""Fixed32 SFWD prior-reuse kernel with derived topology and fixed bases."""

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


# Four 11-bit source-delta entries occupy each 64-bit word. The entry fields
# are d0[3:0], (d1 + 1)[7:4], and (d2 + 2)[10:8].
FIXED32_PACKED_SOURCE_DELTAS = (
    0x0222011100000000,
    0x0455044403330222,
    0x0688057704660466,
    0x059B048A048A0489,
    0x048C048C048C06AC,
    0x048C048C06AE059D,
    0x012601590159048C,
    0x0000000100120126,
)


def fixed32_packed_source_entry(node: int) -> tuple[int, int, int]:
    """Decode the three historical source rows from the packed constants."""
    index = int(node)
    if index < 0 or index >= FIXED32_ROWS:
        raise ValueError("packed SFWD node must be in [0, 32)")
    packed = FIXED32_PACKED_SOURCE_DELTAS[index >> 2]
    entry = (packed >> ((index & 3) << 4)) & 0x7FF
    delta_0 = entry & 0xF
    delta_1 = ((entry >> 4) & 0xF) - 1
    delta_2 = ((entry >> 8) & 0x7) - 2
    return index - delta_0, index - delta_1, index - delta_2


def fixed32_packed_sources() -> tuple[tuple[int, int, int], ...]:
    """Decode every fixed32 source triple for source-level verification."""
    return tuple(fixed32_packed_source_entry(node) for node in range(FIXED32_ROWS))


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


@triton.jit
def _fr13_fixed32_sfwd_prior_reuse_packed_kernel(
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
    """Fuse SFWD using a packed fixed32 topology decoder."""
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

    source_group = offs_n >> 2
    packed = tl.where(
        source_group == 0,
        0x0222011100000000,
        tl.where(
            source_group == 1,
            0x0455044403330222,
            tl.where(
                source_group == 2,
                0x0688057704660466,
                tl.where(
                    source_group == 3,
                    0x059B048A048A0489,
                    tl.where(
                        source_group == 4,
                        0x048C048C048C06AC,
                        tl.where(
                            source_group == 5,
                            0x048C048C06AE059D,
                            tl.where(
                                source_group == 6,
                                0x012601590159048C,
                                0x0000000100120126,
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    entry_shift = (offs_n & 3) << 4
    source_entry = ((packed >> entry_shift) & 0x7FF).to(tl.int32)
    source_0 = offs_n - (source_entry & 0xF)
    source_1 = offs_n - ((source_entry >> 4) & 0xF) + 1
    source_2 = offs_n - ((source_entry >> 8) & 0x7) + 2

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

    current_x = tl.load(x_batch + offs_n * X_STRIDE_ROW + offs_c)
    current_weight = tl.load(weight_channels + (WIDTH - 1)).to(tl.bfloat16)
    current_product = (current_x * current_weight).to(tl.bfloat16).to(tl.float32)
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


@triton.jit
def _fr13_fixed32_sfwd_prior_reuse_packed_xgather_kernel(
    x,
    conv_state,
    spec_state_indices,
    conv_weights,
    bias,
    out,
    source_stage,
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
    """Fuse SFWD and reuse one current-x tile for every convolution tap."""
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
        spec_state_indices + pid_b * N
    ).to(tl.int64)
    stage_offset = pid_b * SOURCE_ROWS * C
    prior_base = (
        conv_state
        + bank_row * C * STATE_LEN
        + offs_c * STATE_LEN
    )
    prior_pair = tl.load(prior_base.to(tl.pointer_type(tl.uint32)))
    prior_0 = prior_pair.to(tl.uint16).to(tl.bfloat16, bitcast=True)
    prior_1 = (prior_pair >> 16).to(tl.uint16).to(
        tl.bfloat16, bitcast=True
    )
    prior_2 = tl.load(prior_base + 2)

    source_group = offs_n >> 2
    packed = tl.where(
        source_group == 0,
        0x0222011100000000,
        tl.where(
            source_group == 1,
            0x0455044403330222,
            tl.where(
                source_group == 2,
                0x0688057704660466,
                tl.where(
                    source_group == 3,
                    0x059B048A048A0489,
                    tl.where(
                        source_group == 4,
                        0x048C048C048C06AC,
                        tl.where(
                            source_group == 5,
                            0x048C048C06AE059D,
                            tl.where(
                                source_group == 6,
                                0x012601590159048C,
                                0x0000000100120126,
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    entry_shift = (offs_n & 3) << 4
    source_entry = ((packed >> entry_shift) & 0x7FF).to(tl.int32)
    source_0 = offs_n - (source_entry & 0xF)
    source_1 = offs_n - ((source_entry >> 4) & 0xF) + 1
    source_2 = offs_n - ((source_entry >> 8) & 0x7) + 2

    current_x = tl.load(x_batch + offs_n * X_STRIDE_ROW + offs_c)
    acc = tl.zeros((ROWS_PER_PROGRAM, BLOCK_C), dtype=tl.float32)
    if HAS_BIAS:
        acc = tl.load(bias + offs_c).to(tl.float32)

    weight_quad = tl.load(weight_channels.to(tl.pointer_type(tl.uint64)))
    for tap in tl.static_range(0, WIDTH - 2):
        source_row = tl.where(
            tap == 0,
            source_0,
            source_1,
        )
        if tap == 0:
            from_prior = offs_n < 9
            prior_value = tl.where(
                offs_n == 0,
                prior_0,
                tl.where(offs_n < 4, prior_1, prior_2),
            )
        else:
            from_prior = offs_n < 4
            prior_value = tl.where(offs_n == 0, prior_1, prior_2)
        x_node = tl.maximum(source_row - (WIDTH - 1), 0)
        x_index = tl.broadcast_to(x_node, ROWS_PER_PROGRAM, BLOCK_C)
        x_value = tl.gather(current_x, x_index, axis=0)
        value = tl.where(from_prior, prior_value, x_value).to(tl.bfloat16)
        weight_bits = (weight_quad >> (tap << 4)).to(tl.uint16)
        weight = weight_bits.to(tl.bfloat16, bitcast=True)
        product = (value * weight).to(tl.bfloat16).to(tl.float32)
        acc = acc + product

    source_row = source_2
    from_prior = offs_n == 0
    x_node = tl.maximum(source_row - (WIDTH - 1), 0)
    x_index = tl.broadcast_to(x_node, ROWS_PER_PROGRAM, BLOCK_C)
    x_value = tl.gather(current_x, x_index, axis=0)
    value = tl.where(from_prior, prior_2, x_value).to(tl.bfloat16)
    weight_2_bits = (weight_quad >> 32).to(tl.uint16)
    weight_2 = weight_2_bits.to(tl.bfloat16, bitcast=True)
    product = (value * weight_2).to(tl.bfloat16).to(tl.float32)
    acc = acc + product

    current_index = tl.broadcast_to(
        offs_n - pid_n_base, ROWS_PER_PROGRAM, BLOCK_C
    )
    current_value = tl.gather(current_x, current_index, axis=0)
    current_weight_bits = (weight_quad >> 48).to(tl.uint16)
    current_weight = current_weight_bits.to(tl.bfloat16, bitcast=True)
    current_product = (current_value * current_weight).to(tl.bfloat16).to(
        tl.float32
    )
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


@triton.jit
def _fr13_fixed32_sfwd_channel_serial_kernel(
    x,
    conv_state,
    spec_state_indices,
    conv_weights,
    bias,
    out,
    source_stage,
    CONV_STRIDE_ROW: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    WIDTH: tl.constexpr,
    STATE_LEN: tl.constexpr,
    SOURCE_ROWS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    X_STRIDE_ROW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Keep every fixed32 row register-local to a coalesced channel lane."""
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    x_batch = x + pid_b * N * X_STRIDE_ROW
    out_batch = out + pid_b * N * C
    stage_batch = source_stage + pid_b * SOURCE_ROWS * C

    bank_row = tl.load(spec_state_indices + pid_b * N).to(tl.int64)
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

    # Load current rows at first use; nodes 17-24 reload their tap-0 ancestor.
    x_0 = tl.load(x_batch + 0 * X_STRIDE_ROW + offs_c)
    product_0 = (prior_0 * weight_0).to(tl.bfloat16).to(tl.float32)
    acc = bias_value + product_0
    product_1 = (prior_1 * weight_1).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_1
    product_2 = (prior_2 * weight_2).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_2
    product_3 = (x_0 * weight_3).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_3
    activated_0 = acc / (1.0 + tl.exp(0.0 - acc))
    tl.store(out_batch + 0 * C + offs_c, activated_0)
    tl.store(stage_batch + ((WIDTH - 1) + 0) * C + offs_c, x_0)
    x_1 = tl.load(x_batch + 1 * X_STRIDE_ROW + offs_c)
    product_0 = (prior_1 * weight_0).to(tl.bfloat16).to(tl.float32)
    acc = bias_value + product_0
    product_1 = (prior_2 * weight_1).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_1
    product_2 = (x_0 * weight_2).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_2
    product_3 = (x_1 * weight_3).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_3
    activated_1 = acc / (1.0 + tl.exp(0.0 - acc))
    tl.store(out_batch + 1 * C + offs_c, activated_1)
    tl.store(stage_batch + ((WIDTH - 1) + 1) * C + offs_c, x_1)
    x_2 = tl.load(x_batch + 2 * X_STRIDE_ROW + offs_c)
    product_0 = (prior_1 * weight_0).to(tl.bfloat16).to(tl.float32)
    acc = bias_value + product_0
    product_1 = (prior_2 * weight_1).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_1
    product_2 = (x_0 * weight_2).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_2
    product_3 = (x_2 * weight_3).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_3
    activated_2 = acc / (1.0 + tl.exp(0.0 - acc))
    tl.store(out_batch + 2 * C + offs_c, activated_2)
    tl.store(stage_batch + ((WIDTH - 1) + 2) * C + offs_c, x_2)
    x_3 = tl.load(x_batch + 3 * X_STRIDE_ROW + offs_c)
    product_0 = (prior_1 * weight_0).to(tl.bfloat16).to(tl.float32)
    acc = bias_value + product_0
    product_1 = (prior_2 * weight_1).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_1
    product_2 = (x_0 * weight_2).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_2
    product_3 = (x_3 * weight_3).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_3
    activated_3 = acc / (1.0 + tl.exp(0.0 - acc))
    tl.store(out_batch + 3 * C + offs_c, activated_3)
    tl.store(stage_batch + ((WIDTH - 1) + 3) * C + offs_c, x_3)
    x_4 = tl.load(x_batch + 4 * X_STRIDE_ROW + offs_c)
    product_0 = (prior_2 * weight_0).to(tl.bfloat16).to(tl.float32)
    acc = bias_value + product_0
    product_1 = (x_0 * weight_1).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_1
    product_2 = (x_1 * weight_2).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_2
    product_3 = (x_4 * weight_3).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_3
    activated_4 = acc / (1.0 + tl.exp(0.0 - acc))
    tl.store(out_batch + 4 * C + offs_c, activated_4)
    tl.store(stage_batch + ((WIDTH - 1) + 4) * C + offs_c, x_4)
    x_5 = tl.load(x_batch + 5 * X_STRIDE_ROW + offs_c)
    product_0 = (prior_2 * weight_0).to(tl.bfloat16).to(tl.float32)
    acc = bias_value + product_0
    product_1 = (x_0 * weight_1).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_1
    product_2 = (x_1 * weight_2).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_2
    product_3 = (x_5 * weight_3).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_3
    activated_5 = acc / (1.0 + tl.exp(0.0 - acc))
    tl.store(out_batch + 5 * C + offs_c, activated_5)
    tl.store(stage_batch + ((WIDTH - 1) + 5) * C + offs_c, x_5)
    x_6 = tl.load(x_batch + 6 * X_STRIDE_ROW + offs_c)
    product_0 = (prior_2 * weight_0).to(tl.bfloat16).to(tl.float32)
    acc = bias_value + product_0
    product_1 = (x_0 * weight_1).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_1
    product_2 = (x_1 * weight_2).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_2
    product_3 = (x_6 * weight_3).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_3
    activated_6 = acc / (1.0 + tl.exp(0.0 - acc))
    tl.store(out_batch + 6 * C + offs_c, activated_6)
    tl.store(stage_batch + ((WIDTH - 1) + 6) * C + offs_c, x_6)
    x_7 = tl.load(x_batch + 7 * X_STRIDE_ROW + offs_c)
    product_0 = (prior_2 * weight_0).to(tl.bfloat16).to(tl.float32)
    acc = bias_value + product_0
    product_1 = (x_0 * weight_1).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_1
    product_2 = (x_2 * weight_2).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_2
    product_3 = (x_7 * weight_3).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_3
    activated_7 = acc / (1.0 + tl.exp(0.0 - acc))
    tl.store(out_batch + 7 * C + offs_c, activated_7)
    tl.store(stage_batch + ((WIDTH - 1) + 7) * C + offs_c, x_7)
    x_8 = tl.load(x_batch + 8 * X_STRIDE_ROW + offs_c)
    product_0 = (prior_2 * weight_0).to(tl.bfloat16).to(tl.float32)
    acc = bias_value + product_0
    product_1 = (x_0 * weight_1).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_1
    product_2 = (x_3 * weight_2).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_2
    product_3 = (x_8 * weight_3).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_3
    activated_8 = acc / (1.0 + tl.exp(0.0 - acc))
    tl.store(out_batch + 8 * C + offs_c, activated_8)
    tl.store(stage_batch + ((WIDTH - 1) + 8) * C + offs_c, x_8)
    x_9 = tl.load(x_batch + 9 * X_STRIDE_ROW + offs_c)
    product_0 = (x_0 * weight_0).to(tl.bfloat16).to(tl.float32)
    acc = bias_value + product_0
    product_1 = (x_1 * weight_1).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_1
    product_2 = (x_4 * weight_2).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_2
    product_3 = (x_9 * weight_3).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_3
    activated_9 = acc / (1.0 + tl.exp(0.0 - acc))
    tl.store(out_batch + 9 * C + offs_c, activated_9)
    tl.store(stage_batch + ((WIDTH - 1) + 9) * C + offs_c, x_9)
    x_10 = tl.load(x_batch + 10 * X_STRIDE_ROW + offs_c)
    product_0 = (x_0 * weight_0).to(tl.bfloat16).to(tl.float32)
    acc = bias_value + product_0
    product_1 = (x_1 * weight_1).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_1
    product_2 = (x_4 * weight_2).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_2
    product_3 = (x_10 * weight_3).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_3
    activated_10 = acc / (1.0 + tl.exp(0.0 - acc))
    tl.store(out_batch + 10 * C + offs_c, activated_10)
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
    activated_11 = acc / (1.0 + tl.exp(0.0 - acc))
    tl.store(out_batch + 11 * C + offs_c, activated_11)
    tl.store(stage_batch + ((WIDTH - 1) + 11) * C + offs_c, x_11)
    x_12 = tl.load(x_batch + 12 * X_STRIDE_ROW + offs_c)
    product_0 = (x_0 * weight_0).to(tl.bfloat16).to(tl.float32)
    acc = bias_value + product_0
    product_1 = (x_2 * weight_1).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_1
    product_2 = (x_7 * weight_2).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_2
    product_3 = (x_12 * weight_3).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_3
    activated_12 = acc / (1.0 + tl.exp(0.0 - acc))
    tl.store(out_batch + 12 * C + offs_c, activated_12)
    tl.store(stage_batch + ((WIDTH - 1) + 12) * C + offs_c, x_12)
    x_13 = tl.load(x_batch + 13 * X_STRIDE_ROW + offs_c)
    product_0 = (x_0 * weight_0).to(tl.bfloat16).to(tl.float32)
    acc = bias_value + product_0
    product_1 = (x_3 * weight_1).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_1
    product_2 = (x_8 * weight_2).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_2
    product_3 = (x_13 * weight_3).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_3
    activated_13 = acc / (1.0 + tl.exp(0.0 - acc))
    tl.store(out_batch + 13 * C + offs_c, activated_13)
    tl.store(stage_batch + ((WIDTH - 1) + 13) * C + offs_c, x_13)
    x_14 = tl.load(x_batch + 14 * X_STRIDE_ROW + offs_c)
    product_0 = (x_1 * weight_0).to(tl.bfloat16).to(tl.float32)
    acc = bias_value + product_0
    product_1 = (x_4 * weight_1).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_1
    product_2 = (x_9 * weight_2).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_2
    product_3 = (x_14 * weight_3).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_3
    activated_14 = acc / (1.0 + tl.exp(0.0 - acc))
    tl.store(out_batch + 14 * C + offs_c, activated_14)
    tl.store(stage_batch + ((WIDTH - 1) + 14) * C + offs_c, x_14)
    x_15 = tl.load(x_batch + 15 * X_STRIDE_ROW + offs_c)
    product_0 = (x_1 * weight_0).to(tl.bfloat16).to(tl.float32)
    acc = bias_value + product_0
    product_1 = (x_4 * weight_1).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_1
    product_2 = (x_9 * weight_2).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_2
    product_3 = (x_15 * weight_3).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_3
    activated_15 = acc / (1.0 + tl.exp(0.0 - acc))
    tl.store(out_batch + 15 * C + offs_c, activated_15)
    tl.store(stage_batch + ((WIDTH - 1) + 15) * C + offs_c, x_15)
    x_16 = tl.load(x_batch + 16 * X_STRIDE_ROW + offs_c)
    product_0 = (x_1 * weight_0).to(tl.bfloat16).to(tl.float32)
    acc = bias_value + product_0
    product_1 = (x_4 * weight_1).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_1
    product_2 = (x_9 * weight_2).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_2
    product_3 = (x_16 * weight_3).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_3
    activated_16 = acc / (1.0 + tl.exp(0.0 - acc))
    tl.store(out_batch + 16 * C + offs_c, activated_16)
    tl.store(stage_batch + ((WIDTH - 1) + 16) * C + offs_c, x_16)
    x_17 = tl.load(x_batch + 17 * X_STRIDE_ROW + offs_c)
    product_0 = (
        tl.load(x_batch + 2 * X_STRIDE_ROW + offs_c) * weight_0
    ).to(tl.bfloat16).to(tl.float32)
    acc = bias_value + product_0
    product_1 = (x_7 * weight_1).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_1
    product_2 = (x_12 * weight_2).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_2
    product_3 = (x_17 * weight_3).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_3
    activated_17 = acc / (1.0 + tl.exp(0.0 - acc))
    tl.store(out_batch + 17 * C + offs_c, activated_17)
    tl.store(stage_batch + ((WIDTH - 1) + 17) * C + offs_c, x_17)
    x_18 = tl.load(x_batch + 18 * X_STRIDE_ROW + offs_c)
    product_0 = (
        tl.load(x_batch + 3 * X_STRIDE_ROW + offs_c) * weight_0
    ).to(tl.bfloat16).to(tl.float32)
    acc = bias_value + product_0
    product_1 = (x_8 * weight_1).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_1
    product_2 = (x_13 * weight_2).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_2
    product_3 = (x_18 * weight_3).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_3
    activated_18 = acc / (1.0 + tl.exp(0.0 - acc))
    tl.store(out_batch + 18 * C + offs_c, activated_18)
    tl.store(stage_batch + ((WIDTH - 1) + 18) * C + offs_c, x_18)
    x_19 = tl.load(x_batch + 19 * X_STRIDE_ROW + offs_c)
    product_0 = (
        tl.load(x_batch + 4 * X_STRIDE_ROW + offs_c) * weight_0
    ).to(tl.bfloat16).to(tl.float32)
    acc = bias_value + product_0
    product_1 = (x_9 * weight_1).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_1
    product_2 = (x_14 * weight_2).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_2
    product_3 = (x_19 * weight_3).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_3
    activated_19 = acc / (1.0 + tl.exp(0.0 - acc))
    tl.store(out_batch + 19 * C + offs_c, activated_19)
    tl.store(stage_batch + ((WIDTH - 1) + 19) * C + offs_c, x_19)
    x_20 = tl.load(x_batch + 20 * X_STRIDE_ROW + offs_c)
    product_0 = (
        tl.load(x_batch + 4 * X_STRIDE_ROW + offs_c) * weight_0
    ).to(tl.bfloat16).to(tl.float32)
    acc = bias_value + product_0
    product_1 = (x_9 * weight_1).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_1
    product_2 = (x_14 * weight_2).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_2
    product_3 = (x_20 * weight_3).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_3
    activated_20 = acc / (1.0 + tl.exp(0.0 - acc))
    tl.store(out_batch + 20 * C + offs_c, activated_20)
    tl.store(stage_batch + ((WIDTH - 1) + 20) * C + offs_c, x_20)
    x_21 = tl.load(x_batch + 21 * X_STRIDE_ROW + offs_c)
    product_0 = (
        tl.load(x_batch + 4 * X_STRIDE_ROW + offs_c) * weight_0
    ).to(tl.bfloat16).to(tl.float32)
    acc = bias_value + product_0
    product_1 = (x_9 * weight_1).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_1
    product_2 = (x_14 * weight_2).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_2
    product_3 = (x_21 * weight_3).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_3
    activated_21 = acc / (1.0 + tl.exp(0.0 - acc))
    tl.store(out_batch + 21 * C + offs_c, activated_21)
    tl.store(stage_batch + ((WIDTH - 1) + 21) * C + offs_c, x_21)
    x_22 = tl.load(x_batch + 22 * X_STRIDE_ROW + offs_c)
    product_0 = (
        tl.load(x_batch + 7 * X_STRIDE_ROW + offs_c) * weight_0
    ).to(tl.bfloat16).to(tl.float32)
    acc = bias_value + product_0
    product_1 = (x_12 * weight_1).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_1
    product_2 = (x_17 * weight_2).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_2
    product_3 = (x_22 * weight_3).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_3
    activated_22 = acc / (1.0 + tl.exp(0.0 - acc))
    tl.store(out_batch + 22 * C + offs_c, activated_22)
    tl.store(stage_batch + ((WIDTH - 1) + 22) * C + offs_c, x_22)
    x_23 = tl.load(x_batch + 23 * X_STRIDE_ROW + offs_c)
    product_0 = (
        tl.load(x_batch + 8 * X_STRIDE_ROW + offs_c) * weight_0
    ).to(tl.bfloat16).to(tl.float32)
    acc = bias_value + product_0
    product_1 = (x_13 * weight_1).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_1
    product_2 = (x_18 * weight_2).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_2
    product_3 = (x_23 * weight_3).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_3
    activated_23 = acc / (1.0 + tl.exp(0.0 - acc))
    tl.store(out_batch + 23 * C + offs_c, activated_23)
    tl.store(stage_batch + ((WIDTH - 1) + 23) * C + offs_c, x_23)
    x_24 = tl.load(x_batch + 24 * X_STRIDE_ROW + offs_c)
    product_0 = (
        tl.load(x_batch + 9 * X_STRIDE_ROW + offs_c) * weight_0
    ).to(tl.bfloat16).to(tl.float32)
    acc = bias_value + product_0
    product_1 = (x_14 * weight_1).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_1
    product_2 = (x_19 * weight_2).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_2
    product_3 = (x_24 * weight_3).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_3
    activated_24 = acc / (1.0 + tl.exp(0.0 - acc))
    tl.store(out_batch + 24 * C + offs_c, activated_24)
    tl.store(stage_batch + ((WIDTH - 1) + 24) * C + offs_c, x_24)
    x_25 = tl.load(x_batch + 25 * X_STRIDE_ROW + offs_c)
    product_0 = (x_13 * weight_0).to(tl.bfloat16).to(tl.float32)
    acc = bias_value + product_0
    product_1 = (x_18 * weight_1).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_1
    product_2 = (x_23 * weight_2).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_2
    product_3 = (x_25 * weight_3).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_3
    activated_25 = acc / (1.0 + tl.exp(0.0 - acc))
    tl.store(out_batch + 25 * C + offs_c, activated_25)
    tl.store(stage_batch + ((WIDTH - 1) + 25) * C + offs_c, x_25)
    x_26 = tl.load(x_batch + 26 * X_STRIDE_ROW + offs_c)
    product_0 = (x_14 * weight_0).to(tl.bfloat16).to(tl.float32)
    acc = bias_value + product_0
    product_1 = (x_19 * weight_1).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_1
    product_2 = (x_24 * weight_2).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_2
    product_3 = (x_26 * weight_3).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_3
    activated_26 = acc / (1.0 + tl.exp(0.0 - acc))
    tl.store(out_batch + 26 * C + offs_c, activated_26)
    tl.store(stage_batch + ((WIDTH - 1) + 26) * C + offs_c, x_26)
    x_27 = tl.load(x_batch + 27 * X_STRIDE_ROW + offs_c)
    product_0 = (x_18 * weight_0).to(tl.bfloat16).to(tl.float32)
    acc = bias_value + product_0
    product_1 = (x_23 * weight_1).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_1
    product_2 = (x_25 * weight_2).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_2
    product_3 = (x_27 * weight_3).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_3
    activated_27 = acc / (1.0 + tl.exp(0.0 - acc))
    tl.store(out_batch + 27 * C + offs_c, activated_27)
    tl.store(stage_batch + ((WIDTH - 1) + 27) * C + offs_c, x_27)
    x_28 = tl.load(x_batch + 28 * X_STRIDE_ROW + offs_c)
    product_0 = (x_19 * weight_0).to(tl.bfloat16).to(tl.float32)
    acc = bias_value + product_0
    product_1 = (x_24 * weight_1).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_1
    product_2 = (x_26 * weight_2).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_2
    product_3 = (x_28 * weight_3).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_3
    activated_28 = acc / (1.0 + tl.exp(0.0 - acc))
    tl.store(out_batch + 28 * C + offs_c, activated_28)
    tl.store(stage_batch + ((WIDTH - 1) + 28) * C + offs_c, x_28)
    x_29 = tl.load(x_batch + 29 * X_STRIDE_ROW + offs_c)
    product_0 = (x_24 * weight_0).to(tl.bfloat16).to(tl.float32)
    acc = bias_value + product_0
    product_1 = (x_26 * weight_1).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_1
    product_2 = (x_28 * weight_2).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_2
    product_3 = (x_29 * weight_3).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_3
    activated_29 = acc / (1.0 + tl.exp(0.0 - acc))
    tl.store(out_batch + 29 * C + offs_c, activated_29)
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
    activated_30 = acc / (1.0 + tl.exp(0.0 - acc))
    tl.store(out_batch + 30 * C + offs_c, activated_30)
    tl.store(stage_batch + ((WIDTH - 1) + 30) * C + offs_c, x_30)
    x_31 = tl.load(x_batch + 31 * X_STRIDE_ROW + offs_c)
    product_0 = (x_28 * weight_0).to(tl.bfloat16).to(tl.float32)
    acc = bias_value + product_0
    product_1 = (x_29 * weight_1).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_1
    product_2 = (x_30 * weight_2).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_2
    product_3 = (x_31 * weight_3).to(tl.bfloat16).to(tl.float32)
    acc = acc + product_3
    activated_31 = acc / (1.0 + tl.exp(0.0 - acc))
    tl.store(out_batch + 31 * C + offs_c, activated_31)
    tl.store(stage_batch + ((WIDTH - 1) + 31) * C + offs_c, x_31)

    tl.store(stage_batch + offs_c, prior_0)
    tl.store(stage_batch + C + offs_c, prior_1)
    tl.store(stage_batch + 2 * C + offs_c, prior_2)
    tl.store(stage_batch + (SOURCE_ROWS - 1) * C + offs_c, 0.0)
