"""Offline-only fixed32 SFWD prior-reuse kernel with int32 descriptors."""

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
DESCRIPTOR_TAPS = CONV_WIDTH - 1
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
        raise ValueError("int32 SFWD addressing requires B1-B4")
    if stride != X_ROW_STRIDE:
        raise ValueError("int32 SFWD addressing requires the fixed padded x stride")
    maxima = {
        "x": (batch * FIXED32_ROWS - 1) * stride + CHANNELS - 1,
        "out": batch * FIXED32_ROWS * CHANNELS - 1,
        "source_stage": batch * SOURCE_ROWS * CHANNELS - 1,
    }
    if max(maxima.values()) > SIGNED_INT32_MAX:
        raise ValueError("int32 SFWD dense-buffer offset would overflow")
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
    source_descriptor_shape: tuple[int, ...],
    source_descriptor_stride: tuple[int, ...],
) -> dict[str, object]:
    """Validate every fixed layout term removed from the kernel signature."""
    batch = int(batch_size)
    if batch not in (1, 2, 3, 4):
        raise ValueError("specialized SFWD layout requires B1-B4")
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
        "source_descriptor_shape": (FIXED32_ROWS * DESCRIPTOR_TAPS,),
        "source_descriptor_stride": (1,),
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
        "source_descriptor_shape": tuple(
            int(value) for value in source_descriptor_shape
        ),
        "source_descriptor_stride": tuple(
            int(value) for value in source_descriptor_stride
        ),
    }
    failures = tuple(
        name for name, value in observed.items() if value != expected[name]
    )
    if failures:
        raise ValueError(
            "specialized SFWD operand layout drift: " + ",".join(failures)
        )
    return {
        "batch_size": batch,
        "layouts": expected,
        "maximum_offsets": fixed32_i32_address_contract(
            batch, x_stride_row=X_ROW_STRIDE
        ),
    }


def fixed32_i32_source_descriptor(width: int = CONV_WIDTH) -> tuple[int, ...]:
    """Return the three non-final source rows for each fixed32 node."""
    if width != CONV_WIDTH:
        raise ValueError("int32 SFWD source descriptor requires width 4")
    descriptor: list[int] = []
    for node in range(len(FIXED32_PARENT)):
        path: list[int] = []
        cursor = node
        while cursor >= 0:
            path.append(cursor)
            cursor = FIXED32_PARENT[cursor]
        path.reverse()
        source = list(range(width - 1)) + [
            width - 1 + path_node for path_node in path
        ]
        descriptor.extend(source[-width:-1])
    return tuple(descriptor)


@triton.jit
def _fr13_fixed32_sfwd_prior_reuse_i32_descriptor_kernel(
    x,
    conv_state,
    spec_state_indices,
    source_descriptor,
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
    """Fuse fixed32 convolution and state motion using a narrow descriptor."""
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
    stage_batch = source_stage + pid_b * SOURCE_ROWS * C

    bank_row = tl.load(
        spec_state_indices + pid_b * ssi_stride_b + 0 * ssi_stride_s
    ).to(tl.int64)
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
    acc = tl.zeros((ROWS_PER_PROGRAM, BLOCK_C), dtype=tl.float32)
    if HAS_BIAS:
        acc = tl.load(bias + offs_c).to(tl.float32)
    for tap in tl.static_range(0, WIDTH - 1):
        source_row = tl.load(
            source_descriptor + offs_n * (WIDTH - 1) + tap
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
        weight = tl.load(
            conv_weights + offs_c * WIDTH + tap
        ).to(tl.bfloat16)
        product = (value * weight).to(tl.bfloat16).to(tl.float32)
        acc = acc + product

    current_x = tl.load(
        x_batch + offs_n * X_STRIDE_ROW + offs_c
    )
    current_weight = tl.load(
        conv_weights + offs_c * WIDTH + (WIDTH - 1)
    ).to(tl.bfloat16)
    current_product = (
        current_x * current_weight
    ).to(tl.bfloat16).to(tl.float32)
    acc = acc + current_product

    activated = acc / (1.0 + tl.exp(0.0 - acc))
    tl.store(out_batch + offs_n * C + offs_c, activated)

    tl.store(
        stage_batch
        + ((WIDTH - 1) + offs_n) * C
        + offs_c,
        current_x,
    )
    source_edge_writer = pid_n_base == 0
    tl.store(
        stage_batch + offs_c,
        prior_0,
        mask=source_edge_writer,
    )
    tl.store(
        stage_batch + C + offs_c,
        prior_1,
        mask=source_edge_writer,
    )
    tl.store(
        stage_batch + 2 * C + offs_c,
        prior_2,
        mask=source_edge_writer,
    )
    tl.store(
        stage_batch
        + (SOURCE_ROWS - 1) * C
        + offs_c,
        0.0,
        mask=source_edge_writer,
    )
