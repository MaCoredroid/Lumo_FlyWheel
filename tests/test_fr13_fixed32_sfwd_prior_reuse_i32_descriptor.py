from __future__ import annotations

import inspect
import sys
import types

import torch

try:
    import triton  # noqa: F401
except ModuleNotFoundError:
    triton_stub = types.ModuleType("triton")

    def _jit(function=None, **_kwargs):
        return (lambda decorated: decorated) if function is None else function

    triton_stub.jit = _jit
    language_stub = types.ModuleType("triton.language")
    triton_stub.language = language_stub
    sys.modules["triton"] = triton_stub
    sys.modules["triton.language"] = language_stub

from lumo_flywheel_serving.fr13_sfwd_prior_reuse_i32_descriptor import (
    CHANNELS,
    CONV_WIDTH,
    DESCRIPTOR_TAPS,
    FIXED32_PARENT,
    FIXED32_ROWS,
    SIGNED_INT32_MAX,
    SOURCE_ROWS,
    X_ROW_STRIDE,
    _fr13_fixed32_sfwd_prior_reuse_i32_descriptor_kernel,
    fixed32_i32_address_contract,
    fixed32_i32_source_descriptor,
    fixed32_specialized_layout_contract,
)


def _source_rows(width: int = 4) -> tuple[tuple[int, ...], ...]:
    rows = []
    for node in range(len(FIXED32_PARENT)):
        path = []
        cursor = node
        while cursor >= 0:
            path.append(cursor)
            cursor = FIXED32_PARENT[cursor]
        path.reverse()
        source = list(range(width - 1)) + [
            width - 1 + path_node for path_node in path
        ]
        rows.append(tuple(source[-width:]))
    return tuple(rows)


def test_i32_descriptor_matches_every_non_final_tap() -> None:
    source = _source_rows()
    descriptor = fixed32_i32_source_descriptor()

    assert len(descriptor) == 32 * 3
    for node in range(32):
        assert descriptor[node * 3 : (node + 1) * 3] == source[node][:-1]
        assert source[node][-1] == node + 3


def test_i32_descriptor_preserves_exact_ordered_conv_math() -> None:
    torch.manual_seed(20260802)
    source = _source_rows()
    descriptor = fixed32_i32_source_descriptor()
    channels = 17
    prior = torch.randn(3, channels).to(torch.bfloat16)
    x = torch.randn(32, channels).to(torch.bfloat16)
    weights = torch.randn(4, channels).to(torch.bfloat16)

    for node in range(32):
        reference = torch.zeros(channels, dtype=torch.float32)
        candidate = torch.zeros(channels, dtype=torch.float32)
        for tap in range(4):
            source_row = source[node][tap]
            value = prior[source_row] if source_row < 3 else x[source_row - 3]
            product = (value * weights[tap]).to(torch.bfloat16).to(torch.float32)
            reference = reference + product

            if tap == 3:
                candidate_value = x[node]
            else:
                decoded = descriptor[node * 3 + tap]
                candidate_value = (
                    prior[decoded] if decoded < 3 else x[decoded - 3]
                )
            candidate_product = (
                candidate_value * weights[tap]
            ).to(torch.bfloat16).to(torch.float32)
            candidate = candidate + candidate_product

        assert torch.equal(candidate, reference)


def test_b4_fixed_offsets_fit_signed_int32() -> None:
    maxima = fixed32_i32_address_contract(4, x_stride_row=X_ROW_STRIDE)

    assert maxima == {
        "x": (4 * 32 - 1) * X_ROW_STRIDE + CHANNELS - 1,
        "out": 4 * 32 * CHANNELS - 1,
        "source_stage": 4 * 36 * CHANNELS - 1,
    }
    assert max(maxima.values()) <= SIGNED_INT32_MAX


def test_i32_address_contract_rejects_wrong_x_stride() -> None:
    try:
        fixed32_i32_address_contract(4, x_stride_row=CHANNELS)
    except ValueError as error:
        assert "padded" in str(error)
    else:
        raise AssertionError("wrong x row stride was accepted")


def _specialized_layout(batch: int) -> dict[str, tuple[int, ...]]:
    return {
        "x_shape": (batch * FIXED32_ROWS, CHANNELS),
        "x_stride": (X_ROW_STRIDE, 1),
        "out_shape": (batch * FIXED32_ROWS, CHANNELS),
        "out_stride": (CHANNELS, 1),
        "source_stage_shape": (batch * SOURCE_ROWS, CHANNELS),
        "source_stage_stride": (CHANNELS, 1),
        "conv_weights_shape": (CHANNELS, CONV_WIDTH),
        "conv_weights_stride": (CONV_WIDTH, 1),
        "source_descriptor_shape": (FIXED32_ROWS * DESCRIPTOR_TAPS,),
        "source_descriptor_stride": (1,),
    }


def test_specialized_layout_contract_accepts_exact_b1_b4() -> None:
    for batch in (1, 4):
        layouts = _specialized_layout(batch)
        contract = fixed32_specialized_layout_contract(batch, **layouts)

        assert contract["batch_size"] == batch
        assert contract["layouts"] == layouts
        assert contract["maximum_offsets"]["x"] == (
            (batch * FIXED32_ROWS - 1) * X_ROW_STRIDE + CHANNELS - 1
        )


def test_specialized_layout_contract_rejects_each_specialized_drift() -> None:
    exact = _specialized_layout(4)
    drifted = {
        "x_stride": (CHANNELS, 1),
        "out_stride": (CHANNELS + 1, 1),
        "source_stage_stride": (CHANNELS + 1, 1),
        "conv_weights_stride": (1, CHANNELS),
        "source_descriptor_stride": (2,),
    }
    for name, value in drifted.items():
        observed = dict(exact)
        observed[name] = value
        try:
            fixed32_specialized_layout_contract(4, **observed)
        except ValueError as error:
            assert name in str(error)
        else:
            raise AssertionError(f"specialized layout drift accepted: {name}")


def test_specialized_batch_bases_match_flat_element_offsets() -> None:
    for batch in range(4):
        for node in (0, 1, FIXED32_ROWS - 1):
            for channel in (0, 63, CHANNELS - 1):
                x_flat = (batch * FIXED32_ROWS + node) * X_ROW_STRIDE + channel
                x_batched = (
                    batch * FIXED32_ROWS * X_ROW_STRIDE
                    + node * X_ROW_STRIDE
                    + channel
                )
                out_flat = (batch * FIXED32_ROWS + node) * CHANNELS + channel
                out_batched = (
                    batch * FIXED32_ROWS * CHANNELS + node * CHANNELS + channel
                )
                assert x_batched == x_flat
                assert out_batched == out_flat

        for source_row in (0, 1, SOURCE_ROWS - 1):
            source_flat = (batch * SOURCE_ROWS + source_row) * CHANNELS
            source_batched = (
                batch * SOURCE_ROWS * CHANNELS + source_row * CHANNELS
            )
            assert source_batched == source_flat


def test_kernel_specializes_only_validated_dense_and_padded_terms() -> None:
    source = inspect.getsource(
        _fr13_fixed32_sfwd_prior_reuse_i32_descriptor_kernel
    )

    assert "X_STRIDE_ROW: tl.constexpr" in source
    assert "x_batch = x + pid_b * N * X_STRIDE_ROW" in source
    assert "out_batch = out + pid_b * N * C" in source
    assert "stage_batch = source_stage + pid_b * SOURCE_ROWS * C" in source
    assert "conv_weights + offs_c * WIDTH + tap" in source
    assert "x_stride_row" not in source
    assert "weight_stride_c" not in source
    assert "weight_stride_w" not in source
    assert ").to(tl.int64)" in source
    assert "bank_row * conv_stride_row" in source
