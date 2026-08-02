from __future__ import annotations

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
    FIXED32_PARENT,
    SIGNED_INT32_MAX,
    fixed32_i32_address_contract,
    fixed32_i32_source_descriptor,
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


def test_b4_dense_offsets_fit_signed_int32() -> None:
    maxima = fixed32_i32_address_contract(4, x_stride_row=CHANNELS)

    assert maxima == {
        "x": 4 * 32 * CHANNELS - 1,
        "out": 4 * 32 * CHANNELS - 1,
        "source_stage": 4 * 36 * CHANNELS - 1,
    }
    assert max(maxima.values()) <= SIGNED_INT32_MAX


def test_i32_address_contract_rejects_unsafe_stride() -> None:
    unsafe_stride = SIGNED_INT32_MAX // (4 * 32 - 1) + 1

    try:
        fixed32_i32_address_contract(4, x_stride_row=unsafe_stride)
    except ValueError as error:
        assert "overflow" in str(error)
    else:
        raise AssertionError("unsafe int32 row stride was accepted")
