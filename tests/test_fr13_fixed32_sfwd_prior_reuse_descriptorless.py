from __future__ import annotations

import ast
from pathlib import Path
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

from lumo_flywheel_serving.fr13_sfwd_prior_reuse_descriptorless import (
    CHANNELS,
    FIXED32_PARENT,
    SIGNED_INT32_MAX,
    fixed32_derived_parent_q,
    fixed32_descriptorless_sources,
    fixed32_i32_address_contract,
)


def _reference_sources() -> tuple[tuple[int, int, int, int], ...]:
    rows = []
    for node in range(len(FIXED32_PARENT)):
        path = []
        cursor = node
        while cursor >= 0:
            path.append(cursor)
            cursor = FIXED32_PARENT[cursor]
        path.reverse()
        source = [0, 1, 2] + [3 + path_node for path_node in path]
        rows.append(tuple(source[-4:]))
    return tuple(rows)


def test_parent_arithmetic_matches_every_fixed32_edge() -> None:
    assert tuple(
        fixed32_derived_parent_q(node) - 1 for node in range(32)
    ) == FIXED32_PARENT


def test_descriptorless_sources_match_every_non_final_tap() -> None:
    reference = _reference_sources()
    derived = fixed32_descriptorless_sources()

    assert len(derived) == 32
    for node in range(32):
        assert derived[node] == reference[node][:-1]
        assert reference[node][-1] == node + 3


def test_descriptorless_sources_preserve_ordered_conv_math() -> None:
    torch.manual_seed(20260802)
    reference = _reference_sources()
    derived = fixed32_descriptorless_sources()
    channels = 17
    prior = torch.randn(3, channels).to(torch.bfloat16)
    x = torch.randn(32, channels).to(torch.bfloat16)
    weights = torch.randn(4, channels).to(torch.bfloat16)

    for node in range(32):
        expected = torch.zeros(channels, dtype=torch.float32)
        candidate = torch.zeros(channels, dtype=torch.float32)
        for tap in range(4):
            expected_row = reference[node][tap]
            expected_value = (
                prior[expected_row]
                if expected_row < 3
                else x[expected_row - 3]
            )
            expected_product = (
                expected_value * weights[tap]
            ).to(torch.bfloat16).to(torch.float32)
            expected = expected + expected_product

            candidate_row = node + 3 if tap == 3 else derived[node][tap]
            candidate_value = (
                prior[candidate_row]
                if candidate_row < 3
                else x[candidate_row - 3]
            )
            candidate_product = (
                candidate_value * weights[tap]
            ).to(torch.bfloat16).to(torch.float32)
            candidate = candidate + candidate_product

        assert torch.equal(candidate, expected)


def test_descriptor_pointer_is_absent_from_kernel_contract() -> None:
    module_path = Path(
        sys.modules[
            "lumo_flywheel_serving.fr13_sfwd_prior_reuse_descriptorless"
        ].__file__
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    kernel = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_fr13_fixed32_sfwd_prior_reuse_descriptorless_kernel"
    )
    argument_names = tuple(argument.arg for argument in kernel.args.args)
    fragment = ast.get_source_segment(module_path.read_text(encoding="utf-8"), kernel)

    assert "source_descriptor" not in argument_names
    assert "source_flat" not in argument_names
    assert "source_descriptor" not in fragment
    assert argument_names[-2:] == ("ROWS_PER_PROGRAM", "BLOCK_C")


def test_b1_b4_dense_offsets_fit_signed_int32() -> None:
    for batch in (1, 4):
        maxima = fixed32_i32_address_contract(batch, x_stride_row=CHANNELS)
        assert maxima == {
            "x": batch * 32 * CHANNELS - 1,
            "out": batch * 32 * CHANNELS - 1,
            "source_stage": batch * 36 * CHANNELS - 1,
        }
        assert max(maxima.values()) <= SIGNED_INT32_MAX


def test_descriptorless_contract_rejects_unsafe_stride() -> None:
    unsafe_stride = SIGNED_INT32_MAX // (4 * 32 - 1) + 1

    try:
        fixed32_i32_address_contract(4, x_stride_row=unsafe_stride)
    except ValueError as error:
        assert "overflow" in str(error)
    else:
        raise AssertionError("unsafe int32 row stride was accepted")
