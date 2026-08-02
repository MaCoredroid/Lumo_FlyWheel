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
    CONV_WIDTH,
    FIXED32_PARENT,
    FIXED32_PACKED_SOURCE_DELTAS,
    FIXED32_ROWS,
    SIGNED_INT32_MAX,
    SOURCE_ROWS,
    X_ROW_STRIDE,
    fixed32_derived_parent_q,
    fixed32_descriptorless_sources,
    fixed32_i32_address_contract,
    fixed32_packed_source_entry,
    fixed32_packed_sources,
    fixed32_specialized_layout_contract,
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


def test_packed_sources_match_every_descriptorless_source() -> None:
    expected = fixed32_descriptorless_sources()

    assert len(FIXED32_PACKED_SOURCE_DELTAS) == 8
    assert fixed32_packed_sources() == expected
    assert tuple(fixed32_packed_source_entry(node) for node in range(32)) == expected


def test_packed_source_decoder_rejects_out_of_range_nodes() -> None:
    for node in (-1, 32):
        try:
            fixed32_packed_source_entry(node)
        except ValueError as error:
            assert "[0, 32)" in str(error)
        else:
            raise AssertionError(f"packed source decoder accepted node {node}")


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


def test_fixed_topology_prior_masks_match_every_source_row() -> None:
    sources = fixed32_descriptorless_sources()
    thresholds = (9, 4, 1)

    for tap, threshold in enumerate(thresholds):
        assert tuple(source[tap] < 3 for source in sources) == tuple(
            node < threshold for node in range(FIXED32_ROWS)
        )

    assert tuple(source[0] for source in sources[:9]) == (
        0,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
    )
    assert tuple(source[1] for source in sources[:4]) == (1, 2, 2, 2)
    assert sources[0][2] == 2


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


def test_b1_b4_live_padded_offsets_fit_signed_int32() -> None:
    for batch in (1, 4):
        maxima = fixed32_i32_address_contract(
            batch,
            x_stride_row=X_ROW_STRIDE,
        )
        assert maxima == {
            "x": (batch * 32 - 1) * X_ROW_STRIDE + CHANNELS - 1,
            "out": batch * 32 * CHANNELS - 1,
            "source_stage": batch * 36 * CHANNELS - 1,
        }
        assert max(maxima.values()) <= SIGNED_INT32_MAX


def test_descriptorless_contract_rejects_wrong_x_stride() -> None:
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


def test_specialized_layout_contract_rejects_each_layout_drift() -> None:
    exact = _specialized_layout(4)
    drifted = {
        "x_shape": (4 * FIXED32_ROWS - 1, CHANNELS),
        "x_stride": (CHANNELS, 1),
        "out_shape": (4 * FIXED32_ROWS - 1, CHANNELS),
        "out_stride": (CHANNELS + 1, 1),
        "source_stage_shape": (4 * SOURCE_ROWS - 1, CHANNELS),
        "source_stage_stride": (CHANNELS + 1, 1),
        "conv_weights_shape": (CHANNELS, CONV_WIDTH - 1),
        "conv_weights_stride": (1, CHANNELS),
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
            source_offset = (
                batch * SOURCE_ROWS * CHANNELS + source_row * CHANNELS
            )
            assert source_offset == source_flat


def test_kernel_retains_descriptorless_and_int64_state_contracts() -> None:
    module_path = Path(
        sys.modules[
            "lumo_flywheel_serving.fr13_sfwd_prior_reuse_descriptorless"
        ].__file__
    )
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    kernel = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_fr13_fixed32_sfwd_prior_reuse_descriptorless_kernel"
    )
    fragment = ast.get_source_segment(source, kernel)
    assert fragment is not None
    argument_names = tuple(argument.arg for argument in kernel.args.args)

    assert "source_descriptor" not in argument_names
    assert "x_stride_row" not in argument_names
    assert "weight_stride_c" not in argument_names
    assert "weight_stride_w" not in argument_names
    assert "X_STRIDE_ROW" in argument_names
    assert "x_batch = x + pid_b * N * X_STRIDE_ROW" in fragment
    assert "out_batch = out + pid_b * N * C" in fragment
    assert "stage_offset = pid_b * SOURCE_ROWS * C" in fragment
    assert ").to(tl.int64)" in fragment
    assert "bank_row * conv_stride_row" in fragment


def test_packed_xgather_loads_current_x_once_and_reuses_it() -> None:
    module_path = Path(
        sys.modules[
            "lumo_flywheel_serving.fr13_sfwd_prior_reuse_descriptorless"
        ].__file__
    )
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    kernel = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_fr13_fixed32_sfwd_prior_reuse_packed_xgather_kernel"
    )
    fragment = ast.get_source_segment(source, kernel)

    assert fragment is not None
    assert fragment.count("tl.load(x_batch") == 1
    assert fragment.count("tl.gather(current_x,") == 3
    assert "tl.gather(current_x, x_index, axis=0)" in fragment
    assert "tl.broadcast_to(x_node, ROWS_PER_PROGRAM, BLOCK_C)" in fragment
    assert "offs_n - pid_n_base, ROWS_PER_PROGRAM, BLOCK_C" in fragment
    assert "tl.gather(current_x, current_index, axis=0)" in fragment
    assert "current_value * current_weight" in fragment
    assert "current_x * current_weight" not in fragment
    assert fragment.index("for tap in tl.static_range(0, WIDTH - 2):") < (
        fragment.index("current_index = tl.broadcast_to(")
    )
    assert fragment.index("current_product =") < fragment.index(
        "acc = acc + current_product"
    )
    assert "((WIDTH - 1) + offs_n) * C" in fragment


def test_packed_xgather_loads_contiguous_weights_as_exact_pairs() -> None:
    module_path = Path(
        sys.modules[
            "lumo_flywheel_serving.fr13_sfwd_prior_reuse_descriptorless"
        ].__file__
    )
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    kernel = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_fr13_fixed32_sfwd_prior_reuse_packed_xgather_kernel"
    )
    fragment = ast.get_source_segment(source, kernel)

    assert fragment is not None
    assert fragment.count("tl.pointer_type(tl.uint64)") == 1
    assert "weight_quad >> (tap << 4)" in fragment
    assert "weight_quad >> 32" in fragment
    assert "weight_quad >> 48" in fragment
    assert "weight_bits.to(tl.bfloat16, bitcast=True)" in fragment
    assert "weight_2_bits.to(tl.bfloat16, bitcast=True)" in fragment
    assert "current_weight_bits.to(tl.bfloat16, bitcast=True)" in fragment
    assert "tl.load(weight_channels + tap)" not in fragment


def test_packed_xgather_loads_first_two_prior_values_as_exact_pair() -> None:
    module_path = Path(
        sys.modules[
            "lumo_flywheel_serving.fr13_sfwd_prior_reuse_descriptorless"
        ].__file__
    )
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    kernel = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_fr13_fixed32_sfwd_prior_reuse_packed_xgather_kernel"
    )
    fragment = ast.get_source_segment(source, kernel)
    argument_names = tuple(argument.arg for argument in kernel.args.args)

    assert fragment is not None
    assert "conv_stride_row" not in argument_names
    assert "conv_stride_c" not in argument_names
    assert "conv_stride_l" not in argument_names
    assert "ssi_stride_b" not in argument_names
    assert "ssi_stride_s" not in argument_names
    assert "spec_state_indices + pid_b * N" in fragment
    assert "bank_row * C * STATE_LEN" in fragment
    assert "offs_c * STATE_LEN" in fragment
    assert "prior_base.to(tl.pointer_type(tl.uint32))" in fragment
    assert "prior_pair.to(tl.uint16).to(tl.bfloat16, bitcast=True)" in fragment
    assert "(prior_pair >> 16).to(tl.uint16).to(" in fragment
    assert "prior_base + 2" in fragment


def test_packed_xgather_specializes_prior_masks_to_fixed_topology() -> None:
    module_path = Path(
        sys.modules[
            "lumo_flywheel_serving.fr13_sfwd_prior_reuse_descriptorless"
        ].__file__
    )
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    kernel = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_fr13_fixed32_sfwd_prior_reuse_packed_xgather_kernel"
    )
    fragment = ast.get_source_segment(source, kernel)

    assert fragment is not None
    assert "if tap == 0:" in fragment
    assert "from_prior = offs_n < 9" in fragment
    assert "from_prior = offs_n < 4" in fragment
    assert "from_prior = offs_n == 0" in fragment
    assert "tl.where(offs_n == 0, prior_1, prior_2)" in fragment
    assert "tl.where(from_prior, prior_2, x_value)" in fragment
    assert "source_row < (WIDTH - 1)" not in fragment
    assert "source_row == 0" not in fragment


def test_channel_serial_split20_math_matches_every_fixed_source() -> None:
    module_path = Path(
        sys.modules[
            "lumo_flywheel_serving.fr13_sfwd_prior_reuse_descriptorless"
        ].__file__
    )
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    kernel = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_fr13_fixed32_sfwd_channel_serial_kernel"
    )
    def source_row(expression: ast.expr) -> int:
        while isinstance(expression, ast.Call):
            assert isinstance(expression.func, ast.Attribute)
            expression = expression.func.value
        assert isinstance(expression, ast.BinOp)
        assert isinstance(expression.op, ast.Mult)
        source = expression.left
        if isinstance(source, ast.Call):
            assert isinstance(source.func, ast.Attribute)
            assert source.func.attr == "load"
            stride_terms = [
                item
                for item in ast.walk(source.args[0])
                if isinstance(item, ast.BinOp)
                and isinstance(item.op, ast.Mult)
                and isinstance(item.left, ast.Constant)
                and isinstance(item.left.value, int)
                and isinstance(item.right, ast.Name)
                and item.right.id == "X_STRIDE_ROW"
            ]
            assert len(stride_terms) == 1
            return CONV_WIDTH - 1 + int(stride_terms[0].left.value)
        assert isinstance(source, ast.Name)
        if source.id.startswith("prior_"):
            return int(source.id.removeprefix("prior_"))
        assert source.id.startswith("x_")
        return CONV_WIDTH - 1 + int(source.id.removeprefix("x_"))

    expected = fixed32_descriptorless_sources()
    for tap in range(CONV_WIDTH - 1):
        products = [
            node.value
            for node in kernel.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == f"product_{tap}"
        ]
        assert len(products) == FIXED32_ROWS
        assert tuple(source_row(item) for item in products) == tuple(
            row[tap] for row in expected
        )
    current_products = [
        node.value
        for node in kernel.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "product_3"
    ]
    assert tuple(source_row(item) for item in current_products) == tuple(
        CONV_WIDTH - 1 + node for node in range(FIXED32_ROWS)
    )


def test_channel_serial_keeps_global_rows_coalesced_and_ordered() -> None:
    module_path = Path(
        sys.modules[
            "lumo_flywheel_serving.fr13_sfwd_prior_reuse_descriptorless"
        ].__file__
    )
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    kernel = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_fr13_fixed32_sfwd_channel_serial_kernel"
    )
    fragment = ast.get_source_segment(source, kernel)
    argument_names = tuple(argument.arg for argument in kernel.args.args)

    assert fragment is not None
    assert "CONV_STRIDE_ROW" in argument_names
    assert "bank_row * CONV_STRIDE_ROW + offs_c" in fragment
    assert "prior_base + C" in fragment
    assert "prior_base + 2 * C" in fragment
    assert "weight_pair_01" in fragment
    assert "weight_pair_23" in fragment
    assert "tl.pointer_type(tl.uint64)" not in fragment
    assert "tl.arange(0, BLOCK_C)" in fragment
    assert "tl.arange(0, BLOCK_C)[:, None]" not in fragment
    assert "tl.arange(0, N)" not in fragment
    assert fragment.count("tl.load(x_batch +") == FIXED32_ROWS + 8
    reload_counts = {2: 1, 3: 1, 4: 3, 7: 1, 8: 1, 9: 1}
    for row in range(FIXED32_ROWS):
        load = f"x_{row} = tl.load(x_batch + {row} * X_STRIDE_ROW + offs_c)"
        product = f"product_3 = (x_{row} * weight_3)"
        assert fragment.count(
            f"x_batch + {row} * X_STRIDE_ROW + offs_c"
        ) == 1 + reload_counts.get(row, 0)
        assert fragment.index(load) < fragment.index(product)
        if row:
            prior_store = (
                "tl.store(stage_batch + ((WIDTH - 1) + "
                f"{row - 1}) * C + offs_c, x_{row - 1})"
            )
            assert fragment.index(prior_store) < fragment.index(load)
    assert "tl.gather" not in fragment
    assert "tl.join" not in fragment
    assert "tl.permute" not in fragment
    assert "tl.reshape" not in fragment
    assert "tl.split" not in fragment
    assert "ROWS_PER_PROGRAM" not in argument_names
    assert "for node in tl.static_range(0, N):" not in fragment
    assert fragment.count("tl.store(out_batch +") == FIXED32_ROWS
    assert fragment.count("tl.store(stage_batch + ((WIDTH - 1) +") == FIXED32_ROWS
    assert fragment.count(
        "x_4 = tl.load(x_batch + 4 * X_STRIDE_ROW + offs_c)"
    ) == 1
    assert fragment.index(
        "tl.store(stage_batch + ((WIDTH - 1) + 19) * C + offs_c, x_19)"
    ) < fragment.index("x_20 = tl.load(")
    ordered = (
        "product_0 =",
        "acc = bias_value + product_0",
        "product_1 =",
        "acc = acc + product_1",
        "product_2 =",
        "acc = acc + product_2",
        "product_3 =",
        "acc = acc + product_3",
    )
    positions = tuple(fragment.index(value) for value in ordered)
    assert positions == tuple(sorted(positions))
