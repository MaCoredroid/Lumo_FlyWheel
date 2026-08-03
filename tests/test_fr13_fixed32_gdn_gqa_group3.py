from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = (
    ROOT / "src/lumo_flywheel_serving/fr13_gdn_gqa_group3.py"
)
SERVED_KERNEL = (
    ROOT / "src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py"
)


def _tree_and_source() -> tuple[ast.Module, str]:
    source = CANDIDATE.read_text(encoding="utf-8")
    return ast.parse(source), source


def _function_source(name: str) -> str:
    tree, source = _tree_and_source()
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    segment = ast.get_source_segment(source, node)
    assert segment is not None
    return segment


def _contract() -> object:
    constants = {
        "CANDIDATE",
        "FIXED32_MODES",
        "PHYSICAL_ROWS",
        "NUM_K_HEADS",
        "NUM_V_HEADS",
        "HEAD_GROUP",
        "DIM_K",
        "DIM_V",
        "BLOCK_V",
        "GDN_LAYERS",
        "BF16_BYTES",
    }
    tree, _source = _tree_and_source()
    body = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id in constants
                for target in node.targets
            )
        )
        or (
            isinstance(node, ast.FunctionDef)
            and node.name == "fixed32_gdn_gqa_group3_contract"
        )
    ]
    namespace: dict[str, object] = {}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
            CANDIDATE,
            "exec",
        ),
        namespace,
    )
    return namespace["fixed32_gdn_gqa_group3_contract"]


@pytest.mark.parametrize(
    ("batch", "mode", "reference", "candidate", "removed", "bytes_removed"),
    (
        (1, "tail6_fixed32", 768, 256, 24_576, 402_653_184),
        (4, "hydra27_fixed32", 3_072, 1_024, 98_304, 1_610_612_736),
    ),
)
def test_contract_closes_exact_b1_b4_physical_work(
    batch: int,
    mode: str,
    reference: int,
    candidate: int,
    removed: int,
    bytes_removed: int,
) -> None:
    contract = _contract()
    result = contract(batch, mode=mode)

    assert result["candidate"] == "fixed32_gdn_single_launch_gqa_group3_v1"
    assert result["physical_rows_per_request"] == 32
    assert result["logical_tree_limit"] == 32
    assert result["fixed_work_for_any_logical_tree_lte"] == 32
    assert result["value_heads_per_key_head"] == 3
    assert result["physical_launches_per_layer"] == 1
    assert result["reference_ctas_per_layer"] == reference
    assert result["candidate_ctas_per_layer"] == candidate
    assert result["ctas_removed_per_event"] == removed
    assert result["qk_bytes_removed_per_event"] == bytes_removed
    assert result["qk_norm_reductions_removed_per_event"] == removed * 64
    assert result["qk_norm_lane_terms_removed_per_event"] == removed * 8_192
    assert result["state_export_writes"] == 0
    assert result["state_parent_reads"] == 0
    assert result["candidate_default_off"] is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"batch_size": 2, "mode": "tail6_fixed32"}, "B1 or B4"),
        ({"batch_size": 1, "mode": "tail6"}, "fixed32 mode"),
        (
            {
                "batch_size": 1,
                "mode": "tail6_fixed32",
                "physical_rows": 31,
            },
            "geometry drift",
        ),
        (
            {
                "batch_size": 1,
                "mode": "tail6_fixed32",
                "num_k_heads": 8,
            },
            "geometry drift",
        ),
        (
            {
                "batch_size": 1,
                "mode": "tail6_fixed32",
                "num_v_heads": 32,
            },
            "geometry drift",
        ),
        (
            {
                "batch_size": 1,
                "mode": "tail6_fixed32",
                "block_v": 16,
            },
            "geometry drift",
        ),
        (
            {
                "batch_size": 1,
                "mode": "tail6_fixed32",
                "layers": 47,
            },
            "geometry drift",
        ),
    ),
)
def test_contract_rejects_any_nonqualified_geometry(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _contract()(**kwargs)


def test_program_mapping_covers_each_value_tile_once() -> None:
    writes = [
        (key_head * 3 + sibling, value_tile)
        for key_head in range(16)
        for value_tile in range(16)
        for sibling in range(3)
    ]
    expected = [
        (value_head, value_tile)
        for value_head in range(48)
        for value_tile in range(16)
    ]

    assert len(writes) == 256 * 3
    assert len(set(writes)) == len(writes)
    assert sorted(writes) == expected


def test_grouped_recurrence_is_exactly_three_independent_value_heads() -> None:
    generator = torch.Generator().manual_seed(20260803)
    state = torch.randn(3, 4, 8, generator=generator)
    q = torch.randn(8, generator=generator)
    k = torch.randn(8, generator=generator)
    value = torch.randn(3, 4, generator=generator)
    beta = torch.sigmoid(torch.randn(3, generator=generator))
    decay = torch.exp(-torch.rand(3, generator=generator))

    grouped_state = state * decay[:, None, None]
    grouped_value = value - torch.sum(
        grouped_state * k[None, None, :], dim=2
    )
    grouped_value *= beta[:, None]
    grouped_state += grouped_value[:, :, None] * k[None, None, :]
    grouped_out = torch.sum(grouped_state * q[None, None, :], dim=2)

    scalar_states = []
    scalar_outputs = []
    for sibling in range(3):
        scalar_state = state[sibling] * decay[sibling]
        scalar_value = value[sibling] - torch.sum(
            scalar_state * k[None, :], dim=1
        )
        scalar_value *= beta[sibling]
        scalar_state += scalar_value[:, None] * k[None, :]
        scalar_states.append(scalar_state)
        scalar_outputs.append(torch.sum(scalar_state * q[None, :], dim=1))

    assert torch.equal(grouped_state, torch.stack(scalar_states))
    assert torch.equal(grouped_out, torch.stack(scalar_outputs))


def test_kernel_reuses_qk_and_preserves_ordered_single_launch_contract() -> None:
    value_head = _function_source(
        "_fr13_fixed32_gdn_gqa_group3_value_head_node"
    )
    node = _function_source("_fr13_fixed32_gdn_gqa_group3_node")
    kernel = _function_source(
        "_fr13_fixed32_gdn_gqa_group3_single_launch_kernel"
    )
    launch = _function_source(
        "launch_fixed32_gdn_gqa_group3_source_candidate"
    )

    assert node.count(
        "b_q = tl.load(\n        q + (global_node * NUM_KH + pid_kh)"
    ) == 1
    assert node.count(
        "b_k_raw = tl.load(\n        k + (global_node * NUM_KH + pid_kh)"
    ) == 1
    assert node.count("_fr13_fixed32_gdn_gqa_group3_value_head_node(") == 3
    assert "tl.arange(0, HEAD_GROUP)" not in node
    assert "axis=1" in value_head
    assert "prior_state = state_i" in value_head
    assert "return tl.where(n_ok, state_i, prior_state)" in value_head
    assert "pid_kh = tl.program_id(0)" in kernel
    assert "pid_vh_0 = pid_kh * HEAD_GROUP" in kernel
    assert "tl.arange(0, HEAD_GROUP)" not in kernel
    assert "for root_index in tl.range(0, ROOT_STEPS):" in kernel
    assert "for member in tl.static_range(0, MAX_GROUP_PATHS):" in kernel
    assert "for path_offset in tl.range(0, path_len):" in kernel
    assert kernel.count("_fr13_fixed32_gdn_gqa_group3_node(") == 2
    assert "if H0_IS_BANK:" in kernel
    assert "grid = (NUM_K_HEADS, DIM_V // BLOCK_V, int(batch_size))" in launch
    assert "num_warps=8" in launch


def test_candidate_is_source_only_and_cannot_change_the_served_arm() -> None:
    served = SERVED_KERNEL.read_text(encoding="utf-8")
    _tree, source = _tree_and_source()

    assert "deliberately not wired into serving" in source
    assert "fixed32_gdn_single_launch_gqa_group3_v1" not in served
    assert "fr13_gdn_gqa_group3" not in served


def test_value_head_helper_calls_exactly_match_the_ast_signature() -> None:
    tree, _source = _tree_and_source()
    helper_name = "_fr13_fixed32_gdn_gqa_group3_value_head_node"
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == helper_name
    )
    group = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_fr13_fixed32_gdn_gqa_group3_node"
    )
    signature = [argument.arg for argument in helper.args.args]
    calls = [
        node
        for node in ast.walk(group)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == helper_name
    ]

    assert "ring_k" not in signature
    assert "ring_k_norm" not in signature
    assert len(calls) == 3
    for sibling, call in enumerate(calls):
        assert len(call.args) + len(call.keywords) == len(signature)
        assert [keyword.arg for keyword in call.keywords] == signature[
            len(call.args) :
        ]
        actual_positionals = [ast.unparse(argument) for argument in call.args]
        expected_positionals = signature[: len(call.args)]
        expected_positionals[0] = f"state_{sibling}"
        expected_positionals[8] = f"b_a_log_{sibling}"
        expected_positionals[9] = f"b_dt_bias_{sibling}"
        expected_positionals[16] = f"pid_vh_{sibling}"
        assert actual_positionals == expected_positionals


class _FakeGdnKernelLaunch:
    def __init__(self) -> None:
        self.calls = 0

    def __getitem__(self, _grid):
        def launch(*_args, **_kwargs) -> None:
            self.calls += 1

        return launch


def _load_gdn_candidate(monkeypatch: pytest.MonkeyPatch):
    fake_triton = types.ModuleType("triton")
    fake_language = types.ModuleType("triton.language")
    fake_triton.jit = lambda function: function
    fake_triton.language = fake_language
    monkeypatch.setitem(sys.modules, "triton", fake_triton)
    monkeypatch.setitem(sys.modules, "triton.language", fake_language)
    spec = importlib.util.spec_from_file_location(
        "fr13_gdn_gqa_group3_launch_test", CANDIDATE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _qualified_gdn_launch_args(module) -> dict[str, object]:
    batch = 1
    rows = batch * module.PHYSICAL_ROWS
    h0 = torch.empty(
        (2, module.NUM_V_HEADS, module.DIM_V, module.DIM_K),
        dtype=torch.float32,
    )
    return {
        "q": torch.empty(
            (rows, module.NUM_K_HEADS, module.DIM_K),
            dtype=torch.bfloat16,
        ),
        "k": torch.empty(
            (rows, module.NUM_K_HEADS, module.DIM_K),
            dtype=torch.bfloat16,
        ),
        "v": torch.empty(
            (rows, module.NUM_V_HEADS, module.DIM_V),
            dtype=torch.bfloat16,
        ),
        "g": torch.empty((rows, module.NUM_V_HEADS), dtype=torch.float32),
        "beta": torch.empty(
            (rows, module.NUM_V_HEADS), dtype=torch.float32
        ),
        "raw_a": torch.empty(
            (rows, module.NUM_V_HEADS), dtype=torch.bfloat16
        ),
        "raw_b": torch.empty(
            (rows, module.NUM_V_HEADS), dtype=torch.bfloat16
        ),
        "A_log": torch.empty((module.NUM_V_HEADS,), dtype=torch.float32),
        "dt_bias": torch.empty(
            (module.NUM_V_HEADS,), dtype=torch.float32
        ),
        "h0": h0,
        "h0_indices": torch.tensor([[0, 1]], dtype=torch.int64),
        "h0_num_accepted_tokens": torch.tensor([1], dtype=torch.int32),
        "invocation_counter": torch.zeros((), dtype=torch.int32),
        "root_nodes": torch.tensor(
            [module._ROOT_NODES], dtype=torch.int32
        ),
        "branch_nodes": torch.tensor(
            module._BRANCH_NODES, dtype=torch.int32
        ),
        "branch_lengths": torch.tensor(
            module._BRANCH_LENGTHS, dtype=torch.int32
        ),
        "group_path_indices": torch.tensor(
            module._GROUP_PATH_INDICES, dtype=torch.int32
        ),
        "group_path_counts": torch.tensor(
            module._GROUP_PATH_COUNTS, dtype=torch.int32
        ),
        "out": torch.empty(
            (rows, module.NUM_V_HEADS, module.DIM_V),
            dtype=torch.bfloat16,
        ),
        "ring_k": torch.empty(
            (rows, module.NUM_K_HEADS, module.DIM_K),
            dtype=torch.bfloat16,
        ),
        "ring_v": torch.empty(
            (rows, module.NUM_V_HEADS, module.DIM_V),
            dtype=torch.bfloat16,
        ),
        "ring_a": torch.empty(
            (rows, module.NUM_V_HEADS), dtype=torch.bfloat16
        ),
        "ring_b": torch.empty(
            (rows, module.NUM_V_HEADS), dtype=torch.bfloat16
        ),
        "flags": torch.zeros((2,), dtype=torch.int32),
        "ring_k_norm": torch.empty(
            (rows, module.NUM_K_HEADS), dtype=torch.float32
        ),
        "ring_gate": torch.empty(
            (rows, module.NUM_V_HEADS, 2), dtype=torch.float32
        ),
        "batch_size": batch,
        "mode": "tail6_fixed32",
        "output_scale": module.DIM_K**-0.5,
        "h0_is_bank": True,
        "h0_index_row": 0,
        "h0_index_batch_stride": 2,
        "h0_batch_index": 0,
        "h0_accepted_batch_stride": 1,
        "h0_bank_stride": int(h0.stride(0)),
        "h0_use_accepted_column": True,
        "use_qk_l2norm_in_kernel": True,
        "raw_gating": True,
        "count_invocation": True,
        "scan_align": False,
        "root_steps": module._ROOT_STEPS,
        "max_path_len": module._MAX_PATH_LEN,
        "max_group_paths": module._MAX_GROUP_PATHS,
        "prescaled_path_base": False,
        "ring_export": True,
        "k_norm_export": True,
        "gate_export": True,
        "decay_export": True,
        "flags_export": True,
        "flags_rows": batch,
    }


def _arm_gdn_cpu_as_cuda(
    module, monkeypatch: pytest.MonkeyPatch
) -> _FakeGdnKernelLaunch:
    launch = _FakeGdnKernelLaunch()
    monkeypatch.setattr(
        torch.Tensor,
        "device",
        property(lambda _self: torch.device("cuda:0")),
    )
    monkeypatch.setattr(
        module,
        "_fr13_fixed32_gdn_gqa_group3_single_launch_kernel",
        launch,
    )
    return launch


@pytest.mark.parametrize("prescaled_path_base", (False, True))
def test_launch_guard_accepts_only_the_exact_qualified_gdn_binding(
    monkeypatch: pytest.MonkeyPatch,
    prescaled_path_base: bool,
) -> None:
    module = _load_gdn_candidate(monkeypatch)
    args = _qualified_gdn_launch_args(module)
    if prescaled_path_base:
        args["prescaled_path_base"] = True
        args["branch_lengths"] = torch.tensor(
            module._PRESCALED_BRANCH_LENGTHS, dtype=torch.int32
        )
        args["group_path_indices"] = torch.tensor(
            module._PRESCALED_GROUP_PATH_BASES, dtype=torch.int32
        )
    launch = _arm_gdn_cpu_as_cuda(module, monkeypatch)

    result = module.launch_fixed32_gdn_gqa_group3_source_candidate(**args)

    assert result["candidate_default_off"] is True
    assert launch.calls == 1


@pytest.mark.parametrize(
    "case",
    (
        "dtype",
        "shape",
        "stride",
        "root_descriptor",
        "branch_descriptor",
        "schedule_extent",
        "bank_index",
        "bank_column",
        "bank_stride",
        "ring_shape",
        "counter_shape",
        "flags_rows",
        "write_overlap",
        "export_dependency",
        "ring_raw_dependency",
    ),
)
def test_launch_guard_fails_closed_before_gdn_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    module = _load_gdn_candidate(monkeypatch)
    args = _qualified_gdn_launch_args(module)
    rows = module.PHYSICAL_ROWS
    if case == "dtype":
        args["k"] = args["k"].to(torch.float32)
    elif case == "shape":
        args["out"] = args["out"][:, :, :-1].contiguous()
    elif case == "stride":
        args["q"] = torch.empty(
            (rows, module.NUM_K_HEADS, 2 * module.DIM_K),
            dtype=torch.bfloat16,
        )[..., ::2]
    elif case == "root_descriptor":
        args["root_nodes"][0, 0] = 1
    elif case == "branch_descriptor":
        args["branch_lengths"][0] = module._MAX_PATH_LEN + 1
    elif case == "schedule_extent":
        args["root_steps"] = module._ROOT_STEPS - 1
    elif case == "bank_index":
        args["h0_indices"][0, 0] = int(args["h0"].shape[0])
    elif case == "bank_column":
        args["h0_num_accepted_tokens"][0] = 3
    elif case == "bank_stride":
        args["h0_bank_stride"] = int(args["h0_bank_stride"]) + 1
    elif case == "ring_shape":
        args["ring_gate"] = args["ring_gate"][:, :-1].contiguous()
    elif case == "counter_shape":
        args["invocation_counter"] = torch.zeros((1,), dtype=torch.int32)
    elif case == "flags_rows":
        args["flags_rows"] = 2
    elif case == "write_overlap":
        args["ring_v"] = args["out"]
    elif case == "export_dependency":
        args["k_norm_export"] = False
    elif case == "ring_raw_dependency":
        args["raw_gating"] = False
    else:  # pragma: no cover - parameter list is closed above
        raise AssertionError(case)
    launch = _arm_gdn_cpu_as_cuda(module, monkeypatch)

    with pytest.raises((TypeError, ValueError)):
        module.launch_fixed32_gdn_gqa_group3_source_candidate(**args)

    assert launch.calls == 0


def test_launch_guard_rejects_cpu_and_cross_device_gdn_operands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_gdn_candidate(monkeypatch)
    args = _qualified_gdn_launch_args(module)
    launch = _FakeGdnKernelLaunch()
    monkeypatch.setattr(
        module,
        "_fr13_fixed32_gdn_gqa_group3_single_launch_kernel",
        launch,
    )
    with pytest.raises(ValueError, match="requires CUDA"):
        module.launch_fixed32_gdn_gqa_group3_source_candidate(**args)

    mismatched = args["ring_gate"]
    monkeypatch.setattr(
        torch.Tensor,
        "device",
        property(
            lambda self: torch.device(
                "cuda:1" if self is mismatched else "cuda:0"
            )
        ),
    )
    with pytest.raises(ValueError, match="share one device"):
        module.launch_fixed32_gdn_gqa_group3_source_candidate(**args)

    assert launch.calls == 0
