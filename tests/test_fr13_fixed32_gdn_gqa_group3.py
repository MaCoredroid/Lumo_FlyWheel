from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = (
    ROOT / "src/lumo_flywheel_serving/fr13_gdn_gqa_group3.py"
)
SERVED_KERNEL = (
    ROOT / "src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py"
)
PATCHER = ROOT / "scripts/fr10_phase4_patch_vllm_tree_gdn.py"
LAUNCHER = ROOT / "scripts/fr13_launch_forked_fa2_tree_server.sh"


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
    candidate_node_visits = candidate * 32 * 48
    assert result["trusted_node_domain"] == (0, 31)
    assert result["source_node_domain_guard_sites_removed_per_visit"] == 4
    assert result["source_node_domain_guard_sites_removed_per_event"] == (
        candidate_node_visits * 4
    )
    assert result["source_node_clamp_sites_removed_per_event"] == (
        candidate_node_visits * 4
    )
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
    assert "node >= 0" not in value_head
    assert "tl.maximum(node, 0)" not in value_head
    assert "n_ok" in value_head
    assert "global_node" in value_head
    assert "return tl.where(n_ok, state_i, prior_state)" in value_head
    assert "if TRUST_FIXED32_NODE_DOMAIN:" in node
    assert "n_ok = True" in node
    assert "global_node = pid_batch * N_ACTUAL + node" in node
    assert "pid_kh = tl.program_id(0)" in kernel
    assert "pid_vh_0 = pid_kh * HEAD_GROUP" in kernel
    assert "tl.arange(0, HEAD_GROUP)" not in kernel
    assert "for root_index in tl.range(0, ROOT_STEPS):" in kernel
    assert "for member in tl.static_range(0, MAX_GROUP_PATHS):" in kernel
    assert "for path_offset in tl.range(0, path_len):" in kernel
    assert kernel.count("_fr13_fixed32_gdn_gqa_group3_node(") == 2
    assert "if H0_IS_BANK:" in kernel
    assert "grid = (NUM_K_HEADS, DIM_V // BLOCK_V, int(batch_size))" in launch
    assert 'launch_options = {"num_warps": 8}' in launch
    assert 'int(maxnreg) != 128' in launch
    assert 'launch_options["maxnreg"] = int(maxnreg)' in launch
    assert "descriptor_execution_sha256 != FIXED32_EXECUTION_SHA256" in launch
    assert "physical32 descriptor provenance drift" in launch
    assert "expected_descriptor_numels" in launch
    assert "immutable physical32 descriptor drift" in launch
    assert "TRUST_FIXED32_NODE_DOMAIN=True" in launch


def test_candidate_is_default_off_and_gate_wired_without_serving() -> None:
    served = SERVED_KERNEL.read_text(encoding="utf-8")
    patcher = PATCHER.read_text(encoding="utf-8")
    launcher = LAUNCHER.read_text(encoding="utf-8")
    _tree, source = _tree_and_source()

    assert "default-off ``gqa_group3`` live" in source
    assert '== _FR13_FIXED32_GDN_GQA_GROUP3_GATE_VALUE' in served
    assert "_FR13_FIXED32_GDN_GQA_GROUP3_LAUNCH = None" in served
    assert "launch_fixed32_gdn_gqa_group3_source_candidate" in served
    assert served.count("descriptor_execution_sha256=str(") == 2
    assert served.count('["execution_sha256"]') >= 2
    assert '"gqa_group3"' in patcher
    assert '"gqa_group3"' in launcher
    assert '"candidate_served": False' in served


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
    assert "node" not in signature
    assert "N_ACTUAL" not in signature
    assert "n_ok" in signature
    assert "global_node" in signature
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
        expected_positionals[15] = f"pid_vh_{sibling}"
        assert actual_positionals == expected_positionals
