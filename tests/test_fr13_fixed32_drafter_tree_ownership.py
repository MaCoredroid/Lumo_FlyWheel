from __future__ import annotations

import ast
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest


PATCHER_PATH = Path("scripts/fr10_phase4_patch_vllm_tree_gdn.py").resolve()
SCRIPTS_DIR = PATCHER_PATH.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

SPEC = importlib.util.spec_from_file_location(
    "fr13_fixed32_drafter_tree_patcher", PATCHER_PATH
)
assert SPEC is not None and SPEC.loader is not None
PATCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCHER)

census = __import__("fr13_fixed32_work_census")


DRAFTER_LAYER = "mtp.layers.0.self_attn.attn"
TARGET_LAYERS = tuple(
    f"language_model.model.layers.{layer}.self_attn.attn"
    for layer in range(3, 64, 4)
)


def _runtime() -> dict[str, object]:
    namespace: dict[str, object] = {
        "_FR13_FIXED32_MODE": "tail6_fixed32",
    }
    exec(PATCHER._FR13_FIXED32_OBSERVED_RUNTIME_SOURCE, namespace)
    return namespace


def _open_drafter_capture(
    batch_size: int, graph_id: int
) -> dict[str, object]:
    namespace = _runtime()
    request_ids = tuple(f"request-{row}" for row in range(batch_size))
    namespace["_fr13_fixed32_drafter_proposal_begin"](
        "tail6_fixed32",
        request_ids,
        batch_size,
        batch_size,
        batch_size,
    )
    namespace["_fr13_fixed32_drafter_graph_capture_begin"](
        graph_id, batch_size
    )
    return namespace


@pytest.mark.parametrize("batch_size", (1, 2, 3, 4))
def test_drafter_tree_attention_is_owned_and_signed(batch_size: int) -> None:
    graph_id = 40_000 + batch_size
    namespace = _open_drafter_capture(batch_size, graph_id)

    for _ in range(4):
        namespace["_fr13_fixed32_observed_tree_attn"](
            DRAFTER_LAYER,
            batch_size,
            (1, 1),
            True,
        )
        namespace["_fr13_fixed32_drafter_mtp_forward"](batch_size, True)

    context = namespace["_FR13_FIXED32_DRAFTER_GRAPH_CAPTURE_CONTEXT"]
    assert context["tree_attn_calls"] == 4
    assert context["tree_attn_rows"] == 4 * batch_size
    assert context["mtp_forward_calls"] == 4
    assert context["mtp_forward_rows"] == 4 * batch_size
    assert namespace["_FR13_FIXED32_CAPTURE_CONTEXT"] is None
    assert namespace["_FR13_FIXED32_OBSERVED_CURRENT"] is None

    signature = namespace["_fr13_fixed32_drafter_graph_capture_end"](
        graph_id, batch_size
    )
    stored_signature, canonical = namespace[
        "_FR13_FIXED32_DRAFTER_GRAPH_MANIFESTS"
    ][graph_id]
    manifest = json.loads(canonical)

    assert stored_signature == signature
    assert signature == census._drafter_graph_signature(batch_size)
    assert manifest == {
        "schema": "fr13-fixed32-drafter-graph-manifest-v2",
        "batch_size": batch_size,
        "mtp_forward_calls": 4,
        "mtp_forward_rows": 4 * batch_size,
        "tree_attn_calls": 4,
        "tree_attn_rows": 4 * batch_size,
        "tree_attn_layer": DRAFTER_LAYER,
        "tree_attn_bias_shape": [1, 1],
    }

    namespace["_fr13_fixed32_drafter_graph_replay"](
        graph_id, signature, batch_size
    )
    proposal = namespace["_FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT"]
    assert proposal["mtp_execution_basis"] == "cudagraph_replay"
    assert proposal["mtp_forward_calls"] == 4
    assert proposal["mtp_forward_rows"] == 4 * batch_size


def test_unscoped_one_row_attention_remains_invalid_for_target() -> None:
    namespace = _runtime()

    with pytest.raises(RuntimeError, match="rows are not fixed32"):
        namespace["_fr13_fixed32_observed_tree_attn"](
            DRAFTER_LAYER, 1, (1, 1), True
        )


@pytest.mark.parametrize("batch_size", (1, 2, 3, 4))
def test_target_tree_attention_requires_exact_layer_set(
    batch_size: int,
) -> None:
    namespace = _runtime()
    namespace["_fr13_fixed32_capture_begin"](
        40_100 + batch_size,
        "FULL",
        32 * batch_size,
        batch_size,
        True,
        False,
        0,
    )

    for layer_name in TARGET_LAYERS:
        namespace["_fr13_fixed32_observed_tree_attn"](
            layer_name,
            32 * batch_size,
            (32, 32),
            True,
        )

    work = namespace["_FR13_FIXED32_CAPTURE_CONTEXT"]["work"]
    assert work["tree_layers"] == set(TARGET_LAYERS)
    assert work["tree_calls"] == 16
    assert work["tree_q_rows"] == 16 * 32 * batch_size
    assert work["tree_bias_shape"] == (32, 32)


@pytest.mark.parametrize(
    "layer_name",
    (
        "attn.0",
        "language_model.model.layers.4.self_attn.attn",
        "language_model.model.layers.3.self_attn",
        DRAFTER_LAYER,
    ),
)
def test_target_correct_geometry_wrong_layer_fails(layer_name: str) -> None:
    namespace = _runtime()
    namespace["_fr13_fixed32_capture_begin"](
        40_200,
        "FULL",
        32,
        1,
        True,
        False,
        0,
    )

    with pytest.raises(RuntimeError, match="tree-attention work drift"):
        namespace["_fr13_fixed32_observed_tree_attn"](
            layer_name,
            32,
            (32, 32),
            True,
        )


@pytest.mark.parametrize(
    ("layer_name", "rows", "bias_shape"),
    (
        ("language_model.model.layers.3.self_attn.attn", 1, (1, 1)),
        (DRAFTER_LAYER, 2, (1, 1)),
        (DRAFTER_LAYER, 1, (32, 32)),
    ),
)
def test_scoped_drafter_geometry_drift_fails(
    layer_name: str,
    rows: int,
    bias_shape: tuple[int, int],
) -> None:
    namespace = _open_drafter_capture(1, 41_000)

    with pytest.raises(RuntimeError, match="drafter tree-attention work drift"):
        namespace["_fr13_fixed32_observed_tree_attn"](
            layer_name, rows, bias_shape, True
        )


def test_partial_and_overlapping_owner_scopes_fail() -> None:
    proposal_only = _runtime()
    proposal_only["_fr13_fixed32_drafter_proposal_begin"](
        "tail6_fixed32", ("proposal-only",), 1, 1, 1
    )
    with pytest.raises(RuntimeError, match="ownership scope drift"):
        proposal_only["_fr13_fixed32_observed_tree_attn"](
            DRAFTER_LAYER, 1, (1, 1), True
        )

    capture_only = _open_drafter_capture(1, 42_001)
    capture_only["_FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT"] = None
    with pytest.raises(RuntimeError, match="ownership scope drift"):
        capture_only["_fr13_fixed32_observed_tree_attn"](
            DRAFTER_LAYER, 1, (1, 1), True
        )

    overlap = _open_drafter_capture(1, 42_002)
    overlap["_FR13_FIXED32_CAPTURE_CONTEXT"] = {}
    with pytest.raises(RuntimeError, match="ownership scope drift"):
        overlap["_fr13_fixed32_observed_tree_attn"](
            DRAFTER_LAYER, 1, (1, 1), True
        )


def test_owner_lifecycle_rejects_cross_scope_begins() -> None:
    target_capture = _runtime()
    target_capture["_FR13_FIXED32_CAPTURE_CONTEXT"] = {}
    with pytest.raises(RuntimeError, match="proposal begin drift"):
        target_capture["_fr13_fixed32_drafter_proposal_begin"](
            "tail6_fixed32", ("target-open",), 1, 1, 1
        )

    drafter_proposal = _runtime()
    drafter_proposal["_fr13_fixed32_drafter_proposal_begin"](
        "tail6_fixed32", ("drafter-open",), 1, 1, 1
    )
    with pytest.raises(RuntimeError, match="capture/recompile"):
        drafter_proposal["_fr13_fixed32_capture_begin"](
            43_001, "FULL", 32, 1, True, False, 0
        )


def test_drafter_ownership_rejects_profile_scope_overlap() -> None:
    for profile_capture, profile_memory in (
        (None, True),
        ({}, False),
        ({}, True),
    ):
        namespace = _runtime()
        namespace["_FR13_FIXED32_PROFILE_CAPTURE_SCOPE"] = profile_capture
        namespace["_FR13_FIXED32_PROFILE_MEMORY_SCOPE"] = profile_memory
        with pytest.raises(RuntimeError, match="proposal begin drift"):
            namespace["_fr13_fixed32_drafter_proposal_begin"](
                "tail6_fixed32", ("profile-open",), 1, 1, 1
            )

    proposal_first = _runtime()
    proposal_first["_fr13_fixed32_drafter_proposal_begin"](
        "tail6_fixed32", ("proposal-first",), 1, 1, 1
    )
    with pytest.raises(RuntimeError, match="outside pristine bootstrap"):
        proposal_first["_fr13_fixed32_profile_memory_scope_begin"]()

    profile_capture_begin = _runtime()
    profile_capture_begin["_FR13_FIXED32_PROFILE_MEMORY_SCOPE"] = True
    profile_capture_begin["_FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT"] = {}
    with pytest.raises(RuntimeError, match="outside pristine bootstrap"):
        profile_capture_begin["_fr13_fixed32_profile_capture_scope_begin"](
            "FULL", 32, 1, True, False, 0
        )

    profile_memory_end = _runtime()
    profile_memory_end["_FR13_FIXED32_PROFILE_MEMORY_SCOPE"] = True
    profile_memory_end["_FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT"] = {}
    with pytest.raises(RuntimeError, match="did not close cleanly"):
        profile_memory_end["_fr13_fixed32_profile_memory_scope_end"]()

    capture_begin = _runtime()
    capture_begin["_fr13_fixed32_drafter_proposal_begin"](
        "tail6_fixed32", ("capture-profile",), 1, 1, 1
    )
    capture_begin["_FR13_FIXED32_PROFILE_CAPTURE_SCOPE"] = {}
    with pytest.raises(RuntimeError, match="lazy/duplicate drafter graph capture"):
        capture_begin["_fr13_fixed32_drafter_graph_capture_begin"](43_100, 1)

    observer = _open_drafter_capture(1, 43_101)
    observer["_FR13_FIXED32_PROFILE_CAPTURE_SCOPE"] = {}
    with pytest.raises(RuntimeError, match="ownership scope drift"):
        observer["_fr13_fixed32_observed_tree_attn"](
            DRAFTER_LAYER, 1, (1, 1), True
        )

    profile_end = _runtime()
    profile_end["_FR13_FIXED32_PROFILE_MEMORY_SCOPE"] = True
    profile_end["_FR13_FIXED32_PROFILE_CAPTURE_SCOPE"] = {
        "descriptor": {
            "runtime_mode": "FULL",
            "num_tokens": 32,
            "num_reqs": 1,
            "uniform": True,
            "has_lora": False,
            "num_active_loras": 0,
        },
        "graph_id": 43_102,
        "completed": True,
    }
    profile_end["_FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT"] = {}
    with pytest.raises(RuntimeError, match="did not close cleanly"):
        profile_end["_fr13_fixed32_profile_capture_scope_end"]()


def _emitted_tree_attention_hook():
    patcher_tree = ast.parse(PATCHER_PATH.read_text())
    patch_function = next(
        node
        for node in patcher_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_patch_tree_attn_op_capture"
    )
    helper_source = next(
        node.value.value
        for node in ast.walk(patch_function)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "helper"
            for target in node.targets
        )
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
        and "def _fr13_tree_attn_op_capture(" in node.value.value
    )
    helper_tree = ast.parse(helper_source)
    hook_node = next(
        node
        for node in helper_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_fr13_tree_attn_op_capture"
    )
    hook_module = ast.fix_missing_locations(
        ast.Module(body=[hook_node], type_ignores=[])
    )
    return compile(hook_module, "<emitted-tree-attn-hook>", "exec")


@pytest.mark.parametrize("batch_size", (1, 2, 3, 4))
def test_emitted_hook_feeds_drafter_and_target_geometry(
    batch_size: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int, tuple[int, int], bool]] = []
    state = {"capturing": True, "event_active": False}
    fake_gdn = types.SimpleNamespace(
        _FR13_FIXED32_MODE="tail6_fixed32",
        _fr13_fixed32_observed_event_active=lambda: state["event_active"],
        _fr13_fixed32_observed_tree_attn=lambda *args: calls.append(args),
    )
    packages = {
        "vllm": types.ModuleType("vllm"),
        "vllm.model_executor": types.ModuleType("vllm.model_executor"),
        "vllm.model_executor.layers": types.ModuleType(
            "vllm.model_executor.layers"
        ),
        "vllm.model_executor.layers.mamba": types.ModuleType(
            "vllm.model_executor.layers.mamba"
        ),
    }
    for package in packages.values():
        package.__path__ = []
    packages[
        "vllm.model_executor.layers.mamba"
    ].gdn_linear_attn = fake_gdn
    for name, package in packages.items():
        monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.delenv("FR13_TREE_ATTN_OP_CAPTURE", raising=False)
    monkeypatch.setenv("FR13_FA2_MAB", "0")

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            is_current_stream_capturing=lambda: state["capturing"],
        )
    )
    namespace = {"os": os, "torch": fake_torch}
    exec(_emitted_tree_attention_hook(), namespace)
    hook = namespace["_fr13_tree_attn_op_capture"]
    layer = types.SimpleNamespace(layer_name=DRAFTER_LAYER)
    query = types.SimpleNamespace(shape=(batch_size, 8, 128))
    metadata = types.SimpleNamespace(
        tree_attn_bias=types.SimpleNamespace(shape=(1, 1))
    )

    hook(None, layer, query, None, None, None, None, None, metadata)
    assert calls == [(DRAFTER_LAYER, batch_size, (1, 1), True)]

    calls.clear()
    state.update(capturing=False, event_active=True)
    target_layer = "language_model.model.layers.3.self_attn.attn"
    target_query = types.SimpleNamespace(shape=(32 * batch_size, 8, 128))
    target_metadata = types.SimpleNamespace(
        tree_attn_bias=types.SimpleNamespace(shape=(32, 32))
    )
    hook(
        None,
        types.SimpleNamespace(layer_name=target_layer),
        target_query,
        None,
        None,
        None,
        None,
        None,
        target_metadata,
    )
    assert calls == [
        (target_layer, 32 * batch_size, (32, 32), False)
    ]

    calls.clear()
    state["event_active"] = False
    hook(
        None,
        types.SimpleNamespace(layer_name=target_layer),
        target_query,
        None,
        None,
        None,
        None,
        None,
        target_metadata,
    )
    assert calls == []
