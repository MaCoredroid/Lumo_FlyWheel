from __future__ import annotations

import ast
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNEL_PATH = ROOT / "src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py"
SOURCE = KERNEL_PATH.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def _function(name: str) -> ast.FunctionDef:
    return next(
        node
        for node in TREE.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _text(name: str) -> str:
    node = _function(name)
    lines = SOURCE.splitlines()
    return "\n".join(lines[node.lineno - 1 : node.end_lineno])


def test_direct_metadata_arm_is_explicit_and_default_off(monkeypatch) -> None:
    node = _function("_fr13_fixed32_committer_direct_metadata_requested")
    namespace = {"os": os}
    exec(
        compile(
            ast.Module(body=[node], type_ignores=[]),
            str(KERNEL_PATH),
            "exec",
        ),
        namespace,
    )
    requested = namespace[node.name]

    monkeypatch.delenv("FR13_FIXED32_COMMITTER_DIRECT_METADATA", raising=False)
    monkeypatch.setattr(os.path, "exists", lambda _path: False)
    assert requested() is False

    monkeypatch.setenv("FR13_FIXED32_COMMITTER_DIRECT_METADATA", "1")
    assert requested() is True
    monkeypatch.delenv("FR13_FIXED32_COMMITTER_DIRECT_METADATA")
    monkeypatch.setattr(
        os.path,
        "exists",
        lambda path: path
        == "/logs/fr13_fixed32_committer_direct_metadata.arm",
    )
    assert requested() is True


def test_preseed_captures_direct_persistent_taw_metadata() -> None:
    preseed = _text("preseed_fixed32_committer_graph")
    body = _text("_fr13_fixed32_committer_graph_body")
    launch = _text("_fr13_fixed32_committer_native_layer_batch")

    assert "direct_metadata and not layer_batch" in preseed
    assert "direct_metadata and metadata_copy_fusion" in preseed
    assert '"direct_accepted_paths": accepted_paths if direct_metadata else None' in preseed
    assert '"direct_accepted_lens": accepted_lens if direct_metadata else None' in preseed
    assert '"metadata_roundtrip_elements_per_request": (' in preseed
    assert '"persistent_taw_publish_buffers"' in preseed
    assert 'state["direct_accepted_paths"]' in body
    assert 'state["direct_accepted_lens"]' in body
    assert 'state["direct_accepted_paths"]' in launch
    assert 'state["direct_accepted_lens"]' in launch


def test_direct_route_selects_smaller_col0_kernel_and_one_shot_lease() -> None:
    conv = _text("launch_fixed32_conv_commit_to_col0")
    start = conv.index("elif direct_metadata is not None:")
    direct = conv[start : conv.index("elif committer_state is None:", start)]

    assert conv.index("validate_fixed32_conv_commit_rows(") < start
    assert "_launch_direct(" in direct
    assert "_fr13_fixed32_conv_direct_col0_metadata_kernel" not in direct
    assert "direct_accepted_paths" in direct
    assert "direct_accepted_lens" in direct
    assert "_fr13_fixed32_committer_publish_direct_metadata_lease(" in direct
    assert "direct_metadata_published_by_batch" in direct


def test_direct_replay_requires_exact_preceding_guard_lease_without_copy() -> None:
    replay = _text("_fr13_fixed32_committer_replay")
    start = replay.index('if state.get("direct_metadata", False):')
    direct = replay[start : replay.index(
        'elif state.get("metadata_copy_fusion", False):', start
    )]
    fast = _text("_fr13_fixed32_committer_fast_state")

    assert "_fr13_fixed32_committer_consume_direct_metadata_lease(" in direct
    assert "requires the preceding guarded" in direct
    assert '"conv lease"' in direct
    assert ".copy_(accepted_paths)" not in direct
    assert ".copy_(accepted_lens)" not in direct
    assert "direct_metadata_consumed_by_batch" in direct
    assert 'state["direct_accepted_paths"].data_ptr()' in fast
    assert 'state["direct_accepted_lens"].data_ptr()' in fast


def test_direct_resolver_pins_capacity_storage_and_batch_view() -> None:
    resolver = _text("_fr13_fixed32_committer_direct_metadata_state")

    assert 'route.get("accepted_paths_data_ptr", -1)' in resolver
    assert 'route.get("accepted_lens_data_ptr", -1)' in resolver
    assert 'tuple(graph_paths.shape) != (int(batch), 16)' in resolver
    assert 'tuple(graph_lens.shape) != (int(batch),)' in resolver
    assert "int(graph_paths.data_ptr()) != int(accepted_paths.data_ptr())" in resolver
    assert "int(graph_lens.data_ptr()) != int(accepted_lens.data_ptr())" in resolver
    assert "validation_bank_rows = min(" in resolver
