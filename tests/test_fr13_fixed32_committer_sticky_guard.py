from __future__ import annotations

import ast
import os
from pathlib import Path

import torch


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


def test_sticky_guard_arm_is_explicit_and_default_off(monkeypatch) -> None:
    node = _function("_fr13_fixed32_committer_sticky_guard_requested")
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

    monkeypatch.delenv("FR13_FIXED32_COMMITTER_STICKY_GUARD", raising=False)
    monkeypatch.setattr(os.path, "exists", lambda _path: False)
    assert requested() is False

    monkeypatch.setenv("FR13_FIXED32_COMMITTER_STICKY_GUARD", "1")
    assert requested() is True
    monkeypatch.delenv("FR13_FIXED32_COMMITTER_STICKY_GUARD")
    monkeypatch.setattr(
        os.path,
        "exists",
        lambda path: path
        == "/logs/fr13_fixed32_committer_sticky_guard.arm",
    )
    assert requested() is True


def test_sticky_kernel_preserves_exact_ownerpath_validation_body() -> None:
    incumbent = _text("_fr13_fixed32_conv_commit_row_guard_kernel")
    candidate = _text("_fr13_fixed32_conv_commit_sticky_guard_kernel")

    incumbent_body = incumbent[
        incumbent.index("    pid = tl.program_id(0)") : incumbent.index(
            "    tl.store(guard_flags + pid, contract_ok)"
        )
    ]
    candidate_body = candidate[
        candidate.index("    pid = tl.program_id(0)") : candidate.index(
            "    # The scalar starts at one"
        )
    ]
    assert candidate_body == incumbent_body
    assert "tl.atomic_xchg(sticky_ok, 0, mask=~contract_ok)" in candidate
    assert "tl.store(" not in candidate


def test_sticky_launcher_removes_only_the_scalar_reduction() -> None:
    incumbent = _text("validate_fixed32_conv_commit_rows")
    candidate = _text("validate_fixed32_conv_commit_rows_sticky")

    assert "_fr13_fixed32_conv_commit_row_guard_kernel[(48 * batch,)]" in incumbent
    assert "guard_flags.all()" in incumbent
    assert (
        "_fr13_fixed32_conv_commit_sticky_guard_kernel[(48 * batch,)]"
        in candidate
    )
    assert "_fr13_fixed32_device_assert(\n        sticky_ok," in candidate
    assert ".all()" not in candidate
    assert "torch.sort" not in candidate
    assert "torch.arange" not in candidate
    assert "torch.where" not in candidate


def test_sticky_route_keeps_incumbent_fallback_and_binds_direct_lease() -> None:
    preseed = _text("preseed_fixed32_committer_graph")
    launch = _text("launch_fixed32_conv_commit_to_col0")
    replay = _text("_fr13_fixed32_committer_replay")
    resolver = _text("_fr13_fixed32_committer_direct_metadata_state")
    fast = _text("_fr13_fixed32_committer_fast_state")

    assert "sticky_guard and not direct_metadata" in preseed
    assert "torch.ones((), dtype=torch.int32, device=device)" in preseed
    assert '"sticky_guard_ok_data_ptr"' in preseed
    assert '"fixed32_ownerpath_warp32_sticky_scalar_v5"' in preseed
    assert '"sticky_guard_scalar_reduction_launches_per_event": 0' in preseed
    assert '"sticky_guard_valid_event_global_stores": 0' in preseed

    guard_start = launch.index("    alias_group_guard = bool(")
    conv_start = launch.index("    conv_c =", guard_start)
    guard_route = launch[guard_start:conv_start]
    assert "if alias_group_guard:" in guard_route
    assert "validate_fixed32_conv_commit_alias_groups_sticky(" in guard_route
    assert "elif sticky_guard:" in guard_route
    assert "validate_fixed32_conv_commit_rows_sticky(" in guard_route
    assert "validate_fixed32_conv_commit_rows(" in guard_route
    assert "validation_guard=(" in launch
    assert 'committer_state["sticky_guard_ok"]' in launch
    assert "validation_guard=(" in replay
    assert 'state["sticky_guard_ok"]' in replay

    for text in (resolver, fast):
        assert 'state.get("sticky_guard_ok_data_ptr", -1)' in text
        assert 'state["sticky_guard_ok"].dtype != torch.int32' in text or (
            "sticky_ok.dtype != torch.int32" in text
        )
        assert "tuple(sticky_ok.shape) != ()" in text or (
            'tuple(state["sticky_guard_ok"].shape) != ()' in text
        )


def test_sticky_lease_key_is_pointer_batch_and_stream_exact() -> None:
    node = _function("_fr13_fixed32_committer_metadata_lease_key")
    namespace = {"torch": torch}
    exec(
        compile(
            ast.Module(body=[node], type_ignores=[]),
            str(KERNEL_PATH),
            "exec",
        ),
        namespace,
    )
    key_fn = namespace[node.name]
    ssi = torch.zeros((48, 4, 32), dtype=torch.int32)
    paths = torch.zeros((4, 16), dtype=torch.int32)
    lens = torch.zeros((4,), dtype=torch.int32)
    sticky_a = torch.ones((), dtype=torch.int32)
    sticky_b = torch.ones((), dtype=torch.int32)

    kwargs = {
        "batch": 4,
        "spec_state_indices": ssi,
        "accepted_paths": paths,
        "accepted_lens": lens,
        "committer_paths": paths,
        "committer_lens": lens,
        "validation_bank_rows": 257,
        "validation_guard": sticky_a,
        "stream_key": ("cuda:0", 17),
    }
    key = key_fn(**kwargs)
    assert key[1] == 4
    assert key[3] == ("cuda:0", 17)
    assert key[-1][0] == "sticky_guard_v1"
    assert key[-1][1] == sticky_a.data_ptr()

    assert key_fn(**{**kwargs, "validation_guard": sticky_b}) != key
    assert key_fn(**{**kwargs, "stream_key": ("cuda:0", 18)}) != key
    assert key_fn(**{**kwargs, "batch": 1}) != key
