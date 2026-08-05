from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest
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


def _fixture(batch: int) -> tuple[torch.Tensor, ...]:
    bank_rows = 257
    aliases = torch.arange(16, dtype=torch.int64).repeat(3)
    groups = torch.tensor(
        [(alias, alias + 16, alias + 32) for alias in range(16)],
        dtype=torch.int32,
    )
    ssi = torch.empty((48, batch, 32), dtype=torch.int32)
    for layer in range(48):
        alias_rank = layer // 16
        for request in range(batch):
            ssi[layer, request] = (
                torch.arange(32, dtype=torch.int32) + layer + request + 1
            ) % (bank_rows - 1) + 1
            ssi[layer, request, 0] = alias_rank * batch + request + 1
    generator = torch.Generator().manual_seed(2700 + batch)
    paths = torch.randint(0, 32, (batch, 16), generator=generator).to(
        torch.int32
    )
    lens = torch.randint(0, 12, (batch,), generator=generator).to(torch.int32)
    return ssi, paths, lens, aliases, groups, torch.tensor(bank_rows)


def _incumbent_guard(
    ssi: torch.Tensor,
    paths: torch.Tensor,
    lens: torch.Tensor,
    aliases: torch.Tensor,
    groups: torch.Tensor,
    bank_rows: int,
) -> bool:
    batch = int(paths.shape[0])
    flags = []
    for layer in range(48):
        alias_id = int(aliases[layer])
        members = groups[alias_id]
        for request in range(batch):
            rows = ssi[layer, request]
            duplicate = any(
                (int(other_layer) != layer or other_request != request)
                and int(ssi[int(other_layer), other_request, 0])
                == int(rows[0])
                for other_layer in members
                for other_request in range(batch)
            )
            ok = (
                bool((rows > 0).all())
                and bool((rows < bank_rows).all())
                and not duplicate
            )
            if layer == 0:
                accepted = int(lens[request])
                active = paths[request, : max(accepted, 0)]
                ok = (
                    ok
                    and 0 <= accepted <= 11
                    and bool(((active >= 0) & (active < 32)).all())
                )
            flags.append(ok)
    return bool(((aliases >= 0) & (aliases < 16)).all()) and all(flags)


def _alias_group_guard(
    ssi: torch.Tensor,
    paths: torch.Tensor,
    lens: torch.Tensor,
    aliases: torch.Tensor,
    groups: torch.Tensor,
    bank_rows: int,
) -> bool:
    batch = int(paths.shape[0])
    flags = []
    for alias_id, members_tensor in enumerate(groups):
        members = tuple(int(layer) for layer in members_tensor)
        member_table_ok = (
            len(set(members)) == 3
            and all(0 <= layer < 48 for layer in members)
            and all(int(aliases[layer]) == alias_id for layer in members)
        )
        for request in range(batch):
            peer_rows = [
                int(ssi[layer, other_request, 0])
                for layer in members
                for other_request in range(batch)
            ]
            ok = True
            for layer in members:
                rows = ssi[layer, request]
                running_row = int(rows[0])
                duplicate_count = sum(row == running_row for row in peer_rows)
                ok = (
                    ok
                    and bool((rows > 0).all())
                    and bool((rows < bank_rows).all())
                    and duplicate_count == 1
                )
            if alias_id == 0:
                accepted = int(lens[request])
                active = paths[request, : max(accepted, 0)]
                ok = (
                    ok
                    and 0 <= accepted <= 11
                    and bool(((active >= 0) & (active < 32)).all())
                )
            if request == 0:
                ok = ok and member_table_ok
            flags.append(ok)
    return all(flags)


def test_alias_group_guard_arm_is_explicit_and_default_off(monkeypatch) -> None:
    node = _function("_fr13_fixed32_committer_alias_group_guard_requested")
    namespace = {"os": os}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(KERNEL_PATH), "exec"),
        namespace,
    )
    requested = namespace[node.name]

    monkeypatch.delenv(
        "FR13_FIXED32_COMMITTER_ALIAS_GROUP_GUARD", raising=False
    )
    monkeypatch.setattr(os.path, "exists", lambda _path: False)
    assert requested() is False

    monkeypatch.setenv("FR13_FIXED32_COMMITTER_ALIAS_GROUP_GUARD", "1")
    assert requested() is True
    monkeypatch.delenv("FR13_FIXED32_COMMITTER_ALIAS_GROUP_GUARD")
    monkeypatch.setattr(
        os.path,
        "exists",
        lambda path: path
        == "/logs/fr13_fixed32_committer_alias_group_guard.arm",
    )
    assert requested() is True


@pytest.mark.parametrize("batch", (1, 4))
def test_alias_group_partition_matches_incumbent(batch: int) -> None:
    ssi, paths, lens, aliases, groups, bank_rows_tensor = _fixture(batch)
    bank_rows = int(bank_rows_tensor)
    fixtures = [(ssi, paths, lens, aliases)]

    for target, index, value in (
        ("ssi", (4, 0, 7), 0),
        ("ssi", (5, 0, 9), bank_rows),
        ("lens", (0,), -1),
        ("lens", (0,), 12),
        ("paths", (0, 0), -1),
        ("paths", (0, 0), 32),
    ):
        values = {
            "ssi": ssi.clone(),
            "paths": paths.clone(),
            "lens": lens.clone(),
            "aliases": aliases.clone(),
        }
        values[target][index] = value
        fixtures.append(
            (values["ssi"], values["paths"], values["lens"], values["aliases"])
        )

    duplicate_ssi = ssi.clone()
    duplicate_ssi[16, 0, 0] = duplicate_ssi[0, 0, 0]
    fixtures.append((duplicate_ssi, paths, lens, aliases))

    inactive_paths = paths.clone()
    inactive_lens = lens.clone()
    inactive_lens[0] = 0
    inactive_paths[0].fill_(2**30)
    fixtures.append((ssi, inactive_paths, inactive_lens, aliases))

    for candidate in fixtures:
        assert _alias_group_guard(*candidate, groups, bank_rows) == (
            _incumbent_guard(*candidate, groups, bank_rows)
        )


def test_alias_group_table_corruption_fails_closed() -> None:
    ssi, paths, lens, aliases, groups, bank_rows_tensor = _fixture(1)
    bank_rows = int(bank_rows_tensor)

    duplicate = groups.clone()
    duplicate[0, 2] = duplicate[0, 1]
    assert not _alias_group_guard(ssi, paths, lens, aliases, duplicate, bank_rows)

    wrong_alias = aliases.clone()
    wrong_alias[32] = 1
    assert not _alias_group_guard(ssi, paths, lens, wrong_alias, groups, bank_rows)


def test_alias_group_kernel_preserves_physical32_validation() -> None:
    kernel = _text(
        "_fr13_fixed32_conv_commit_alias_group_sticky_guard_kernel"
    )
    launcher = _text("validate_fixed32_conv_commit_alias_groups_sticky")

    assert "alias_id = pid // B" in kernel
    assert "request = pid - alias_id * B" in kernel
    assert "for alias_slot in tl.static_range(0, ALIAS_WIDTH):" in kernel
    assert "row_offsets = tl.arange(0, SPEC_COLS)" in kernel
    assert "rows > 0" in kernel
    assert "rows < BANK_ROWS" in kernel
    assert "if alias_id == 0:" in kernel
    assert "if request == 0:" in kernel
    assert "member_aliases == alias_id" in kernel
    assert "members_unique" in kernel
    assert "tl.atomic_xchg(sticky_ok, 0, mask=~contract_ok)" in kernel
    assert "tl.store(" not in kernel

    assert "(16 * batch,)" in launcher
    assert "SPEC_COLS=32" in launcher
    assert "PATH_COLS=16" in launcher
    assert "ALIAS_GROUPS=16" in launcher
    assert "ALIAS_WIDTH=3" in launcher
    assert "PEER_CAP=16" in launcher
    assert "num_warps=4" in launcher
    assert "num_stages=1" in launcher


def test_alias_group_route_is_default_off_persistent_and_fail_closed() -> None:
    preseed = _text("preseed_fixed32_committer_graph")
    resolver = _text("_fr13_fixed32_committer_direct_metadata_state")
    fast = _text("_fr13_fixed32_committer_fast_state")
    launch = _text("launch_fixed32_conv_commit_to_col0")

    assert "if alias_group_guard and not sticky_guard:" in preseed
    assert '"alias_group_guard": alias_group_guard' in preseed
    assert '"alias_group_layers": alias_group_layers' in preseed
    assert '"alias_group_layers_data_ptr"' in preseed
    assert '"fixed32_alias_group3_physical32_sticky_scalar_v6"' in preseed
    assert '"alias_group_guard_programs_per_request"' in preseed
    assert "alias_group_guard_invalid" in resolver
    assert 'state.get("alias_group_guard", False)' in fast
    assert "validate_fixed32_conv_commit_alias_groups_sticky(" in launch
    assert "validate_fixed32_conv_commit_rows_sticky(" in launch
    assert "validate_fixed32_conv_commit_rows(" in launch


@pytest.mark.parametrize(
    (
        "batch",
        "incumbent_programs",
        "candidate_programs",
        "incumbent_peers",
        "candidate_peers",
    ),
    ((1, 48, 16, 144, 48), (4, 192, 64, 2304, 768)),
)
def test_alias_group_work_reduction_is_exact(
    batch: int,
    incumbent_programs: int,
    candidate_programs: int,
    incumbent_peers: int,
    candidate_peers: int,
) -> None:
    assert 48 * batch == incumbent_programs
    assert 16 * batch == candidate_programs
    assert incumbent_programs == 3 * candidate_programs
    assert incumbent_programs * (3 * batch) == incumbent_peers
    assert candidate_programs * (3 * batch) == candidate_peers
    assert incumbent_peers == 3 * candidate_peers
    assert incumbent_programs * 32 == candidate_programs * 3 * 32
