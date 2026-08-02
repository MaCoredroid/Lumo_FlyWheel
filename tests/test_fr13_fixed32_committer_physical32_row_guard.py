from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import fr13_fixed32_work_census as census  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
KERNEL_PATH = ROOT / "src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py"
PATCHER_PATH = ROOT / "scripts/fr10_phase4_patch_vllm_tree_gdn.py"
SOURCE = KERNEL_PATH.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def _function_text(name: str) -> str:
    node = next(
        item
        for item in TREE.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    lines = SOURCE.splitlines()
    return "\n".join(lines[node.lineno - 1 : node.end_lineno])


def _legacy_guard(
    ssi: torch.Tensor,
    paths: torch.Tensor,
    lens: torch.Tensor,
    aliases: torch.Tensor,
    bank_rows: int,
) -> bool:
    batch = int(paths.shape[0])
    active_rows = ssi[:, :batch, :].to(torch.long)
    running_rows = active_rows[:, :, 0]
    destinations = (
        aliases.view(48, 1) * bank_rows + running_rows
    ).reshape(-1)
    sorted_destinations = torch.sort(destinations).values
    distinct = bool(
        (sorted_destinations[1:] != sorted_destinations[:-1]).all()
    )
    lens_long = lens[:batch].to(torch.long).view(batch, 1)
    paths_long = paths[:batch].to(torch.long)
    positions = torch.arange(16, dtype=torch.long).view(1, 16)
    active_paths = positions < lens_long
    leaf_pos = (lens_long.view(-1) - 1).clamp(min=0, max=15)
    leaf_nodes = paths_long.gather(1, leaf_pos.view(batch, 1)).view(batch)
    leaf_nodes = torch.where(
        lens_long.view(-1) > 0,
        leaf_nodes,
        torch.zeros_like(leaf_nodes),
    ).clamp(min=0, max=31)
    selected_rows = active_rows.gather(
        2,
        leaf_nodes.view(1, batch, 1).expand(48, batch, 1),
    ).view(48, batch)
    return bool(
        (active_rows > 0).all()
        and (active_rows < bank_rows).all()
        and (aliases >= 0).all()
        and (aliases < 16).all()
        and distinct
        and (lens_long >= 0).all()
        and (lens_long <= 11).all()
        and ((~active_paths) | ((paths_long >= 0) & (paths_long < 32))).all()
        and (selected_rows >= 0).all()
        and (selected_rows < bank_rows).all()
    )


def _fixed_program_guard(
    ssi: torch.Tensor,
    paths: torch.Tensor,
    lens: torch.Tensor,
    aliases: torch.Tensor,
    bank_rows: int,
) -> bool:
    batch = int(paths.shape[0])
    flags = []
    for layer in range(48):
        for request in range(batch):
            rows = ssi[layer, request]
            accepted = int(lens[request])
            active = paths[request, : max(accepted, 0)]
            leaf_pos = min(max(accepted - 1, 0), 15)
            leaf_node = int(paths[request, leaf_pos]) if accepted > 0 else 0
            leaf_node = min(max(leaf_node, 0), 31)
            duplicate = any(
                other_layer != layer or other_request != request
                for other_layer in range(48)
                for other_request in range(batch)
                if int(aliases[other_layer]) == int(aliases[layer])
                and int(ssi[other_layer, other_request, 0]) == int(rows[0])
            )
            flags.append(
                bool((rows > 0).all())
                and bool((rows < bank_rows).all())
                and 0 <= accepted <= 11
                and bool(((active >= 0) & (active < 32)).all())
                and 0 <= int(rows[leaf_node]) < bank_rows
                and 0 <= int(aliases[layer]) < 16
                and not duplicate
            )
    return all(flags)


def _valid_fixture(batch: int) -> tuple[torch.Tensor, ...]:
    bank_rows = 257
    aliases = torch.arange(16, dtype=torch.int64).repeat_interleave(3)
    ssi = torch.empty((48, batch, 32), dtype=torch.int32)
    for layer in range(48):
        alias_rank = layer % 3
        for request in range(batch):
            ssi[layer, request] = (
                torch.arange(32, dtype=torch.int32) + layer + request + 1
            ) % (bank_rows - 1) + 1
            ssi[layer, request, 0] = alias_rank * batch + request + 1
    generator = torch.Generator().manual_seed(9100 + batch)
    paths = torch.randint(0, 32, (batch, 16), generator=generator).to(
        torch.int32
    )
    lens = torch.randint(0, 12, (batch,), generator=generator).to(torch.int32)
    return ssi, paths, lens, aliases, torch.tensor(bank_rows)


@pytest.mark.parametrize("batch", (1, 4))
def test_fixed_program_partition_matches_prior_guard(batch: int) -> None:
    ssi, paths, lens, aliases, bank_rows_tensor = _valid_fixture(batch)
    bank_rows = int(bank_rows_tensor)
    fixtures = [(ssi, paths, lens, aliases)]

    for target, index, value in (
        ("ssi", (4, 0, 7), 0),
        ("ssi", (5, 0, 9), bank_rows),
        ("aliases", (6,), -1),
        ("aliases", (9,), 16),
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
    duplicate_ssi[1, 0, 0] = duplicate_ssi[0, 0, 0]
    fixtures.append((duplicate_ssi, paths, lens, aliases))

    inactive_paths = paths.clone()
    inactive_lens = lens.clone()
    inactive_lens[0] = 0
    inactive_paths[0].fill_(2**30)
    fixtures.append((ssi, inactive_paths, inactive_lens, aliases))

    for candidate in fixtures:
        assert _fixed_program_guard(*candidate, bank_rows) == _legacy_guard(
            *candidate, bank_rows
        )


def test_guard_kernel_owns_fixed_physical32_and_path16_domains() -> None:
    kernel = _function_text("_fr13_fixed32_conv_commit_row_guard_kernel")
    launcher = _function_text("validate_fixed32_conv_commit_rows")

    assert "row_offsets = tl.arange(0, SPEC_COLS)" in kernel
    assert "path_offsets = tl.arange(0, PATH_COLS)" in kernel
    assert "other_offsets = tl.arange(0, OTHER_CAP)" in kernel
    assert "duplicate_destination" in kernel
    assert "selected_row = tl.load" in kernel
    assert "SPEC_COLS=32" in launcher
    assert "PATH_COLS=16" in launcher
    assert "LAYERS=48" in launcher
    assert "OTHER_CAP=256" in launcher
    assert "[(48 * batch,)]" in launcher


def test_event_launcher_has_no_pytorch_index_transform_chain() -> None:
    launcher = _function_text("validate_fixed32_conv_commit_rows")

    assert launcher.count("_fr13_fixed32_conv_commit_row_guard_kernel[") == 1
    assert "_fr13_fixed32_device_assert(" in launcher
    assert "guard_flags.all()" in launcher
    for banned in (
        "torch.sort",
        "torch.arange",
        "torch.where",
        ".gather(",
        ".to(torch.long)",
    ):
        assert banned not in launcher


def test_guard_workspace_is_preseeded_warmed_and_source_bound() -> None:
    preseed = _function_text("preseed_fixed32_conv_col0_pregather")
    capture = _function_text("launch_fixed32_conv_col0_pregather")
    commit = _function_text("launch_fixed32_conv_commit_to_col0")

    assert '"row_guard_flags_by_batch": row_guard_flags' in preseed
    assert "48 * batch" in preseed
    assert "validate_fixed32_conv_commit_rows(" in preseed
    assert '"commit_row_guard_route": "fixed32_triton_physical32_v1"' in preseed
    assert '"commit_row_guard_kernel_launches_per_event": 1' in preseed
    assert '"commit_row_guard_compare_capacity": 256' in preseed
    assert '"commit_row_guard_torch_index_transforms": 0' in preseed
    assert '"commit_row_guard_async_scalar_reductions": 1' in preseed
    assert '"commit_row_guard_async_assertions": 1' in preseed
    assert "row_guard_flags_by_batch" in capture
    assert 'guard_flags=state["row_guard_flags_by_batch"][batch]' in commit


def test_observer_binds_guard_workspace_and_fixed32_contract() -> None:
    patcher = PATCHER_PATH.read_text(encoding="utf-8")

    assert 'row_guard_flags_by_batch = state.get("row_guard_flags_by_batch")' in patcher
    assert '!= (48 * guard_batch,)' in patcher
    assert 'str(row_guard_flags_by_batch[guard_batch].dtype) != "torch.bool"' in patcher
    assert "id(row_guard_flags_by_batch[guard_batch])" in patcher
    assert "int(row_guard_flags_by_batch[guard_batch].data_ptr())" in patcher
    assert 'contract.get("commit_row_guard_physical_rows", -1)' in patcher
    assert 'contract.get("commit_row_guard_compare_capacity", -1)' in patcher
    assert 'int(commit_spec_state_indices.shape[2]) != 32' in patcher


@pytest.mark.parametrize("batch", (1, 4))
def test_work_census_binds_fixed_row_guard_work(batch: int) -> None:
    event = census.reference_event(
        census.HYDRA_MODE,
        batch,
        f"physical32-row-guard-b{batch}",
    )
    commit = event["conv_commit"]

    assert census.SCHEMA == "fr13-fixed32-work-census-v10"
    assert commit["row_guard_route"] == "fixed32_triton_physical32_v1"
    assert commit["row_guard_kernel_launches"] == 1
    assert commit["row_guard_programs"] == 48 * batch
    assert commit["row_guard_physical_rows"] == 32
    assert commit["row_guard_path_capacity"] == 16
    assert commit["row_guard_compare_capacity"] == 256
    assert commit["row_guard_torch_index_transforms"] == 0
    assert commit["row_guard_async_scalar_reductions"] == 1
    assert commit["row_guard_async_assertions"] == 1

    normalized = census.validate_event(
        event, source=f"physical32-row-guard-b{batch}"
    ).normalized_work["conv_commit"]
    assert normalized["row_guard_programs_per_request"] == 48
    assert normalized["row_guard_physical_rows"] == 32
    assert normalized["row_guard_compare_capacity"] == 256
