from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch


KERNEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "lumo_flywheel_serving"
    / "fr10_gdn_tree_kernel.py"
)
GUARD_NAME = "validate_fixed32_conv_commit_rows"


def _load_guard(assert_fn):
    tree = ast.parse(KERNEL_PATH.read_text(encoding="utf-8"))
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == GUARD_NAME
    ]
    assert len(definitions) == 1
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            definitions[0],
        ],
        type_ignores=[],
    )
    namespace = {
        "torch": torch,
        "_FR13_FIXED32_COMMITTER_MAX_ACCEPTED_LENGTH": 11,
        "_fr13_fixed32_device_assert": assert_fn,
    }
    exec(compile(ast.fix_missing_locations(module), KERNEL_PATH, "exec"), namespace)
    return namespace[GUARD_NAME]


def _guard_inputs() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    spec_state_indices = torch.zeros((48, 4, 32), dtype=torch.int32)
    for layer in range(48):
        alias_rank = layer // 16
        for batch_index in range(4):
            spec_state_indices[layer, batch_index, :].fill_(
                alias_rank * 4 + batch_index + 1
            )
    accepted_paths = torch.zeros((4, 16), dtype=torch.int32)
    accepted_lens = torch.ones((4,), dtype=torch.int32)
    bank_alias_ids = torch.arange(16, dtype=torch.int64).repeat(3)
    return (
        spec_state_indices,
        accepted_paths,
        accepted_lens,
        bank_alias_ids,
    )


def _guard_result(
    spec_state_indices: torch.Tensor,
    accepted_paths: torch.Tensor,
    accepted_lens: torch.Tensor,
    bank_alias_ids: torch.Tensor,
    *,
    batch: int = 4,
    bank_rows: int = 64,
) -> bool:
    observed: list[bool] = []

    def record_assert(condition: torch.Tensor, message: str) -> None:
        assert message == (
            "FR13_FIXED32_CONV_COMMIT precommit row/path contract violation"
        )
        observed.append(bool(condition.item()))

    guard = _load_guard(record_assert)
    guard(
        spec_state_indices=spec_state_indices,
        accepted_paths=accepted_paths,
        accepted_lens=accepted_lens,
        bank_alias_ids=bank_alias_ids,
        batch=batch,
        bank_rows=bank_rows,
    )
    assert len(observed) == 1
    return observed[0]


def test_precommit_guard_accepts_valid_capacity_backed_b1_and_b4() -> None:
    operands = _guard_inputs()
    assert _guard_result(*operands, batch=1)
    assert _guard_result(*operands, batch=4)


def test_precommit_guard_ignores_poisoned_inactive_capacity_slots_for_b1() -> None:
    (
        spec_state_indices,
        accepted_paths,
        accepted_lens,
        bank_alias_ids,
    ) = _guard_inputs()
    spec_state_indices[:, 1:, :].fill_(64)
    accepted_paths[1:, :].fill_(32)
    accepted_lens[1:].fill_(16)

    assert _guard_result(
        spec_state_indices,
        accepted_paths,
        accepted_lens,
        bank_alias_ids,
        batch=1,
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "oob_nonselected_source",
        "oob_selected_source",
        "null_nonselected_source",
        "duplicate_destination",
        "cross_alias_destination",
        "invalid_active_path",
        "invalid_accepted_length",
        "invalid_alias_id",
    ),
)
def test_precommit_guard_rejects_unsafe_dynamic_rows_and_paths(
    mutation: str,
) -> None:
    (
        spec_state_indices,
        accepted_paths,
        accepted_lens,
        bank_alias_ids,
    ) = _guard_inputs()
    if mutation == "oob_nonselected_source":
        spec_state_indices[17, 2, 31] = 64
    elif mutation == "oob_selected_source":
        accepted_paths[2, 0] = 5
        spec_state_indices[17, 2, 5] = 64
    elif mutation == "null_nonselected_source":
        spec_state_indices[17, 2, 31] = 0
    elif mutation == "duplicate_destination":
        spec_state_indices[:, 1, 0] = spec_state_indices[:, 0, 0]
    elif mutation == "cross_alias_destination":
        spec_state_indices[16, 3, 0] = spec_state_indices[0, 0, 0]
    elif mutation == "invalid_active_path":
        accepted_paths[2, 0] = 32
    elif mutation == "invalid_accepted_length":
        accepted_lens[2] = 12
    elif mutation == "invalid_alias_id":
        bank_alias_ids[17] = 16
    else:
        raise AssertionError(mutation)

    assert not _guard_result(
        spec_state_indices,
        accepted_paths,
        accepted_lens,
        bank_alias_ids,
    )


def test_precommit_guard_allows_source_aliases_after_gather_barrier() -> None:
    operands = list(_guard_inputs())
    spec_state_indices = operands[0]
    accepted_paths = operands[1]
    spec_state_indices[16, 3, 7] = spec_state_indices[0, 0, 7]
    accepted_paths[3, 0] = 7

    assert _guard_result(*operands)
