from __future__ import annotations

import ast
import importlib.util
import inspect
from pathlib import Path

import pytest
import torch


CANDIDATE_PATH = Path(
    "scripts/fr13_cfwd_packed_walk_active_depth_kernel.py"
)
BASE_PATH = Path("scripts/fr13_cfwd_packed_walk_node_trust_kernel.py")
PRODUCER_PATH = Path("scripts/fr13_cfwd_logit_direct_decision_kernel.py")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


candidate = _load("fr13_active_depth_walk_test", CANDIDATE_PATH)
base = _load("fr13_node_trust_walk_test", BASE_PATH)
producer = _load("fr13_active_depth_producer_test", PRODUCER_PATH)


def _function_source(name: str) -> str:
    source = CANDIDATE_PATH.read_text(encoding="ascii")
    tree = ast.parse(source)
    node = next(
        item
        for item in ast.walk(tree)
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    return ast.get_source_segment(source, node) or ""


def _base_contract(mode: str) -> dict[str, object]:
    return {
        "candidate": candidate.BASE_CANDIDATE,
        "candidate_schema": candidate.BASE_CANDIDATE_SCHEMA,
        "candidate_source_sha256": candidate.BASE_CANDIDATE_SOURCE_SHA256,
        "producer_candidate": candidate.PRODUCER_CANDIDATE,
        "producer_schema": candidate.PRODUCER_SCHEMA,
        "producer_source_sha256": candidate.PRODUCER_SOURCE_SHA256,
        "mode": mode,
        "physical_drafts": 31,
        "physical_rows": 32,
        "walk_levels": 12,
    }


@pytest.mark.parametrize("mode", sorted(candidate.FIXED32_MODES))
def test_contract_binds_exact_node_trust_and_packed_producer(mode: str) -> None:
    contract = candidate.active_depth_walk_contract(mode)
    assert contract["maximum_walk_iterations"] == 12
    assert contract["physical_rows"] == 32
    assert contract["topology_size_controls_loop_bound"] is False
    assert contract["candidate_default_off"] is True
    candidate.validate_active_depth_base_contract(
        _base_contract(mode), mode=mode
    )


def test_contract_fails_closed_on_every_bound_field() -> None:
    contract = _base_contract("hydra27_fixed32")
    for name, value in tuple(contract.items()):
        drifted = dict(contract)
        drifted[name] = -1 if isinstance(value, int) else "drift"
        with pytest.raises(RuntimeError, match="exact reviewed"):
            candidate.validate_active_depth_base_contract(
                drifted, mode="hydra27_fixed32"
            )
    with pytest.raises(ValueError, match="unsupported fixed32 mode"):
        candidate.validate_active_depth_base_contract(
            contract, mode="hydra31"
        )


def test_kernel_is_bounded_dynamic_loop_without_domain_clamps() -> None:
    source = _function_source(
        "_fr13_fixed32_taw_packed_active_depth_commit_kernel"
    )
    assert "while alive & (level < WALK_CAP):" in source
    assert "tl.static_range" not in source
    assert "parent_slot = current + 1" in source
    assert "tl.maximum" not in source
    assert "tl.minimum" not in source
    assert "mask=leaf" in source
    assert "alive = is_accepted" in source
    assert "level += 1" in source
    assert "sampled_bonus" not in source


def _physical_events(mode: str, batch: int, seed: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    source = torch.zeros((batch, 32), dtype=torch.long)
    selected = torch.zeros((batch, 32), dtype=torch.long)
    rejected = torch.zeros((batch, 32), dtype=torch.long)
    accepted = torch.zeros((batch, 32), dtype=torch.bool)
    table = torch.full((batch, 32, 3), -1, dtype=torch.long)
    counts = torch.zeros((batch, 32), dtype=torch.long)
    for parent, children in producer.MODE_CHILDREN[mode].items():
        child_tensor = torch.tensor(children, dtype=torch.long)
        table[:, parent, : len(children)] = child_tensor
        counts[:, parent] = len(children)
        source[:, parent] = torch.randint(
            0, len(children), (batch,), generator=generator
        )
        selected[:, parent] = torch.randint(
            0, producer.VOCAB_SIZE, (batch,), generator=generator
        )
        rejected[:, parent] = torch.randint(
            0, producer.VOCAB_SIZE, (batch,), generator=generator
        )
        accepted[:, parent] = torch.rand(batch, generator=generator) < 0.72
    event = producer.pack_physical_event_oracle(
        source,
        selected,
        rejected,
        accepted,
        table,
        counts,
    )
    self_token = torch.randint(
        0, producer.VOCAB_SIZE, (batch, 31), generator=generator
    )
    bonus_token = torch.randint(
        0, producer.VOCAB_SIZE, (batch,), generator=generator
    )
    return self_token, event, bonus_token


@pytest.mark.parametrize("mode", sorted(candidate.FIXED32_MODES))
@pytest.mark.parametrize("batch", (1, 4))
def test_oracle_matches_node_trust_for_valid_events(
    mode: str, batch: int
) -> None:
    for seed in range(64):
        inputs = _physical_events(mode, batch, seed + 1_000 * batch)
        expected = base.packed_walk_node_trust_oracle(*inputs)
        observed = candidate.active_depth_packed_walk_oracle(*inputs)
        assert len(expected) == 5
        assert len(observed) == 6
        assert all(
            torch.equal(left, right)
            for left, right in zip(expected, observed[:5], strict=True)
        )
        assert torch.all(observed[5] >= 1)
        assert torch.all(observed[5] <= candidate.WALK_CAP)


@pytest.mark.parametrize("batch", (1, 4))
def test_oracle_stops_after_root_rejection(batch: int) -> None:
    self_token, event, bonus = _physical_events(
        "hydra27_fixed32", batch, 60_000 + batch
    )
    event[:, 0] = 37 | candidate.PACKED_EVENT_PARENT_MASK
    observed = candidate.active_depth_packed_walk_oracle(
        self_token, event, bonus
    )
    assert torch.equal(observed[1], torch.ones(batch, dtype=torch.long))
    assert torch.equal(observed[3], torch.zeros(batch, dtype=torch.long))
    assert torch.equal(observed[5], torch.ones(batch, dtype=torch.long))


@pytest.mark.parametrize("batch", (1, 4))
def test_oracle_preserves_full_depth_cap(batch: int) -> None:
    self_token, event, bonus = _physical_events(
        "hydra27_fixed32", batch, 90_210 + batch
    )
    table = producer.MODE_CHILDREN["hydra27_fixed32"]
    current = -1
    expected_iterations = 0
    for _level in range(candidate.WALK_CAP):
        parent = current + 1
        expected_iterations += 1
        children = table.get(parent, ())
        if not children:
            break
        accepted_row = int(children[0]) + 1
        event[:, parent] = (
            (17 + parent)
            | (accepted_row << candidate.PACKED_EVENT_ACCEPTED_ROW_SHIFT)
            | candidate.PACKED_EVENT_PARENT_MASK
        )
        current = accepted_row - 1
    expected = base.packed_walk_node_trust_oracle(
        self_token, event, bonus
    )
    observed = candidate.active_depth_packed_walk_oracle(
        self_token, event, bonus
    )
    assert all(
        torch.equal(left, right)
        for left, right in zip(expected, observed[:5], strict=True)
    )
    assert torch.all(observed[5] == expected_iterations)


def test_oracle_rejects_root_leaf_and_non_b1_b4() -> None:
    self_token = torch.zeros((1, 31), dtype=torch.long)
    event = torch.zeros((1, 32), dtype=torch.long)
    bonus = torch.tensor([7], dtype=torch.long)
    with pytest.raises(RuntimeError, match="root cannot be a leaf"):
        candidate.active_depth_packed_walk_oracle(self_token, event, bonus)
    with pytest.raises(ValueError, match="B1/B4 only"):
        candidate.active_depth_packed_walk_oracle(
            self_token.expand(2, -1).clone(),
            event.expand(2, -1).clone(),
            bonus.expand(2).clone(),
        )


def test_launch_stays_source_only_and_one_program_per_request() -> None:
    source = inspect.getsource(candidate.launch_active_depth_packed_walk)
    assert (
        "_fr13_fixed32_taw_packed_active_depth_commit_kernel[(batch,)]"
        in source
    )
    assert "num_warps=1" in source
    assert "batch not in (1, 4)" in source
    assert "validate_active_depth_base_contract" in source
    assert "os.environ" not in CANDIDATE_PATH.read_text(encoding="ascii")
