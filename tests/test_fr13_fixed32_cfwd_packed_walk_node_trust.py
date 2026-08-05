from __future__ import annotations

import ast
import importlib.util
import inspect
from pathlib import Path

import pytest
import torch


CANDIDATE_PATH = Path(
    "scripts/fr13_cfwd_packed_walk_node_trust_kernel.py"
)
PRODUCER_PATH = Path("scripts/fr13_cfwd_logit_direct_decision_kernel.py")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


candidate = _load("fr13_packed_walk_node_trust_test", CANDIDATE_PATH)
producer = _load("fr13_packed_walk_producer_test", PRODUCER_PATH)


def _function_source(name: str) -> str:
    source = CANDIDATE_PATH.read_text(encoding="ascii")
    tree = ast.parse(source)
    node = next(
        item
        for item in ast.walk(tree)
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    return ast.get_source_segment(source, node) or ""


def _producer_contract(mode: str) -> dict[str, object]:
    return {
        "candidate": candidate.BASE_CANDIDATE,
        "candidate_schema": candidate.BASE_CANDIDATE_SCHEMA,
        "candidate_source_sha256": candidate.BASE_CANDIDATE_SOURCE_SHA256,
        "integration_source_schema": candidate.BASE_INTEGRATION_SOURCE_SCHEMA,
        "integration_source_sha256": (
            candidate.BASE_INTEGRATION_SOURCE_SHA256
        ),
        "mode": mode,
        "physical_drafts": 31,
        "physical_rows": 32,
    }


@pytest.mark.parametrize("mode", sorted(candidate.FIXED32_MODES))
def test_contract_binds_exact_v3_physical32_producer(mode: str) -> None:
    contract = candidate.packed_walk_node_trust_contract(mode)
    assert contract["loop_bound_topology_constant"] is True
    assert contract["physical_rows"] == 32
    assert contract["walk_levels"] == 12
    assert contract["accepted_row_domain"] == [1, 31]
    assert contract["node_domain_clamps_per_request_before"] == 48
    assert contract["node_domain_clamps_per_request_after"] == 0
    assert contract["leaf_domain_comparisons_per_request_before"] == 24
    assert contract["leaf_domain_comparisons_per_request_after"] == 0
    assert contract["unconditional_self_token_loads_per_request_before"] == 12
    assert contract["unconditional_self_token_loads_per_request_after"] == 0
    assert contract["leaf_self_token_loads_per_request_after_max"] == 1
    candidate.validate_packed_walk_producer_contract(
        _producer_contract(mode), mode=mode
    )


def test_producer_contract_fails_closed_on_every_trusted_field() -> None:
    base = _producer_contract("hydra27_fixed32")
    for name in tuple(base):
        drifted = dict(base)
        drifted[name] = "drift" if not isinstance(base[name], int) else -1
        with pytest.raises(RuntimeError, match="exact reviewed physical32"):
            candidate.validate_packed_walk_producer_contract(
                drifted, mode="hydra27_fixed32"
            )
    with pytest.raises(ValueError, match="unsupported fixed32 mode"):
        candidate.validate_packed_walk_producer_contract(
            base, mode="hydra31"
        )


def test_kernel_consumes_validated_node_domain_without_hot_clamps() -> None:
    source = _function_source(
        "_fr13_fixed32_taw_packed_node_trust_commit_kernel"
    )
    assert "tl.static_range(0, WALK_CAP)" in source
    assert "parent_slot = current + 1" in source
    assert "tl.maximum" not in source
    assert "tl.minimum" not in source
    assert "mask=alive" in source
    assert "mask=leaf" in source
    assert "alive = is_accepted" in source
    assert "sampled_bonus" not in source
    assert "current_valid" not in source


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
def test_oracle_matches_credentialed_walk_across_valid_events(
    mode: str, batch: int
) -> None:
    for seed in range(64):
        self_token, event, bonus_token = _physical_events(
            mode, batch, seed + 1_000 * batch
        )
        reference = producer.packed_physical_walk_oracle(
            self_token, event, bonus_token
        )
        observed = candidate.packed_walk_node_trust_oracle(
            self_token, event, bonus_token
        )
        assert len(reference) == len(observed) == 5
        assert all(
            torch.equal(expected, actual)
            for expected, actual in zip(reference, observed, strict=True)
        )


@pytest.mark.parametrize("batch", (1, 4))
def test_oracle_covers_full_depth_and_rejection_boundaries(batch: int) -> None:
    self_token, event, bonus_token = _physical_events(
        "hydra27_fixed32", batch, 90210 + batch
    )
    # Follow child lane zero from root to the full depth-11 leaf.
    table = producer.MODE_CHILDREN["hydra27_fixed32"]
    current = -1
    for _level in range(candidate.WALK_CAP):
        parent = current + 1
        children = table.get(parent, ())
        if not children:
            break
        accepted_row = int(children[0]) + 1
        token = 17 + parent
        event[:, parent] = (
            token
            | (accepted_row << candidate.PACKED_EVENT_ACCEPTED_ROW_SHIFT)
            | candidate.PACKED_EVENT_PARENT_MASK
        )
        current = accepted_row - 1
    reference = producer.packed_physical_walk_oracle(
        self_token, event, bonus_token
    )
    observed = candidate.packed_walk_node_trust_oracle(
        self_token, event, bonus_token
    )
    assert all(
        torch.equal(expected, actual)
        for expected, actual in zip(reference, observed, strict=True)
    )
    assert torch.all(observed[3] <= 11)


def test_oracle_rejects_the_removed_root_leaf_fallback_domain() -> None:
    self_token = torch.zeros((1, 31), dtype=torch.long)
    event = torch.zeros((1, 32), dtype=torch.long)
    bonus = torch.tensor([7], dtype=torch.long)
    with pytest.raises(RuntimeError, match="root cannot be a leaf"):
        candidate.packed_walk_node_trust_oracle(self_token, event, bonus)


def test_launch_is_default_off_and_uses_one_fixed_program_per_request() -> None:
    source = inspect.getsource(candidate.launch_packed_walk_node_trust)
    assert "_fr13_fixed32_taw_packed_node_trust_commit_kernel[(batch,)]" in source
    assert "num_warps=1" in source
    assert "batch not in (1, 4)" in source
    assert "validate_packed_walk_producer_contract" in source
    assert "os.environ" not in CANDIDATE_PATH.read_text(encoding="ascii")
