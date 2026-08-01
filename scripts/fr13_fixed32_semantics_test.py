#!/usr/bin/env python3
"""CPU-only semantic gate for the fixed-32 Tail6/Hydra27 contract.

Run:
    .venv/bin/python scripts/fr13_fixed32_semantics_test.py

This is an offline contract test. It does not import the serving runtime,
launch a model, or make performance claims.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import fr13_fixed32_topology as topology  # noqa: E402


Distribution = tuple[float, ...]
Path = tuple[int, ...]
Uniforms = tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class StepDistribution:
    children: tuple[int, ...]
    child_tokens: tuple[int, ...]
    weights: tuple[float, ...]
    q_mix: Distribution
    accept_probs: tuple[float, ...]
    residual: Distribution
    all_reject: bool


@dataclass(frozen=True)
class WalkResult:
    output_tokens: tuple[int, ...]
    accepted_physical_nodes: tuple[int, ...]
    terminal: str


def _normalize(values: Sequence[float]) -> Distribution:
    result = tuple(float(value) for value in values)
    if not result:
        raise ValueError("probability row must not be empty")
    if not all(math.isfinite(value) and value >= 0.0 for value in result):
        raise ValueError("probability row must be finite and nonnegative")
    total = sum(result)
    if total <= 0.0:
        raise ValueError("probability row must have positive mass")
    return tuple(value / total for value in result)


def _inverse_cdf(weights: Sequence[float], uniform: float) -> int:
    if not 0.0 <= uniform < 1.0:
        raise ValueError(f"uniform must be in [0, 1), got {uniform}")
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("inverse CDF needs positive mass")
    threshold = uniform * total
    cumulative = 0.0
    for index, weight in enumerate(weights):
        cumulative += weight
        if cumulative > threshold:
            return index
    return len(weights) - 1


def _step_distribution(
    target_row: Sequence[float],
    children: Sequence[int],
    draft_tokens: Sequence[int],
) -> StepDistribution:
    if not children:
        raise ValueError("step distribution needs at least one child")

    target = _normalize(target_row)
    child_ids = tuple(int(child) for child in children)
    child_tokens = tuple(int(draft_tokens[child]) for child in child_ids)
    if not all(0 <= token < len(target) for token in child_tokens):
        raise ValueError("active draft token is outside the target vocabulary")

    overlaps = tuple(target[token] for token in child_tokens)
    overlap_mass = sum(overlaps)
    if overlap_mass <= 0.0:
        return StepDistribution(
            children=child_ids,
            child_tokens=child_tokens,
            weights=(),
            q_mix=(0.0,) * len(target),
            accept_probs=(),
            residual=target,
            all_reject=True,
        )

    weights = tuple(overlap / overlap_mass for overlap in overlaps)
    q_mix_list = [0.0] * len(target)
    for token, weight in zip(child_tokens, weights, strict=True):
        q_mix_list[token] += weight
    q_mix = tuple(q_mix_list)
    accept_probs = tuple(
        min(1.0, target[token] / q_mix[token]) for token in child_tokens
    )

    residual_mass = tuple(
        max(probability - mixed, 0.0)
        for probability, mixed in zip(target, q_mix, strict=True)
    )
    if sum(residual_mass) == 0.0:
        residual = target
    else:
        residual = _normalize(residual_mass)

    return StepDistribution(
        children=child_ids,
        child_tokens=child_tokens,
        weights=weights,
        q_mix=q_mix,
        accept_probs=accept_probs,
        residual=residual,
        all_reject=False,
    )


def _probability_row(seed: int, vocab_size: int) -> Distribution:
    return _normalize(
        tuple(
            float(((seed + 7 * token + token * token) % 29) + 1)
            for token in range(vocab_size)
        )
    )


def _fixture(
    vocab_size: int = 17,
) -> tuple[tuple[int, ...], tuple[Distribution, ...], tuple[Distribution, ...]]:
    draft_tokens = tuple(
        (3 + 7 * node) % vocab_size for node in range(topology.PHYSICAL_DRAFTS)
    )
    target_rows = tuple(
        _probability_row(11 + 13 * node, vocab_size)
        for node in range(topology.PHYSICAL_DRAFTS)
    )
    self_rows = tuple(
        _probability_row(23 + 17 * node, vocab_size)
        for node in range(topology.PHYSICAL_DRAFTS)
    )
    return draft_tokens, target_rows, self_rows


def _uniforms(seed: int) -> Uniforms:
    return tuple(
        (
            ((seed + 17 * level) % 97 + 0.25) / 97.0,
            ((3 * seed + 19 * level) % 89 + 0.25) / 89.0,
            ((5 * seed + 23 * level) % 83 + 0.25) / 83.0,
        )
        for level in range(topology.WALK_CAP)
    )


def _fixed_children(
    mode: topology.Mode,
) -> tuple[dict[int, tuple[int, ...]], tuple[tuple[int, ...], ...]]:
    table, counts = topology.sampler_child_table(mode)
    children = {
        parent: tuple(table[parent + 1][: counts[parent + 1]])
        for parent in range(-1, topology.PHYSICAL_DRAFTS)
        if counts[parent + 1]
    }
    return children, table


def _logical_choices(mode: topology.Mode) -> tuple[Path, ...]:
    if mode == "tail6_fixed32":
        choices = topology.TAIL6_FIXED32_CHOICES
    elif mode == "hydra27_fixed32":
        choices = topology.HYDRA27_CHOICES
    else:
        raise ValueError(f"unknown compact logical mode {mode!r}")
    return tuple(sorted(choices, key=lambda path: (len(path), path)))


def _compact_children(mode: topology.Mode) -> dict[Path, tuple[Path, ...]]:
    children: dict[Path, list[Path]] = {}
    for path in _logical_choices(mode):
        children.setdefault(path[:-1], []).append(path)
    return {parent: tuple(nodes) for parent, nodes in children.items()}


def _walk_fixed(
    mode: topology.Mode,
    draft_tokens: Sequence[int],
    target_rows: Sequence[Sequence[float]],
    self_rows: Sequence[Sequence[float]],
    uniforms: Uniforms,
) -> WalkResult:
    children_by_parent, _table = _fixed_children(mode)
    current = -1
    output_tokens: list[int] = []
    accepted_nodes: list[int] = []

    for source_u, accept_u, residual_u in uniforms:
        children = children_by_parent.get(current, ())
        if not children:
            if current < 0:
                raise AssertionError("fixed tree unexpectedly has a root leaf")
            output_tokens.append(
                _inverse_cdf(_normalize(self_rows[current]), residual_u)
            )
            return WalkResult(tuple(output_tokens), tuple(accepted_nodes), "leaf")

        step = _step_distribution(
            target_rows[children[0]],
            children,
            draft_tokens,
        )
        if step.all_reject:
            output_tokens.append(_inverse_cdf(step.residual, residual_u))
            return WalkResult(tuple(output_tokens), tuple(accepted_nodes), "zero-mass")

        source = _inverse_cdf(step.weights, source_u)
        if accept_u < step.accept_probs[source]:
            selected = children[source]
            output_tokens.append(draft_tokens[selected])
            accepted_nodes.append(selected)
            current = selected
            continue

        output_tokens.append(_inverse_cdf(step.residual, residual_u))
        return WalkResult(tuple(output_tokens), tuple(accepted_nodes), "reject")

    raise AssertionError("fixed walk exceeded WALK_CAP")


def _walk_compact(
    mode: topology.Mode,
    draft_tokens: Sequence[int],
    target_rows: Sequence[Sequence[float]],
    self_rows: Sequence[Sequence[float]],
    uniforms: Uniforms,
) -> WalkResult:
    children_by_path = _compact_children(mode)
    physical_by_path = {
        path: node for node, path in enumerate(topology.FIXED32_CHOICES)
    }
    current: Path = ()
    output_tokens: list[int] = []
    accepted_nodes: list[int] = []

    for source_u, accept_u, residual_u in uniforms:
        child_paths = children_by_path.get(current, ())
        if not child_paths:
            if not current:
                raise AssertionError("compact tree unexpectedly has a root leaf")
            current_node = physical_by_path[current]
            output_tokens.append(
                _inverse_cdf(_normalize(self_rows[current_node]), residual_u)
            )
            return WalkResult(tuple(output_tokens), tuple(accepted_nodes), "leaf")

        children = tuple(physical_by_path[path] for path in child_paths)
        step = _step_distribution(
            target_rows[children[0]],
            children,
            draft_tokens,
        )
        if step.all_reject:
            output_tokens.append(_inverse_cdf(step.residual, residual_u))
            return WalkResult(tuple(output_tokens), tuple(accepted_nodes), "zero-mass")

        source = _inverse_cdf(step.weights, source_u)
        if accept_u < step.accept_probs[source]:
            current = child_paths[source]
            selected = children[source]
            output_tokens.append(draft_tokens[selected])
            accepted_nodes.append(selected)
            continue

        output_tokens.append(_inverse_cdf(step.residual, residual_u))
        return WalkResult(tuple(output_tokens), tuple(accepted_nodes), "reject")

    raise AssertionError("compact walk exceeded WALK_CAP")


def _fixed_distribution_snapshot(
    mode: topology.Mode,
    draft_tokens: Sequence[int],
    target_rows: Sequence[Sequence[float]],
) -> dict[Path, StepDistribution]:
    children_by_parent, _table = _fixed_children(mode)
    snapshot = {}
    for parent, children in children_by_parent.items():
        parent_path = () if parent < 0 else topology.FIXED32_CHOICES[parent]
        snapshot[parent_path] = _step_distribution(
            target_rows[children[0]],
            children,
            draft_tokens,
        )
    return snapshot


def _compact_distribution_snapshot(
    mode: topology.Mode,
    draft_tokens: Sequence[int],
    target_rows: Sequence[Sequence[float]],
) -> dict[Path, StepDistribution]:
    physical_by_path = {
        path: node for node, path in enumerate(topology.FIXED32_CHOICES)
    }
    snapshot = {}
    for parent_path, child_paths in _compact_children(mode).items():
        children = tuple(physical_by_path[path] for path in child_paths)
        snapshot[parent_path] = _step_distribution(
            target_rows[children[0]],
            children,
            draft_tokens,
        )
    return snapshot


def _sampler_cache_key(mode: topology.Mode) -> tuple[Any, ...]:
    return (
        topology.DRAFT_PARENT,
        topology.VALID_MASK_BY_MODE[mode],
        topology.SAMPLER_TABLE_SHAPE,
        topology.WALK_CAP,
    )


class _SamplerTableCache:
    def __init__(self) -> None:
        self.entries: dict[
            tuple[Any, ...],
            tuple[tuple[tuple[int, ...], ...], tuple[int, ...]],
        ] = {}

    def get(
        self,
        mode: topology.Mode,
    ) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
        key = _sampler_cache_key(mode)
        if key not in self.entries:
            self.entries[key] = topology.sampler_child_table(mode)
        return self.entries[key]


@contextmanager
def _patched_contract(**replacements: object) -> Iterator[None]:
    original = {name: getattr(topology, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(topology, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(topology, name, value)


def _assert_raises(
    error_type: type[BaseException],
    message: str,
    operation: Callable[[], object],
) -> None:
    try:
        operation()
    except error_type as exc:
        if message not in str(exc):
            raise AssertionError(
                f"expected {message!r} in {error_type.__name__}: {exc}"
            ) from exc
    else:
        raise AssertionError(f"expected {error_type.__name__}: {message}")


def test_exact_physical_shape() -> None:
    topology.validate_contract()
    assert len(topology.FIXED32_CHOICES) == topology.PHYSICAL_DRAFTS == 31
    assert len(topology.DRAFT_PARENT) == topology.PHYSICAL_DRAFTS
    assert len(topology.PHYSICAL_PARENT) == topology.PHYSICAL_ROWS == 32
    assert topology.PHYSICAL_PARENT[0] == -1
    assert topology.SAMPLER_TABLE_SHAPE == (32, 3)

    for node, parent in enumerate(topology.DRAFT_PARENT):
        expected = 0 if parent < 0 else parent + 1
        assert topology.PHYSICAL_PARENT[node + 1] == expected

    scheduled_rows = {
        node
        for level in topology.SUBTREE_LEVELS
        for path, _parent in level
        for node in path
    }
    assert scheduled_rows == set(range(topology.PHYSICAL_ROWS))

    expected_active = {
        "tail6_fixed32": topology.TAIL6_ACTIVE_DRAFTS,
        "hydra27_fixed32": topology.HYDRA27_ACTIVE_DRAFTS,
    }
    for mode, active_count in expected_active.items():
        table, counts = topology.sampler_child_table(mode)
        assert len(table) == topology.PHYSICAL_ROWS
        assert all(len(row) == topology.SAMPLER_MAX_FANOUT for row in table)
        assert len(counts) == topology.PHYSICAL_ROWS
        assert sum(counts) == active_count
        for row, count in zip(table, counts, strict=True):
            assert 0 <= count <= topology.SAMPLER_MAX_FANOUT
            assert all(0 <= node < topology.PHYSICAL_DRAFTS for node in row[:count])
            assert row[count:] == (-1,) * (topology.SAMPLER_MAX_FANOUT - count)


def test_compact_logical_reference_equivalence() -> None:
    draft_tokens, target_rows, self_rows = _fixture()
    expected_active = {
        "tail6_fixed32": 23,
        "hydra27_fixed32": 27,
    }
    for mode, expected_count in expected_active.items():
        compact_choices = _logical_choices(mode)
        compact_index = {path: index for index, path in enumerate(compact_choices)}
        physical_index = {
            path: index for index, path in enumerate(topology.FIXED32_CHOICES)
        }
        compact_parents = tuple(
            -1 if len(path) == 1 else compact_index[path[:-1]]
            for path in compact_choices
        )
        translated_physical_parents = tuple(
            (
                -1
                if topology.DRAFT_PARENT[physical_index[path]] < 0
                else compact_index[
                    topology.FIXED32_CHOICES[
                        topology.DRAFT_PARENT[physical_index[path]]
                    ]
                ]
            )
            for path in compact_choices
        )

        assert len(compact_choices) == expected_count
        assert compact_choices == topology.active_choices(mode)
        assert compact_parents == translated_physical_parents
        assert _fixed_distribution_snapshot(
            mode,
            draft_tokens,
            target_rows,
        ) == _compact_distribution_snapshot(mode, draft_tokens, target_rows)

        uniform_sets = [_uniforms(seed) for seed in (3, 19, 71)] + [
            ((0.01, 0.0, 0.73),) * topology.WALK_CAP,
        ]
        for uniform_set in uniform_sets:
            assert _walk_fixed(
                mode,
                draft_tokens,
                target_rows,
                self_rows,
                uniform_set,
            ) == _walk_compact(
                mode,
                draft_tokens,
                target_rows,
                self_rows,
                uniform_set,
            )


def test_invalid_node_poison_invariance() -> None:
    draft_tokens, target_rows, self_rows = _fixture()
    for mode in topology.VALID_BY_MODE:
        valid = topology.valid_for_mode(mode)
        poisoned_tokens = list(draft_tokens)
        poisoned_target = list(target_rows)
        poisoned_self = list(self_rows)
        for node, enabled in enumerate(valid):
            if enabled:
                continue
            poisoned_tokens[node] = len(target_rows[0]) + 1000 + node
            poisoned_target[node] = (math.nan,) * len(target_rows[0])
            poisoned_self[node] = (math.nan,) * len(target_rows[0])

        assert _fixed_distribution_snapshot(
            mode,
            poisoned_tokens,
            poisoned_target,
        ) == _fixed_distribution_snapshot(mode, draft_tokens, target_rows)

        for seed in (5, 29, 73):
            uniform_set = _uniforms(seed)
            assert _walk_fixed(
                mode,
                poisoned_tokens,
                poisoned_target,
                poisoned_self,
                uniform_set,
            ) == _walk_fixed(
                mode,
                draft_tokens,
                target_rows,
                self_rows,
                uniform_set,
            )


def test_duplicate_token_masking_and_q_mix() -> None:
    vocab_size = 8
    target = [0.02] * vocab_size
    target[1] = 0.30
    target[2] = 0.20
    target[3] = 0.10

    for mode in topology.VALID_BY_MODE:
        children_by_parent, _table = _fixed_children(mode)
        root_children = children_by_parent[-1]
        assert root_children == (0, 1, 2)

        tokens = [(node + 4) % vocab_size for node in range(topology.PHYSICAL_DRAFTS)]
        tokens[0:3] = [1, 2, 3]
        baseline = _step_distribution(target, root_children, tokens)

        inactive = next(
            node
            for node, enabled in enumerate(topology.valid_for_mode(mode))
            if not enabled
        )
        invalid_duplicate = list(tokens)
        invalid_duplicate[inactive] = 1
        assert _step_distribution(target, root_children, invalid_duplicate) == baseline

        valid_duplicate = list(tokens)
        valid_duplicate[root_children[1]] = 1
        duplicated = _step_distribution(target, root_children, valid_duplicate)
        token_one_sources = tuple(
            index for index, token in enumerate(duplicated.child_tokens) if token == 1
        )
        expected_q_mix = sum(duplicated.weights[index] for index in token_one_sources)
        assert token_one_sources == (0, 1)
        assert math.isclose(duplicated.q_mix[1], expected_q_mix)
        assert duplicated.q_mix[1] > baseline.q_mix[1]
        assert duplicated.accept_probs[0] == duplicated.accept_probs[1]
        assert duplicated.accept_probs[0] < baseline.accept_probs[0]


def test_tail_hydra_tail_cache_stability() -> None:
    cache = _SamplerTableCache()
    tail_first = cache.get("tail6_fixed32")
    hydra = cache.get("hydra27_fixed32")
    tail_again = cache.get("tail6_fixed32")

    tail_key = _sampler_cache_key("tail6_fixed32")
    hydra_key = _sampler_cache_key("hydra27_fixed32")
    assert tail_first is tail_again
    assert tail_first == topology.sampler_child_table("tail6_fixed32")
    assert hydra == topology.sampler_child_table("hydra27_fixed32")
    assert tail_first != hydra
    assert tail_key != hydra_key
    assert tail_key[0] == hydra_key[0] == topology.DRAFT_PARENT
    assert tail_key[2:] == hydra_key[2:]
    assert tail_key[1] == topology.TAIL6_VALID_MASK
    assert hydra_key[1] == topology.HYDRA27_VALID_MASK
    assert len(cache.entries) == 2


def test_mask_is_sampler_only_boundary() -> None:
    physical_contract = (
        topology.FIXED32_CHOICES,
        topology.DRAFT_PARENT,
        topology.PHYSICAL_PARENT,
        topology.SUBTREE_LEVELS,
        topology.FIXED_EXECUTION_SIGNATURE,
    )
    projections = {}
    for mode in topology.VALID_BY_MODE:
        table, counts = topology.sampler_child_table(mode)
        sampled_nodes = {
            node
            for row, count in zip(table, counts, strict=True)
            for node in row[:count]
        }
        projections[mode] = {
            "physical": physical_contract,
            "sampler": (table, counts),
            "sampled_nodes": sampled_nodes,
        }
        assert sampled_nodes == {
            node
            for node, enabled in enumerate(topology.valid_for_mode(mode))
            if enabled
        }
        assert projections[mode]["physical"] == physical_contract

    assert (
        projections["tail6_fixed32"]["physical"]
        == projections["hydra27_fixed32"]["physical"]
    )
    assert (
        projections["tail6_fixed32"]["sampler"]
        != projections["hydra27_fixed32"]["sampler"]
    )
    assert projections["hydra27_fixed32"]["sampled_nodes"] - projections[
        "tail6_fixed32"
    ]["sampled_nodes"] == {11, 12, 16, 21}
    rescue_nodes = tuple(
        topology.FIXED32_CHOICES.index(path)
        for path in topology.TAIL6_FIXED32_RESCUE_CHOICES
    )
    assert rescue_nodes == (6, 7)
    assert all(topology.TAIL6_VALID[node] for node in rescue_nodes)
    assert all(topology.HYDRA27_VALID[node] for node in rescue_nodes)
    signature = topology.FIXED_EXECUTION_SIGNATURE
    assert signature["physical_pack_width"] == 31
    assert signature["target_rows"] == 32
    assert signature["tree_attention_rows"] == 32
    assert signature["gdn_rows"] == 32
    assert signature["gdn_launches"] == 2
    assert signature["sampler_walk_iterations"] == 12
    assert signature["committer_path_capacity"] == 16


def test_fail_loud_parent_closure_and_fanout() -> None:
    inactive_parent = [False] * topology.PHYSICAL_DRAFTS
    inactive_parent[3] = True
    parent_modes = {
        **topology.VALID_BY_MODE,
        "broken_parent": tuple(inactive_parent),
    }
    with _patched_contract(VALID_BY_MODE=parent_modes):
        _assert_raises(
            ValueError,
            "active node 3 has inactive parent 0",
            lambda: topology.active_child_lists("broken_parent"),
        )

    fanout_parent = list(topology.DRAFT_PARENT)
    fanout_parent[:4] = [-1, -1, -1, -1]
    fanout_valid = [False] * topology.PHYSICAL_DRAFTS
    fanout_valid[:4] = [True, True, True, True]
    fanout_modes = {
        **topology.VALID_BY_MODE,
        "fanout_four": tuple(fanout_valid),
    }
    with _patched_contract(
        DRAFT_PARENT=tuple(fanout_parent),
        VALID_BY_MODE=fanout_modes,
    ):
        _assert_raises(
            ValueError,
            "parent -1 exceeds fixed sampler fanout",
            lambda: topology.sampler_child_table("fanout_four"),
        )

    _assert_raises(
        ValueError,
        "missing physical parent",
        lambda: topology._draft_parents(((0,), (1, 0))),
    )


TESTS = (
    test_exact_physical_shape,
    test_compact_logical_reference_equivalence,
    test_invalid_node_poison_invariance,
    test_duplicate_token_masking_and_q_mix,
    test_tail_hydra_tail_cache_stability,
    test_mask_is_sampler_only_boundary,
    test_fail_loud_parent_closure_and_fanout,
)


def main() -> None:
    for test in TESTS:
        test()
    print(
        "PASS fr13_fixed32_semantics_test "
        f"tests={len(TESTS)} modes={len(topology.VALID_BY_MODE)} "
        f"physical={topology.PHYSICAL_DRAFTS}/{topology.PHYSICAL_ROWS}"
    )


if __name__ == "__main__":
    main()
