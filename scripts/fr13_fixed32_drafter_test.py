#!/usr/bin/env python3
"""Offline contract tests for the common fixed-32 drafter payload."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest import mock

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import fr13_fixed32_topology as topology  # noqa: E402
import fr13_merged_drafter as drafter  # noqa: E402
import fr13_merged_fill as fill  # noqa: E402
from fr13_fixed32_work_census import reference_event  # noqa: E402


class MockDraft:
    def __init__(self, token_ids):
        self.token_ids = list(token_ids)


class ExactCache:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def speculate(self, req_id, pattern, **kwargs):
        key = (str(req_id), tuple(int(token) for token in pattern))
        self.calls.append((key, dict(kwargs)))
        return MockDraft(self.responses.get(key, []))


def _run_common_pack():
    drafter.reset_for_test()
    drafter._COMMITTED.update({"ra": [9, 8], "rb": [7]})
    head = [
        [100, 200],
        [101, 201],
        [102, 202],
        [103, 203],
        [104, 204],
    ]
    seeds = {1: [301, 302], 2: [401, 402]}
    responses = {
        ("ra", (9, 8, 100, 101, 102, 103, 104)): range(500, 506),
        ("ra", (9, 8, 301)): [310, 311, 312, 313],
        ("ra", (9, 8, 401)): [410, 411],
        ("rb", (7, 200, 201, 202, 203, 204)): range(600, 606),
        ("rb", (7, 302)): [320, 321, 322, 323],
        ("rb", (7, 402)): [420, 421],
    }
    cache = ExactCache(responses)
    with mock.patch.object(fill.torch, "tensor", wraps=torch.tensor) as tensor_spy:
        tail = drafter.decide_fixed32(
            cache,
            ["ra", "rb"],
            head,
            seeds,
            torch.device("cpu"),
            0,
            vocab_size=1000,
        )
    return cache, tail, dict(drafter.get_tail_path_tokens()), tensor_spy.call_count


def test_identical_common_work_for_both_modes():
    first = _run_common_pack()
    second = _run_common_pack()
    first_cache, first_tail, first_paths, first_transfers = first
    second_cache, second_tail, second_paths, second_transfers = second

    assert first_transfers == second_transfers == 1
    assert [column.tolist() for column in first_tail] == [
        column.tolist() for column in second_tail
    ]
    assert {
        path: column.tolist() for path, column in first_paths.items()
    } == {
        path: column.tolist() for path, column in second_paths.items()
    }

    for cache in (first_cache, second_cache):
        assert len(cache.calls) == 2 * topology.ARCTIC_LOOKUP_CALLS_PER_REQUEST
        assert [
            call[1]["max_spec_tokens"] for call in cache.calls
        ] == [6, 4, 2, 6, 4, 2]
        assert all(call[1]["use_tree_spec"] is False for call in cache.calls)

    assert len(first_tail) == topology.ARCTIC_MAIN_TAIL_LENGTH
    assert set(first_paths) == set(
        topology.branch_paths(topology.PHYSICAL_BRANCH_CHAINS)
    )
    assert (
        5 * 3 + len(first_tail) + len(first_paths)
        == topology.PHYSICAL_DRAFTS
    )

    rank2 = [
        (2,) + (0,) * length
        for length in range(1, topology.PHYSICAL_BRANCH_CHAINS[1][1] + 1)
    ]
    assert [first_paths[path].tolist() for path in rank2] == [
        [410, 420],
        [411, 421],
        [411, 421],
        [411, 421],
        [411, 421],
        [411, 421],
    ]
    assert drafter.STATS["fixed32_events"] == 1
    assert drafter.STATS["fixed32_rows"] == 2
    assert drafter.STATS["fixed32_carry_slots"] == 8


def test_exact_work_census():
    expected = reference_event(
        "tail6_fixed32", batch_size=4, event_id="tail-b4"
    )["drafter"]
    assert drafter.get_fixed32_drafter_work(4) == expected
    assert drafter.get_fixed32_drafter_work(1) == reference_event(
        "hydra27_fixed32", batch_size=1, event_id="hydra-b1"
    )["drafter"]


def test_failure_is_fail_closed():
    class BrokenCache:
        def speculate(self, *_args, **_kwargs):
            raise RuntimeError("lookup failed")

    drafter.reset_for_test()
    drafter._COMMITTED["r"] = [1]
    head = [[10], [11], [12], [13], [14]]
    seeds = {1: [20], 2: [30]}
    try:
        drafter.decide_fixed32(
            BrokenCache(),
            ["r"],
            head,
            seeds,
            torch.device("cpu"),
            0,
            vocab_size=100,
        )
    except RuntimeError as exc:
        assert "lookup failed" in str(exc)
    else:
        raise AssertionError("fixed32 Arctic failure must not take a fallback")


if __name__ == "__main__":
    tests = (
        test_identical_common_work_for_both_modes,
        test_exact_work_census,
        test_failure_is_fail_closed,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print("PASS fixed32 drafter: common 31-column 6/4/2 strict pack")
