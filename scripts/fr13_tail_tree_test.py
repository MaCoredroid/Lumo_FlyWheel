#!/usr/bin/env python3
"""CPU unit tests for the accept>5 tail-tree assembly (fr13_mtp_suffix_assembly.assemble_tail_tree).

Verifies: (1) topology consistency (head == CAT33333_ORDER, 31 nodes -> n_pad=32, pure chain tail);
(2) NEVER-REGRESS (mtp_k=head_depth cold -> head == baseline cat33333, tail = pad-repeat, never matches
past the head so accept==baseline); (3) warm arctic tail fills the chain; (4) partial arctic fills then pads.
Run: .venv/bin/python scripts/fr13_tail_tree_test.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from fr13_mtp_suffix_assembly import (assemble_tail_tree, tail_tree_order,
                                       CAT33333_ORDER, _pure_mtp)


def test_topology():
    order = tail_tree_order()                      # head_depth=5, tail_len=16, branches=2
    assert order[:15] == CAT33333_ORDER, f"head mismatch: {order[:15]}"
    assert len(order) == 31, f"want 31 nodes got {len(order)}"
    assert order[15] == (0,) * 6 and order[30] == (0,) * 21, f"tail order: {order[15]}..{order[30]}"


def test_never_regress_cold():
    ms = [101, 102, 103, 104, 105]
    tk = {0: [201, 202], 1: [211, 212], 2: [221, 222], 3: [231, 232], 4: [241, 242]}
    nodes, meta = assemble_tail_tree(ms, tk, {}, mtp_k=5)
    assert nodes[:15] == _pure_mtp(ms, tk), "cold head must == baseline cat33333"
    assert all(t == 105 for t in nodes[15:]), "cold tail must pad-repeat last spine"
    assert all(s == "pad" for s in meta["tail_src"])


def test_warm_tail():
    ms = [101, 102, 103, 104, 105]
    tk = {0: [201, 202], 1: [211, 212], 2: [221, 222], 3: [231, 232], 4: [241, 242]}
    sr = {j: [9000 + j] for j in range(16)}
    nodes, meta = assemble_tail_tree(ms, tk, sr, mtp_k=5)
    assert nodes[:15] == _pure_mtp(ms, tk), "warm head must still == baseline"
    assert nodes[15:] == [9000 + j for j in range(16)], "tail must be arctic-filled"
    assert all(s == "suffix" for s in meta["tail_src"])


def test_partial_then_pad():
    ms = [101, 102, 103, 104, 105]
    tk = {0: [201, 202], 1: [211, 212], 2: [221, 222], 3: [231, 232], 4: [241, 242]}
    sr = {j: [7000 + j] for j in range(6)}
    nodes, _ = assemble_tail_tree(ms, tk, sr, mtp_k=5)
    assert nodes[15:21] == [7000 + j for j in range(6)]
    assert all(t == 7005 for t in nodes[21:]), "must pad-repeat last filled (7005)"


if __name__ == "__main__":
    for fn in (test_topology, test_never_regress_cold, test_warm_tail, test_partial_then_pad):
        fn(); print(f"OK {fn.__name__}")
    print("ALL TAIL-TREE ASSEMBLY TESTS PASS")
