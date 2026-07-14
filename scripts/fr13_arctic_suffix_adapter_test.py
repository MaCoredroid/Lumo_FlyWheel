#!/usr/bin/env python3
"""CPU unit test for fr13_arctic_suffix_adapter (isolated, no arctic/torch/vllm import).

Proves:
  (A) arctic_draft_to_suffix_rel maps a FLAT arctic draft.token_ids chain -> spine-only suffix_rel
      {i: [token_ids[i]]}, robust to a MOCK draft object (.token_ids), a plain list, a tuple, an
      array-like (.tolist), None, and an empty draft.
  (B) feeding that suffix_rel through fr13_mtp_suffix_assembly.assemble_cat33333 yields the
      MTP near-spine (d<mtp_k) + the Arctic deep-spine (d>=mtp_k) with MTP-topk BRANCH FALLBACK
      (Arctic supplies no per-position alternatives in the first version) -- for mtp_k in {1,2}.
  (C) a cold/None draft -> {} -> assemble_cat33333 == pure-MTP baseline (never regresses).

Run: /home/mark/shared/lumoFlyWheel/.venv/bin/python scripts/fr13_arctic_suffix_adapter_test.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fr13_arctic_suffix_adapter import (  # noqa: E402
    arctic_draft_to_suffix_rel,
    extract_draft_token_ids,
)
from fr13_mtp_suffix_assembly import (  # noqa: E402
    N_DEPTH,
    assemble_cat33333,
    assemble_pure_mtp,
)

_PASS = 0
_FAIL = 0


def check(cond, msg):
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  PASS  {msg}")
    else:
        _FAIL += 1
        print(f"  FAIL  {msg}")


# ---- mock draft objects (stand in for arctic_inference SuffixDecodingCache.speculate() result) ----
class MockDraft:
    """Mimics the real arctic draft: a flat .token_ids list. (vLLM proposer reads draft.token_ids.)"""

    def __init__(self, token_ids):
        self.token_ids = token_ids


class ArrayLikeDraft:
    """Draft whose .token_ids is tensor/ndarray-like (exposes .tolist()) -- robustness probe."""

    class _Arr:
        def __init__(self, data):
            self._d = list(data)

        def tolist(self):
            return list(self._d)

    def __init__(self, token_ids):
        self.token_ids = ArrayLikeDraft._Arr(token_ids)


# ---- fixed MTP inputs shared across assembly checks ----
# MTP argmax spine (the current parallel-drafter spine) and its branch runners-up per depth.
MTP_SPINE = [10, 11, 12, 13, 14]
MTP_TOPK = {0: [110, 111], 1: [112, 113], 2: [114, 115], 3: [116, 117], 4: [118, 119]}


def test_adapter_shape():
    print("[A] adapter: flat token_ids -> spine-only suffix_rel {i:[tok]}")
    d = MockDraft([101, 102, 103, 104])
    sr = arctic_draft_to_suffix_rel(d)
    check(sr == {0: [101], 1: [102], 2: [103], 3: [104]}, f"MockDraft -> {sr}")
    check(all(isinstance(k, int) for k in sr), "keys are ints")
    check(all(isinstance(v, list) and len(v) == 1 and isinstance(v[0], int)
              for v in sr.values()), "values are single-int lists")

    # robust: plain list
    sr2 = arctic_draft_to_suffix_rel([201, 202, 203])
    check(sr2 == {0: [201], 1: [202], 2: [203]}, f"plain list -> {sr2}")

    # robust: tuple
    sr3 = arctic_draft_to_suffix_rel((301, 302))
    check(sr3 == {0: [301], 1: [302]}, f"tuple -> {sr3}")

    # robust: array-like .tolist()
    sr4 = arctic_draft_to_suffix_rel(ArrayLikeDraft([401, 402, 403]))
    check(sr4 == {0: [401], 1: [402], 2: [403]}, f"array-like .tolist() -> {sr4}")

    # robust: coercion (numpy/torch-ish scalars would arrive; simulate with bool/str-free ints)
    check(extract_draft_token_ids(MockDraft([True, 2, 3])) == [1, 2, 3], "int-coercion via int()")

    # cold / empty / None -> {}
    check(arctic_draft_to_suffix_rel(None) == {}, "None draft -> {}")
    check(arctic_draft_to_suffix_rel(MockDraft([])) == {}, "empty .token_ids -> {}")
    check(arctic_draft_to_suffix_rel([]) == {}, "empty list -> {}")
    check(arctic_draft_to_suffix_rel(MockDraft(None)) == {}, ".token_ids is None -> {}")

    # max_rel cap keeps only the deep-spine positions the live drafter grows
    srcap = arctic_draft_to_suffix_rel(MockDraft([501, 502, 503, 504, 505]), max_rel=3)
    check(srcap == {0: [501], 1: [502], 2: [503]}, f"max_rel=3 caps -> {srcap}")


def test_assembly_mtp_k1():
    print("[B] assemble_cat33333: mtp_k=1  (near-spine d0 MTP; deep d1..d4 Arctic; branches MTP)")
    # Arctic deep-spine continuation: rel i=0 -> depth mtp_k+0 = 1, ... i=3 -> depth 4
    draft = MockDraft([900, 901, 902, 903])
    suffix_rel = arctic_draft_to_suffix_rel(draft)
    nodes, meta = assemble_cat33333(MTP_SPINE, MTP_TOPK, suffix_rel, mtp_k=1)

    # per-depth: [spine, branch_a, branch_b]
    # d0 (near): MTP spine 10; root branches ALWAYS MTP topk[0]
    # d1..d4 (deep): Arctic spine 900..903; branches fall back to MTP topk[d]
    expected = [
        10, 110, 111,      # d0  MTP near-spine + MTP root branches
        900, 112, 113,     # d1  Arctic deep-spine + MTP branch fallback
        901, 114, 115,     # d2
        902, 116, 117,     # d3
        903, 118, 119,     # d4
    ]
    check(nodes == expected, f"nodes == expected ({nodes})")
    check(meta["spine_src"] == ["mtp", "suffix", "suffix", "suffix", "suffix"],
          f"spine provenance {meta['spine_src']}")
    check(meta["branch_src"] == ["mtp", "mtp_fallback", "mtp_fallback", "mtp_fallback", "mtp_fallback"],
          f"branch provenance (all MTP fallback past root) {meta['branch_src']}")
    # explicit: deep spine came from Arctic, near spine from MTP
    check(nodes[0] == MTP_SPINE[0], "d0 spine is MTP near token")
    check([nodes[3], nodes[6], nodes[9], nodes[12]] == [900, 901, 902, 903],
          "d1..d4 spine are the Arctic continuation tokens")


def test_assembly_mtp_k2():
    print("[B] assemble_cat33333: mtp_k=2  (near-spine d0,d1 MTP; deep d2..d4 Arctic)")
    # rel i=0 -> depth 2, i=1 -> depth 3, i=2 -> depth 4
    draft = MockDraft([900, 901, 902])
    suffix_rel = arctic_draft_to_suffix_rel(draft)
    nodes, meta = assemble_cat33333(MTP_SPINE, MTP_TOPK, suffix_rel, mtp_k=2)
    expected = [
        10, 110, 111,      # d0 MTP near
        11, 112, 113,      # d1 MTP near
        900, 114, 115,     # d2 Arctic deep + MTP branch fallback
        901, 116, 117,     # d3
        902, 118, 119,     # d4
    ]
    check(nodes == expected, f"nodes == expected ({nodes})")
    check(meta["spine_src"] == ["mtp", "mtp", "suffix", "suffix", "suffix"],
          f"spine provenance {meta['spine_src']}")
    check([nodes[6], nodes[9], nodes[12]] == [900, 901, 902],
          "d2..d4 spine are the Arctic continuation tokens")


def test_cold_equals_baseline():
    print("[C] cold/None draft -> {} -> assemble_cat33333 == pure-MTP baseline (no regression)")
    for mtp_k in (1, 2):
        suffix_rel = arctic_draft_to_suffix_rel(None)
        check(suffix_rel == {}, f"None -> empty suffix_rel (mtp_k={mtp_k})")
        nodes, meta = assemble_cat33333(MTP_SPINE, MTP_TOPK, suffix_rel, mtp_k=mtp_k)
        baseline = assemble_pure_mtp(MTP_SPINE, MTP_TOPK)
        check(nodes == baseline, f"mtp_k={mtp_k}: assembled == pure-MTP baseline ({nodes})")
        check(all(s in ("mtp", "mtp_fallback") for s in meta["spine_src"]),
              f"mtp_k={mtp_k}: no suffix provenance when cold")


def test_short_draft_partial_fill():
    print("[B] short Arctic draft: only fills as many deep-spine positions as it has, rest MTP")
    # mtp_k=1, Arctic returns only 2 tokens -> depths 1,2 Arctic; depths 3,4 fall back to MTP spine
    draft = MockDraft([900, 901])
    suffix_rel = arctic_draft_to_suffix_rel(draft)
    nodes, meta = assemble_cat33333(MTP_SPINE, MTP_TOPK, suffix_rel, mtp_k=1)
    check(nodes[0] == 10 and nodes[3] == 900 and nodes[6] == 901, "d0 MTP, d1/d2 Arctic")
    check(nodes[9] == MTP_SPINE[3] and nodes[12] == MTP_SPINE[4], "d3/d4 fall back to MTP spine")
    check(meta["spine_src"] == ["mtp", "suffix", "suffix", "mtp_fallback", "mtp_fallback"],
          f"partial-fill provenance {meta['spine_src']}")
    check(len(nodes) == 3 * N_DEPTH, "still exactly 15 cat33333 nodes")


def main():
    test_adapter_shape()
    test_assembly_mtp_k1()
    test_assembly_mtp_k2()
    test_cold_equals_baseline()
    test_short_draft_partial_fill()
    print(f"\nRESULT: {_PASS} passed, {_FAIL} failed")
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
