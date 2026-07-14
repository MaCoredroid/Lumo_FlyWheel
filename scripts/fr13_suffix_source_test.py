#!/usr/bin/env python3
"""CPU unit test for scripts/fr13_suffix_source.py (SuffixSource).

Boot-free, deterministic (fixed-seed random.Random only), no torch/GPU/network.
Run:  /home/mark/shared/lumoFlyWheel/.venv/bin/python scripts/fr13_suffix_source_test.py
Exit 0 = all PASS.
"""
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fr13_suffix_source import SuffixSource  # noqa: E402

_P = [0, 0]  # [passed, failed]


def check(cond, name):
    ok = bool(cond)
    _P[0 if ok else 1] += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return ok


class _FakeTensor:
    """Minimal 1-D int-tensor stand-in exposing .tolist() (no torch dep)."""
    def __init__(self, data):
        self._d = list(data)

    def tolist(self):
        return list(self._d)


# --- 1. REPETITIVE: current suffix has a prior occurrence -> copy-forward ----
def t1_repetitive():
    print("[1] repetitive stream")
    rng = random.Random(0)
    noise = [rng.randint(100, 200) for _ in range(30)]
    stream = [10, 11, 12, 13, 14, 15] + noise + [10, 11, 12]
    s = SuffixSource()
    s.update("r1", stream)
    r = s.propose("r1", 6)
    check(r["matched"] is True, "matched True on repeated pattern")
    check(r["match_len"] >= 3, f"match_len>=3 (got {r['match_len']})")
    check(r["spine"][:3] == [13, 14, 15], f"spine starts [13,14,15] (got {r['spine'][:3]})")


# --- 2. RANDOM/cold: no hallucinated match ----------------------------------
def t2_cold():
    print("[2] random/cold stream")
    rng = random.Random(1234)
    stream = [rng.randint(0, 1_000_000) for _ in range(300)]
    s = SuffixSource()
    s.update("cold", stream)
    r = s.propose("cold", 8)
    check(r["matched"] is False, f"matched False on random stream (got {r})")
    check(r["spine"] == [] and r["match_len"] == 0, "empty spine / match_len 0 when cold")


# --- 3. BRANCH candidates: one k-gram, two distinct followers ----------------
def t3_branch():
    print("[3] branch candidates (two followers)")
    # (20,21) -> 30 twice, -> 31 once ; suffix ends with (20,21).
    # Distinct sentinels (700..703) before each pair keep the 3-gram context
    # unique so match_len == 2 and the depth-0 frequency ranking is exercised.
    stream = [700, 20, 21, 30, 701, 20, 21, 30, 702, 20, 21, 31, 703, 20, 21]
    s = SuffixSource()
    s.update("b", stream)
    r = s.propose("b", 4)
    b0 = r["branch_candidates"][0] if r["branch_candidates"] else []
    check(r["matched"] is True, "matched True")
    check(r["match_len"] == 2, f"match_len==2 (got {r['match_len']})")
    check(30 in b0 and 31 in b0, f"both followers present in branch[0] (got {b0})")
    check(b0.index(30) < b0.index(31), f"30 ranked before 31 by frequency (got {b0})")
    check(r["spine"][0] == 30, f"spine[0]==30 (most frequent) (got {r['spine'][:1]})")


# --- 4. INCREMENTAL == batch (rolling index is correct) ---------------------
def t4_incremental():
    print("[4] incremental == batch")
    rng = random.Random(7)
    base = [10, 11, 12, 13, 14, 15]
    stream = []
    for _ in range(40):
        stream += base + [rng.randint(50, 90)]
    stream += [10, 11, 12]                    # matchable suffix

    a = SuffixSource()
    a.update("r", stream)
    b = SuffixSource()
    for t in stream:                          # one token at a time
        b.update("r", [t])

    check(a.reqs["r"].tokens == b.reqs["r"].tokens, "token streams identical")
    check(a.reqs["r"].index == b.reqs["r"].index, "rolling index identical to batch")
    check(a.propose("r", 8) == b.propose("r", 8), "propose() identical")

    # boundary: list input == 1-D tensor-like input
    c = SuffixSource()
    c.update("r", _FakeTensor(stream))
    check(c.reqs["r"].index == a.reqs["r"].index, "tensor-input path == list-input path")


# --- 5. EVICT frees state (KeyError-safe afterwards) -------------------------
def t5_evict():
    print("[5] evict frees state")
    s = SuffixSource()
    s.update("gone", [1, 2, 3, 1, 2, 3])
    check(s.propose("gone", 4)["matched"] is True, "matched before evict")
    s.evict("gone")
    check("gone" not in s.reqs, "state removed from reqs dict")
    r = s.propose("gone", 4)
    check(r["matched"] is False and r["spine"] == [], "propose after evict is safe / no-match")
    s.evict("never-existed")                  # must not raise
    check(True, "evict of unknown req_id does not raise")


# --- 6. match_rate needle reflects matched/total ----------------------------
def t6_match_rate():
    print("[6] match_rate needle")
    s = SuffixSource()
    s.update("hot", [5, 6, 7, 5, 6, 7])       # (5,6,7)-> matchable
    s.propose("hot", 4)                        # match
    s.propose("hot", 4)                        # match
    s.propose("ghost", 4)                      # unknown -> no match
    s.propose("ghost", 4)                      # unknown -> no match
    check(s.total_proposals == 4, f"total_proposals==4 (got {s.total_proposals})")
    check(s.matched_proposals == 2, f"matched_proposals==2 (got {s.matched_proposals})")
    check(abs(s.match_rate - 0.5) < 1e-12, f"match_rate==0.5 (got {s.match_rate})")


# --- 7. amortised cost: 50k tokens, no O(n^2) blowup ------------------------
def t7_amortised():
    print("[7] amortised cost (50k tokens)")
    rng = random.Random(7)
    big = [rng.randint(0, 5000) for _ in range(50_000)]
    s = SuffixSource()
    t0 = time.perf_counter()
    s.update("big", big)
    s.propose("big", 8)
    el = time.perf_counter() - t0
    print(f"    elapsed={el:.3f}s (soft target < 2s)")
    check(el < 20.0, f"completes < 20s hard cap (got {el:.3f}s)")


def main():
    for fn in (t1_repetitive, t2_cold, t3_branch, t4_incremental,
               t5_evict, t6_match_rate, t7_amortised):
        fn()
    passed, failed = _P
    print(f"\n{passed}/{passed + failed} checks PASS", end="")
    print("" if not failed else f"  ({failed} FAIL)")
    if failed:
        print(">>> FAIL")
        return 1
    print(">>> PASS — SuffixSource match logic + rolling index verified (CPU)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
