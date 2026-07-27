#!/usr/bin/env python3
"""CPU gate for the merged-drafter orchestration (mock SuffixDecodingCache) — TAIL MODE.

Verifies lifecycle (start/add/stop/evict + rolling buffer + non-gappy delta ingest) and
decide_tail (the shipped tail6 path: Arctic chain appended past the native MTP head;
hit / cold / pattern-seed / pad semantics + TAIL engagement counters).

decide_and_fill (head-merge) tests were removed 2026-07-27 with the path itself
(cleanup+bake, FR13_CLEANUP_BAKE_PLAN.md).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fr13_merged_drafter as md  # noqa: E402

PASS = FAIL = 0
DEV = torch.device("cpu")


def check(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  [PASS] {msg}")
    else:
        FAIL += 1; print(f"  [FAIL] {msg}")


class MockDraft:
    def __init__(self, toks): self.token_ids = toks


class MockCache:
    def __init__(self, spec_map):
        self.active_requests = set()
        self.cached_requests = set()
        self.spec_map = spec_map            # req_id -> list[int] returned by speculate
        self.calls = []

    def start_request(self, req_id, prompt):
        self.active_requests.add(req_id); self.calls.append(("start", req_id, list(prompt)))

    def add_active_response(self, req_id, ids):
        self.calls.append(("add", req_id, list(ids)))

    def speculate(self, req_id, pattern, **kw):
        self.calls.append(("spec", req_id, list(pattern)))
        return MockDraft(list(self.spec_map.get(req_id, [])))

    def stop_request(self, req_id):
        self.active_requests.discard(req_id); self.calls.append(("stop", req_id))

    def evict_cached_response(self, req_id):
        self.cached_requests.discard(req_id); self.calls.append(("evict", req_id))


print("[1] lifecycle: start(+evict), NON-GAPPY delta ingest (prompt not re-fed), stop")
md.reset_for_test()
c = MockCache({})
c.cached_requests.add("r_old")
md.note_new_requests(c, {"r_old": [1, 2, 3], "r_new": [4, 5]})   # prompts len 3 and 2
check(("evict", "r_old") in c.calls, "cached req evicted before start")
check(c.active_requests == {"r_old", "r_new"}, "both reqs started (active)")
check(md.STATS["started"] == 2, f"started counter == 2 (got {md.STATS['started']})")
check(md._INGESTED_LEN["r_new"] == 2, "ingested-len seeded to prompt length (prompt not re-fed)")
# runner ingest: full sequence = prompt [4,5] + generated [10,11]; only the DELTA [10,11] fed to arctic
md.ingest_from_sequence(c, "r_new", [4, 5, 10, 11], num_tokens=4, max_tree_depth=24)
check(("add", "r_new", [10, 11]) in c.calls, "add_active_response fed ONLY the generated delta (non-gappy)")
check(md._COMMITTED["r_new"] == [4, 5, 10, 11], "rolling suffix = recent full sequence (for pattern)")
md.ingest_from_sequence(c, "r_new", [4, 5, 10, 11, 12], num_tokens=5, max_tree_depth=3)  # +1 token
adds = [x for x in c.calls if x[0] == "add" and x[1] == "r_new"]
check(adds[-1] == ("add", "r_new", [12]), f"next step feeds only the new token [12] (got {adds[-1]})")
check(md._COMMITTED["r_new"] == [10, 11, 12], f"suffix capped to max_tree_depth=3 (got {md._COMMITTED['r_new']})")
md.retire_requests(c, ["r_old"])
check(("stop", "r_old") in c.calls and "r_old" not in md._COMMITTED, "retire -> stop + buffer dropped")

print("[2] decide_tail: hit rows get the Arctic chain, short rows pad, pattern = committed++head")
md.reset_for_test()
c = MockCache({"ra": [50, 51, 52, 53, 54, 55], "rb": [60, 61]})   # ra full 6-deep, rb only 2
c.active_requests = {"ra", "rb"}
md._COMMITTED["ra"] = [9, 8]; md._COMMITTED["rb"] = [7]
# native MTP head, depths 0..4; per-depth per-row indexables (row order == req order)
head = [[100, 200], [101, 201], [102, 202], [103, 203], [104, 204]]
tail = md.decide_tail(c, ["ra", "rb"], head, head_depth=5, tail_len=6, device=DEV, pad_token=0)
check(tail is not None and len(tail) == 6, f"returns tail_len=6 columns (got {None if tail is None else len(tail)})")
check(tail[0].tolist() == [50, 60], f"tail d0 = arctic rel0 per row (got {tail[0].tolist()})")
check(tail[1].tolist() == [51, 61], "tail d1 = arctic rel1 per row")
check(tail[2].tolist()[0] == 52 and tail[2].tolist()[1] == 0, "short row pads past its match (pad_token)")
check(tail[5].tolist()[0] == 55, "deep row carries the full 6-token chain")
spec_calls = [x for x in c.calls if x[0] == "spec"]
check(("spec", "ra", [9, 8, 100, 101, 102, 103, 104]) in spec_calls,
      f"pattern ra = committed++native-head ({[x for x in spec_calls if x[1] == 'ra']})")
check(("spec", "rb", [7, 200, 201, 202, 203, 204]) in spec_calls, "pattern rb = committed++native-head")
check(md.STATS["tail_speculate_fired"] == 2, "tail_speculate_fired == 2")
check(md.STATS["tail_hit"] == 2, f"tail_hit == 2 (both rows matched; got {md.STATS['tail_hit']})")
check(md.STATS["tail_all_cold"] == 0, "no all-cold step")

print("[3] decide_tail: all-cold step -> pad columns + tail_all_cold counted")
md.reset_for_test()
c = MockCache({})                      # speculate returns empty draft for every row
c.active_requests = {"ra"}
md._COMMITTED["ra"] = [9]
tail = md.decide_tail(c, ["ra"], [[100], [101], [102], [103], [104]], head_depth=5, tail_len=6,
                      device=DEV, pad_token=0)
check(tail is not None and all(t.tolist() == [0] for t in tail), "cold row -> all-pad columns (never-regress)")
check(md.STATS["tail_all_cold"] == 1 and md.STATS["tail_hit"] == 0, "tail_all_cold counted, no hit")

print("[4] cache=None / empty batch -> graceful no-op")
md.reset_for_test()
check(md.decide_tail(None, ["ra"], [[1]] * 5, 5, 6, DEV, 0) is None, "no cache -> None (head-only fallback)")
check(md.decide_tail(MockCache({}), [], [], 5, 6, DEV, 0) is None, "empty batch -> None")
md.note_new_requests(None, {"r": [1]}); md.ingest_from_sequence(None, "r", [1, 2], 2); md.retire_requests(None, ["r"])
check(md._COMMITTED.get("r") is None, "lifecycle no-ops safely with cache=None (buffer maintained then cleared on retire)")

print(f"\n{PASS}/{PASS+FAIL} checks PASS")
if FAIL == 0:
    print(">>> PASS — merged-drafter TAIL orchestration: lifecycle + rolling buffer, decide_tail "
          "(hit/pad/cold, committed++head pattern seed), TAIL engagement counters.")
    sys.exit(0)
sys.exit(1)
