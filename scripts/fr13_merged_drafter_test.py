#!/usr/bin/env python3
"""CPU gate for the merged-drafter orchestration (mock SuffixDecodingCache).

Verifies lifecycle (start/add/stop/evict + rolling buffer), the ADAPTIVE gate (all-rows full-depth
match -> skip+columns; any miss -> None = never-regress), pattern = committed ++ mtp-near, and the
engagement counters (needle for the live ENGAGED assert).
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

print("[2] adaptive gate: ALL rows full-depth match -> skip + columns")
md.reset_for_test()
# mtp_k=1, need = 5-1 = 4 deep tokens; both rows get >=4 from arctic
c = MockCache({"ra": [50, 51, 52, 53], "rb": [60, 61, 62, 63]})
c.active_requests = {"ra", "rb"}
md._COMMITTED["ra"] = [9, 8]; md._COMMITTED["rb"] = [7]
near = [[1000, 2000]]                        # depth0 (root) per-row: ra=1000, rb=2000
topk = {d: [[3000 + d, 3001 + d], [4000 + d, 4001 + d]] for d in range(5)}  # per-depth [rank2,rank3] cols
spine, wide, skip = md.decide_and_fill(c, ["ra", "rb"], near, topk, mtp_k=1, device=DEV, pad_token=0)
check(skip is True, "all-full-match -> do_skip True")
check(spine is not None and len(spine) == 5, "returns 5 spine columns")
check(spine[0].tolist() == [1000, 2000], "spine d0 = MTP root (row-major)")
check(spine[1].tolist() == [50, 60], "spine d1 = arctic rel0 (row-major, req-keyed)")
check(spine[4].tolist() == [53, 63], "spine d4 = arctic rel3")
# pattern = committed ++ near
spec_calls = [x for x in c.calls if x[0] == "spec"]
check(("spec", "ra", [9, 8, 1000]) in spec_calls, f"pattern ra = committed++near ({[x for x in spec_calls if x[1]=='ra']})")
check(("spec", "rb", [7, 2000]) in spec_calls, "pattern rb = committed++near")
check(md.STATS["skip_fired"] == 1 and md.STATS["assembler_engaged"] == 1, "engagement counters incremented")

print("[3] adaptive gate: ANY row short match -> None = never-regress")
md.reset_for_test()
c = MockCache({"ra": [50, 51, 52, 53], "rb": [60]})   # rb only 1 token < need 4
c.active_requests = {"ra", "rb"}
spine, wide, skip = md.decide_and_fill(c, ["ra", "rb"], [[1000, 2000]], topk, mtp_k=1, device=DEV, pad_token=0)
check(skip is False and spine is None and wide is None, "partial match -> (None,None,False) never-regress")
check(md.STATS["match_partial_norun"] == 1 and md.STATS["skip_fired"] == 0, "no skip fired; partial-norun counted")
check(md.STATS["speculate_fired"] == 2, "speculate still fired for both rows (needle non-vacuous)")

print("[4] cache=None (arctic unavailable) -> graceful no-op, never-regress")
md.reset_for_test()
spine, wide, skip = md.decide_and_fill(None, ["ra"], [[1000]], topk, mtp_k=1, device=DEV, pad_token=0)
check(skip is False and spine is None, "no cache -> None (full MTP fallback)")
md.note_new_requests(None, {"r": [1]}); md.ingest_from_sequence(None, "r", [1, 2], 2); md.retire_requests(None, ["r"])
check(md._COMMITTED.get("r") is None, "lifecycle no-ops safely with cache=None (buffer maintained then cleared on retire)")

print("[5] mtp_k=2: near covers d0,d1; arctic fills d2,d3,d4 (need=3)")
md.reset_for_test()
c = MockCache({"ra": [70, 71, 72]})           # 3 tokens == need for mtp_k=2
c.active_requests = {"ra"}
near2 = [[1000], [1001]]                       # d0=1000, d1=1001 for row0
spine, wide, skip = md.decide_and_fill(c, ["ra"], near2, topk, mtp_k=2, device=DEV, pad_token=0)
check(skip is True, "mtp_k=2 all-full -> skip")
check(spine[0].tolist() == [1000] and spine[1].tolist() == [1001], "d0,d1 = MTP near")
check(spine[2].tolist() == [70] and spine[4].tolist() == [72], "d2..d4 = arctic (need=3)")

print(f"\n{PASS}/{PASS+FAIL} checks PASS")
if FAIL == 0:
    print(">>> PASS — merged-drafter orchestration: lifecycle + rolling buffer, adaptive gate "
          "(all-match->skip, any-miss->never-regress), req-keyed pattern, engagement needle.")
    sys.exit(0)
sys.exit(1)
