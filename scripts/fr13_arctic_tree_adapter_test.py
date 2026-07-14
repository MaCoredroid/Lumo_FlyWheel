#!/usr/bin/env python3
"""v2 tree-adapter test: arctic use_tree_spec draft (token_ids/parents/probs) -> per-depth ranked
suffix_rel -> assemble_cat33333 fills BRANCHES from the trie (the user's trie-branches mapping)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fr13_arctic_suffix_adapter import arctic_tree_to_suffix_rel
from fr13_mtp_suffix_assembly import assemble_cat33333, N_DEPTH

PASS = FAIL = 0
def check(c, m):
    global PASS, FAIL
    PASS, FAIL = (PASS+1, FAIL) if c else (PASS, FAIL+1)
    print(("  [PASS] " if c else "  [FAIL] ") + m)

class TreeDraft:
    def __init__(self, token_ids, parents, probs):
        self.token_ids, self.parents, self.probs = token_ids, parents, probs

print("[1] real arctic branching example: [1,2,3] -> children [4,7]")
# from the live container: token_ids=[4,7,5,8,6,9,1,2] parents=[-1,-1,0,1,2,3,5,6] probs all 0.5
d = TreeDraft([4,7,5,8,6,9,1,2], [-1,-1,0,1,2,3,5,6], [0.5]*8)
sr = arctic_tree_to_suffix_rel(d)
check(set(sr.get(0, [])) == {4,7}, f"depth0 = the two branch continuations {{4,7}} (got {sr.get(0)})")
check(4 in sr.get(0, []) and 7 in sr.get(0, []), "both branches present at depth 0")
check(sr.get(1) is not None and 5 in sr.get(1, []) and 8 in sr.get(1, []), f"depth1 = {{5,8}} (got {sr.get(1)})")

print("[2] ranked by prob: higher-prob token becomes the spine (rank 0)")
# depth0: token 100 prob 0.9, token 200 prob 0.1 -> 100 first
d2 = TreeDraft([100,200,101], [-1,-1,0], [0.9,0.1,0.9])
sr2 = arctic_tree_to_suffix_rel(d2)
check(sr2[0][0] == 100, f"depth0 rank0 = higher-prob 100 (got {sr2[0]})")
check(sr2[0][1] == 200, "depth0 rank1 = lower-prob 200 (the branch)")

print("[3] tree adapter -> assemble_cat33333: BRANCHES now come from the trie (mtp_k=1)")
# depth0 {4,7}, depth1 {5,8}, depth2 {6,9}, depth3 {1}, depth4 {2}
mtp_spine = [900, 901, 902, 903, 904]      # MTP argmax spine (only [0] used for mtp_k=1)
mtp_topk = {d: [800+d, 810+d] for d in range(N_DEPTH)}
nodes, meta = assemble_cat33333(mtp_spine, mtp_topk, sr, mtp_k=1)
# spine: d0=MTP(900), d1..=suffix rank0
check(nodes[0] == 900, "spine d0 = MTP root")
# d1 (abs depth1) uses suffix_rel[0] (rel index 0) -> ranked [4,7] -> spine=4, branch=7 (from TRIE not MTP)
check(nodes[3] == 4, f"spine d1 = trie rank0 = 4 (got {nodes[3]})")
check(7 in [nodes[4], nodes[5]], f"d1 BRANCH = trie sibling 7 (from trie, not MTP fallback) (got {nodes[4:6]})")
check(meta["branch_src"][1] == "suffix", "d1 branch provenance = suffix (v2 trie branches engaged)")

print("[4] flat draft (no parents) -> falls back to v1 spine-only cleanly")
class FlatDraft:
    def __init__(self, t): self.token_ids = t
srf = arctic_tree_to_suffix_rel(FlatDraft([11,12,13,14]))
check(srf == {0:[11],1:[12],2:[13],3:[14]}, f"flat draft -> v1 spine-only suffix_rel (got {srf})")

print(f"\n{PASS}/{PASS+FAIL} checks PASS")
sys.exit(0 if FAIL==0 else 1)
