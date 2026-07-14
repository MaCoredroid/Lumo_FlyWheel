#!/usr/bin/env python3
"""CPU gate for the MTP-k + Arctic-suffix -> cat33333 assembly core.

Proves the correctness properties (source-agnostic; real Arctic tokens enter via the live
adapter): cold-fallback == pure MTP (byte-identical baseline), MTP/suffix provenance by mtp_k,
per-node dedup, valid 15-node tree, and that spine/branch tokens land in the right slots.
suffix_rel is RELATIVE to the pattern end (Arctic's .token_ids): suffix_rel[i] -> abs depth mtp_k+i.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fr13_mtp_suffix_assembly import (  # noqa: E402
    assemble_cat33333, assemble_pure_mtp, CAT33333_ORDER, N_DEPTH,
)

PASS = FAIL = 0


def check(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  [PASS] {msg}")
    else:
        FAIL += 1; print(f"  [FAIL] {msg}")


def spine_slots(nodes):   # slot 0 of each depth-triple
    return [nodes[3 * d] for d in range(N_DEPTH)]


def branch_slots(nodes, d):
    return [nodes[3 * d + 1], nodes[3 * d + 2]]


# distinct MTP tokens so provenance is unambiguous
MTP_SPINE = [1000, 1001, 1002, 1003, 1004]
MTP_TOPK = {d: [2000 + 10 * d, 2001 + 10 * d] for d in range(N_DEPTH)}
# relative suffix continuation: rel[i] = candidates for the i-th token past the MTP prefix
SUFFIX_REL = {
    0: [5000, 5001, 5002],
    1: [5010, 5011, 5012],
    2: [5020, 5021, 5022],
    3: [5030, 5031, 5032],
    4: [5040, 5041, 5042],
}

print("[1] cold context (no suffix) == pure MTP baseline (byte-identical), any mtp_k")
pure = assemble_pure_mtp(MTP_SPINE, MTP_TOPK)
check(len(pure) == 15, f"pure MTP produces 15 nodes (got {len(pure)})")
for k in (1, 2):
    cold, meta = assemble_cat33333(MTP_SPINE, MTP_TOPK, {}, mtp_k=k)
    check(cold == pure, f"mtp_k={k}: cold assemble == pure MTP (byte-identical fallback)")
    check(all(s in ("mtp", "mtp_fallback") for s in meta["spine_src"]),
          f"mtp_k={k}: cold provenance all-MTP ({meta['spine_src']})")
check(spine_slots(pure) == MTP_SPINE, "pure MTP spine == MTP argmax spine")
for d in range(N_DEPTH):
    check(branch_slots(pure, d) == MTP_TOPK[d], f"pure MTP branches[d{d}] == MTP topk")

print("[2] mtp_k=1: spine[0]=MTP, spine[1..4]=suffix rel[0..3]; suffix branches used (d>=1)")
nodes, meta = assemble_cat33333(MTP_SPINE, MTP_TOPK, SUFFIX_REL, mtp_k=1)
sp = spine_slots(nodes)
check(sp == [1000, 5000, 5010, 5020, 5030],
      f"mtp_k=1 spine = [MTP0, rel0_0, rel1_0, rel2_0, rel3_0] (got {sp})")
check(meta["spine_src"] == ["mtp", "suffix", "suffix", "suffix", "suffix"],
      f"provenance mtp then suffix ({meta['spine_src']})")
check(branch_slots(nodes, 0) == MTP_TOPK[0],
      f"d0 branches = MTP topk (root runners-up ALWAYS MTP) ({branch_slots(nodes,0)})")
check(branch_slots(nodes, 1) == [5001, 5002],
      f"d1 branches = suffix rel0 rank1/2 ({branch_slots(nodes,1)})")
check(meta["branch_src"][1] == "suffix", "d1 branch src = suffix")

print("[3] mtp_k=2: spine[0,1]=MTP, spine[2..4]=suffix rel[0..2]")
nodes2, meta2 = assemble_cat33333(MTP_SPINE, MTP_TOPK, SUFFIX_REL, mtp_k=2)
sp2 = spine_slots(nodes2)
check(sp2 == [1000, 1001, 5000, 5010, 5020],
      f"mtp_k=2 spine = [MTP0, MTP1, rel0_0, rel1_0, rel2_0] (got {sp2})")
check(meta2["spine_src"] == ["mtp", "mtp", "suffix", "suffix", "suffix"],
      f"provenance mtp,mtp,suffix... ({meta2['spine_src']})")
check(branch_slots(nodes2, 1) == MTP_TOPK[1], f"d1 (still MTP region) branches = MTP topk ({branch_slots(nodes2,1)})")

print("[4] per-node dedup: suffix branch == spine token is dropped (mtp_k=1, d1 uses rel0)")
dupsuf = {0: [7000, 7000, 7001]}   # rel0: rank0=7000 (spine d1), rank1=7000 (dup!), rank2=7001
nodes3, _ = assemble_cat33333(MTP_SPINE, MTP_TOPK, dupsuf, mtp_k=1)
b1 = branch_slots(nodes3, 1)
check(nodes3[3 * 1] == 7000, f"d1 spine = 7000 (got {nodes3[3*1]})")
check(7000 not in b1, f"d1 branches exclude the spine token 7000 (got {b1})")
check(7001 in b1, f"d1 branches include the distinct suffix alt 7001 (got {b1})")
check(len(set(nodes3[3:6])) == 3, f"d1 triple all-distinct ({nodes3[3:6]})")

print("[5] every depth-triple internally distinct (no self-collision), all suffix shapes/k")
CASES = [("cold", {}), ("norm", SUFFIX_REL), ("dup", dupsuf), ("degenerate", {1: [9, 9, 9, 9]})]
for name, src in CASES:
    for k in (1, 2):
        nn, _ = assemble_cat33333(MTP_SPINE, MTP_TOPK, src, mtp_k=k)
        ok = all(len(set(nn[3 * d:3 * d + 3])) == 3 for d in range(N_DEPTH))
        check(ok, f"suffix={name} k={k}: all 5 triples distinct")

print("[6] partial suffix (short match): rel covers 0..1 only, deeper spine falls back to MTP")
partial = {0: [6000, 6001, 6002], 1: [6010, 6011]}   # only 2 continuation positions
nodes4, meta4 = assemble_cat33333(MTP_SPINE, MTP_TOPK, partial, mtp_k=1)
sp4 = spine_slots(nodes4)
check(sp4 == [1000, 6000, 6010, 1003, 1004],
      f"spine: MTP0, rel0, rel1, then MTP-fallback d3,d4 (got {sp4})")
check(meta4["spine_src"] == ["mtp", "suffix", "suffix", "mtp_fallback", "mtp_fallback"],
      f"provenance switches to fallback when suffix runs out ({meta4['spine_src']})")
check(branch_slots(nodes4, 3) == MTP_TOPK[3], f"d3 (past suffix) branches fall back to MTP topk ({branch_slots(nodes4,3)})")

print("[7] output shape / order invariants")
check(len(CAT33333_ORDER) == 15 and len(nodes) == 15, "15 nodes matches cat33333 tree_choices")
check(all(isinstance(x, int) for x in nodes), "all node tokens are ints")

print(f"\n{PASS}/{PASS + FAIL} checks PASS")
if FAIL == 0:
    print(">>> PASS — assembly core: cold==pure-MTP (byte-identical), MTP/suffix provenance by "
          "mtp_k, per-node dedup, MTP-fallback when suffix short, valid 15-node cat33333.")
    sys.exit(0)
sys.exit(1)
