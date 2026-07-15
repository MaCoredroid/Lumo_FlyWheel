#!/usr/bin/env python3
"""CPU gate for the merged-drafter fill (assembled nodes -> packer tensors).

Verifies the red-team tensor-discipline requirements: device/dtype, [batch] & [batch,>=3] shapes,
row-major ordering preserved, col0=spine (dedup-safe) + cols1,2=branches, PAD-densified None rows,
and an END-TO-END path (assemble_cat33333 -> build_cat33333_columns) matching the packer's
_fr10_wide_plan read (spine_tokens[pp] for rk0, wide_topk[pp][:,rk] for rk>0).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fr13_merged_fill import build_cat33333_columns, N_DEPTH  # noqa: E402
from fr13_mtp_suffix_assembly import assemble_cat33333        # noqa: E402
from fr13_arctic_suffix_adapter import arctic_draft_to_suffix_rel  # noqa: E402

PASS = FAIL = 0
DEV = torch.device("cpu")


def check(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  [PASS] {msg}")
    else:
        FAIL += 1; print(f"  [FAIL] {msg}")


print("[1] shapes/device/dtype (batch=3, width=3)")
# 3 distinct rows, each a 15-int CAT33333_ORDER list (spine_d, ba_d, bb_d) x5
rows = [[100 + 10 * d + r for d in range(N_DEPTH) for r in range(3)],  # row0
        [200 + 10 * d + r for d in range(N_DEPTH) for r in range(3)],  # row1
        [300 + 10 * d + r for d in range(N_DEPTH) for r in range(3)]]  # row2
spine, wide = build_cat33333_columns(rows, DEV, pad_token=0)
check(len(spine) == N_DEPTH, f"spine_tokens has {N_DEPTH} entries (got {len(spine)})")
check(all(t.shape == (3,) and t.dtype == torch.int64 and t.device == DEV for t in spine),
      "each spine col = int64 [batch=3] on device")
check(set(wide.keys()) == set(range(N_DEPTH)), f"wide_topk keys 0..{N_DEPTH-1} ({sorted(wide.keys())})")
check(all(wide[d].shape == (3, 3) and wide[d].dtype == torch.int64 for d in wide),
      "each wide_topk[d] = int64 [batch=3, width=3]")

print("[2] row-major ordering preserved + col0=spine / cols1,2=branches")
for d in range(N_DEPTH):
    # spine col == node[3d] of each row, in row order
    check(spine[d].tolist() == [rows[b][3 * d] for b in range(3)],
          f"d{d} spine col row-major == rows' node[{3*d}]")
    # wide_topk[d][:,0]==spine, [:,1]==branch_a, [:,2]==branch_b
    check(wide[d][:, 0].tolist() == [rows[b][3 * d] for b in range(3)], f"d{d} wide col0 == spine (dedup-safe)")
    check(wide[d][:, 1].tolist() == [rows[b][3 * d + 1] for b in range(3)], f"d{d} wide col1 == branch_a")
    check(wide[d][:, 2].tolist() == [rows[b][3 * d + 2] for b in range(3)], f"d{d} wide col2 == branch_b")

print("[3] PAD-densify: a None row is filled with pad_token everywhere")
rows2 = [rows[0], None, rows[2]]
sp2, wd2 = build_cat33333_columns(rows2, DEV, pad_token=7)
ok_pad = all(sp2[d][1].item() == 7 for d in range(N_DEPTH)) and all((wd2[d][1] == 7).all().item() for d in range(N_DEPTH))
check(ok_pad, "None row -> pad_token(7) at every spine + wide slot")
check(sp2[0][0].item() == rows[0][0] and sp2[0][2].item() == rows[2][0], "non-None rows unaffected by the pad row")

print("[4] width>4 pads extra cols; width<3 rejected")
sp5, wd5 = build_cat33333_columns([rows[0]], DEV, pad_token=9, width=5)
check(wd5[0].shape == (1, 5), "width=5 -> [batch,5]")
check(wd5[0][0, 3].item() == 9 and wd5[0][0, 4].item() == 9, "cols 3,4 padded")
try:
    build_cat33333_columns([rows[0]], DEV, pad_token=0, width=2); raised = False
except AssertionError:
    raised = True
check(raised, "width<3 raises (packer reads ranks 1,2)")

print("[5] END-TO-END: adapter -> assemble -> fill matches packer _fr10_wide_plan read")
# simulate MTP per-row scalars + an arctic flat-chain draft, mtp_k=1
class MockDraft:
    def __init__(self, t): self.token_ids = t
B = 2
mtp_spine_rows = [[1000, 1001, 1002, 1003, 1004], [1100, 1101, 1102, 1103, 1104]]
mtp_topk_rows = [{d: [2000 + 10 * d, 2001 + 10 * d] for d in range(N_DEPTH)},
                 {d: [2100 + 10 * d, 2101 + 10 * d] for d in range(N_DEPTH)}]
arctic_rows = [MockDraft([5000, 5001, 5002, 5003]), MockDraft([6000, 6001, 6002, 6003])]  # 4 deep tokens
assembled = []
for b in range(B):
    sr = arctic_draft_to_suffix_rel(arctic_rows[b])
    nodes, _ = assemble_cat33333(mtp_spine_rows[b], mtp_topk_rows[b], sr, mtp_k=1)
    assembled.append(nodes)
spine3, wide3 = build_cat33333_columns(assembled, DEV, pad_token=0)
# mtp_k=1: spine[0]=MTP, spine[1..4]=arctic rel[0..3]
check(spine3[0].tolist() == [1000, 1100], "e2e spine d0 = MTP argmax (row-major)")
check(spine3[1].tolist() == [5000, 6000], "e2e spine d1 = arctic rel0 (row-major)")
check(spine3[4].tolist() == [5003, 6003], "e2e spine d4 = arctic rel3")
# packer _fr10_wide_plan read: rk0 -> spine_tokens[pp], rk>0 -> wide_topk[pp][:,rk]
# d0 branches are MTP topk (root runners-up always MTP)
check(wide3[0][:, 1].tolist() == [2000, 2100], "e2e d0 branch1 = MTP topk (root always MTP)")
# d1 branches: arctic flat-chain has no alts -> MTP topk fallback
check(wide3[1][:, 1].tolist() == [2010, 2110], "e2e d1 branch1 = MTP topk fallback (flat-chain adapter)")
# every triple distinct per row (no self-collision)
allok = all(len({spine3[d][b].item(), wide3[d][b, 1].item(), wide3[d][b, 2].item()}) == 3
            for d in range(N_DEPTH) for b in range(B))
check(allok, "e2e all (spine,b1,b2) triples distinct per row/depth")

# ---- BOUNDS GUARD: OOB arctic token -> pad (prevents the CUDA device-side assert; merge16c fix) ----
from fr13_merged_fill import get_oob_stats  # noqa: E402
VOCAB = 152064
_n0, _ = get_oob_stats()
# row 0 has an in-range deep spine; row 1 injects OOB values: >=vocab at (d1 spine) and negative at (d2 b1)
good_row = [10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42, 50, 51, 52]          # all valid
oob_row = [10, 11, 12, VOCAB + 7, 21, 22, 30, -5, 32, 40, 41, 42, 50, 51, 52]    # idx3=OOB hi, idx7=OOB neg
sp_g, wt_g = build_cat33333_columns([good_row, oob_row], DEV, pad_token=99, vocab_size=VOCAB)
check(sp_g[0].tolist() == [10, 10], "guard: valid spine d0 passes through")
check(sp_g[1].tolist() == [20, 99], "guard: OOB(>=vocab) spine d1 row1 -> pad(99)")   # idx3 was oob_row spine d1
check(wt_g[2][:, 1].tolist() == [31, 99], "guard: OOB(negative) branch d2b1 row1 -> pad(99)")  # idx7
check(wt_g[0][:, 1].tolist() == [11, 11], "guard: valid branch d0b1 both rows pass through")
_n1, _last = get_oob_stats()
check(_n1 - _n0 == 2, f"guard: exactly 2 OOB dropped (got {_n1 - _n0})")
check(_last is not None, "guard: last-OOB recorded for the needle")
# vocab_size=None (CPU tests / default) => NO bounds check, OOB passes through unchanged (back-compat)
sp_n, _ = build_cat33333_columns([oob_row], DEV, pad_token=99, vocab_size=None)
check(sp_n[1].tolist() == [VOCAB + 7], "guard: vocab_size=None disables check (back-compat)")

print(f"\n{PASS}/{PASS+FAIL} checks PASS")
if FAIL == 0:
    print(">>> PASS — merged fill: device/dtype/shape correct, row-major preserved, PAD-densified, "
          "end-to-end adapter->assemble->fill matches the wide packer read.")
    sys.exit(0)
sys.exit(1)
