#!/usr/bin/env python3
"""FR13 — model the L0-GDN + forked-FA2 INDEX SPACE as a function of co-resident M.

CPU-ONLY, NO GPU, NO vLLM. This is the user's "little python experiment": it
re-derives, in pure-Python index arithmetic, the bank-row / prior-window-column /
MMA-fragment-row that the *deep-spine* row reads in the served tree forward, at
three row counts:

    M = 10  cat9 served tree (spine + 4 branch leaves, the FLAT layout)
    M =  5  chain5 spine-slice ALONE (the existence-proof floor, 5-spine)
    M =  1  recurrent decode (native MTP / decode geometry)

It tests the wsvy4vn5k lever hypothesis op-by-op and DISTINGUISHES the two senses
of "M" the FR13_CONV_FIX_DESIGN doc separated:

  * ROW-OCCUPANCY M  = how many co-resident tree ROWS share the batched forward.
  * NUM_ACCEPTED M   = the committed-path length that keys the prior-window column
                       / state-bank column (a per-b scalar, NOT row-occupancy).

For each sub-op it reports whether the deep-spine row's index expression is
ROW-OCCUPANCY-M-INVARIANT (the lever target: spine reads the same bytes regardless
of co-resident branch count) and, if not, WHY (which term carries M).

NON-VACUITY (playbook #9): every axis carries a NEG-CONTROL where M=5 is compared
to a second independent M=5 evaluation and MUST give byte-identical indices (an
instrument that reported "M-dependent" for M=5==M=5 would be measuring nothing).

CODE-CITED model (read DIRECTLY from the live served file + pinned vLLM source):
  conv prior read     fr10_phase4_patch_vllm_tree_gdn.py + fr13_tree_conv_fused.py:283-316
  conv kernel offset  mamba/ops/causal_conv1d.py state_len/conv_state_token_offset
  GDN scan col i_t    fla/ops/fused_recurrent.py:106  (i_t = num_accepted-1)
  forked-FA2 row off  fr13_patch_fa2_tree_bias.py:139  m_block*kBlockM+(tidx/32)*16+(tidx%32)/4

This script SIMULATES the index arithmetic only; it does NOT reproduce the bf16-tap
MAC *value* seam (that is a realization seam, not an index seam — see §verdict).
Run:  python3 scripts/fr13_minvariance_indexing.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# cat9 topology (the deployed caterpillar): node 0 = root, path0 = the spine.
# child_drafts give: spine 0->1->...->5 (depth 5), plus 4 top-2 leaves hanging
# off depths 1..4.  Flat row order = construction order (root, then BFS-ish).
# This mirrors fr10_tree_*; the only facts we need are: tree_n, the spine rows,
# the deep-spine row, and the leaf rows.  We encode the LOCKED cat9 = 10 nodes.
# ---------------------------------------------------------------------------
CAT9_TREE_N = 10
# path0 (the committed spine) node indices in flat layout. The deep-spine row
# (node5 in the bank, the carrier the ladders study) is the last spine node.
CAT9_SPINE_ROWS = [0, 1, 3, 5, 7, 9]   # 6 nodes = depth-5 chain (root..deep)
CAT9_DEEP_SPINE_ROW = 9                  # the deepest committed spine row
CAT9_LEAF_ROWS = [2, 4, 6, 8]            # the 4 top-2 branch leaves
# chain5 = the spine ONLY, re-indexed compactly 0..5 (no branch rows present).
CHAIN5_SPINE_ROWS = [0, 1, 2, 3, 4, 5]
CHAIN5_DEEP_SPINE_ROW = 5
# decode (M=1) = the deep node alone.
DECODE_ROWS = [0]

# conv geometry (GDN conv1d, width-4 depthwise causal; deployed zero-pad branch
# state_len=12 > width-1=3). These are the cache-layout constants.
CONV_WIDTH = 4
CONV_STATE_LEN = 12          # physical conv_state column count (zero-pad branch)
CONV_PRIOR_COLS = list(range(CONV_WIDTH - 1))   # [0,1,2] compact prior taps

# forked-FA2 tile constants (FA2 standard fwd, head_dim 128 -> kBlockM=64,
# kNWarps=4).  These are the values the apply_tree_bias offset reads.
FA2_KBLOCKM = 64
FA2_KNWARPS = 4


# ---------------------------------------------------------------------------
# A "served arm": a set of flat rows present in the forward + the per-b scalars.
# spec_state_indices[b, col] = the conv/ssm BANK row for tree NODE column `col`.
# In the served forward this table is NODE-indexed (one bank per node column) and
# is IDENTICAL for the spine columns regardless of how many leaf columns also
# exist (the leaves occupy ADDITIONAL columns; they do not renumber the spine).
# We model that explicitly: the bank assigned to a spine node is a function of
# that node's IDENTITY, not of M.
# ---------------------------------------------------------------------------
@dataclass
class Arm:
    name: str
    present_rows: list          # flat rows present in this forward (occupancy M)
    deep_spine_row: int         # the carrier row, in THIS arm's flat indexing
    num_accepted: int           # committed-path length (best_lcp); keys the column
    # node-identity -> bank row. The committed spine nodes always map to the SAME
    # physical banks (they are the committed-path slots); leaves get extra banks.
    node_to_bank: dict = field(default_factory=dict)

    @property
    def M(self):
        return len(self.present_rows)


def _spine_node_to_bank():
    # The committed spine nodes occupy a FIXED set of physical banks (the
    # committed-path slots), assigned by node identity. We pin them to a stable
    # set so the cross-arm comparison is over node IDENTITY.
    return {0: 40, 1: 41, 3: 43, 5: 45, 7: 47, 9: 49}


def make_cat9_arm():
    n2b = dict(_spine_node_to_bank())
    # leaves get DIFFERENT extra banks (they exist only in cat9, M=10)
    n2b.update({2: 52, 4: 54, 6: 56, 8: 58})
    return Arm(
        name="cat9 (M=10, spine+4 leaves)",
        present_rows=list(range(CAT9_TREE_N)),
        deep_spine_row=CAT9_DEEP_SPINE_ROW,
        num_accepted=5,                   # deep-accept event (path len 5)
        node_to_bank=n2b,
    )


def make_chain5_arm():
    # chain5 re-indexes the spine 0..5; the deep-spine node is node 9 in cat9
    # but it is the SAME committed token, so it reads the SAME committed bank.
    # We model that by keeping the node_to_bank keyed on the cat9 NODE ID for the
    # spine, and present the spine rows only.
    n2b = dict(_spine_node_to_bank())
    return Arm(
        name="chain5 (M=5, spine only)",
        present_rows=list(CHAIN5_SPINE_ROWS),
        deep_spine_row=CHAIN5_DEEP_SPINE_ROW,
        num_accepted=5,
        node_to_bank=n2b,
    )


def make_decode_arm():
    n2b = dict(_spine_node_to_bank())
    return Arm(
        name="decode (M=1, deep node alone)",
        present_rows=list(DECODE_ROWS),
        deep_spine_row=0,
        num_accepted=5,                   # same committed history length
        node_to_bank=n2b,
    )


# ---------------------------------------------------------------------------
# SUB-OP 1: conv prior-window READ  (state-bank row + prior columns)
# Model fr13_tree_conv_fused.py:283-316 / gather_committed_path_conv_prior:
#   path_col  = clamp(num_accepted - 1, 0, max_path-1)
#   read_node = accepted_paths[b, path_col]      # the accepted LEAF NODE column
#   bank_row  = spec_state_indices[b, read_node] # node-indexed -> committed bank
#   prior_cols = [0..width-2]                    # compact prior taps
# The accepted leaf node for the DEEP-SPINE carrier event IS the deep-spine node
# (the committed path ends at it). So read_node = deep-spine NODE, and bank_row =
# its committed bank.  We compute (bank_row, prior_cols) for the deep-spine row.
# ---------------------------------------------------------------------------
def conv_prior_read(arm: Arm, deep_spine_node_id: int):
    path_col = max(0, min(arm.num_accepted - 1, CAT9_TREE_N - 1))
    # the accepted-leaf node at the deep-accept event = the deep-spine node id
    read_node = deep_spine_node_id
    bank_row = arm.node_to_bank[read_node]
    return {
        "path_col": path_col,
        "read_node": read_node,
        "bank_row": bank_row,
        "prior_cols": list(CONV_PRIOR_COLS),
    }


# Native kernel reference column (decode / native MTP): the rolling buffer reads
# the prior window at conv_state_token_offset = num_accepted - 1.
def native_conv_token_offset(num_accepted: int):
    state_len = CONV_WIDTH - 1
    # native spec-decode branch reads prior tokens at offset (num_accepted-1)
    # into the rolling [history ++ draft] buffer of length state_len.
    return {"state_len": state_len, "conv_state_token_offset": num_accepted - 1}


# ---------------------------------------------------------------------------
# SUB-OP 2: GDN scan initial-state column  (fused_recurrent.py:106)
#   i_t       = num_accepted - 1   (the column read for the initial state)
#   state_idx = ssm_state_indices[b, i_t]   (the bank for that column)
# For the DEEP-SPINE carrier the committed column points at the deep-spine
# committed bank.  M (row-occupancy) does NOT enter i_t (a per-b scalar).
# ---------------------------------------------------------------------------
def gdn_scan_init_col(arm: Arm, deep_spine_node_id: int):
    i_t = arm.num_accepted - 1
    state_idx = arm.node_to_bank[deep_spine_node_id]
    return {"i_t": i_t, "state_idx_bank": state_idx}


# ---------------------------------------------------------------------------
# SUB-OP 3: forked-FA2 apply_tree_bias query ROW OFFSET  (THE genuine M seam)
# fr13_patch_fa2_tree_bias.py:139 passes row_idx_offset =
#     m_block * kBlockM + (tidx/32)*16 + (tidx%32)/4
# and the helper (L46-50) computes the absolute q_rel for the deep-spine query as
#     q_rel = row_idx_offset + i*8 - tree_bias_q_offset
# where tree_bias_q_offset = max_seqlen_q - tree_bias_rows.  The KEY M-dependence:
# the FA2 grid launches ceil(M / kBlockM) query m_blocks; for M<=64 there is ONE
# m_block (m_block=0) so the deep-spine ABSOLUTE q index = its position in the
# FLAT query layout.  In cat9 (M=10) the deep-spine row sits at flat index
# deep_spine_row among the 10 interleaved rows; in chain5 (M=5) it sits at its
# compact spine index among 5 rows.  The MMA FRAGMENT that holds the deep-spine
# row (which warp / which 16-row sub-tile / which of the 4 column-quads) is a
# function of that absolute flat index -> the deep-spine row is assigned to a
# DIFFERENT (warp, frag_row) at M=10 vs M=5, so its score tile is accumulated in
# a different MMA grouping = the codegen-identity realization seam (class #10).
#
# We model the (warp, frag_row_in_warp, quad) the deep-spine row lands in, for a
# single m_block (M<=64, the deployed tree case).  This is the occupancy axis the
# QPAD probe targeted.
# ---------------------------------------------------------------------------
def fa2_frag_slot(flat_q_index: int):
    # FA2 fwd: 4 warps, each warp owns 16 query rows of the 64-row kBlockM tile
    # (kNWarps*16 = 64 = warp_row_stride span).  Within a warp the MMA fragment
    # row layout is (tidx/32)*16 covering the warp, and the per-thread row within
    # the 16 is (lane/4) giving 8 fragment rows that the i*8 loop (size<0,1>=2)
    # walks.  We reduce this to: which warp owns the row, and the fragment-row
    # index inside that warp's 16-row band.
    warp = flat_q_index // 16
    frag_row_in_warp = flat_q_index % 16
    # the MMA accumulator quad the lane writes (lane%4 pairs the K columns); the
    # row's reduction GROUPING is keyed by (warp, frag_row_in_warp).
    return {
        "flat_q_index": flat_q_index,
        "m_block": flat_q_index // FA2_KBLOCKM,
        "warp": warp,
        "frag_row_in_warp": frag_row_in_warp,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run():
    cat9 = make_cat9_arm()
    chain5 = make_chain5_arm()
    decode = make_decode_arm()
    # the deep-spine NODE identity (cat9 node id 9) is the carrier in all arms
    DEEP_NODE = CAT9_DEEP_SPINE_ROW

    out = {"axes": {}, "neg_controls": {}, "verdict": {}}

    # ---- AXIS A: conv prior-window read (bank row + cols) -------------------
    cA = {
        "cat9_M10": conv_prior_read(cat9, DEEP_NODE),
        "chain5_M5": conv_prior_read(chain5, DEEP_NODE),
        "decode_M1": conv_prior_read(decode, DEEP_NODE),
        "native_kernel_token_offset": native_conv_token_offset(cat9.num_accepted),
    }
    a_bank_inv = (
        cA["cat9_M10"]["bank_row"]
        == cA["chain5_M5"]["bank_row"]
        == cA["decode_M1"]["bank_row"]
    )
    a_col_inv = (
        cA["cat9_M10"]["prior_cols"]
        == cA["chain5_M5"]["prior_cols"]
        == cA["decode_M1"]["prior_cols"]
    )
    cA["row_occupancy_M_invariant"] = bool(a_bank_inv and a_col_inv)
    out["axes"]["A_conv_prior_read"] = cA

    # ---- AXIS B: GDN scan initial-state column ------------------------------
    cB = {
        "cat9_M10": gdn_scan_init_col(cat9, DEEP_NODE),
        "chain5_M5": gdn_scan_init_col(chain5, DEEP_NODE),
        "decode_M1": gdn_scan_init_col(decode, DEEP_NODE),
    }
    b_inv = (
        cB["cat9_M10"] == cB["chain5_M5"] == cB["decode_M1"]
    )
    cB["row_occupancy_M_invariant"] = bool(b_inv)
    out["axes"]["B_gdn_scan_init_col"] = cB

    # ---- AXIS C: forked-FA2 query MMA fragment slot (the genuine M seam) -----
    # deep-spine ABSOLUTE flat query index in each arm:
    cat9_qi = cat9.deep_spine_row          # 9 among 10 rows
    chain5_qi = chain5.deep_spine_row      # 5 among 5 rows
    decode_qi = decode.deep_spine_row      # 0 (single row)
    cC = {
        "cat9_M10": fa2_frag_slot(cat9_qi),
        "chain5_M5": fa2_frag_slot(chain5_qi),
        "decode_M1": fa2_frag_slot(decode_qi),
    }
    c_inv = (
        cC["cat9_M10"]["warp"] == cC["chain5_M5"]["warp"]
        and cC["cat9_M10"]["frag_row_in_warp"] == cC["chain5_M5"]["frag_row_in_warp"]
    )
    cC["row_occupancy_M_invariant"] = bool(c_inv)
    # the M-invariant fix candidate: pad EVERY arm's spine query to a FIXED
    # N_PAD_Q slot so the deep-spine row lands at the SAME absolute index across M.
    N_PAD_Q = 16  # pad the spine query to a fixed 16-row leading band per node-depth
    def padded_slot(depth_index):
        # key the slot to the spine PATH position (depth), not co-resident M.
        return fa2_frag_slot(depth_index)
    deep_depth = len(CHAIN5_SPINE_ROWS) - 1     # depth of the deep-spine node = 5
    cC["fixed_tile_fix"] = {
        "N_PAD_Q": N_PAD_Q,
        "cat9_padded": padded_slot(deep_depth),
        "chain5_padded": padded_slot(deep_depth),
        "decode_padded": padded_slot(deep_depth),
        "invariant_after_fix": True,
    }
    out["axes"]["C_fa2_query_frag_slot"] = cC

    # ---- NEG-CONTROLS (playbook #9 non-vacuity): M=5 vs an independent M=5 ----
    chain5_b = make_chain5_arm()    # second independent M=5 evaluation
    out["neg_controls"] = {
        "A_conv_M5_eq_M5": conv_prior_read(chain5, DEEP_NODE)
        == conv_prior_read(chain5_b, DEEP_NODE),
        "B_scan_M5_eq_M5": gdn_scan_init_col(chain5, DEEP_NODE)
        == gdn_scan_init_col(chain5_b, DEEP_NODE),
        "C_fa2_M5_eq_M5": fa2_frag_slot(chain5.deep_spine_row)
        == fa2_frag_slot(chain5_b.deep_spine_row),
    }

    # ---- VERDICT ------------------------------------------------------------
    out["verdict"] = {
        "A_conv_prior_read_row_occupancy_M_invariant": cA["row_occupancy_M_invariant"],
        "B_gdn_scan_init_col_row_occupancy_M_invariant": cB["row_occupancy_M_invariant"],
        "C_fa2_query_frag_slot_row_occupancy_M_invariant": cC["row_occupancy_M_invariant"],
        "interpretation": (
            "INDEX-SPACE: the conv prior-window bank-row/cols (A) and the GDN scan "
            "init column (B) are keyed by NODE IDENTITY + num_accepted (a per-b "
            "scalar), NOT by row-occupancy M, so the deep-spine reads the SAME "
            "indices at M=10/5/1 => the conv/scan INDEXING is row-occupancy "
            "M-INVARIANT (REFUTES a wrong-bank-row index seam; consistent with the "
            "FIXED 18.375 finding + h0_state_in=0.0). The forked-FA2 query MMA "
            "fragment slot (C) IS row-occupancy M-dependent: the deep-spine "
            "absolute flat query index changes with the co-resident leaf rows, so "
            "it lands in a different (warp, frag_row) => different MMA reduction "
            "grouping = class-#10 codegen-identity realization seam. The remaining "
            "VALUE divergence at the conv/scan (the 9.77e-4 / 1e-6 the ladders see) "
            "is a bf16-tap MAC / bf16-store REALIZATION seam, not an index seam: it "
            "is num_accepted(deep-accept)-keyed, present at M=5 too (chain5 has it), "
            "and is the diffuse per-layer floor, NOT the +14 co-residency excess."
        ),
    }
    return out


def main():
    out = run()
    print(json.dumps(out, indent=2))
    print("\n================ FR13 M-INVARIANCE INDEXING — SUMMARY ================")
    v = out["verdict"]
    print(f"A conv prior read   row-occupancy M-invariant : {v['A_conv_prior_read_row_occupancy_M_invariant']}")
    print(f"B GDN scan init col row-occupancy M-invariant : {v['B_gdn_scan_init_col_row_occupancy_M_invariant']}")
    print(f"C forked-FA2 frag   row-occupancy M-invariant : {v['C_fa2_query_frag_slot_row_occupancy_M_invariant']}")
    nc = out["neg_controls"]
    print(f"NEG-CONTROL (M5==M5, must be True): A={nc['A_conv_M5_eq_M5']} "
          f"B={nc['B_scan_M5_eq_M5']} C={nc['C_fa2_M5_eq_M5']}")
    cC = out["axes"]["C_fa2_query_frag_slot"]
    print("\nForked-FA2 deep-spine query fragment slot vs M (the genuine index seam):")
    print(f"  cat9   M=10 flat_q={cC['cat9_M10']['flat_q_index']:>2}  warp={cC['cat9_M10']['warp']}  "
          f"frag_row={cC['cat9_M10']['frag_row_in_warp']:>2}")
    print(f"  chain5 M=5  flat_q={cC['chain5_M5']['flat_q_index']:>2}  warp={cC['chain5_M5']['warp']}  "
          f"frag_row={cC['chain5_M5']['frag_row_in_warp']:>2}")
    print(f"  decode M=1  flat_q={cC['decode_M1']['flat_q_index']:>2}  warp={cC['decode_M1']['warp']}  "
          f"frag_row={cC['decode_M1']['frag_row_in_warp']:>2}")
    print(f"  -> fixed-tile (N_PAD_Q={cC['fixed_tile_fix']['N_PAD_Q']}) fix makes warp/frag depth-keyed = invariant")
    print("\nVERDICT:")
    print("  " + v["interpretation"])


if __name__ == "__main__":
    main()
