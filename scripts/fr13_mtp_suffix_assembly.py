#!/usr/bin/env python3
"""FR13 MTP-k + Arctic-suffix -> cat33333 tree assembly (CPU core, correctness-critical).

User architecture (2026-07-14): use native MTP for the confident near-spine (mtp_k in {1,2}),
then Arctic suffix decoding to GROW the deep spine + branches into the fixed t33333 (16-node)
tree; OUR committer verifies (Path B, lossless by Gate 1). Motivation: the FR13 drafter is
PARALLEL (one forward) so its deep spine tokens (depth 2-4) are weak parallel-drafts, whereas
Arctic RETRIEVES the real historical continuation (high accept on repetitive context) for FREE.

This module is the SOURCE-AGNOSTIC assembly logic + its gate. Arctic can't run on the host
(C++ ext + torch build), so the real .speculate() output is adapted (in the live drafter) into
the abstract `suffix_per_depth` form this function consumes; here we unit-test with mock inputs.

cat33333 (t33333) tree_choices, sorted (len, path) -- 15 nodes, depth-5 spine + 2 branches/level:
  d0:(0,)  (1,)  (2,)          d1:(0,0)   (0,1)   (0,2)      d2:(0,0,0)   (0,0,1)   (0,0,2)
  d3:(0,0,0,0) (0,0,0,1) (0,0,0,2)   d4:(0,0,0,0,0) (0,0,0,0,1) (0,0,0,0,2)
So per depth d in 0..4: slot 0 = spine (rank0), slots 1,2 = the two branches (rank1,rank2).
"""
from __future__ import annotations

N_DEPTH = 5                      # cat33333 spine depth
BRANCHES_PER_DEPTH = 2           # (1,)/(2,) style runners-up per spine node

# tree_choices order this function returns, for reference / the live packer:
CAT33333_ORDER = [
    (0,), (1,), (2,),
    (0, 0), (0, 1), (0, 2),
    (0, 0, 0), (0, 0, 1), (0, 0, 2),
    (0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2),
    (0, 0, 0, 0, 0), (0, 0, 0, 0, 1), (0, 0, 0, 0, 2),
]


def _pick_distinct(ranked, used, need):
    """Take up to `need` tokens from `ranked` not already in `used` (dedup, order-preserving)."""
    out = []
    for t in ranked:
        if t is None:
            continue
        t = int(t)
        if t in used:
            continue
        used.add(t)
        out.append(t)
        if len(out) == need:
            break
    return out


def assemble_cat33333(mtp_spine, mtp_topk_per_depth, suffix_rel, mtp_k):
    """Assemble the 15 cat33333 node tokens (in CAT33333_ORDER) from MTP + suffix.

    mtp_spine:           [t0, t1, t2, t3, t4] MTP argmax spine tokens (parallel-drafted); the
                         current drafter's spine. len must be >= N_DEPTH.
    mtp_topk_per_depth:  {d: [rank2_tok, rank3_tok]} the MTP branch runners-up at each depth d
                         (0..4). The current drafter's branch tokens; the BASELINE + fallback.
    suffix_rel:          {i: [ranked suffix tokens]} Arctic's continuation candidates RELATIVE to
                         the pattern end -- i=0 is the FIRST token past the MTP prefix (Arctic's
                         .token_ids are pattern-relative). suffix_rel[i] maps to ABSOLUTE spine
                         depth (mtp_k + i). May be empty/missing (cold) -> pure-MTP fallback.
    mtp_k:               1 or 2 -- how many near-spine tokens come from MTP (the rest from suffix).

    Returns (nodes, meta): nodes = 15 ints in CAT33333_ORDER; meta = per-depth source provenance
    (for the engagement needle). SPINE precedence: d<mtp_k -> MTP; else suffix_rel[d-mtp_k][0] if
    present else MTP parallel-spine[d] (never worse than baseline). BRANCHES: suffix_rel[d-mtp_k][1:]
    deduped vs the chosen spine, else MTP topk[d]. Root branches (d=0 (1,),(2,)) are ALWAYS MTP topk
    (root runners-up are a property of the MTP forward, not the suffix continuation).
    """
    assert mtp_k in (1, 2), "mtp_k must be 1 or 2"
    assert len(mtp_spine) >= N_DEPTH, "need >=5 MTP spine tokens (parallel-drafted)"
    nodes = []
    meta = {"spine_src": [], "branch_src": []}

    for d in range(N_DEPTH):
        used = set()
        # suffix candidates for absolute depth d live at relative index (d - mtp_k)
        suf = ([int(x) for x in suffix_rel.get(d - mtp_k, []) if x is not None]
               if d >= mtp_k else [])

        # --- spine token at depth d ---
        if d < mtp_k:
            spine_tok = int(mtp_spine[d]); src = "mtp"
        elif suf:
            spine_tok = suf[0]; src = "suffix"
        else:
            spine_tok = int(mtp_spine[d]); src = "mtp_fallback"
        used.add(spine_tok)
        meta["spine_src"].append(src)

        # --- 2 branch tokens at depth d ---
        if d == 0:
            # root runners-up are ALWAYS MTP topk (property of the MTP forward)
            branch_pool = list(mtp_topk_per_depth.get(d, []))
            bsrc = "mtp"
        elif len(suf) > 1:
            branch_pool = suf[1:] + list(mtp_topk_per_depth.get(d, []))  # suffix first, MTP backfill
            bsrc = "suffix"
        else:
            branch_pool = list(mtp_topk_per_depth.get(d, []))
            bsrc = "mtp_fallback"
        branches = _pick_distinct(branch_pool, used, BRANCHES_PER_DEPTH)
        # if still short (rare: pools exhausted after dedup), backfill from MTP spine-neighbours
        while len(branches) < BRANCHES_PER_DEPTH:
            filler = _pick_distinct(mtp_spine, used, 1)
            if not filler:
                # last resort: a distinct sentinel-free duplicate-avoiding pad (repeat spine is
                # harmless -- committer just treats a duplicate candidate as adding no mass).
                branches.append(spine_tok)
            else:
                branches.append(filler[0])
        meta["branch_src"].append(bsrc)

        nodes.extend([spine_tok, branches[0], branches[1]])

    assert len(nodes) == len(CAT33333_ORDER), "must produce exactly 15 cat33333 node tokens"
    return nodes, meta


def assemble_pure_mtp(mtp_spine, mtp_topk_per_depth):
    """The BASELINE / cold-fallback: current drafter's cat33333 (spine=argmax, branches=MTP topk).
    Used when suffix is entirely absent -> byte-identical to the MTP-only t33333 drafter.
    Equal to assemble_cat33333(..., suffix_per_depth={}, mtp_k=anything) by construction."""
    return _pure_mtp(mtp_spine, mtp_topk_per_depth)


def _pure_mtp(mtp_spine, mtp_topk_per_depth):
    nodes = []
    for d in range(N_DEPTH):
        used = {int(mtp_spine[d])}
        br = _pick_distinct(list(mtp_topk_per_depth.get(d, [])), used, BRANCHES_PER_DEPTH)
        while len(br) < BRANCHES_PER_DEPTH:
            f = _pick_distinct(mtp_spine, used, 1)
            br.append(f[0] if f else int(mtp_spine[d]))
        nodes.extend([int(mtp_spine[d]), br[0], br[1]])
    return nodes
