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


## ---------------------------------------------------------------------------
## TAIL TREE (accept>5): cat33333 head (depths 0-4) + a deep arctic-filled spine
## chain (depths 6..5+TAIL_LEN). The chain is the ONLY path past accept=5 (MTP has
## 5 heads; the tail spine comes from Arctic's historical continuation). Pure chain,
## NO tail branches: ladder-step-1 (ctrace1 n=25080) showed the stream is 94.9%
## argmax_prob>0.9 (peaked) and branches earn ~10% only at SHALLOW depths, so a deep
## chain maximizes reach per node. mtp_k=5 => the head is BYTE-IDENTICAL to the
## baseline cat33333 (pure MTP), and the tail only ADDS candidates => NEVER-REGRESS
## by the monotone committer (accept=p(S), source/depth-blind). Cold Arctic -> tail
## nodes pad-fill (repeat last spine tok) -> never match past the head -> accept==baseline.
TAIL_HEAD_DEPTH = 5          # cat33333 head depths 0..4 (MTP-drafted, 3 nodes/depth)
TAIL_LEN = 16                # Arctic chain nodes past the head (depths 6..21); 15+16=31 -> n_pad=32


def tail_tree_order(head_depth: int = TAIL_HEAD_DEPTH, tail_len: int = TAIL_LEN,
                    branches_per_depth: int = BRANCHES_PER_DEPTH):
    """The node-path list (sorted head then chain) that drives the SPEC tree, the packer,
    and the assembly -- ONE topology source. Head = spine+branches for depths 0..head_depth-1
    (== CAT33333_ORDER when head_depth=5,branches=2); tail = pure spine chain (0,)*(head_depth+1+j).
    """
    order = []
    for d in range(head_depth):
        order.append((0,) * (d + 1))                       # spine at depth d
        for r in range(1, branches_per_depth + 1):
            order.append((0,) * d + (r,))                  # branch rank r at depth d
    for j in range(tail_len):
        order.append((0,) * (head_depth + 1 + j))          # chain node, absolute depth head_depth+1+j
    return order


def assemble_tail_tree(mtp_spine, mtp_topk_per_depth, suffix_rel, mtp_k,
                       head_depth: int = TAIL_HEAD_DEPTH, tail_len: int = TAIL_LEN,
                       branches_per_depth: int = BRANCHES_PER_DEPTH):
    """Assemble the tail-tree node tokens (in tail_tree_order) from MTP head + Arctic chain.

    HEAD (depths 0..head_depth-1): spine = MTP argmax when d<mtp_k else Arctic suffix_rel[d-mtp_k][0]
      (never worse than MTP parallel-spine); 2 branches = suffix_rel[d-mtp_k][1:] deduped, MTP topk
      backfill; root (d=0) branches are ALWAYS MTP topk. (mtp_k=head_depth => pure-MTP head = baseline.)
    TAIL (chain node j, absolute depth head_depth+j): spine ONLY = Arctic suffix_rel[head_depth-mtp_k+j][0]
      if present, else repeat the previous spine token (pad; never matches past the head => lossless).

    Returns (nodes, meta). len(nodes) == head_depth*(1+branches_per_depth) + tail_len.
    """
    assert 1 <= mtp_k <= head_depth, "mtp_k in [1, head_depth]"
    assert len(mtp_spine) >= mtp_k, "need >= mtp_k MTP spine tokens"
    nodes, meta = [], {"spine_src": [], "branch_src": [], "tail_src": []}

    # --- HEAD: depths 0..head_depth-1 (spine + branches) ---
    last_spine = None
    for d in range(head_depth):
        used = set()
        suf = ([int(x) for x in suffix_rel.get(d - mtp_k, []) if x is not None]
               if d >= mtp_k else [])
        if d < mtp_k:
            spine_tok = int(mtp_spine[d]); src = "mtp"
        elif suf:
            spine_tok = suf[0]; src = "suffix"
        else:
            spine_tok = int(mtp_spine[d]) if d < len(mtp_spine) else last_spine; src = "mtp_fallback"
        used.add(spine_tok); last_spine = spine_tok
        meta["spine_src"].append(src)

        if d == 0:
            branch_pool = list(mtp_topk_per_depth.get(d, [])); bsrc = "mtp"
        elif len(suf) > 1:
            branch_pool = suf[1:] + list(mtp_topk_per_depth.get(d, [])); bsrc = "suffix"
        else:
            branch_pool = list(mtp_topk_per_depth.get(d, [])); bsrc = "mtp_fallback"
        branches = _pick_distinct(branch_pool, used, branches_per_depth)
        while len(branches) < branches_per_depth:
            filler = _pick_distinct(mtp_spine, used, 1)
            branches.append(filler[0] if filler else spine_tok)
        meta["branch_src"].append(bsrc)
        nodes.append(spine_tok)
        nodes.extend(branches[:branches_per_depth])

    # --- TAIL: pure Arctic spine chain, depths head_depth..head_depth+tail_len-1 ---
    for j in range(tail_len):
        rel = head_depth - mtp_k + j          # suffix_rel index for this chain depth
        suf = [int(x) for x in suffix_rel.get(rel, []) if x is not None]
        if suf:
            tail_tok = suf[0]; tsrc = "suffix"
        else:
            tail_tok = last_spine; tsrc = "pad"   # cold -> repeat; never matches past head (lossless)
        last_spine = tail_tok
        meta["tail_src"].append(tsrc)
        nodes.append(tail_tok)

    expected = head_depth * (1 + branches_per_depth) + tail_len
    assert len(nodes) == expected, f"tail tree must produce {expected} nodes, got {len(nodes)}"
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
