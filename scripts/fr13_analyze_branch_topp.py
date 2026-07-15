#!/usr/bin/env python3
"""Analyze a committer commit-trace to answer: (1) do the rank-2/3 BRANCHES help? (2) MTP top-1 p
(argmax_prob) nucleus distribution => how much node-budget can be reallocated from wasted branches to the
suffix TAIL via a top-p-adaptive width rule. Input = LUMO_TREE_SAMPLER_DEBUG_LOG JSONL (.commit).

Record fields: node, argmax_token, argmax_prob (top-1 p), argmax_drafted (argmax in child_drafts),
child_drafts (candidate tokens at this node's children), child_probs, overlap_mass (sum child_probs),
accepted, committed_token, committed_is_argmax.

cat33333 flat node order (index -> (depth, rank)); rank 0 = spine, 1/2 = branch.
"""
import sys, json
from collections import defaultdict

CAT33333 = [(0,),(1,),(2,),(0,0),(0,1),(0,2),(0,0,0),(0,0,1),(0,0,2),
            (0,0,0,0),(0,0,0,1),(0,0,0,2),(0,0,0,0,0),(0,0,0,0,1),(0,0,0,0,2)]
NODE_DR = {i: (len(t), t[-1]) for i, t in enumerate(CAT33333)}  # node idx -> (depth, rank)

def bucket_p(p):
    return (">0.9" if p > 0.9 else "0.7-0.9" if p > 0.7 else "0.5-0.7" if p > 0.5 else "<0.5")

def main(path):
    n = 0
    p_hist = defaultdict(int)                    # argmax_prob bucket -> count
    cover = defaultdict(int)                     # 'spine'|'branch'|'missed' (where the argmax falls)
    cover_by_pbucket = defaultdict(lambda: defaultdict(int))
    branch_help_by_depth = defaultdict(lambda: defaultdict(int))  # depth -> {spine,branch,missed}
    overlap_hist = defaultdict(int)
    accepted_by_rank = defaultdict(int)          # rank of committed/accepted -> count (branch value)
    depths_seen = set()
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if "argmax_prob" not in d:
                continue
            n += 1
            ap = float(d.get("argmax_prob", 0.0))
            pb = bucket_p(ap)
            p_hist[pb] += 1
            om = float(d.get("overlap_mass", 0.0))
            overlap_hist[(">0.9" if om > 0.9 else "0.5-0.9" if om > 0.5 else "<0.5")] += 1
            cd = d.get("child_drafts") or []
            amax = d.get("argmax_token")
            # where does the model's argmax fall among the node's children?
            if not d.get("argmax_drafted", False) or amax is None:
                where = "missed"
            elif cd and amax == cd[0]:
                where = "spine"        # top child (rank 0)
            else:
                where = "branch"       # a rank>0 child caught it
            cover[where] += 1
            cover_by_pbucket[pb][where] += 1
            node = d.get("node")
            if isinstance(node, int) and node in NODE_DR:
                depth = NODE_DR[node][0]
                depths_seen.add(depth)
                branch_help_by_depth[depth][where] += 1

    def pct(x, tot): return f"{100*x/max(1,tot):.1f}%"
    print(f"=== commit-trace analysis: {n} node-records ({path}) ===\n")
    print("MTP top-1 prob (argmax_prob) distribution  [the top-p / nucleus signal]:")
    for b in [">0.9","0.7-0.9","0.5-0.7","<0.5"]:
        print(f"  argmax_prob {b:8s}: {p_hist[b]:8d}  {pct(p_hist[b],n)}")
    dom = p_hist[">0.9"]
    print(f"  => top-1 dominant (>0.9) {pct(dom,n)} of nodes: at those, rank-2/3 branches are WASTED -> reallocate to TAIL")
    print()
    print("Where the model's argmax falls among the node's tree children  [do branches help?]:")
    tot = sum(cover.values())
    for w in ["spine","branch","missed"]:
        print(f"  {w:7s}: {cover[w]:8d}  {pct(cover[w],tot)}")
    print(f"  => BRANCH (rank>0) catches the argmax {pct(cover['branch'],tot)} of nodes = the branch's coverage value")
    print()
    print("Branch value CONDITIONED on top-1 confidence  [top-p-adaptive width justification]:")
    for b in [">0.9","0.7-0.9","0.5-0.7","<0.5"]:
        sub = cover_by_pbucket[b]; st = sum(sub.values())
        print(f"  argmax_prob {b:8s} (n={st:7d}): spine {pct(sub['spine'],st)}  branch {pct(sub['branch'],st)}  missed {pct(sub['missed'],st)}")
    print("  => branches earn their nodes ONLY in the low-argmax_prob rows; high-p rows should be width-1 (spine) + tail")
    print()
    if depths_seen:
        print("Per-depth (node->cat33333 map):")
        for depth in sorted(depths_seen):
            sub = branch_help_by_depth[depth]; st = sum(sub.values())
            print(f"  depth {depth} (n={st:7d}): spine {pct(sub['spine'],st)}  branch {pct(sub['branch'],st)}  missed {pct(sub['missed'],st)}")
    print()
    print("overlap_mass (tree candidates' nucleus coverage):")
    ot = sum(overlap_hist.values())
    for b in [">0.9","0.5-0.9","<0.5"]:
        print(f"  overlap_mass {b:7s}: {overlap_hist[b]:8d}  {pct(overlap_hist[b],ot)}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/dev/stdin")
