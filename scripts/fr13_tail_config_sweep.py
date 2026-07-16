#!/usr/bin/env python3
"""Offline accept-model sweep over the MERGED two-proposer tree.

Framing (user 2026-07-16): the tree is a MERGE of TWO tree-proposers, each proposing top-k per depth:
  - MTP proposer   : strong, but only 5 heads -> covers depths 1..n  (n = mtp_k, n<=5). x0 => pure suffix.
  - Suffix/Arctic  : weaker per-token but UNBOUNDED depth + catches MTP misses -> covers depths 1..x. n0 => pure MTP.
Regions by depth d:
  - MTP-only  (d<=n, d>x)            : spine + MTP branches            -> cond ~ HEAD (0.85 flat)
  - OVERLAP   (d<=n and d<=x)        : spine + (MTP u ARCTIC) branches -> cond = HEAD + comp_uplift  (arctic COMPLEMENTS MTP; head-miss rescue)
  - ARCTIC-only / TAIL (d>n, d<=x)   : arctic spine (+opt branches)    -> cond = TAIL[pos]; pos0 = the MTP->arctic HANDOFF (weak 0.666)
Endpoints: x=0 pure MTP (=cat33333, measured accept 3.56); n=0 pure arctic (weaker, ~flat 0.5-0.6).

accept = Sum survival. Two UNMEASURED uplifts (calibrate from live A/Bs, then re-run):
  comp_uplift  : how much an ARCTIC complement branch raises an MTP-covered depth's conditional (head-rescue).
  tail_uplift  : how much ARCTIC branches raise a tail depth's conditional (d6-handoff rescue).

Node budget (n_pad): sum over depths of (1 + branches_at_depth) <= NPAD (32). Branch widths per region are args.
"""
import argparse

HEAD = [0.970, 0.864, 0.851, 0.849, 0.866]           # MTP depth d1..5 (branched)
TAIL = [0.666, 0.848, 0.895, 0.906, 0.908, 0.950]     # arctic tail position j=0.. (j0=handoff)
TAIL_PLATEAU = 0.95
ARCTIC_PURE = 0.55                                    # n=0 pure-arctic flat conditional (design's ~0.5-0.6 deep)
NPAD, CAP = 32, 0.97

def tailc(j): return TAIL[j] if j < len(TAIL) else TAIL_PLATEAU

def model(n, x, w_over, w_tail, tail_bd, comp_uplift, tail_uplift):
    """(accept, nodes). n = MTP HEAD depth (<=5), x = TAIL length (arctic-only depths past the head).
    w_over = arctic COMPLEMENT branch added in the head region (0/1). w_tail = arctic branches in the
    first tail_bd tail depths. Endpoints: x=0 pure MTP head; n=0 pure arctic tail (no MTP prefix)."""
    surv, acc, nodes = 1.0, 0.0, 0
    for d in range(1, n + 1):                           # HEAD: MTP spine + 2 MTP branches (+ w_over arctic complement)
        c = HEAD[d-1] + (comp_uplift if w_over else 0.0)
        surv *= min(CAP, c); acc += surv; nodes += 3 + w_over
    for j in range(x):                                  # TAIL: arctic spine (+ w_tail branches at first tail_bd depths)
        c = ARCTIC_PURE if n == 0 else tailc(j)         # n=0 => pure arctic (no MTP handoff)
        branched = j < tail_bd and w_tail > 0
        if branched: c = min(CAP, c + tail_uplift)
        surv *= c; acc += surv; nodes += 1 + (w_tail if branched else 0)
    return acc, nodes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp_uplift", type=float, default=0.06, help="arctic-complement uplift in overlap (calibrate: head-complement A/B)")
    ap.add_argument("--tail_uplift", type=float, default=0.12, help="arctic-branch uplift in tail (calibrate: tail6-vs-tail6b A/B)")
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()
    rows = []
    for n in range(0, 6):                     # MTP HEAD depth (0=pure arctic .. 5)
        for x in range(0, 22):                # TAIL length (0=pure MTP)
            if n == 0 and x == 0: continue
            for w_over in ((0,) if n == 0 else (0, 1)):
                for w_tail in ((0,) if x == 0 else (0, 1, 2, 3)):
                    for tail_bd in range(0, x + 1):
                        if w_tail == 0 and tail_bd > 0: continue
                        acc, nodes = model(n, x, w_over, w_tail, tail_bd, a.comp_uplift, a.tail_uplift)
                        if nodes > NPAD or nodes == 0: continue
                        rows.append((acc, nodes, n, x, w_over, w_tail, tail_bd))
    rows.sort(reverse=True)
    print(f"# merged 2-proposer sweep  comp_uplift={a.comp_uplift} tail_uplift={a.tail_uplift}  nodes<={NPAD}")
    print(f"  [ref] pure-MTP (n5,x0) accept={model(5,0,0,0,0,0,0)[0]:.3f}(nodes {model(5,0,0,0,0,0,0)[1]})  |  shipped tail6 (n5,x6,no branch) accept={model(5,6,0,0,0,0,0)[0]:.3f}(measured ~5.2)")
    print(f"{'accept':>7} {'nodes':>5}  n=MTP  x=TAIL  w_over  w_tail  tail_bd")
    for acc, nodes, n, x, w_over, w_tail, tail_bd in rows[:a.top]:
        tag = " <- shipped tail6" if (n,x,w_over,w_tail,tail_bd)==(5,6,0,0,0) else ""
        print(f"{acc:7.3f} {nodes:5d}  {n:4d}  {x:5d}  {w_over:5d}  {w_tail:5d}  {tail_bd:6d}{tag}")

if __name__ == "__main__":
    main()
