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

# b7-REGIME conditionals (recalibrated 2026-07-18 from the b7 tail6 arm's 756
# per-position windows, 15603 events, token-weighted; Sum-survival check 4.307
# vs bracketed 4.317 = 0.2% agreement). Old-regime values in comments.
HEAD = [0.967, 0.819, 0.795, 0.801, 0.815]           # MTP d1..5 (old: .970/.864/.851/.849/.866)
TAIL = [0.593, 0.817, 0.860, 0.865, 0.851, 0.901]     # arctic j0..5; j0=handoff (old: .666 ...)
TAIL_PLATEAU = 0.15  # tailx10 MEASURED refutation: deep tail is COLD (accept flat at x=10,
                     # tps -21%); 0.95 was the extrapolation artifact that ranked n1/x21 fantasies
ARCTIC_PURE = 0.55                                    # n=0 pure-arctic flat conditional (design's ~0.5-0.6 deep)
NPAD, CAP = 32, 0.97
CHAIN_SLOTS = 8   # piggyback chain consumes 8 of NPAD (pb era only)

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

def survival_at(n, x, w_over, w_tail, tail_bd, comp_uplift, tail_uplift, depth):
    """P(accept >= depth) under the same conditionals as model() (for overflow prob)."""
    surv = 1.0
    d_all = [("h", d) for d in range(1, n + 1)] + [("t", j) for j in range(x)]
    for i, (kind, idx) in enumerate(d_all):
        if i >= depth:
            break
        if kind == "h":
            c = HEAD[idx - 1] + (comp_uplift if w_over else 0.0)
        else:
            c = ARCTIC_PURE if n == 0 else tailc(idx)
            if idx < tail_bd and w_tail > 0:
                c = min(CAP, c + tail_uplift)
        surv *= min(CAP, c)
    return surv


def cost_model(a, n, x, w_over, w_tail, tail_bd, accept, nodes):
    """PB-era (or replay-era) step cost in ms -> (tps, step_ms, committer_ms, coverage).

    ALL inputs are provisional CLI params until measured (FR13_BEAT_NATIVE_LADDER R5):
      c_pb        <- pbmech cat9pb CFWD (target ~16; UNMEASURED)
      c_replay    <- cng16 measured 70.7 (native committer baked)
      v_base/v_node <- refit from the pbmech 18-vs-10-stream delta + confirm campaign
      d_head      <- ~20ms/MTP forward (drafter decomp); d_arctic host adder
    PB constraints: chain consumes CHAIN_SLOTS of NPAD (tree budget 24); configs with
    max committed depth n+x > 6 pay the HYBRID blend via P(accept > 6) = survival(7).
    """
    era_pb = a.era == "pb"
    streams = 1 + nodes + (CHAIN_SLOTS if era_pb else 0)
    verify = a.v_base + a.v_node * nodes + (a.v_chain * CHAIN_SLOTS if era_pb else 0.0)
    drafter = a.d_head * n + (a.d_arctic if x > 0 else 0.0)
    if era_pb:
        if n + x <= 6:
            committer, cov = a.c_pb, 1.0
        else:
            p_ov = survival_at(n, x, w_over, w_tail, tail_bd,
                               a.comp_uplift, a.tail_uplift, 7)
            committer = (1 - p_ov) * a.c_pb + p_ov * a.c_replay
            cov = 1 - p_ov
    else:
        committer, cov = a.c_replay, 0.0
    step = drafter + verify + committer + a.g_ms
    tps = (accept + 1.0) / step * 1000.0
    return tps, step, committer, cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp_uplift", type=float, default=0.06, help="arctic-complement uplift in overlap (calibrate: head-complement A/B)")
    ap.add_argument("--tail_uplift", type=float, default=0.12, help="arctic-branch uplift in tail (calibrate: tail6-vs-tail6b A/B)")
    ap.add_argument("--top", type=int, default=15)
    # ---- cost model (R5): rank by tps under an era's committer cost ----------
    ap.add_argument("--rank", choices=("accept", "tps"), default="accept")
    ap.add_argument("--era", choices=("replay", "pb"), default="pb",
                    help="replay = CALIBRATION mode (must reproduce measured tail6 ~18.5 fullstep before trusting pb rankings)")
    ap.add_argument("--c_pb", type=float, default=16.0, help="UNMEASURED until pbmech cat9pb CFWD")
    ap.add_argument("--c_replay", type=float, default=70.7, help="measured cng16 (native committer baked)")
    ap.add_argument("--v_base", type=float, default=24.2, help="verify base ms (fit: tail6 88ms @ 21 nodes, v_node 2.9)")
    ap.add_argument("--v_node", type=float, default=2.9, help="verify ms/node (b7 same-session tail6b-vs-tail6 delta)")
    ap.add_argument("--v_chain", type=float, default=1.0, help="verify ms/chain-slot (identity rows; MEASURE via pbmech 18-vs-10 stream delta)")
    ap.add_argument("--d_head", type=float, default=20.0, help="drafter ms per MTP head forward (drafter decomp ~100/5)")
    ap.add_argument("--d_arctic", type=float, default=5.0, help="arctic host adder ms when x>0")
    ap.add_argument("--g_ms", type=float, default=26.0, help="inter-phase gaps/host overhead ms (calibrated so replay-era tail6 == measured 18.5)")
    a = ap.parse_args()
    budget = NPAD - (CHAIN_SLOTS if (a.rank == "tps" and a.era == "pb") else 0)
    rows = []
    for n in range(0, 6):                     # MTP HEAD depth (0=pure arctic .. 5)
        for x in range(0, 22):                # TAIL length (0=pure MTP)
            if n == 0 and x == 0: continue
            for w_over in ((0,) if n == 0 else (0, 1)):
                for w_tail in ((0,) if x == 0 else (0, 1, 2, 3)):
                    for tail_bd in range(0, x + 1):
                        if w_tail == 0 and tail_bd > 0: continue
                        acc, nodes = model(n, x, w_over, w_tail, tail_bd, a.comp_uplift, a.tail_uplift)
                        if nodes > budget or nodes == 0: continue
                        tps, step, cmt, cov = cost_model(a, n, x, w_over, w_tail, tail_bd, acc, nodes)
                        key = tps if a.rank == "tps" else acc
                        rows.append((key, acc, tps, step, cmt, cov, nodes, n, x, w_over, w_tail, tail_bd))
    rows.sort(reverse=True)
    print(f"# merged 2-proposer sweep  rank={a.rank} era={a.era}  comp_uplift={a.comp_uplift} tail_uplift={a.tail_uplift}  nodes<={budget}")
    print(f"  [ref] pure-MTP (n5,x0) accept={model(5,0,0,0,0,0,0)[0]:.3f}(nodes {model(5,0,0,0,0,0,0)[1]})  |  shipped tail6 (n5,x6,no branch) accept={model(5,6,0,0,0,0,0)[0]:.3f}(measured ~5.2 old-regime / 4.32 b7)")
    if a.rank == "tps":
        _t6 = model(5, 6, 0, 0, 0, a.comp_uplift, a.tail_uplift)
        _t6c = cost_model(a, 5, 6, 0, 0, 0, *_t6)
        print(f"  [calib] tail6 under era={a.era}: tps={_t6c[0]:.1f} step={_t6c[1]:.0f}ms committer={_t6c[2]:.0f}ms"
              f"  (replay-era MUST land near the measured 18.5 fullstep; native bar = 27.9)")
    print(f"{'tps':>6} {'accept':>7} {'step':>5} {'cmtms':>5} {'cov':>4} {'nodes':>5}  n=MTP  x=TAIL  w_over  w_tail  tail_bd")
    for key, acc, tps, step, cmt, cov, nodes, n, x, w_over, w_tail, tail_bd in rows[:a.top]:
        tag = " <- shipped tail6" if (n,x,w_over,w_tail,tail_bd)==(5,6,0,0,0) else ""
        print(f"{tps:6.1f} {acc:7.3f} {step:5.0f} {cmt:5.0f} {cov:4.2f} {nodes:5d}  {n:4d}  {x:5d}  {w_over:5d}  {w_tail:5d}  {tail_bd:6d}{tag}")

if __name__ == "__main__":
    main()
