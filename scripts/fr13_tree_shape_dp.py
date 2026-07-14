#!/usr/bin/env python3
"""S2: offline tree-shape optimizer for the FR13 caterpillar family (CPU-only).

MODEL (designed 2026-07-14, calibrated on the 4-arm live campaign):
  Shapes: per-depth widths [w1..wD]; the spine continues, non-spine siblings are
  LEAVES (matches cat6root/cat8/t33333). A rescue at depth d commits the rescued
  leaf and ENDS the walk; only spine acceptance descends.

  E[accepted_len] = sum_d P(len >= d), with
    P(len >= d) = (prod_{k<d} p_k) * (p_d + (1-p_d) * rescue_d(w_d))
    rescue_d(w) = 1 - (1 - q_d)^(w-1)        (independent-sibling saturation)
  p_d = conditional spine accept at depth d — measured (cat8 brhist, 199 events):
    p = [0.884, 0.801, 0.858, 0.777, 0.915]  (depths 1..5; d>5 -> p5)
  q_d = single-sibling rescue efficiency — measured from cat8 branch rows:
    q = [0.65, 0.43, 0.50] (depths 1..3; d>3 -> q3)

  COST: step_ms(S) = verify(rows) + DRAFTER + C
    verify(rows): sfwd fit — 204.3 @7, 206.5 @9, 233.9 @16
      => 203.4 + 0.9*(r-7) for r<=9 ; 206.5 + 3.9*(r-9) for r>9
    DRAFTER: measured 88-100 ms/step for cat8 and — CRITICAL CORRECTION —
      the fork uses PARALLEL drafting (one draft pass, split5/6 path-map
      elimination), so drafter cost is ~SHAPE-FLAT to first order. Depth does
      NOT buy drafter savings (the earlier "20ms/level" model was WRONG for
      this config). Held constant; only RELATIVE ranking is meaningful.
    C: committer+gaps, ~shape-flat, held constant.
  Objective: committed tokens/sec ∝ (E[len] + 1) / step_ms(S).

  VALIDATION: the model must reproduce the measured ordering and approximate
  accept values of cat6root/cat8/t33333 before its ranking is trusted.
  KNOWN LIMITS: q_d beyond depth 3 unmeasured; sibling independence is an
  approximation; single-boot calibration; resolve-quality (t33333 11/16) NOT
  modeled — shapes near the top should be A/B'd live, and quality tracked.
"""
import itertools

P = [0.884, 0.801, 0.858, 0.777, 0.915]
Q = [0.65, 0.43, 0.50]
DRAFTER_C = 94.0   # ms, shape-flat (parallel drafting)
OTHER_C = 50.0     # ms, committer+gaps, shape-flat


def p_d(d):
    return P[min(d - 1, len(P) - 1)]


def q_d(d):
    return Q[min(d - 1, len(Q) - 1)]


def e_len(widths):
    reach = 1.0
    total = 0.0
    for d, w in enumerate(widths, 1):
        pd = p_d(d)
        resc = 1.0 - (1.0 - q_d(d)) ** max(0, w - 1)
        total += reach * (pd + (1.0 - pd) * resc)
        reach *= pd
    return total


def verify_ms(rows):
    if rows <= 9:
        return 203.4 + 0.9 * max(0, rows - 7)
    return 206.5 + 3.9 * (rows - 9)


def score(widths):
    rows = 1 + sum(widths)
    el = e_len(widths)
    ms = verify_ms(rows) + DRAFTER_C + OTHER_C
    return (el + 1.0) / ms * 1000.0, el, rows, ms


KNOWN = {
    "cat6root": [2, 1, 1, 1, 1],
    "cat8": [2, 2, 2, 1, 1],
    "t33333": [3, 3, 3, 3, 3],
    "chain5(native-ish)": [1, 1, 1, 1, 1],
}


def main():
    print("=== calibration check (measured accept_per_event minus ~bonus) ===")
    meas = {"cat6root": 3.333, "cat8": 3.500, "t33333": 3.567}
    for name, w in KNOWN.items():
        s, el, rows, ms = score(w)
        m = meas.get(name)
        # measured accept_per_event ≈ E[len] + bonus-ish share; compare SHAPE deltas
        print(f"  {name:<20} widths={w} rows={rows} E[len]={el:.3f} "
              f"model_ms={ms:.1f} tok/s∝{s:.3f}"
              + (f"  measured_accept={m}" if m else ""))
    print()
    cands = []
    for depth in (3, 4, 5, 6):
        for widths in itertools.product((1, 2, 3, 4), repeat=depth):
            rows = 1 + sum(widths)
            if rows < 7 or rows > 17:
                continue
            if sum(1 for w in widths if w > 1) < 1:
                continue  # never below the branch class (branches = deliverable)
            s, el, r, ms = score(list(widths))
            cands.append((s, el, r, ms, list(widths)))
    cands.sort(reverse=True)
    print("=== top 12 shapes by modeled tokens/sec ===")
    for s, el, r, ms, w in cands[:12]:
        tag = ""
        for name, kw in KNOWN.items():
            if list(kw) == w:
                tag = f"  <= {name}"
        print(f"  widths={w} rows={r:2d} E[len]={el:.3f} ms={ms:.1f} tok/s∝{s:.4f}{tag}")
    print()
    for name, w in KNOWN.items():
        s, el, r, ms = score(w)
        rank = 1 + sum(1 for c in cands if c[0] > s)
        print(f"  {name}: rank {rank}/{len(cands)} (tok/s∝{s:.4f})")


if __name__ == "__main__":
    main()
