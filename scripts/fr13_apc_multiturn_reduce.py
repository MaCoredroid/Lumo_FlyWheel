#!/usr/bin/env python3
# FR13 APC MULTI-TURN ORACLE GATE — reducer.
#
# Aligns the three per-boot replays by turn (seq), and for each turn computes the
# first-divergence offset of cache-OFF vs the oracle (the FLOOR = tree-spec drift from
# no-spec) and of cache-ON vs the oracle (the TEST). The headline question:
#
#   Does cache-ON diverge from the oracle EARLIER / on MORE turns than the cache-OFF
#   floor, and does that excess GROW with turn index (the compounding signal that a
#   per-turn-below-floor cache defect accumulates over a multi-turn agentic solve)?
#
# Compare target is US (cache-ON) vs the no-spec recurrent oracle, with cache-OFF as
# the calibrated floor — the standing rule (never a proxy / backend name). At temp=0
# the first-divergence offset is a true argmax-flip position.
import argparse, json
from pathlib import Path


def first_div(a, b):
    """char offset of first difference between two canon strings; None if identical
    (one a prefix of the other counts as divergence at the shorter length)."""
    if a is None or b is None:
        return None
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    if i == n and len(a) == len(b):
        return None  # byte-identical = lossless this turn
    return i


def load(p):
    d = json.load(open(p))
    return {r.get("seq", r.get("idx")): r for r in d.get("records", [])}, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--off", required=True)
    ap.add_argument("--on", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    orc, dorc = load(a.oracle)
    off, doff = load(a.off)
    on, don = load(a.on)
    keys = sorted(set(orc) & set(off) & set(on), key=lambda k: (k is None, k))

    rows = []
    for k in keys:
        ro, rf, rn = orc[k], off[k], on[k]
        co, cf, cn = ro.get("canon"), rf.get("canon"), rn.get("canon")
        d_off = first_div(co, cf)   # FLOOR: tree-spec vs no-spec
        d_on = first_div(co, cn)    # TEST: cache-ON vs no-spec
        d_onoff = first_div(cf, cn) # direct cache effect: ON vs OFF
        rows.append({
            "seq": k,
            "oracle_len": len(co) if co else None,
            "off_div_vs_oracle": d_off,
            "on_div_vs_oracle": d_on,
            "on_div_vs_off": d_onoff,
            "on_cached_tokens": rn.get("cached_tokens"),
            "off_cached_tokens": rf.get("cached_tokens"),
            # ON flips EARLIER than the floor this turn? (None == infinity == lossless)
            "on_worse_than_floor": _earlier(d_on, d_off),
            "input_tokens": rn.get("input_tokens"),
        })

    n = len(rows)
    on_loss = [r for r in rows if r["on_div_vs_oracle"] is not None]       # ON diverged from oracle
    off_loss = [r for r in rows if r["off_div_vs_oracle"] is not None]     # OFF (floor) diverged from oracle
    on_worse = [r for r in rows if r["on_worse_than_floor"]]               # ON strictly worse than floor
    on_cache_engaged = sum(1 for r in rows if (r["on_cached_tokens"] or 0) > 0)
    off_cache_clean = sum(1 for r in rows if (r["off_cached_tokens"] or 0) == 0)

    # compounding: among ON-worse-than-floor turns, are they concentrated LATE? Compare
    # the mean turn-ordinal of ON-worse turns vs all turns. Ordinal = position in keys.
    ordinal = {k: i for i, k in enumerate(keys)}
    mean_ord_all = (sum(ordinal[r["seq"]] for r in rows) / n) if n else None
    mean_ord_worse = (sum(ordinal[r["seq"]] for r in on_worse) / len(on_worse)) if on_worse else None
    # split half: divergence rate in first half vs second half
    half = n // 2
    on_worse_first = sum(1 for r in on_worse if ordinal[r["seq"]] < half)
    on_worse_second = len(on_worse) - on_worse_first

    if on_cache_engaged == 0:
        verdict = "INVALID: cache never engaged on the ON arm (cached_tokens==0 every turn) — boot/keying bug, rerun"
    elif not on_worse and len(on_loss) <= len(off_loss):
        verdict = ("CACHE-ON MULTI-TURN LOSSLESS (eager/BI/temp0): ON tracks the oracle no worse than the "
                   "cache-OFF floor on every turn. The temp=0.6 rate-gate failure is NOT a greedy argmax "
                   "defect here -> escalate to cuda-graph + temp-0.6 same-seed regime.")
    elif on_worse and on_worse_second > on_worse_first:
        verdict = (f"COMPOUNDING CACHE DRIFT LOCALIZED: ON diverges from the oracle earlier than the floor on "
                   f"{len(on_worse)}/{n} turns, concentrated LATE ({on_worse_second} in 2nd half vs {on_worse_first} "
                   f"in 1st). Per-turn-below-floor cache defect accumulates over the multi-turn solve.")
    else:
        verdict = (f"CACHE-ON WORSE THAN FLOOR but not clearly late-concentrated: ON worse on {len(on_worse)}/{n} "
                   f"turns ({on_worse_first} early / {on_worse_second} late). Cache implicated; compounding "
                   f"trend inconclusive at this turn count.")

    summary = {
        "n_turns": n,
        "on_cache_engaged_turns": on_cache_engaged,
        "off_cache_clean_turns": off_cache_clean,
        "off_diverged_from_oracle_turns": len(off_loss),   # the floor
        "on_diverged_from_oracle_turns": len(on_loss),
        "on_strictly_worse_than_floor_turns": len(on_worse),
        "on_worse_first_half": on_worse_first,
        "on_worse_second_half": on_worse_second,
        "mean_ordinal_all": round(mean_ord_all, 1) if mean_ord_all is not None else None,
        "mean_ordinal_on_worse": round(mean_ord_worse, 1) if mean_ord_worse is not None else None,
        "VERDICT": verdict,
    }
    json.dump({"summary": summary, "rows": rows}, open(a.out, "w"), indent=1)

    print("\n=== FR13 APC MULTI-TURN ORACLE GATE — VERDICT ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("\n  per-turn (seq | oracle_len | off_div(floor) | on_div(test) | on_vs_off | on_cached):")
    for r in rows:
        flag = "  <-- ON worse than floor" if r["on_worse_than_floor"] else ""
        print(f"    {str(r['seq'])[:14]:>14} | {str(r['oracle_len']):>6} | {str(r['off_div_vs_oracle']):>7} "
              f"| {str(r['on_div_vs_oracle']):>7} | {str(r['on_div_vs_off']):>7} | {str(r['on_cached_tokens']):>7}{flag}")
    print(f"\nsaved -> {a.out}")


def _earlier(test, floor):
    """True if `test` diverges strictly earlier than `floor` (None == infinity)."""
    if test is None:
        return False               # ON lossless this turn -> never worse
    if floor is None:
        return True                # OFF lossless but ON diverged -> ON strictly worse
    return test < floor


if __name__ == "__main__":
    main()
