#!/usr/bin/env python3
"""Reduce the FR13_REPLAY_MULTISTREAM A/B (task #43).

Reads the two CF2 whole-committer GPU-timer sidecars (one sync/step => captures stream
overlap, unlike the per-launch REPLAY_GPU_TIMER whose per-launch synchronize serializes):
  ms_base = multistream OFF (tail6_cf2)  -> output/fr13_sfwd_sidecar/tail6_cf2_commit.json
  ms_strm = multistream ON  (tail6_ms )  -> output/fr13_sfwd_sidecar/tail6_ms_commit.json
Each json = {"gpu_seconds": float, "n_spans": int}  (n_spans = committer CALLS = per-step).

Reports committer ms/step for each arm + the delta. The replay is ~66ms of the committer; if
multistream overlaps the 48 latency-bound per-layer replays it should collapse toward the ~5ms
bandwidth floor => committer ms/step drops by tens of ms. If ms_strm ~= ms_base => multistream
is dead on GB10 (SM-occupancy / memory-serialization) => revert, replay IS the floor.

Also pulls accept from each arm's deploy_speed json for the behavioral-lossless sanity check
(cross-boot autotune forbids byte-identity; accept must be COMPARABLE, not identical-to-the-ULP).
"""
import json
import os
import sys

SIDE = "output/fr13_sfwd_sidecar"
AB = "output/fr13_ms_ab"


def load(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception as e:  # noqa: BLE001
        return {"_err": f"{type(e).__name__}: {e}", "_path": path}


def committer_ms(js):
    if "gpu_seconds" not in js or "n_spans" not in js or not js["n_spans"]:
        return None
    return 1000.0 * js["gpu_seconds"] / js["n_spans"]


def accept_of(arm):
    # deploy_speed json written by fr13_measure.py deploy-speed
    p = f"{AB}/{arm}/deploy_speed_ms.json"
    js = load(p)
    for k in ("acceptance_per_event", "accept_per_event", "acceptance", "accept"):
        if isinstance(js, dict) and k in js:
            return js[k], p
    return None, p


def main():
    base = load(f"{SIDE}/tail6_cf2_commit.json")
    strm = load(f"{SIDE}/tail6_ms_commit.json")
    print("== FR13_REPLAY_MULTISTREAM A/B — committer GPU time (CF2, per-step) ==")
    for name, js in (("ms_base (OFF)", base), ("ms_strm (ON) ", strm)):
        ms = committer_ms(js)
        if ms is None:
            print(f"  {name}: NO DATA ({js})")
        else:
            print(f"  {name}: {ms:8.2f} ms/step   (gpu_s={js['gpu_seconds']:.3f}, "
                  f"n_spans={js['n_spans']})")
    mb, msn = committer_ms(base), committer_ms(strm)
    if mb and msn:
        delta = mb - msn
        pct = 100.0 * delta / mb
        print(f"\n  DELTA (base - strm): {delta:+.2f} ms/step  ({pct:+.1f}%)")
        if delta > 15:
            print("  => MULTISTREAM WINS: replay overlap collapsed the committer. Gate lossless, sweep N.")
        elif delta < 5:
            print("  => NO WIN (<5ms): multistream dead on GB10 (occupancy/mem-serialize). Revert; replay IS floor.")
        else:
            print("  => marginal (5-15ms): sweep N=2/8, confirm before claiming.")
    print("\n== behavioral-lossless (accept COMPARABLE, not byte-identical: cross-boot autotune) ==")
    for arm in ("ms_base", "ms_strm"):
        acc, p = accept_of(arm)
        print(f"  {arm}: accept={acc}   ({p})")
    print("\n== engagement (multistream must have FIRED, not vacuous) ==")
    eng = os.popen(
        "grep -rl 'FR13_REPLAY_MULTISTREAM] ENGAGED' " + AB + " 2>/dev/null"
    ).read().strip()
    print(f"  ENGAGED marker in: {eng or '(NOT FOUND — ms_strm may be vacuous, DISTRUST)'}")


if __name__ == "__main__":
    sys.exit(main())
