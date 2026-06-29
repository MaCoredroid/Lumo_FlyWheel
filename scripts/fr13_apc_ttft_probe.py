#!/usr/bin/env python3
# FR13 APC TTFT probe — quantify the prefix-cache TTFT speedup at a given mamba_block_size.
#
# TTFT is approximated by the wall latency of a max_output_tokens=1 /v1/responses request
# (one decode step is negligible vs the ~28K-token seq49 prefill), measured:
#   COLD  = right after a prefix-cache reset  -> full prefill (cache MISS) = the no-cache baseline
#   WARM  = same request repeated             -> restore + mamba tail re-prefill (cache HIT)
# speedup = ttft_cold / ttft_warm.  Larger mamba_block_size => bigger tail re-prefill on a hit
# => smaller warm speedup (the losslessness<->TTFT dial). cached_tokens>0 on WARM confirms the
# hit is real (else the point is VACUOUS). Reuses the replay's exact request plumbing so the
# prompt + endpoint match the deployed condition. Realistic prefix (real 12907 seq49), temp 0.6.
import argparse, json, time, sys
from fr13_apc_multiturn_replay import post, reset_cache, usage_field, load_trajectory, _REPLAY_TEMP


def _probe_once(port, req, timeout):
    r = dict(req)
    r["temperature"] = _REPLAY_TEMP
    r["top_p"] = 1.0
    r["store"] = False
    r["stream"] = False
    r["max_output_tokens"] = 1            # ~TTFT: prefill + 1 decode step
    r.pop("previous_response_id", None)
    t0 = time.time()
    resp = post(port, "/v1/responses", r, timeout=timeout)
    dt = time.time() - t0
    return dt, usage_field(resp, "input_tokens"), usage_field(resp, "input_tokens_details", "cached_tokens")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--dumps-dir", required=True)
    ap.add_argument("--seq", type=int, default=49)
    ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=float, default=2400)
    ap.add_argument("--warm-reps", type=int, default=3)   # steady-state warm (take the min)
    a = ap.parse_args()

    turns = load_trajectory(a.dumps_dir, 0)
    if not turns:
        print("FAIL: no turns", file=sys.stderr); return 2
    turn = next((t for t in turns if t[0] == a.seq), turns[-1])
    seq, fname, req = turn

    # COLD: fresh cache -> full prefill (the no-cache baseline)
    reset_cache(a.port); time.sleep(1.0)
    cold_dt, in_tok, cold_cached = _probe_once(a.port, req, a.timeout)

    # WARM: repeat (prefix-cache hit); take the min over reps as steady-state
    warm = []
    for _ in range(a.warm_reps):
        wdt, _wt, wcached = _probe_once(a.port, req, a.timeout)
        warm.append((wdt, wcached or 0))
    warm_dt = min(w[0] for w in warm)
    warm_cached = max(w[1] for w in warm)

    rec = {
        "block": a.block, "seq": seq, "input_tokens": in_tok,
        "ttft_cold_s": round(cold_dt, 3), "ttft_warm_s": round(warm_dt, 3),
        "cold_cached": cold_cached or 0, "warm_cached": warm_cached,
        "speedup": round(cold_dt / warm_dt, 3) if warm_dt > 0 else None,
        "warm_reps_s": [round(w[0], 3) for w in warm],
        "vacuous_warm_miss": warm_cached <= 0,
    }
    json.dump(rec, open(a.out, "w"))
    vac = "  !!VACUOUS(warm cache MISS — speedup meaningless)" if warm_cached <= 0 else ""
    print(f"[ttft block={a.block}] in_tok={in_tok}  cold={cold_dt:.2f}s  warm={warm_dt:.2f}s  "
          f"speedup={rec['speedup']}x  warm_cached={warm_cached}{vac}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
