#!/usr/bin/env python3
"""Mid-run windowed speed read from /metrics counter deltas.

Same math as the deploy record (measured wall over engine counters), windowed:
  tps       = d(generation_tokens) / dt      (committed tokens per wall second)
  step_wall = dt / d(iterations)             (ms)
  eps       = d(spec_drafts) / d(iterations) (events per step)
  accept/d  = d(accepted) / d(spec_drafts) + 1

Diagnostic-tier: quote WITH window length + prefill context; the deploy record
stays the record. Usage: fr13_midrun_window.py [window_s] [port]
"""
import re, sys, time, urllib.request

def snap(port):
    txt = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10).read().decode()
    def g(name):
        m = re.search(rf"^vllm:{name}\S*\s+([\d.e+]+)$", txt, re.M)
        return float(m.group(1)) if m else 0.0
    return {
        "gen": g("generation_tokens_total"),
        "steps": g("iteration_tokens_total_count"),
        "drafts": g("spec_decode_num_drafts_total"),
        "acc": g("spec_decode_num_accepted_tokens_total"),
        "prompt": g("prompt_tokens_total"),
        "t": time.monotonic(),
    }

def main():
    win = float(sys.argv[1]) if len(sys.argv) > 1 else 90.0
    port = sys.argv[2] if len(sys.argv) > 2 else "9950"
    a = snap(port)
    time.sleep(win)
    b = snap(port)
    dt = b["t"] - a["t"]
    ds = b["steps"] - a["steps"]
    dd = b["drafts"] - a["drafts"]
    dg = b["gen"] - a["gen"]
    da = b["acc"] - a["acc"]
    dp = b["prompt"] - a["prompt"]
    if ds <= 0 or dd <= 0:
        print(f"window {dt:.0f}s: no decode activity (dsteps={ds:.0f} ddrafts={dd:.0f})")
        return
    print(
        f"window {dt:.0f}s: tps={dg/dt:.2f} step_wall={dt/ds*1000:.1f}ms "
        f"eps={dd/ds:.2f} accept/draft+1={da/dd+1:.3f} "
        f"prefill_toks/s={dp/dt:.0f} (windowed diagnostic; deploy record = the record)"
    )

if __name__ == "__main__":
    main()
