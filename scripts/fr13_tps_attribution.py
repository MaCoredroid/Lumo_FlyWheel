#!/usr/bin/env python3
"""FR13 per-length TPS attribution (RELIABLE global-rate formula model).

Built ONLY on engine-global rates (deploy_speed json) + per-request COUNTS
(offload completion_tokens, spec_decode_num_drafts). NOT the offload proxy's
per-request time sums -- those over-count badly (one req logged 616s prefill+
decode vs a 243s wall), so they cannot carry a per-request decomposition.

Per-step decode cost is N-independent, so the per-length split is EXACT analytic
(not noisy per-request):
    tau_step    = s_per_fwd            (engine decode-WALL per step; idle-incl)
    tau_verify  = s_per_fwd_gpu        (verify forward GPU/step)
    tau_drafter = drafter sidecar gpu_seconds/n_spans   (FR13_DFWD_GPU_TIMER)
    tau_commit  = committer sidecar gpu_seconds/n_spans  (FR13_CFWD_GPU_TIMER)
    tau_idle    = tau_step - tau_verify - tau_drafter - tau_commit   (sched/sync)
    accept      = accept_per_event
    TTFT(avg)   = prefill_frac * (total_drafts * tau_step) / n_requests

Per request of output length N (steps = N/(accept+1)):
    decode = steps*tau_step ; wall = TTFT + decode ; TPS(N) = N/wall
    wall split: prefill=TTFT ; verify=steps*tau_verify ; drafter=steps*tau_drafter
                commit=steps*tau_commit ; idle=steps*tau_idle
Bucket requests by N -> token-weighted component shares + TPS per bucket.

Until a forward-cost campaign runs FR13_DFWD_GPU_TIMER, tau_drafter/tau_commit
are 0 and "rest" = drafter+committer+idle is shown as one column.

Usage:
  fr13_tps_attribution.py --arm cat555_b1 \
      --deploy output/fr13_wide_swe/cat555_b1_deploy_speed.json
  (optional) --dfwd-sidecar output/fr13_sfwd_sidecar/cat555_b1.dfwd.json
             --cfwd-sidecar output/fr13_sfwd_sidecar/cat555_b1.cfwd.json
"""
import argparse
import glob
import json
from pathlib import Path

BUCKETS = [("<=64", 0, 64), ("65-256", 65, 256), (">256", 257, 1 << 30)]


def load_counts(metrics_path):
    """Per-request N + drafts (reliable counts; ignore the proxy time sums)."""
    ns, drafts = [], 0
    for line in Path(metrics_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        n = r.get("completion_tokens") or 0
        d = r.get("spec_decode_num_drafts") or 0
        # real decode requests: an output + at least one spec draft
        if n > 0 and d > 0:
            ns.append(int(n))
            drafts += int(d)
    return ns, drafts


def sidecar_tau(path_or_glob):
    if not path_or_glob:
        return 0.0
    files = glob.glob(path_or_glob) if any(c in path_or_glob for c in "*?") \
        else [path_or_glob]
    sec, spans = 0.0, 0
    for f in files:
        if not Path(f).exists():
            continue
        d = json.loads(Path(f).read_text())
        sec += float(d.get("gpu_seconds") or 0.0)
        spans += int(d.get("n_spans") or 0)
    return (sec / spans) if spans else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm")
    ap.add_argument("--deploy", required=True, help="deploy_speed json")
    ap.add_argument("--metrics", help="offload_request_metrics.jsonl (default from arm)")
    ap.add_argument("--root", default="output/fr13_bigdenom_swe")
    ap.add_argument("--dfwd-sidecar", help="FR13_DFWD_GPU_TIMER sidecar (or glob)")
    ap.add_argument("--cfwd-sidecar", help="FR13_CFWD_GPU_TIMER sidecar (or glob)")
    args = ap.parse_args()

    dep = json.loads(Path(args.deploy).read_text())
    tau_step = float(dep["s_per_fwd"])
    tau_verify = float(dep["s_per_fwd_gpu"])
    accept = float(dep["accept_per_event"])
    prefill_frac = float(dep.get("prefill_frac") or 0.0)

    tau_draft = sidecar_tau(args.dfwd_sidecar)
    tau_commit = sidecar_tau(args.cfwd_sidecar)
    tau_idle = max(0.0, tau_step - tau_verify - tau_draft - tau_commit)
    # "rest" lumps whatever isn't separately measured (drafter+committer+idle
    # when no campaign yet; just committer+idle once the drafter timer ran)
    tau_rest = max(0.0, tau_step - tau_verify - tau_draft - tau_commit)

    metrics = args.metrics or f"{args.root}/{args.arm}/offload_request_metrics.jsonl"
    ns, total_drafts = load_counts(metrics)
    n_req = len(ns)
    if not n_req:
        print(f"ERR: no decode requests in {metrics}")
        return 2
    total_decode = total_drafts * tau_step
    ttft = (prefill_frac * total_decode / n_req) if n_req else 0.0

    label = args.arm or Path(args.deploy).stem
    print(f"\n=== TPS attribution (formula): {label} ===")
    print(f"  per-step (s): step={tau_step:.4f} verify={tau_verify:.4f} "
          f"drafter={tau_draft:.4f} commit={tau_commit:.4f} idle={tau_idle:.4f}"
          f"  | accept={accept:.2f} TTFT={ttft:.2f}s  ({n_req} req, "
          f"{total_drafts} steps)")
    split = (tau_draft > 0 or tau_commit > 0)
    if split:
        hdr = "prefill verify draft commit idle"
    else:
        hdr = "prefill verify  rest(draft+commit+idle)"
    print(f"\nbucket    nreq  outtok tok%  TPS  meanN | wall-share: {hdr}")
    for name, lo, hi in BUCKETS + [("ALL", 0, 1 << 30)]:
        bn = ns if name == "ALL" else [n for n in ns if lo <= n <= hi]
        if not bn:
            continue
        tok = sum(bn)
        meanN = tok / len(bn)
        steps = meanN / (accept + 1.0)
        decode = steps * tau_step
        wall = ttft + decode
        tps = meanN / wall if wall else 0.0

        def pc(x):
            return 100.0 * x / wall if wall else 0.0
        pre = pc(ttft)
        ver = pc(steps * tau_verify)
        if split:
            dr = pc(steps * tau_draft)
            cm = pc(steps * tau_commit)
            idl = pc(steps * tau_idle)
            comp = f"{pre:6.0f}% {ver:5.0f}% {dr:4.0f}% {cm:5.0f}% {idl:4.0f}%"
        else:
            rest = pc(steps * tau_rest)
            comp = f"{pre:6.0f}% {ver:5.0f}%  {rest:4.0f}%"
        print(f"{name:7s} {len(bn):5d} {tok:7d} {100.0*tok/sum(ns):3.0f}% "
              f"{tps:5.1f} {meanN:6.0f} | {comp}")
    print("\nTPS(N)=N/(TTFT + N/(accept+1)*tau_step). short=prefill-dominated "
          "(spec/kernel ~irrelevant), long=decode-dominated (accept+per-step "
          "stack drive it). tau_step is engine decode-WALL (idle-incl).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
