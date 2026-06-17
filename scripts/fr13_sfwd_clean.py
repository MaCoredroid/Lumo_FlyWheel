#!/usr/bin/env python3
"""FR13_SFWD_GPU_TIMER: compute the CLEAN prefill-independent s_per_fwd_gpu for an arm
DIRECTLY from the worker timer's JSON sidecar(s), bypassing the reducer's /metrics-
bracket path.

WHY: cmd_deploy_speed historically divided the (pure-decode-only) decode-forward GPU
seconds by the GLOBAL spec_decode_num_drafts_total. That global counter includes drafts
on mixed prefill+decode steps the timer EXCLUDES from the numerator (~49% of drafts at
B=4 deployment), so the ratio is prefill-load-dependent and arm-dependent -> it defeats
prefill-independence. The sidecar carries the MATCHED denominators
(n_pure_decode_steps_timed, n_drafts_in_timed_steps), both restricted to the same
pure-decode steps as the numerator, so the ratios below ARE prefill-independent.

Use this for sweep runs whose run_swe_bench predates the matched-denominator synthetic
lines (fr13_measure.py is fixed going forward; this re-derives for in-flight runs).

Usage:  fr13_sfwd_clean.py <sidecar_glob> [<sidecar_glob> ...]
  e.g.  fr13_sfwd_clean.py 'output/fr13_sfwd_sidecar/nativeE5_b4.json.*'
"""
import glob
import json
import sys


def clean(globs):
    secs = steps = drafts = 0.0
    files = []
    for g in globs:
        files += glob.glob(g)
    for f in sorted(set(files)):
        try:
            d = json.loads(open(f, encoding="utf-8").read())
            secs += float(d.get("decode_forward_gpu_seconds", 0.0))
            steps += float(d.get("n_pure_decode_steps_timed", 0.0))
            drafts += float(d.get("n_drafts_in_timed_steps", 0.0))
        except Exception as e:  # noqa: BLE001
            print(f"  skip {f}: {e}", file=sys.stderr)
    return secs, steps, drafts, sorted(set(files))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    secs, steps, drafts, files = clean(sys.argv[1:])
    out = {
        "sidecar_files": files,
        "decode_forward_gpu_seconds": secs,
        "n_pure_decode_steps_timed": steps,
        "n_drafts_in_timed_steps": drafts,
        # MATCHED bases (prefill-independent). per-spec-event matches cmd_speed's basis;
        # per-forward matches the metric name + the banked B=1 ~0.218.
        "s_per_fwd_gpu_per_specevent": (secs / drafts) if drafts > 0 else None,
        "s_per_fwd_gpu_per_forward": (secs / steps) if steps > 0 else None,
        "basis": "MATCHED pure-decode-only (numerator + denominator same steps); "
                 "NOT the global spec_decode_num_drafts_total",
    }
    print(json.dumps(out, indent=2))
