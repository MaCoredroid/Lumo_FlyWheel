#!/usr/bin/env python3
"""Per-arm summary for the drafter-pass sweep: pooled decode rate (paper Eq. 5), accepted
tokens per step and survival by depth (spec_delta per-position counters), and the per-step
GPU split from the timer sidecars copied next to each replay.
Usage: sweep_summary.py RUN_DIR [RUN_DIR ...]   (or a glob such as replay/*-sw*)"""
import collections, glob, json, os, re, sys


def timers(d):
    out = {}
    for f in glob.glob(os.path.join(d, "*.json.*")):
        if f.endswith(tuple(f".samples.{x}" for x in range(10000))) or ".samples." in f:
            continue
        try:
            j = json.load(open(f))
        except Exception:
            continue
        if j.get("schema", "").startswith("fr13.sfwd_gpu_timer"):
            st = j.get("n_wall_steps") or 0
            out["steps_timed"] = j.get("n_pure_decode_steps_timed")
            out["wall_ms"] = 1e3 * j["decode_step_wall_seconds"] / st if st else None
            out["target_ms"] = 1e3 * j["decode_forward_gpu_seconds"] / max(1, j["n_pure_decode_steps_timed"])
        elif j.get("label") == "drafter":
            out["drafter_ms"] = 1e3 * j["gpu_seconds"] / max(1, j["n_spans"])
        elif j.get("label") == "committer":
            out["commit_ms"] = 1e3 * j["gpu_seconds"] / max(1, j["n_spans"])
    return out


def summarize(d):
    f = os.path.join(d, "replay.jsonl")
    if not os.path.exists(f):
        return None
    recs = [json.loads(l) for l in open(f)]
    ok = [r for r in recs if not r.get("error")]
    N = sum(r.get("completion_tokens") or 0 for r in ok)
    E = sum(r["t_e2e_s"] for r in ok)
    T = sum(r.get("t_ttft_s") or 0 for r in ok)
    acc = collections.Counter(); drafts = 0
    for r in ok:
        for k, v in (r.get("spec_delta") or {}).items():
            m = re.search(r'per_pos\{.*position="(\d+)"', k)
            if m:
                acc[int(m.group(1))] += v
            elif "num_drafts{" in k:
                drafts += v
    s = {"run": os.path.basename(d), "n": len(ok), "errors": len(recs) - len(ok),
         "rate": (N - len(ok)) / (E - T) if E > T else None, "tokens": N}
    if drafts:
        s["steps"] = int(drafts); s["acc_per_step"] = sum(acc.values()) / drafts
        s["survival"] = [round(acc[i] / drafts, 3) for i in range(max(acc) + 1)]
    s.update(timers(d))
    return s


def main():
    dirs = []
    for a in sys.argv[1:]:
        dirs += sorted(glob.glob(a)) if any(c in a for c in "*?[") else [a]
    rows = [x for x in (summarize(d) for d in dirs) if x and x["n"] and x["rate"]]
    for s in rows:
        t = " ".join(f"{k}={s[k]:.1f}" for k in ("wall_ms", "target_ms", "drafter_ms", "commit_ms") if s.get(k))
        print(f"{s['run'][:46]:46s} n={s['n']:2d} err={s['errors']} rate={s['rate']:.2f} "
              + (f"acc/step={s['acc_per_step']:.3f} steps={s['steps']} " if 'steps' in s else "") + t)
        if "survival" in s:
            print("   survival:", s["survival"])
    json.dump(rows, sys.stdout if False else open(os.devnull, "w"))
    return rows


if __name__ == "__main__":
    main()
