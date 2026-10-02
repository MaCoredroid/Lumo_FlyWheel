#!/usr/bin/env python3
"""Per-optimization attribution on identical replay inputs (same 43-request corpus, same image and vehicle; one
lever toggled per arm). Reports pooled decode rate, accepted tokens/step, per-step GPU spans from the engine's own
timers (target forward, drafter, committer = accepted-path publication) and boot memory (available KV budget,
CUDA-graph pool) as the peak-allocation proxy.
Usage: attribution.py OUT.json label=RUN_DIR[,RUN_DIR...] ..."""
import glob, json, os, re, sys

SIDECAR = "/home/mark/shared/lumoFlyWheel-nvfp4-port-20260816/output/fr13_sfwd_sidecar"


def rate(d):
    recs = [json.loads(l) for l in open(os.path.join(d, "replay.jsonl"))]
    ok = [r for r in recs if not r.get("error")]
    N = sum(r["completion_tokens"] for r in ok); E = sum(r["t_e2e_s"] for r in ok); T = sum(r["t_ttft_s"] for r in ok)
    acc = drafts = 0.0
    for r in ok:
        for k, v in (r.get("spec_delta") or {}).items():
            if "num_accepted_tokens{" in k:
                acc += v
            elif "num_drafts{" in k:
                drafts += v
    return {"n": len(ok), "rate": (N - len(ok)) / (E - T), "acc_per_step": acc / drafts if drafts else None}


def timers(d):
    rr = open(os.path.join(d, "serve_runroot.txt")).read().strip()
    arms = [os.path.basename(p.rstrip("/")) for p in glob.glob(rr + "/*/") if os.path.basename(p.rstrip("/")).startswith("hydra")]
    out = {}
    for f in glob.glob(os.path.join(SIDECAR, arms[0] + "*.json.*")) if arms else []:
        if ".samples." in f:
            continue
        j = json.load(open(f))
        if j.get("schema", "").startswith("fr13.sfwd_gpu_timer"):
            out["wall_ms"] = 1e3 * j["decode_step_wall_seconds"] / max(1, j["n_wall_steps"])
            out["target_ms"] = 1e3 * j["decode_forward_gpu_seconds"] / max(1, j["n_pure_decode_steps_timed"])
        elif j.get("label") == "drafter":
            out["drafter_ms"] = 1e3 * j["gpu_seconds"] / max(1, j["n_spans"])
        elif j.get("label") == "committer":
            out["commit_ms"] = 1e3 * j["gpu_seconds"] / max(1, j["n_spans"])
    return out


def memory(d):
    out = {}
    try:
        s = open(os.path.join(d, "container_after_teardown.log"), errors="ignore").read()
    except FileNotFoundError:
        return out
    for key, pat in (("weights_gib", r"Model loading took ([\d.]+) GiB"), ("kv_avail_gib", r"Available KV cache memory: ([\d.]+) GiB"),
                     ("graph_pool_gib", r"CUDA graph pool memory: ([\d.]+) GiB"), ("kv_tokens", r"GPU KV cache size: ([\d,]+) tokens")):
        m = re.findall(pat, s)
        if m:
            out[key] = float(m[0].replace(",", ""))
    return out


def main():
    out_path = sys.argv[1]; rows = {}
    for arg in sys.argv[2:]:
        label, dirs = arg.split("=", 1)
        reps = []
        for d in dirs.split(","):
            x = {"run": os.path.basename(d)}; x.update(rate(d)); x.update(timers(d)); x.update(memory(d)); reps.append(x)
        mean = {k: sum(r[k] for r in reps if r.get(k) is not None) / max(1, sum(1 for r in reps if r.get(k) is not None))
                for k in ("rate", "acc_per_step", "wall_ms", "target_ms", "drafter_ms", "commit_ms", "kv_avail_gib", "graph_pool_gib")}
        rows[label] = {"replicates": reps, "mean": mean}
    base = rows[list(rows)[0]]["mean"]
    print(f"{'arm':28s} {'n':>2s} {'rate':>7s} {'Δrate':>7s} {'acc':>6s} {'wall':>7s} {'target':>7s} {'drafter':>7s} {'commit':>7s} {'KVavail':>8s} {'graphs':>7s}")
    for label, r in rows.items():
        m = r["mean"]
        d = (m["rate"] / base["rate"] - 1) * 100
        f = lambda k, fmt="%7.1f": (fmt % m[k]) if m.get(k) else "      -"
        print(f"{label:28s} {len(r['replicates']):2d} {m['rate']:7.2f} {d:+6.1f}% {m['acc_per_step']:6.3f} {f('wall_ms')} {f('target_ms')} {f('drafter_ms')} {f('commit_ms')} {f('kv_avail_gib','%8.2f')} {f('graph_pool_gib','%7.2f')}")
    json.dump(rows, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
