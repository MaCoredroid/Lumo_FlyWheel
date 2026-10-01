#!/usr/bin/env python3
"""Distributional-correctness sampler.

For a fixed subset of recorded agent requests, draw many short sampled continuations
from one serving arm. Every arm gets byte-identical request bodies (deployed sampling
settings, no per-request seed). Analysis compares per-position token marginals across
arms/boots.

Usage: dist_sample.py --arm NAME --requests DIR --out FILE [--samples 40] [--subset 20]
                      [--every 2] [--max-tokens 24] [--temperature 0.6]
                      [--order sample-outer|request-outer] [--auth-hook module:function]
"""
import argparse, glob, importlib, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replay as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--requests", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--subset", type=int, default=20)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--auth-hook", default="")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--order", choices=["sample-outer", "request-outer"], default="sample-outer")
    a = ap.parse_args()
    hook = None
    if a.auth_hook:
        mod, fn = a.auth_hook.split(":")
        hook = getattr(importlib.import_module(mod), fn)
    files = sorted(glob.glob(os.path.join(a.requests, "*.json")))[:: a.every][: a.subset]
    if a.order == "sample-outer":
        plan = [(s, i, f) for s in range(a.samples) for i, f in enumerate(files)]
    else:
        plan = [(s, i, f) for i, f in enumerate(files) for s in range(a.samples)]
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                r = json.loads(line); done.add((r["request"], r["sample"]))
            except Exception:
                pass
    n = 0; t0 = time.time(); total = len(plan)
    with open(a.out, "a") as fo:
        for s, i, f in plan:
            name = os.path.basename(f)
            if (name, s) in done:
                continue
            rec = json.load(open(f))
            body = {"model": R.MODEL, "messages": rec["messages"], "max_tokens": a.max_tokens,
                    "stream": True, "stream_options": {"include_usage": True},
                    "temperature": a.temperature, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                    "presence_penalty": 1.0}
            if rec.get("tools"):
                body["tools"] = rec["tools"]
            headers = hook(i, name, body) if hook else {}
            res = R.stream_chat(body, headers)
            res.update({"arm": a.arm, "request": name, "sample": s, "max_tokens": a.max_tokens,
                        "order": a.order, "temperature": a.temperature})
            fo.write(json.dumps(res) + "\n"); fo.flush()
            n += 1
            if res.get("error"):
                print(f"[{a.arm} dist] error {name} s={s}: {res['error'][:200]}", flush=True)
                sys.exit(2)
            if n % 40 == 0:
                print(f"[{a.arm} dist] {n}/{total} requests; {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
