#!/usr/bin/env python3
"""FR13 TV GATE — the agent-free, nudge-free, non-teacher-forced lossless gate that every prior conclusion
skipped. On ONE cache-ON server it measures whether the EXACT_SEED RESTORE reproduces the cold FLA prefill,
by comparing the temp-0.6 first-new-token DISTRIBUTION of a cache-MISS vs a cache-HIT on the SAME prompt:

  miss  = reset_prefix_cache, then POST prompt (cold FLA prefill) -> first-token top_logprobs = q_fresh
  hit   = POST the SAME prompt again (APC HIT -> EXACT_SEED restore)     -> first-token top_logprobs = q_restore
  TV    = 0.5 * sum_t |q_fresh(t) - q_restore(t)|  over the union of top-K tokens

No teacher-forcing (only the prompt, one position). No agent, no nudge. Temp 0.6 (distribution, not sample).
The ~20k-token prompts mean the first-new-token already reflects the FULL restored prefix state, so a non-bit-exact
restore shows up as TV>0. Run on native (control, expect TV~0), cat8 (tree), chain5 (spine).
PASS(lossless) = TV within fp round-off (~<1e-3) on all prompts. FAIL(lossy) = TV elevated.
GPU work is the server's; this is a thin client. Requires the server booted with FR13_ENABLE_APC=1 EXACT_SEED=1
(and FR13_SERVE_LOG=1 so ES_GATE/ES_WRITE/ES_SEED_APPLIED are bankable for the non-vacuity guard).
"""
import argparse, glob, json, math, os, subprocess, sys, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fr13_carrier_locator_probe import load_prompt, post  # reuse plumbing


def _reset_cache(base):
    try:
        subprocess.run(["curl", "-s", "-m", "10", "-X", "POST", base + "/reset_prefix_cache"],
                       capture_output=True, timeout=15)
    except Exception:
        pass


def _first_token_dist(resp):
    """Extract the first generated token's {token_logprob} dict from a chat/completions logprobs response."""
    ch = (resp.get("choices") or [{}])[0]
    lp = ch.get("logprobs") or {}
    content = lp.get("content") or []
    if not content:
        return None
    top = (content[0] or {}).get("top_logprobs") or []
    return {e.get("token"): float(e.get("logprob")) for e in top if e.get("token") is not None}


def _tv(a, b):
    """Total variation between two logprob dicts, over the union of tokens (missing => prob 0)."""
    if not a or not b:
        return None
    toks = set(a) | set(b)
    pa = {t: math.exp(a[t]) for t in a}
    pb = {t: math.exp(b[t]) for t in b}
    return 0.5 * sum(abs(pa.get(t, 0.0) - pb.get(t, 0.0)) for t in toks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:9950/v1/chat/completions")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", nargs="+", required=True)
    ap.add_argument("--prompt-limit", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-logprobs", type=int, default=20)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base = a.endpoint.rsplit("/v1/", 1)[0]

    files = []
    for p in a.prompts:
        files += sorted(glob.glob(os.path.join(p, "chatreq_*.json"))) if os.path.isdir(p) else [p]
    files = files[: a.prompt_limit] if a.prompt_limit else files

    rows = []
    for pf in files:
        pr = load_prompt(pf)
        if not pr:
            continue
        body = {"model": a.model, "messages": pr["messages"], "temperature": a.temperature,
                "max_tokens": 1, "stream": False, "logprobs": True, "top_logprobs": a.top_logprobs}
        if pr["tools"]:
            body["tools"] = pr["tools"]
        try:
            _reset_cache(base)                     # force a cold MISS
            q_fresh = _first_token_dist(post(a.endpoint, body, timeout=300))
            q_restore = _first_token_dist(post(a.endpoint, body, timeout=300))  # 2nd = HIT -> ES restore
        except Exception as e:
            rows.append({"prompt": os.path.basename(pf), "tv": None, "err": str(e)[:120]})
            continue
        tv = _tv(q_fresh, q_restore)
        rows.append({"prompt": os.path.basename(pf), "tv": tv,
                     "argmax_fresh": max(q_fresh, key=q_fresh.get) if q_fresh else None,
                     "argmax_restore": max(q_restore, key=q_restore.get) if q_restore else None})

    tvs = [r["tv"] for r in rows if r.get("tv") is not None]
    argmax_flips = sum(1 for r in rows if r.get("argmax_fresh") is not None
                       and r.get("argmax_fresh") != r.get("argmax_restore"))
    summary = {
        "cell": a.cell, "n_prompts": len(files), "n_ok": len(tvs),
        "tv_max": max(tvs) if tvs else None,
        "tv_mean": (sum(tvs) / len(tvs)) if tvs else None,
        "argmax_flips": argmax_flips,
        "verdict": ("LOSSLESS(restore==fresh)" if tvs and max(tvs) < 1e-3
                    else "LOSSY(restore!=fresh)" if tvs else "NO_DATA"),
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"summary": summary, "rows": rows}, open(a.out, "w"), indent=1)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
