#!/usr/bin/env python3
"""FR13 node-5 per-layer ladder driver (same-boot).

PHASE LIVE: drive the 4 pinned SWE prompts at the canonical rider battery
(/v1/completions RAW text, max_tokens=128, temp 0.0, top_p 1.0, seed 1313,
return_token_ids, fr10_decode_mode=tree_mtp). This reproduces the gold-gate
served stream incl. the p3 step-103 deep-accept event. The FR10_LAYER_HIDDEN
capture (num_tokens=10, ROWS=6 = node-5 self row, SKIP window) + the
FR13_FINAL_LOGIT capture record the live tree-verify node-5-row hidden at every
layer + projected final logits.

PHASE CLEAN: tokenize prompt[3], build ctx = pids + served[3][:73] (the byte
identical accepted prefix [0,1,3,5] spine tokens), send max_tokens=1
teacher-force. The FR10_ROOT_HIDDEN capture (num_tokens=ctx_len, ROOT_ROW=last)
+ FR10_ROOT_LOGIT capture record the clean single-forward last-row hidden at
every layer + projected logits. This is the RIGHT reference per
FR13_SERVING_PATH_SCAN_BIND.

Two reps of the clean request to assert within-boot determinism.
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path


def post(ep, path, payload, timeout=600):
    req = urllib.request.Request(
        ep.rstrip("/") + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        b = r.read().decode()
    return json.loads(b) if b else None


def get_text(ep, path, timeout=10):
    with urllib.request.urlopen(ep.rstrip("/") + path, timeout=timeout) as r:
        return r.read().decode()


def wait_health(ep, timeout_s):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            get_text(ep, "/health", timeout=5)
            return
        except Exception:
            time.sleep(3)
    raise RuntimeError("health timeout")


def tokenize(ep, model, prompt):
    d = post(ep, "/tokenize", {"model": model, "prompt": prompt})
    return [int(t) for t in d["tokens"]]


def one_completion(ep, model, prompt, max_tokens, seed, top_k=20):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": seed,
        "logprobs": top_k,
        "return_token_ids": True,
        "vllm_xargs": {"fr10_decode_mode": "tree_mtp"},
    }
    d = post(ep, "/v1/completions", payload)
    ch = d["choices"][0]
    lp = ch.get("logprobs") or {}
    return {
        "served_token_ids": ch.get("token_ids") or [],
        "served_tokens": lp.get("tokens") or [],
        "finish_reason": ch.get("finish_reason"),
        "text": ch.get("text"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:9950")
    ap.add_argument("--model", default="qwen3.6-27b")
    ap.add_argument("--prompts", default="output/fr13_acceptance_ladder/prompts_swe4.json")
    ap.add_argument("--tree-capture", default="output/fr13_commit_argmax/tree_capture.json")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1313)
    ap.add_argument("--cut", type=int, default=73)
    ap.add_argument("--out", default="output/fr13_node5_ladder/drive_result.json")
    args = ap.parse_args()

    ep = args.endpoint
    wait_health(ep, 600)
    try:
        post(ep, "/reset_prefix_cache", {}, 30)
    except Exception:
        pass

    prompts = json.loads(Path(args.prompts).read_text())
    tc = json.load(open(args.tree_capture))
    served3 = tc["records"][3]["served_token_ids"]

    result = {"ts": time.time(), "endpoint": ep, "model": args.model}

    # ---- PHASE LIVE: 4 SWE prompts, single rep (reproduces step 103) ----
    # IMPORTANT: exactly one pass of the 4 prompts so the live tree-verify-forward
    # sequence (and the FR13_CAG step counter / FR10 layer-capture SEEN counter)
    # match the banked channel_split_summary.json (step 103). No re-runs here, or
    # the SKIP window would mis-align.
    live = []
    served_all = []
    for pid, prompt in enumerate(prompts):
        rec = one_completion(ep, args.model, prompt, args.max_tokens, args.seed)
        served_all.append(rec["served_token_ids"])
        live.append({
            "pid": pid,
            "served_len": len(rec["served_token_ids"]),
            "served_first8": rec["served_token_ids"][:8],
        })
        print(f"[live] pid={pid} served_len={len(rec['served_token_ids'])}", flush=True)
    result["live"] = live

    # same-boot reproduction check: live served[3][69:74] vs banked accepted prefix + flip
    result["live3_served_69_74"] = served_all[3][69:74]
    print(f"[live] pid=3 served[69:74]={served_all[3][69:74]} "
          f"(banked [3269,198,71093,271,9764])", flush=True)

    # ---- PHASE CLEAN: teacher-force the accepted prefix ----
    pids3 = tokenize(ep, args.model, prompts[3])
    ctx = pids3 + served3[: args.cut]
    print(f"[clean] prompt3_tok_len={len(pids3)} cut={args.cut} ctx_len={len(ctx)} "
          f"(banked context_len=1687)", flush=True)
    result["clean_ctx_len"] = len(ctx)
    result["clean_prompt3_tok_len"] = len(pids3)
    result["clean_served3_70_74"] = served3[args.cut - 4 : args.cut + 1]

    clean_reps = []
    for rep in range(2):
        try:
            post(ep, "/reset_prefix_cache", {}, 30)
        except Exception:
            pass
        payload = {
            "model": args.model,
            "prompt": ctx,
            "max_tokens": 1,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": args.seed,
            "logprobs": 20,
            "return_token_ids": True,
            "vllm_xargs": {"fr10_decode_mode": "tree_mtp"},
        }
        d = post(ep, "/v1/completions", payload)
        ch = d["choices"][0]
        lp = ch.get("logprobs") or {}
        emitted = (ch.get("token_ids") or [None])[0]
        clean_reps.append({
            "rep": rep,
            "emitted_id": emitted,
            "emitted_tok": (lp.get("tokens") or [None])[0],
            "emitted_lp": (lp.get("token_logprobs") or [None])[0],
            "top": (lp.get("top_logprobs") or [{}])[0],
        })
        print(f"[clean] rep{rep} emitted={emitted} tok={clean_reps[-1]['emitted_tok']!r} "
              f"lp={clean_reps[-1]['emitted_lp']} (banked clean argmax=71093 '```')", flush=True)
    result["clean_reps"] = clean_reps
    result["clean_within_boot_det"] = (
        clean_reps[0]["emitted_id"] == clean_reps[1]["emitted_id"]
    )

    try:
        result["metrics_excerpt"] = get_text(ep, "/metrics", timeout=10)[:4000]
    except Exception:
        pass

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print("wrote", args.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
