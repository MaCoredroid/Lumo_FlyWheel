#!/usr/bin/env python3
"""Identical-token replay for SGLang: send the vLLM-rendered prompt token IDs to SGLang /generate
(streaming) with the deployed sampling settings, no per-request seed, max_new_tokens 1024.
Records the same timing fields as replay.py so analyze.py/final_stats.py apply unchanged.
Usage: replay_ids.py --arm NAME --ids corpus/vllm_prompt_ids.json --out run.jsonl [--max-tokens 1024]"""
import argparse, http.client, json, os, time


def run_one(ids, max_new):
    body = {"input_ids": ids, "stream": True,
            "sampling_params": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                                "presence_penalty": 1.0, "max_new_tokens": max_new}}
    c = http.client.HTTPConnection("127.0.0.1", 9950, timeout=3600)
    t0 = time.perf_counter()
    c.request("POST", "/generate", body=json.dumps(body), headers={"Content-Type": "application/json"})
    r = c.getresponse()
    if r.status != 200:
        return {"error": f"HTTP {r.status}: {r.read()[:300].decode(errors='replace')}"}
    t_first = None; last = None; prev_ct = 0
    while True:
        line = r.readline()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if payload == b"[DONE]":
            break
        ev = json.loads(payload); last = ev
        ct = (ev.get("meta_info") or {}).get("completion_tokens") or 0
        if ct > prev_ct and t_first is None:
            t_first = time.perf_counter()
        prev_ct = ct
    t_end = time.perf_counter()
    mi = (last or {}).get("meta_info") or {}
    return {"t_ttft_s": (t_first - t0) if t_first else None, "t_decode_s": (t_end - t_first) if t_first else None,
            "t_e2e_s": t_end - t0, "completion_tokens": mi.get("completion_tokens"), "prompt_tokens": mi.get("prompt_tokens"),
            "cached_tokens": mi.get("cached_tokens"), "finish_reason": str((mi.get("finish_reason") or {}).get("type") if isinstance(mi.get("finish_reason"), dict) else mi.get("finish_reason")),
            "content": (last or {}).get("text", ""), "reasoning": "", "tool_calls": "", "spec_delta": {}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True); ap.add_argument("--ids", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=1024); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    corpus = json.load(open(a.ids)); names = sorted(corpus)
    if a.limit:
        names = names[: a.limit]
    with open(a.out, "a") as fo:
        for i, name in enumerate(names):
            res = run_one(corpus[name]["ids"], a.max_tokens)
            res.update({"arm": a.arm, "mode": "sampled-ids", "request": name, "index": i, "max_tokens": a.max_tokens,
                        "prompt_ids_sha256": corpus[name]["sha256"], "wall_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            fo.write(json.dumps(res) + "\n"); fo.flush()
            ct = res.get("completion_tokens"); dr = (ct - 1) / res["t_decode_s"] if ct and res.get("t_decode_s") else None
            print(f"[{a.arm} ids] {i+1}/{len(names)} {name} tok={ct} prompt={res.get('prompt_tokens')} decode={dr and round(dr,2)} tok/s {res.get('error','')}", flush=True)


if __name__ == "__main__":
    main()
