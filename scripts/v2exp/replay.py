#!/usr/bin/env python3
"""Replay recorded agent chat requests against one serving arm and record decode timing.

Same inputs for every arm: messages + tools from recorded agent requests, fixed max_tokens.
Per request we record client-side TTFT, decode time, completion tokens, the full output
(for cross-arm greedy comparison) and the delta of the engine's spec-decode counters.

Usage:
  replay.py --arm ar --requests DIR --out run.jsonl [--mode greedy|sampled] [--max-tokens 1024]
            [--auth-hook module:function]   (optional per-request header provider, used for
                                             servers that require signed requests)
Stdlib only.
"""
import argparse, glob, http.client, importlib, json, os, re, sys, time

HOST, PORT = "127.0.0.1", 9950
MODEL = "qwen3.8-27b-nvfp4-radixark"
SPEC_RE = re.compile(r'^(vllm:spec_decode_[a-z_]+?)(?:_total)?(\{[^}]*\})?\s+([0-9.eE+-]+)$')


def metrics():
    c = http.client.HTTPConnection(HOST, PORT, timeout=30)
    c.request("GET", "/metrics")
    txt = c.getresponse().read().decode()
    out = {}
    for line in txt.splitlines():
        if line.startswith("#"):
            continue
        m = SPEC_RE.match(line)
        if m:
            key = m.group(1) + (m.group(2) or "")
            out[key] = out.get(key, 0.0) + float(m.group(3))
    return out


def stream_chat(body, headers):
    c = http.client.HTTPConnection(HOST, PORT, timeout=3600)
    data = json.dumps(body).encode()
    h = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    h.update(headers or {})
    t0 = time.perf_counter()
    c.request("POST", "/v1/chat/completions", body=data, headers=h)
    r = c.getresponse()
    if r.status != 200:
        return {"error": f"HTTP {r.status}: {r.read()[:500].decode(errors='replace')}"}
    t_first = None
    content, reasoning, tool_args = [], [], []
    usage, finish = None, None
    n_chunks = 0
    buf = b""
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
        ev = json.loads(payload)
        if ev.get("usage"):
            usage = ev["usage"]
        for ch in ev.get("choices", []):
            d = ch.get("delta", {}) or {}
            got = False
            if d.get("content"):
                content.append(d["content"]); got = True
            rc = d.get("reasoning_content") or d.get("reasoning")
            if rc:
                reasoning.append(rc); got = True
            for tc in d.get("tool_calls") or []:
                fn = tc.get("function") or {}
                tool_args.append((fn.get("name") or "") + (fn.get("arguments") or "")); got = True
            if got:
                n_chunks += 1
                if t_first is None:
                    t_first = time.perf_counter()
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
    t_end = time.perf_counter()
    return {
        "t_ttft_s": (t_first - t0) if t_first else None,
        "t_decode_s": (t_end - t_first) if t_first else None,
        "t_e2e_s": t_end - t0,
        "completion_tokens": (usage or {}).get("completion_tokens"),
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "cached_tokens": ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
        "finish_reason": finish,
        "n_chunks": n_chunks,
        "reasoning": "".join(reasoning),
        "content": "".join(content),
        "tool_calls": "".join(tool_args),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--requests", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["greedy", "sampled"], default="greedy")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--auth-hook", default="")
    a = ap.parse_args()

    hook = None
    if a.auth_hook:
        mod, fn = a.auth_hook.split(":")
        hook = getattr(importlib.import_module(mod), fn)

    files = sorted(glob.glob(os.path.join(a.requests, "*.json")))
    if a.limit:
        files = files[: a.limit]
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                done.add(json.loads(line)["request"])
            except Exception:
                pass
    with open(a.out, "a") as fo:
        for i, f in enumerate(files):
            name = os.path.basename(f)
            if name in done:
                continue
            rec = json.load(open(f))
            body = {
                "model": MODEL,
                "messages": rec["messages"],
                "tools": rec.get("tools") or None,
                "max_tokens": a.max_tokens,
                "stream": True,
                "stream_options": {"include_usage": True},
                "ignore_eos": False,
            }
            if body["tools"] is None:
                del body["tools"]
            if a.mode == "greedy":
                body.update({"temperature": 0.0, "top_p": 1.0, "top_k": -1, "seed": 0})
            else:
                # Deployed agent-proxy sampling; no per-request seed (the proxy sends none,
                # and the fixed32 route forbids per-request generators).
                body.update({"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                             "presence_penalty": 1.0})
            headers = hook(i, name, body) if hook else {}
            m0 = metrics()
            res = stream_chat(body, headers)
            m1 = metrics()
            res["spec_delta"] = {k: m1.get(k, 0.0) - m0.get(k, 0.0) for k in set(m0) | set(m1)
                                 if m1.get(k, 0.0) != m0.get(k, 0.0)}
            res.update({"arm": a.arm, "mode": a.mode, "request": name, "index": i,
                        "max_tokens": a.max_tokens, "wall_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            fo.write(json.dumps(res) + "\n"); fo.flush()
            ct = res.get("completion_tokens")
            dr = (ct - 1) / res["t_decode_s"] if ct and res.get("t_decode_s") else None
            print(f"[{a.arm} {a.mode}] {i+1}/{len(files)} {name} tok={ct} decode={dr and round(dr,2)} tok/s "
                  f"ttft={res.get('t_ttft_s') and round(res['t_ttft_s'],1)}s {res.get('error','')}", flush=True)


if __name__ == "__main__":
    main()
