#!/usr/bin/env python3
"""Controlled decode probe against a running vLLM server (port 9950).

Implements the GOOD speed metric (FR13_CAT6_CAT8_ACCEPT_INVESTIGATION.md):
  decode_tps_wall = committed_tokens / (t_last_token - t_first_token)   [B=1, from first gen token]
= REAL wall-clock single-stream decode TPS. Nets the accept gain against tree overhead (forward + drafter +
committer + gaps), excludes prefill (measured from first token) and agent/network (direct fixed prompt).
NOT derived_tps (which is a forward-only upper bound).

Also measures accept/forward from the /metrics spec_decode counter delta (the diagnostic/lever), and supports
GREEDY (temp 0) mode for the superset-bound test (cat8 must accept >= cat6 on the same deterministic output).

Usage:
  fr13_accept_speed_probe.py --mode greedy  --prompt-file P --n 512 --out out.json
  fr13_accept_speed_probe.py --mode temp06  --prompt-file P --n 512 --out out.json
  fr13_accept_speed_probe.py --selftest
"""
import argparse, json, sys, time, urllib.request

BASE = "http://127.0.0.1:9950"


def _get_metrics():
    """Return {name: value} for the spec_decode counters we need."""
    txt = urllib.request.urlopen(BASE + "/metrics", timeout=10).read().decode()
    want = ("vllm:spec_decode_num_accepted_tokens_total",
            "vllm:spec_decode_num_drafts_total",
            "vllm:generation_tokens_total")
    out = {}
    for line in txt.splitlines():
        if line.startswith("#"):
            continue
        for w in want:
            if line.startswith(w):
                try:
                    out[w] = out.get(w, 0.0) + float(line.rsplit(" ", 1)[1])
                except Exception:
                    pass
    return out


def _accept_per_forward(m0, m1):
    da = m1.get("vllm:spec_decode_num_accepted_tokens_total", 0) - m0.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    dd = m1.get("vllm:spec_decode_num_drafts_total", 0) - m0.get("vllm:spec_decode_num_drafts_total", 0)
    return (da / dd) if dd > 0 else None, da, dd


def probe(mode, prompt, n_tokens, messages=None):
    """Run ONE completion; return metrics dict. greedy=temp0 (bound test); temp06/temp10=seed0.
    messages != None => real chat prompt via /v1/chat/completions (faithful ship-config content)."""
    temperature = {"greedy": 0.0, "temp06": 0.6, "temp10": 1.0}[mode]
    m0 = _get_metrics()
    if messages is not None:
        endpoint = "/v1/chat/completions"
        body = {"model": "qwen3.6-27b", "messages": messages, "max_tokens": n_tokens,
                "temperature": temperature, "seed": 0, "stream": True, "ignore_eos": True,
                "stream_options": {"include_usage": True}}
    else:
        endpoint = "/v1/completions"
        body = {"model": "qwen3.6-27b", "prompt": prompt, "max_tokens": n_tokens,
                "temperature": temperature, "seed": 0, "stream": True, "ignore_eos": True,
                "stream_options": {"include_usage": True}}
    req = urllib.request.Request(BASE + endpoint,
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t_first = t_last = None
    n_tok = 0
    usage = None
    t_send = time.monotonic()
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                ev = json.loads(payload)
            except Exception:
                continue
            if ev.get("usage"):
                usage = ev["usage"]
            ch = (ev.get("choices") or [{}])[0]
            piece = ch.get("text") or (ch.get("delta") or {}).get("content")  # completions | chat
            if piece:
                now = time.monotonic()
                if t_first is None:
                    t_first = now
                t_last = now
                n_tok += 1
    m1 = _get_metrics()
    apf, acc, drafts = _accept_per_forward(m0, m1)
    committed = usage.get("completion_tokens") if usage else n_tok
    decode_wall = (t_last - t_first) if (t_first and t_last and t_last > t_first) else None
    ttft = (t_first - t_send) if t_first else None
    # decode_tps_wall = COMMITTED tokens / decode-wall (NOT SSE-chunk count: spec-decode packs ~accept+1
    # committed tokens per chunk, so chunk-count massively undercounts). Window [t_first, t_last] excludes
    # TTFT/prefill. Slight <1% overcount (first chunk's tokens land at t_first) — negligible at n=512.
    decode_tps_wall = (committed / decode_wall) if (decode_wall and committed) else None
    return {
        "mode": mode, "temperature": temperature,
        "committed_tokens": committed, "streamed_tokens": n_tok,
        "accept_per_forward": apf, "accepted_delta": acc, "drafts_delta": drafts,
        "decode_wall_s": decode_wall, "ttft_s": ttft,
        "decode_tps_wall": decode_tps_wall,
    }


def selftest():
    # accept/forward math
    m0 = {"vllm:spec_decode_num_accepted_tokens_total": 100, "vllm:spec_decode_num_drafts_total": 40}
    m1 = {"vllm:spec_decode_num_accepted_tokens_total": 100 + 331, "vllm:spec_decode_num_drafts_total": 40 + 100}
    apf, acc, dr = _accept_per_forward(m0, m1)
    assert abs(apf - 3.31) < 1e-9 and acc == 331 and dr == 100, (apf, acc, dr)
    # decode_tps_wall = committed / (t_last - t_first): 512 committed over 25.6s => 20.0 tok/s
    # (committed, NOT SSE-chunk count — spec-decode packs ~accept+1 tokens per chunk)
    committed, first, last = 512, 10.0, 35.6
    tps = committed / (last - first)
    assert tps == 20.0, tps
    print("selftest OK: accept/fwd=3.31; decode_tps_wall=committed/(t_last-t_first)=20.0 tok/s")


def main():
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["greedy", "temp06", "temp10"])
    ap.add_argument("--prompt-file")
    ap.add_argument("--chat-messages", help="JSON file with {messages:[...]} => real chat prompt (ship-config)")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--out")
    ap.add_argument("--base-url", default=BASE)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    BASE = a.base_url
    messages = None
    if a.chat_messages:
        d = json.load(open(a.chat_messages))
        messages = d.get("messages", d) if isinstance(d, dict) else d
    prompt = open(a.prompt_file).read() if a.prompt_file else "def fibonacci(n):\n    "
    r = probe(a.mode, prompt, a.n, messages=messages)
    print(json.dumps(r, indent=2))
    if a.out:
        json.dump(r, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
