#!/usr/bin/env python3
"""FR13 MULTI-TURN TV GATE — the decisive gate the single-prompt TV probe MISSED.

The single-prompt fr13_tv_gate_probe.py restores a PREFILL checkpoint (POST prompt=miss, POST
prompt=hit). That is the BENIGN part (native has MORE of it, TV 0.881, yet engages). The give-up
carrier is the MULTI-TURN DECODE-DRAIN restore: on a real 2nd turn the cache restores turn-1's
generated (tree-decode) state[leaf], NOT a prefill checkpoint. This gate exercises THAT:

  1. reset cache; POST prompt, GENERATE G tokens at temp 0.6 (tree decode) -> caches prompt-prefill
     + the G-token DECODE-DRAIN (state[leaf], the tree-kernel realization).
  2. extended = prompt + generated-G.
  3. q_hit  = POST extended, max_tokens=1  -> HIT: restores the decode-drain state[leaf].
  4. q_cold = reset; POST extended, max_tokens=1 -> cold prefill of the SAME tokens (chunked-FLA).
  5. floor  = reset; POST extended, max_tokens=1 (a 2nd cold) vs q_cold (within-boot determinism).
  TV(q_cold, q_hit) = decode-drain-restore(state[leaf]) vs chunked-FLA-cold. LOSSY if >> floor.

Run on TREE (cat8) and NATIVE. Prediction if the tree-kernel decode-drain is the carrier:
  native TV ~ floor (native decode-drain IS chunked-FLA); tree TV >> floor (state[leaf] != chunked-FLA).
Temp-0.6 distributional (valid per the lossless-gate-miss rule); floor 0 = within-boot determinism.
Client-side only; the server's GPU does the work.
"""
import argparse, glob, json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fr13_carrier_locator_probe import load_prompt, post
from fr13_tv_gate_probe import _reset_cache, _first_token_dist, _tv


def _gen_continuation(endpoint, model, messages, tools, temperature, gen_tokens):
    """POST the prompt and GENERATE gen_tokens (real tree-decode) -> caches the decode-drain.
    Returns the assistant continuation text (to append as the extended context)."""
    body = {"model": model, "messages": messages, "temperature": temperature,
            "max_tokens": gen_tokens, "stream": False}
    if tools:
        body["tools"] = tools
    resp = post(endpoint, body, timeout=600)
    ch = (resp.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    return msg.get("content") or ""


def _first_dist(endpoint, model, messages, tools, temperature, top_logprobs):
    body = {"model": model, "messages": messages, "temperature": temperature,
            "max_tokens": 1, "stream": False, "logprobs": True, "top_logprobs": top_logprobs}
    if tools:
        body["tools"] = tools
    return _first_token_dist(post(endpoint, body, timeout=300))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:9950/v1/chat/completions")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", nargs="+", required=True)
    ap.add_argument("--prompt-limit", type=int, default=2)
    ap.add_argument("--gen-tokens", type=int, default=256,
                    help="tokens to generate first so a DECODE-DRAIN state[leaf] is cached")
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
        msgs, tools = pr["messages"], pr["tools"]
        try:
            # 1) generate G tokens -> cache the DECODE-DRAIN state[leaf] (tree kernel)
            _reset_cache(base)
            cont = _gen_continuation(a.endpoint, a.model, msgs, tools, a.temperature, a.gen_tokens)
            if not cont.strip():
                rows.append({"prompt": os.path.basename(pf), "tv": None, "err": "empty continuation"})
                continue
            # 2) extended context = prompt + generated continuation
            ext = list(msgs) + [{"role": "assistant", "content": cont}]
            # 3) HIT: restore the decode-drain state[leaf] (prefix prompt+cont is warm)
            q_hit = _first_dist(a.endpoint, a.model, ext, tools, a.temperature, a.top_logprobs)
            # 4) COLD: reset -> cold prefill (chunked-FLA) of the SAME extended tokens
            _reset_cache(base)
            q_cold = _first_dist(a.endpoint, a.model, ext, tools, a.temperature, a.top_logprobs)
            # 5) FLOOR: a 2nd cold (within-boot determinism)
            _reset_cache(base)
            q_cold2 = _first_dist(a.endpoint, a.model, ext, tools, a.temperature, a.top_logprobs)
        except Exception as e:
            rows.append({"prompt": os.path.basename(pf), "tv": None, "err": str(e)[:160]})
            continue
        tv = _tv(q_cold, q_hit)       # decode-drain restore (state[leaf]) vs chunked-FLA cold
        floor = _tv(q_cold, q_cold2)  # cold-vs-cold determinism floor
        rows.append({"prompt": os.path.basename(pf), "tv": tv, "floor": floor,
                     "gen_tokens": a.gen_tokens, "cont_chars": len(cont),
                     "argmax_cold": max(q_cold, key=q_cold.get) if q_cold else None,
                     "argmax_hit": max(q_hit, key=q_hit.get) if q_hit else None})

    tvs = [r["tv"] for r in rows if r.get("tv") is not None]
    floors = [r["floor"] for r in rows if r.get("floor") is not None]
    flips = sum(1 for r in rows if r.get("argmax_cold") is not None
                and r.get("argmax_cold") != r.get("argmax_hit"))
    tv_max = max(tvs) if tvs else None
    floor_max = max(floors) if floors else None
    lossy = bool(tvs and floors and tv_max > max(floor_max * 3, floor_max + 1e-3, 5e-3))
    summary = {
        "cell": a.cell, "n_prompts": len(files), "n_ok": len(tvs), "gen_tokens": a.gen_tokens,
        "tv_max": tv_max, "tv_mean": (sum(tvs) / len(tvs)) if tvs else None,
        "floor_max": floor_max, "floor_mean": (sum(floors) / len(floors)) if floors else None,
        "argmax_flips": flips,
        "verdict": ("NO_DATA" if not tvs else
                    "LOSSY(decode-drain restore != cold, ABOVE floor)" if lossy else
                    "LOSSLESS(decode-drain restore == cold, within floor)"),
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"summary": summary, "rows": rows}, open(a.out, "w"), indent=1)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
