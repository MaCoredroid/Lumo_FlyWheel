#!/usr/bin/env python3
# FR13 APC SHADOW GATE — replay ONE real captured /v1/responses turn through the
# SAME boot two ways and compare the generated output:
#   cache-OFF = reset_prefix_cache THEN send  (fresh prefill = the incumbent / ground truth)
#   cache-ON  = send AGAIN          (warm hit on the prefix = the arm under test)
# Only the cache differs (single variable). First divergence in the generated stream
# = where the cache breaks the output. Confounds controlled:
#   - VACUOUS guard: the cache-ON pass MUST report cached_tokens>0 (cache actually engaged),
#     and the cache-OFF pass MUST report cached_tokens==0 (truly fresh). Else abort.
#   - AUTOTUNE noise: cache-OFF is run TWICE; if off1 != off2 the position is non-reproducible
#     (BATCH_INVARIANT=0) and any ON-vs-OFF diff there is noise, not a cache defect.
#   - temp=0 greedy (deterministic sampling); the captured request already uses temperature 0.
# Reference = cache-OFF on THIS boot (the real incumbent path), never a proxy.
import json, sys, time, argparse, urllib.request, urllib.error, difflib

def post(port, path, payload, timeout=1800):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def reset_cache(port):
    try:
        post(port, "/reset_prefix_cache", {}, timeout=60)
        return True
    except Exception as e:
        print(f"  [warn] reset_prefix_cache failed: {e}", file=sys.stderr)
        return False

def canon_output(resp):
    """Flatten response.output into a canonical, comparable token-ish string + the
    function_call arguments (where the malformed-JSON / char-8 defect lives)."""
    parts, fcalls = [], []
    for it in resp.get("output", []) or []:
        t = it.get("type")
        if t == "reasoning":
            for c in (it.get("content") or []):
                parts.append("R:" + (c.get("text") or ""))
            if it.get("summary"):
                parts.append("RS:" + json.dumps(it["summary"]))
        elif t == "message":
            for c in (it.get("content") or []):
                parts.append("M:" + (c.get("text") or ""))
        elif t in ("function_call", "tool_call"):
            args = it.get("arguments") or ""
            parts.append(f"F:{it.get('name')}({args})")
            fcalls.append({"name": it.get("name"), "arguments": args})
        else:
            parts.append(f"{t}:" + json.dumps(it)[:200])
    return "\n".join(parts), fcalls

def cached_tokens(resp):
    return ((resp.get("usage") or {}).get("input_tokens_details") or {}).get("cached_tokens", None)

def input_tokens(resp):
    return (resp.get("usage") or {}).get("input_tokens", None)

def first_divergence(a, b):
    """char offset of first difference + surrounding context."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    if i == n and len(a) == len(b):
        return None
    lo = max(0, i - 60)
    return {"offset": i, "off_ctx": repr(a[lo:i+60]), "on_ctx": repr(b[lo:i+60]),
            "len_off": len(a), "len_on": len(b)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--request-json", required=True)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--max-output-tokens", type=int, default=2048)
    ap.add_argument("--mode", choices=["shadow", "determinism"], default="shadow",
                    help="shadow=cache-ON-vs-OFF on one boot; determinism=send N times, NO reset, "
                         "measure cross-request stability (boot cache-OFF to isolate the cache as cause)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    req = json.load(open(a.request_json))
    # sanitize for a deterministic, bounded, non-streaming replay
    req["temperature"] = 0.0
    req["top_p"] = 1.0
    req["store"] = False
    req["stream"] = False
    req["max_output_tokens"] = a.max_output_tokens
    req.pop("previous_response_id", None)

    print(f"[shadow] request input≈{req.get('max_output_tokens')} cap; replaying through port {a.port}", flush=True)

    def send(tag):
        t0 = time.time()
        r = post(a.port, "/v1/responses", req)
        dt = time.time() - t0
        txt, fc = canon_output(r)
        ct, it = cached_tokens(r), input_tokens(r)
        print(f"  [{tag}] input_tokens={it} cached_tokens={ct} out_len={len(txt)} fcalls={len(fc)} {dt:.1f}s", flush=True)
        return {"text": txt, "fcalls": fc, "cached_tokens": ct, "input_tokens": it}

    if a.mode == "determinism":
        # Send the SAME request a.reps times with NO reset between (boot cache-OFF to test
        # whether the cuda-graph/spec-decode serve is reproducible absent the cache machinery).
        seq = []
        for k in range(1, a.reps + 1):
            r = send(f"DET {k}")
            seq.append({"out_len": len(r["text"]), "text": r["text"], "cached": r["cached_tokens"]})
        texts = [s["text"] for s in seq]
        all_ident = all(t == texts[0] for t in texts)
        degrading = [s["out_len"] for s in seq]
        verdict = {
            "mode": "determinism",
            "out_lens": degrading,
            "all_identical": all_ident,
            "DETERMINISM_VERDICT": ("DETERMINISTIC (all reps byte-identical)" if all_ident
                                    else f"NON-DETERMINISTIC / degrading: out_lens={degrading}"),
        }
        json.dump({"verdict": verdict, "out_heads": [t[:300] for t in texts]}, open(a.out, "w"), indent=1)
        print("\n=== DETERMINISM VERDICT ===")
        for kk, vv in verdict.items():
            print(f"  {kk}: {vv}")
        print(f"saved -> {a.out}")
        return

    # WARMUP: settle autotune/flashinfer kernel selection so the comparison passes are
    # reproducible. Without this, two identical greedy requests diverge (BATCH_INVARIANT=0
    # autotune-state nondeterminism) and the cache test is confounded.
    print("[shadow] warmup (settle autotune kernels)...", flush=True)
    for w in range(2):
        reset_cache(a.port); time.sleep(1); send(f"WARMUP{w+1}")

    runs = {}
    # round-robin: each rep does (reset->OFF, then ON-hit), to keep the comparison paired
    for rep in range(1, a.reps + 1):
        reset_cache(a.port); time.sleep(1)
        runs[f"off{rep}"] = send(f"OFF rep{rep} (post-reset, fresh)")
        runs[f"on{rep}"] = send(f"ON  rep{rep} (warm hit)")

    off1 = runs["off1"]; on1 = runs["on1"]
    # ---- guards ----
    vacuous = (on1["cached_tokens"] in (None, 0)) or (off1["cached_tokens"] not in (0, None) and off1["cached_tokens"] > 0)
    det_off = all(runs[f"off{r}"]["text"] == off1["text"] for r in range(2, a.reps + 1)) if a.reps > 1 else None
    det_on = all(runs[f"on{r}"]["text"] == on1["text"] for r in range(2, a.reps + 1)) if a.reps > 1 else None

    div = first_divergence(off1["text"], on1["text"])
    # function-call level: did cache-ON malform the args (char-8) vs cache-OFF?
    fc_off = off1["fcalls"][0]["arguments"] if off1["fcalls"] else None
    fc_on = on1["fcalls"][0]["arguments"] if on1["fcalls"] else None
    fc_div = (fc_off != fc_on)

    verdict = {
        "vacuous": bool(vacuous),
        "off_cached_tokens": off1["cached_tokens"], "on_cached_tokens": on1["cached_tokens"],
        "deterministic_off": det_off, "deterministic_on": det_on,
        "outputs_identical_on_vs_off": (off1["text"] == on1["text"]),
        "first_divergence": div,
        "function_call_off_valid_json": _valid_json(fc_off),
        "function_call_on_valid_json": _valid_json(fc_on),
        "function_call_diverged": bool(fc_div),
    }
    if vacuous:
        verdict["LOSSLESS_VERDICT"] = "VACUOUS (cache not engaged — on_cached=%s off_cached=%s)" % (on1["cached_tokens"], off1["cached_tokens"])
    elif det_off is False:
        verdict["LOSSLESS_VERDICT"] = "INCONCLUSIVE (cache-OFF non-deterministic across reps = autotune noise)"
    elif off1["text"] == on1["text"]:
        verdict["LOSSLESS_VERDICT"] = "LOSSLESS (cache-ON output byte-identical to cache-OFF)"
    else:
        verdict["LOSSLESS_VERDICT"] = "NOT-LOSSLESS (cache-ON diverges from cache-OFF at the localized site)"

    json.dump({"verdict": verdict, "runs": {k: {kk: vv for kk, vv in v.items() if kk != "text"} for k, v in runs.items()},
               "off_text_head": off1["text"][:400], "on_text_head": on1["text"][:400],
               "fc_off": fc_off, "fc_on": fc_on}, open(a.out, "w"), indent=1)
    print("\n=== SHADOW VERDICT ===")
    for k, v in verdict.items():
        print(f"  {k}: {v}")
    print(f"saved -> {a.out}")

def _valid_json(s):
    if s is None:
        return None
    try:
        json.loads(s); return True
    except Exception:
        return False

if __name__ == "__main__":
    main()
