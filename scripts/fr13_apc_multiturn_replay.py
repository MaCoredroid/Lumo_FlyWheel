#!/usr/bin/env python3
# FR13 APC MULTI-TURN ORACLE GATE — replay arm.
#
# Replays a FIXED reference trajectory (a directory of captured /v1/responses
# request dumps from a *resolved* rollout, in seq order) through the CURRENT boot
# at temperature=0 (greedy == argmax), capturing each turn's generated output.
#
# Run this script once per boot (oracle / cache-OFF / cache-ON). The reducer then
# compares, PER TURN, each arm's output against the oracle (ground truth) and finds
# the first argmax divergence — tracking whether cache-ON diverges EARLIER than the
# cache-OFF floor and whether that gap GROWS with turn index (the compounding signal).
#
# Why this is a true argmax measurement (not free-running drift): at temperature=0
# every boot decodes greedily, so feeding the IDENTICAL turn input to two boots and
# taking the first position where their outputs differ is exactly the first argmax
# flip caused by the only thing that differs between them — the serve config / cache.
#
# Cache faithfulness: the cache-ON boot is replayed IN SEQUENCE with NO reset, so
# each turn's shared conversation prefix is a genuine APC restore from the prior
# turns' blocks (the deployed multi-turn condition). cached_tokens is logged per turn
# to PROVE the cache engaged on the ON arm and stayed 0 on oracle/OFF.
import argparse, json, sys, time, urllib.request
from pathlib import Path


def post(port, path, payload, timeout=2400):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def reset_cache(port):
    try:
        post(port, "/reset_prefix_cache", {}, timeout=60)
    except Exception as e:
        print(f"  [warn] reset_prefix_cache failed: {e}", file=sys.stderr)


def canon_output(resp):
    """Flatten response.output into a canonical, comparable token-ish string. Identical
    to the single-turn shadow gate's canon so the two instruments agree. The first char
    offset where two arms' canon strings differ is the localized argmax divergence."""
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


def usage_field(resp, *path):
    cur = resp.get("usage") or {}
    for p in path:
        cur = (cur or {}).get(p)
        if cur is None:
            return None
    return cur


def load_trajectory(dumps_dir, limit_turns):
    """Load captured turns sorted by seq. Accepts pair dumps (have .request/.response)
    or request dumps (the bare Responses payload). Returns [(seq, request_payload)]."""
    turns = []
    for f in sorted(Path(dumps_dir).glob("*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if isinstance(d, dict) and "request" in d and isinstance(d["request"], dict):
            req, seq = d["request"], d.get("seq")
        elif isinstance(d, dict) and ("input" in d or "instructions" in d):
            req, seq = d, None
        else:
            continue
        # only real model turns (must have an input conversation)
        if not (req.get("input") or req.get("instructions")):
            continue
        turns.append((seq if seq is not None else len(turns), f.name, req))
    turns.sort(key=lambda x: (x[0] if isinstance(x[0], int) else 1 << 60, x[1]))
    if limit_turns:
        turns = turns[:limit_turns]
    return turns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--dumps-dir", required=True, help="dir of captured /v1/responses (pair or request) dumps")
    ap.add_argument("--arm", required=True, help="oracle | off | on (label only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-output-tokens", type=int, default=384)
    ap.add_argument("--limit-turns", type=int, default=0)
    ap.add_argument("--reset-each", action="store_true",
                    help="reset prefix cache before each turn (NEVER for the ON arm — it destroys the multi-turn cache state)")
    ap.add_argument("--per-turn-timeout", type=float, default=2400)
    a = ap.parse_args()

    turns = load_trajectory(a.dumps_dir, a.limit_turns)
    print(f"[replay arm={a.arm}] {len(turns)} turns from {a.dumps_dir}  cap={a.max_output_tokens}  reset_each={a.reset_each}", flush=True)
    if not turns:
        print("FAIL: no turns loaded", file=sys.stderr); return 2

    # fresh cache at the very start so turn 0 is a clean miss on every arm
    reset_cache(a.port); time.sleep(1)

    records = []
    for idx, (seq, fname, req) in enumerate(turns):
        if a.reset_each:
            reset_cache(a.port); time.sleep(0.5)
        r = dict(req)
        r["temperature"] = 0.0
        r["top_p"] = 1.0
        r["store"] = False
        r["stream"] = False
        r["max_output_tokens"] = a.max_output_tokens
        r.pop("previous_response_id", None)
        t0 = time.time()
        try:
            resp = post(a.port, "/v1/responses", r, timeout=a.per_turn_timeout)
        except Exception as e:
            print(f"  [turn {idx} seq={seq}] ERROR {e}", file=sys.stderr)
            records.append({"idx": idx, "seq": seq, "fname": fname, "error": str(e)})
            json.dump({"arm": a.arm, "records": records}, open(a.out, "w"))
            continue
        dt = time.time() - t0
        canon, fcalls = canon_output(resp)
        rec = {
            "idx": idx, "seq": seq, "fname": fname,
            "input_tokens": usage_field(resp, "input_tokens"),
            "cached_tokens": usage_field(resp, "input_tokens_details", "cached_tokens"),
            "output_tokens": usage_field(resp, "output_tokens"),
            "canon": canon, "canon_len": len(canon),
            "n_fcalls": len(fcalls),
            "first_fcall": fcalls[0] if fcalls else None,
            "status": resp.get("status"),
            "secs": round(dt, 1),
        }
        records.append(rec)
        print(f"  [turn {idx:>3} seq={seq}] in={rec['input_tokens']} cached={rec['cached_tokens']} "
              f"out_tok={rec['output_tokens']} canon_len={rec['canon_len']} fcalls={rec['n_fcalls']} {dt:.1f}s", flush=True)
        json.dump({"arm": a.arm, "max_output_tokens": a.max_output_tokens, "records": records}, open(a.out, "w"))

    eng = sum(1 for r in records if (r.get("cached_tokens") or 0) > 0)
    print(f"[replay arm={a.arm}] DONE {len(records)} turns, {eng} with cached_tokens>0 -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
