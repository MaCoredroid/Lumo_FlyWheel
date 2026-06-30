#!/usr/bin/env python3
"""
fr13_apc_replay.py — Deterministic multi-turn replay harness for diagnosing a
vLLM Automatic-Prefix-Caching (APC) recurrent-state poison on Qwen3-Next GDN.

WHAT THIS DOES
--------------
The codex SWE agent loop is non-deterministic (~17 min, tool exec, model
sampling) which makes it useless for A/B testing an APC fix. This harness
*replays* the EXACT multi-turn Responses-API conversation that the agent
already sent for one SWE task, captured in proxy_pair_dumps/. Each turn N+1's
banked `input` is a (near-)superset of turn N's, so POSTing the turns in order
against a live vLLM at temperature 0 reproduces the growing-prompt cache-hit
geometry that poisons the GDN recurrent state on cache-ON — deterministically,
in ~8 min, with NO tool execution (the banked input already contains the real
tool outputs).

DUMP STRUCTURE (schema lumo.fr13.proxy_pair_dump.v1)
----------------------------------------------------
Each pair_<idnum>_<seq>_<initial|auto_continue>.json has top-level keys:
  schema, ts_ns, seq, kind, request, response
  request:  model, instructions, input[], tools[], tool_choice,
            parallel_tool_calls, reasoning, store, stream, include,
            prompt_cache_key, client_metadata, temperature, max_output_tokens
  response: a full Responses object (id, output[], status, usage, ...)

GROUPING + ORDERING (validated offline against the real dumps)
--------------------------------------------------------------
  * GROUP key  = request['prompt_cache_key']  (a per-conversation UUID; the
                 codex agent reuses it for every turn of ONE task's session).
                 Each prompt_cache_key maps to exactly ONE SWE task id (0
                 collisions over the full 3756-file corpus).
  * ORDER key  = the <idnum> in the filename (pair_<idnum>_...), which is a
                 global, strictly-monotonic capture counter. The <seq> field
                 RESETS per proxy batch so it is NOT a reliable global order;
                 idnum is. Within one prompt_cache_key group, idnum order ==
                 the natural turn order and the input[] array grows.
  * The input[] array grows monotonically per turn; consecutive turns share a
    92-99% literal char-prefix (the only intra-prefix diffs are opaque
    `reasoning.id` strings the proxy regenerated). That shared prefix IS the
    APC cache-hit geometry.

USAGE (replay against a live server)
------------------------------------
  python3 scripts/fr13_apc_replay.py \
      --task astropy__astropy-12907 \
      --dumps output/fr13_bigdenom_swe/cat9_apc_fix/proxy_pair_dumps \
      --base http://127.0.0.1:9950 --endpoint /v1/responses \
      --temperature 0 --model qwen3.6-27b \
      --out output/fr13_apc_replay_12907.jsonl

OFFLINE RECONSTRUCTION CHECK (no server needed)
-----------------------------------------------
  python3 scripts/fr13_apc_replay.py --task astropy__astropy-12907 \
      --dumps <dir> --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict

FILE_RE = re.compile(r"pair_(\d+)_(\d+)_(initial|auto_continue)\.json$")
TASK_RE = re.compile(r"SWE-Bench task:\s*([A-Za-z0-9_.\-]+)")


# --------------------------------------------------------------------------- #
# Reconstruction
# --------------------------------------------------------------------------- #
def iter_text(inp):
    """Yield every text string reachable inside a Responses `input` array."""
    if not isinstance(inp, list):
        return
    for m in inp:
        c = m.get("content")
        if isinstance(c, str):
            yield c
        elif isinstance(c, list):
            for p in c:
                if isinstance(p, dict) and isinstance(p.get("text"), str):
                    yield p["text"]
        o = m.get("output")
        if isinstance(o, str):
            yield o
        elif isinstance(o, list):
            for p in o:
                if isinstance(p, dict) and isinstance(p.get("text"), str):
                    yield p["text"]


def task_id_of(req):
    for t in iter_text(req.get("input")):
        mm = TASK_RE.search(t)
        if mm:
            return mm.group(1)
    return None


def load_dumps(dumps_dir):
    """Return list of dicts: {idnum, seq, kind, path, schema_doc}."""
    out = []
    for name in os.listdir(dumps_dir):
        m = FILE_RE.match(name)
        if not m:
            continue
        out.append(
            {
                "idnum": int(m.group(1)),
                "seq": int(m.group(2)),
                "kind": m.group(3),
                "path": os.path.join(dumps_dir, name),
                "name": name,
            }
        )
    out.sort(key=lambda r: r["idnum"])
    return out


def group_by_cache_key(records):
    """Read each dump, group by prompt_cache_key, attach task id + input."""
    groups = defaultdict(list)
    for r in records:
        with open(r["path"]) as fh:
            doc = json.load(fh)
        req = doc.get("request", {})
        r = dict(r)
        r["doc"] = doc
        r["cache_key"] = req.get("prompt_cache_key")
        r["task"] = task_id_of(req)
        r["n_msgs"] = len(req.get("input") or [])
        groups[r["cache_key"]].append(r)
    for k in groups:
        groups[k].sort(key=lambda x: x["idnum"])
    return groups


def select_conversation(groups, task, cache_key=None):
    """Pick the conversation for a task. Default = the LONGEST one (most cache
    accumulation -> best chance of exposing the poison). Override via cache_key.
    """
    if cache_key:
        if cache_key not in groups:
            raise SystemExit(f"cache_key {cache_key} not found in dumps")
        return cache_key, groups[cache_key]
    candidates = [
        (k, g)
        for k, g in groups.items()
        if any(r["task"] == task for r in g)
    ]
    if not candidates:
        raise SystemExit(f"no conversation found for task {task}")
    # longest by turn count; tiebreak by earliest idnum for determinism
    candidates.sort(key=lambda kg: (len(kg[1]), -kg[1][0]["idnum"]), reverse=True)
    return candidates[0]


# --------------------------------------------------------------------------- #
# Garbage detection
# --------------------------------------------------------------------------- #
GARBAGE_TOKENS = ("</parameter>", "output_text", "<|endoftext|>", "0000")


def detect_garbage(text):
    """Return (is_garbage, reason) for a model output string."""
    if text is None:
        return True, "null_output"
    s = text.strip()
    if s == "":
        return True, "empty_output"
    # repeated marker spam (degenerate token salad)
    for tok in GARBAGE_TOKENS:
        if s.count(tok) >= 5:
            return True, f"repeat_marker:{tok}x{s.count(tok)}"
    # long run of a single repeated char (e.g. 0000000..., aaaaa...)
    run = re.search(r"(.)\1{40,}", s)
    if run:
        return True, f"char_run:{run.group(1)!r}x{len(run.group(0))}"
    # short repeating n-gram looped many times (token-salad loops).
    # The {1,40} non-greedy capture covers 1..40-char repeating units.
    m = re.search(r"(.{1,40}?)\1{8,}", s)
    if m and len(m.group(0)) > 60:
        return True, f"ngram_loop:{m.group(1)[:20]!r}"
    # very low unique-token ratio over a long-ish output = degenerate
    words = s.split()
    if len(words) >= 30:
        uniq = len(set(words))
        if uniq / len(words) < 0.15:
            return True, f"low_unique_ratio:{uniq}/{len(words)}"
    return False, "ok"


# --------------------------------------------------------------------------- #
# Response text extraction (Responses API + Completions fallback)
# --------------------------------------------------------------------------- #
def responses_output_text(resp):
    """Concatenate assistant message text from a Responses object."""
    if not isinstance(resp, dict):
        return ""
    if isinstance(resp.get("output_text"), str):
        return resp["output_text"]
    parts = []
    for item in resp.get("output", []) or []:
        if item.get("type") == "message":
            for p in item.get("content", []) or []:
                if isinstance(p, dict) and isinstance(p.get("text"), str):
                    parts.append(p["text"])
        elif item.get("type") == "function_call":
            # surface tool-call intent so it isn't misread as empty/garbage
            parts.append(f"[function_call {item.get('name')} {item.get('arguments','')}]")
    return "".join(parts)


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def http_post_json(url, payload, timeout):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode())


def endpoint_available(base, endpoint, model, timeout=10):
    """Probe whether /v1/responses accepts a tiny request. Returns bool."""
    url = base.rstrip("/") + endpoint
    probe = {
        "model": model,
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "ping"}]}
        ],
        "temperature": 0,
        "max_output_tokens": 1,
        "stream": False,
        "store": False,
    }
    try:
        status, _ = http_post_json(url, probe, timeout)
        return status == 200
    except urllib.error.HTTPError as e:
        # 400 = endpoint exists but rejected our probe; 404 = not present
        return e.code != 404
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Request builders
# --------------------------------------------------------------------------- #
def build_responses_payload(req, model, temperature, max_tokens, seed,
                            drop_cache_key=False):
    """Build a /v1/responses payload from a banked request, preserving
    instructions + input + tools + message structure verbatim."""
    payload = {
        "model": model,
        "instructions": req.get("instructions"),
        "input": req.get("input"),
        "tools": req.get("tools"),
        "tool_choice": req.get("tool_choice", "auto"),
        "parallel_tool_calls": req.get("parallel_tool_calls", False),
        "temperature": temperature,
        "max_output_tokens": max_tokens,
        "stream": False,
        "store": False,
    }
    # carry reasoning config if present (codex used effort=high)
    if req.get("reasoning") is not None:
        payload["reasoning"] = req["reasoning"]
    # keep the same prompt_cache_key so the server's APC keys identically
    if not drop_cache_key and req.get("prompt_cache_key") is not None:
        payload["prompt_cache_key"] = req["prompt_cache_key"]
    if seed is not None:
        payload["seed"] = seed
    return {k: v for k, v in payload.items() if v is not None}


def flatten_to_prompt(req):
    """Flatten instructions + input into a single text prompt for the
    /v1/completions fallback. Loses tool-call structure but still drives the
    same prefill / recurrent-state path with the same growing prefix."""
    chunks = []
    ins = req.get("instructions")
    if ins:
        chunks.append(f"<|system|>\n{ins}")
    for m in req.get("input") or []:
        role = m.get("role") or m.get("type") or "item"
        texts = []
        c = m.get("content")
        if isinstance(c, str):
            texts.append(c)
        elif isinstance(c, list):
            for p in c:
                if isinstance(p, dict) and isinstance(p.get("text"), str):
                    texts.append(p["text"])
        o = m.get("output")
        if isinstance(o, str):
            texts.append(o)
        elif isinstance(o, list):
            for p in o:
                if isinstance(p, dict) and isinstance(p.get("text"), str):
                    texts.append(p["text"])
        if m.get("type") == "function_call":
            texts.append(f"[call {m.get('name')} {m.get('arguments','')}]")
        if texts:
            chunks.append(f"<|{role}|>\n" + "\n".join(texts))
    chunks.append("<|assistant|>\n")
    return "\n\n".join(chunks)


def build_completions_payload(req, model, temperature, max_tokens, seed):
    payload = {
        "model": model,
        "prompt": flatten_to_prompt(req),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if seed is not None:
        payload["seed"] = seed
    return payload


def completions_output_text(resp):
    try:
        return resp["choices"][0].get("text", "")
    except Exception:
        return ""


# --------------------------------------------------------------------------- #
# Token-count estimate (chars/4 — no tokenizer dependency)
# --------------------------------------------------------------------------- #
def est_input_tokens(req):
    n = 0
    for t in iter_text(req.get("input")):
        n += len(t)
    ins = req.get("instructions") or ""
    return (n + len(ins)) // 4


def first_cat_separable_turn(turns):
    """Index of the first turn whose input contains the separable.py cat blob."""
    for i, r in enumerate(turns):
        joined = "\n".join(iter_text(r["doc"]["request"].get("input")))
        if "def _cstack" in joined or (
            "separable.py" in joined and "def separability_matrix" in joined
        ):
            return i
    return None


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="astropy__astropy-12907")
    ap.add_argument("--dumps", required=True,
                    help="proxy_pair_dumps directory")
    ap.add_argument("--cache-key", default=None,
                    help="override: replay this exact prompt_cache_key group")
    ap.add_argument("--base", default="http://127.0.0.1:9950")
    ap.add_argument("--endpoint", default="/v1/responses",
                    help="/v1/responses (preferred) or /v1/completions")
    ap.add_argument("--model", default="qwen3.6-27b")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens-cap", type=int, default=512,
                    help="max_tokens = min(request.max_output_tokens, cap)")
    ap.add_argument("--seed", type=int, default=0,
                    help="pinned seed (set <0 to omit)")
    ap.add_argument("--drop-cache-key", action="store_true",
                    help="do NOT send the banked prompt_cache_key (lets the "
                         "server key APC purely on prefix tokens)")
    ap.add_argument("--out", default=None, help="output jsonl path")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--dry-run", action="store_true",
                    help="reconstruct + report only; POST nothing")
    args = ap.parse_args()

    seed = None if args.seed < 0 else args.seed

    print(f"[reconstruct] scanning {args.dumps}", file=sys.stderr)
    records = load_dumps(args.dumps)
    print(f"[reconstruct] {len(records)} pair files", file=sys.stderr)
    groups = group_by_cache_key(records)
    print(f"[reconstruct] {len(groups)} distinct prompt_cache_key groups",
          file=sys.stderr)

    cache_key, turns = select_conversation(groups, args.task, args.cache_key)
    task = next((r["task"] for r in turns if r["task"]), args.task)

    # ---- reconstruction report ----
    print("=" * 70)
    print(f"TASK            : {task}")
    print(f"prompt_cache_key: {cache_key}")
    print(f"turns           : {len(turns)}")
    cat_turn = first_cat_separable_turn(turns)
    print(f"first cat-separable.py blob at turn: {cat_turn}")
    print("-" * 70)
    print(f"{'turn':>4} {'kind':14} {'seq':>4} {'n_msgs':>6} "
          f"{'~in_tok':>8} {'growth':>7} {'cat':>4}  {'file'}")
    prev = 0
    for i, r in enumerate(turns):
        req = r["doc"]["request"]
        tok = est_input_tokens(req)
        joined = "\n".join(iter_text(req.get("input")))
        has_cat = "def _cstack" in joined
        growth = tok - prev
        print(f"{i:>4} {r['kind']:14} {r['seq']:>4} {r['n_msgs']:>6} "
              f"{tok:>8} {growth:>+7} {'YES' if has_cat else '   ':>4}  {r['name']}")
        prev = tok
    print("=" * 70)

    if args.dry_run:
        print("[dry-run] reconstruction only; no POST performed.")
        return

    # ---- live replay ----
    base = args.base.rstrip("/")
    endpoint = args.endpoint
    use_completions = endpoint.endswith("/completions")

    if not use_completions:
        if not endpoint_available(base, endpoint, args.model):
            print(f"[fallback] {endpoint} not available -> /v1/completions "
                  f"(flattened prompt; loses tool structure)", file=sys.stderr)
            endpoint = "/v1/completions"
            use_completions = True
    url = base + endpoint
    print(f"[replay] POST -> {url}  temperature={args.temperature} "
          f"seed={seed}", file=sys.stderr)

    out_path = args.out or f"fr13_apc_replay_{task}.jsonl"
    records_out = []
    first_garbage = None

    for i, r in enumerate(turns):
        req = r["doc"]["request"]
        cap = min(req.get("max_output_tokens") or args.max_tokens_cap,
                  args.max_tokens_cap)
        n_in_tok = est_input_tokens(req)
        try:
            if use_completions:
                payload = build_completions_payload(
                    req, args.model, args.temperature, cap, seed)
                status, resp = http_post_json(url, payload, args.timeout)
                text = completions_output_text(resp)
            else:
                payload = build_responses_payload(
                    req, args.model, args.temperature, cap, seed,
                    drop_cache_key=args.drop_cache_key)
                status, resp = http_post_json(url, payload, args.timeout)
                text = responses_output_text(resp)
            err = None
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:300]
            status, text, err = e.code, "", f"HTTPError {e.code}: {body}"
        except Exception as e:
            status, text, err = 0, "", f"{type(e).__name__}: {e}"

        is_garbage, reason = detect_garbage(text) if err is None else (True, err)
        if is_garbage and first_garbage is None:
            first_garbage = i

        rec = {
            "turn": i,
            "seq": r["seq"],
            "kind": r["kind"],
            "cache_key": cache_key,
            "http_status": status,
            "n_input_tokens": n_in_tok,
            "n_input_msgs": r["n_msgs"],
            "output_text_head": (text or "")[:200],
            "output_len": len(text or ""),
            "garbage": is_garbage,
            "garbage_reason": reason,
        }
        records_out.append(rec)
        flag = "GARBAGE" if is_garbage else "ok"
        print(f"[turn {i:>3}] msgs={r['n_msgs']:>3} ~in_tok={n_in_tok:>6} "
              f"http={status} {flag:8} {reason}", file=sys.stderr)

    with open(out_path, "w") as fh:
        for rec in records_out:
            fh.write(json.dumps(rec) + "\n")

    # ---- summary ----
    print("=" * 70)
    print(f"REPLAY SUMMARY  task={task}  cache_key={cache_key}")
    print(f"turns replayed  : {len(records_out)}")
    flags = [("G" if r["garbage"] else ".") for r in records_out]
    print(f"per-turn flags  : {''.join(flags)}  (. = coherent, G = garbage)")
    if first_garbage is None:
        print("FIRST GARBAGE   : all coherent")
    else:
        print(f"FIRST GARBAGE   : turn {first_garbage}  "
              f"(reason: {records_out[first_garbage]['garbage_reason']})")
    print(f"jsonl written   : {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
