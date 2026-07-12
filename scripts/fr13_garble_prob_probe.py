"""Track the per-token PROBABILITY at a reproduced garble, tree-arm vs no-spec-arm — the DIRECT
sufficiency test for 'does the ~9e-4 scan drift flip the accept'.

Reuses the proven teacher-force primitive (fr13_gold_margin_probe.teacher_force / fr13_oracle_capture._tf_one):
force context = prompt_ids + garbled_continuation[:i], decode max_tokens=1 with top-k logprobs. This yields
the CLEAN single-token target distribution at each position (avoids the documented spec-decode
prompt_logprobs contamination: tree attaches unreliable logprobs to accepted DRAFT tokens).

Run TWICE against the SAME garbled continuation:
  --arm tree     : endpoint of a booted cat8 tree server (ship config)
  --arm nospec   : endpoint of a booted no-spec server (num_speculative_tokens=0), same weights/seed
Teacher-forced argmax is temperature-INDEPENDENT (fixed inputs), so cross-boot tree-vs-nospec is valid
(same rationale as fr13_apc_teacher_forced_logit_gate).

At each position report: argmax token + its logprob, and the FORCED (actually-emitted-by-tree) token's
rank + logprob. Offline diff of the two arms' per-position records answers, at the garbled position:
  - is the garbled token's NO-SPEC logprob near-impossible (big gap => 9e-4 drift insufficient => look
    elsewhere) or a small-gap near-neighbor (drift plausibly sufficient),
  - and did the TREE arm's argmax/logprob INFLATE the garbled token vs no-spec (the drift's fingerprint).

Usage:
  # 1. reproduce garble: run the garble gate, grab a sample whose text has an undefined/misspelled name.
  #    Save its prompt + generated continuation to a JSON: {"prompt": "...", "continuation": "...garbled..."}
  # 2. python fr13_garble_prob_probe.py --sample garble_sample.json --endpoint http://127.0.0.1:PORT \
  #        --model qwen3.6-27b --arm tree  --out probe_tree.jsonl
  # 3. reboot no-spec, then --arm nospec --out probe_nospec.jsonl
  # 4. python fr13_garble_prob_probe.py --diff probe_tree.jsonl probe_nospec.jsonl
"""
import argparse, json, urllib.request


def _post(endpoint, path, payload, timeout=120):
    r = urllib.request.Request(endpoint.rstrip("/") + path, data=json.dumps(payload).encode(),
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=timeout).read())


def _tokenize(endpoint, model, text, timeout=60):
    d = _post(endpoint, "/tokenize", {"model": model, "prompt": text}, timeout)
    return [int(t) for t in d.get("tokens", d.get("token_ids", []))]


def probe(args):
    s = json.load(open(args.sample))
    prompt_ids = _tokenize(args.endpoint, args.model, s["prompt"])
    cont_ids = _tokenize(args.endpoint, args.model, s["continuation"])
    recs = []
    for i in range(len(cont_ids)):
        ctx = prompt_ids + cont_ids[:i]
        forced = cont_ids[i]
        try:
            j = _post(args.endpoint, "/v1/completions", {
                "model": args.model, "prompt": ctx, "max_tokens": 1, "temperature": 0.0,
                "logprobs": args.top_k, "seed": 1313, "return_token_ids": True,
            }, timeout=args.timeout)
            ch = j["choices"][0]
            lp = ch.get("logprobs", {}) or {}
            top = (lp.get("top_logprobs") or [{}])[0]  # dict token->logprob at position 0
            # argmax = highest-logprob entry
            am = max(top.items(), key=lambda kv: kv[1]) if top else (None, None)
            # forced token's logprob/rank within top-k (best-effort by token string)
            forced_tok = _detok(args.endpoint, args.model, forced)
            forced_lp = top.get(forced_tok)
            ranked = sorted(top.items(), key=lambda kv: -kv[1])
            forced_rank = next((r for r, (t, _) in enumerate(ranked) if t == forced_tok), None)
            recs.append({"pos": i, "forced_id": forced, "forced_tok": forced_tok,
                         "argmax_tok": am[0], "argmax_lp": am[1],
                         "forced_lp": forced_lp, "forced_rank": forced_rank,
                         "topk": ranked[:8]})
        except Exception as e:
            recs.append({"pos": i, "error": str(e)[:120]})
    with open(args.out, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    flips = [r for r in recs if r.get("argmax_tok") is not None and r.get("forced_tok") != r.get("argmax_tok")]
    print(f"[{args.arm}] wrote {len(recs)} positions -> {args.out}; "
          f"{len(flips)} positions where forced(tree-emitted) != this-arm argmax", flush=True)


def _detok(endpoint, model, tok_id):
    try:
        d = _post(endpoint, "/detokenize", {"model": model, "tokens": [tok_id]}, 30)
        return d.get("prompt", d.get("text", ""))
    except Exception:
        return f"<id{tok_id}>"


def diff(a_path, b_path):
    A = [json.loads(l) for l in open(a_path)]
    B = [json.loads(l) for l in open(b_path)]
    print(f"=== DIFF tree({a_path}) vs nospec({b_path}) ===", flush=True)
    print("pos | forced_tok | tree: argmax(lp) forced_lp/rank | nospec: argmax(lp) forced_lp/rank", flush=True)
    for a, b in zip(A, B):
        if "error" in a or "error" in b:
            continue
        flip = a.get("argmax_tok") != b.get("argmax_tok")
        # the garble signature: tree commits forced token (its own emission) but nospec's argmax is DIFFERENT
        # (the right token), and nospec ranks the forced token LOW (near-impossible) or CLOSE (small gap).
        mark = " <== ARGMAX DIFFERS" if flip else ""
        print(f"{a['pos']:3d} | {str(a.get('forced_tok'))[:14]:14s} | "
              f"T:{str(a.get('argmax_tok'))[:8]:8s}({_f(a.get('argmax_lp'))}) f={_f(a.get('forced_lp'))}/r{a.get('forced_rank')} | "
              f"N:{str(b.get('argmax_tok'))[:8]:8s}({_f(b.get('argmax_lp'))}) f={_f(b.get('forced_lp'))}/r{b.get('forced_rank')}{mark}",
              flush=True)


def _f(x):
    return f"{x:.2f}" if isinstance(x, (int, float)) else "NA"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample"); ap.add_argument("--endpoint"); ap.add_argument("--model", default="qwen3.6-27b")
    ap.add_argument("--arm", choices=["tree", "nospec"]); ap.add_argument("--out")
    ap.add_argument("--top_k", type=int, default=20); ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--diff", nargs=2)
    a = ap.parse_args()
    if a.diff:
        diff(a.diff[0], a.diff[1])
    else:
        probe(a)
