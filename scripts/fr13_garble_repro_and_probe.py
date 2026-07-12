"""Reproduce a garble on the SHIP server, then teacher-force the continuation to read the garbled token's
CLEAN probability — the direct empirical test of 'is the garbled token near-impossible (drift insufficient)
or a small-gap flip (wrong-accept)'.

Single boot. Consistent tokenization: /tokenize(messages) -> prompt_ids; /v1/completions(prompt=ids,
return_token_ids) for both free-run generation AND per-position teacher-force. The teacher-forced max_tokens=1
distribution is the CLEAN single-token target (dodges the spec-decode prompt_logprobs contamination). At each
generated position we compare the tree's committed token to the clean argmax; committed!=argmax = a wrong
accept, and the committed token's rank/logprob in the clean top-k says near-impossible vs small-gap.

Run against a healthy ship endpoint (port 9955). CPU-side (HTTP only) — uses the server's single GPU job.
"""
import json, sys, urllib.request
sys.path.insert(0, "scripts")
from fr13_garble_gate import PROMPTS, undefined_names


def post(ep, path, payload, timeout=300):
    r = urllib.request.Request(ep.rstrip("/") + path, data=json.dumps(payload).encode(),
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=timeout).read())


def tokenize_messages(ep, model, content):
    # templated prompt ids (chat template + generation prompt), thinking OFF like the gate
    for body in (
        {"model": model, "messages": [{"role": "user", "content": content}],
         "add_generation_prompt": True, "chat_template_kwargs": {"enable_thinking": False}},
        {"model": model, "messages": [{"role": "user", "content": content}], "add_generation_prompt": True},
    ):
        try:
            d = post(ep, "/tokenize", body, 60)
            toks = d.get("tokens", d.get("token_ids"))
            if toks:
                return [int(t) for t in toks]
        except Exception as e:
            print("tokenize(messages) attempt failed:", str(e)[:120], flush=True)
    raise RuntimeError("tokenize messages failed")


def gen(ep, model, prompt_ids, seed):
    d = post(ep, "/v1/completions", {"model": model, "prompt": prompt_ids, "max_tokens": 700,
                                     "temperature": 0.6, "seed": seed, "return_token_ids": True,
                                     "logprobs": 1})
    ch = d["choices"][0]
    text = ch.get("text", "")
    tids = ch.get("token_ids") or (ch.get("logprobs", {}) or {}).get("token_ids")
    return text, ([int(t) for t in tids] if tids else None)


def tf_argmax(ep, model, ctx_ids, k=20):
    d = post(ep, "/v1/completions", {"model": model, "prompt": ctx_ids, "max_tokens": 1,
                                     "temperature": 0.0, "logprobs": k, "return_token_ids": True})
    ch = d["choices"][0]
    lp = ch.get("logprobs", {}) or {}
    top = (lp.get("top_logprobs") or [{}])[0]
    emitted_id = None
    tids = ch.get("token_ids") or lp.get("token_ids")
    if tids:
        emitted_id = int(tids[0])
    ranked = sorted(top.items(), key=lambda kv: -kv[1])  # [(tokstr, logprob)]
    return emitted_id, ranked


def detok(ep, model, tid):
    try:
        return post(ep, "/detokenize", {"model": model, "tokens": [tid]}, 30).get("prompt", f"<{tid}>")
    except Exception:
        return f"<{tid}>"


def main():
    ep = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:9955/v1"
    model = post(ep.replace("/v1", ""), "/v1/models" if False else "/v1/models", None) if False else None
    # get model id
    md = json.loads(urllib.request.urlopen(ep.rstrip("/") + "/models", timeout=30).read())
    model = md["data"][0]["id"]
    print("model:", model, flush=True)

    # 1. reproduce a garble
    garbled = None
    for (name, content) in PROMPTS:
        pids = tokenize_messages(ep, model, content)
        for seed in range(8):
            text, gids = gen(ep, model, pids, seed)
            loads, nundef, undef = undefined_names(text)
            print(f"  [{name} seed{seed}] len(text)={len(text)} undefined={nundef} {undef[:4]}", flush=True)
            if nundef > 0 and gids:
                garbled = {"name": name, "prompt_ids": pids, "gen_ids": gids, "text": text, "undef": undef}
                break
        if garbled:
            break
    if not garbled:
        print("NO GARBLE reproduced in this sample set (try more seeds/prompts).", flush=True)
        return

    print(f"\n=== GARBLED sample: {garbled['name']} undefined={garbled['undef'][:6]} ===", flush=True)
    pids, gids = garbled["prompt_ids"], garbled["gen_ids"]

    # 2. teacher-force each generated position; clean argmax vs committed token
    flips = []
    for i in range(len(gids)):
        ctx = pids + gids[:i]
        committed = gids[i]
        try:
            _, ranked = tf_argmax(ep, model, ctx)
        except Exception as e:
            continue
        committed_str = detok(ep, model, committed)
        argmax_str = ranked[0][0] if ranked else None
        argmax_lp = ranked[0][1] if ranked else None
        # committed token's rank/logprob in the clean top-k (by string match)
        crank = next((r for r, (t, _) in enumerate(ranked) if t == committed_str), None)
        clp = next((lp for (t, lp) in ranked if t == committed_str), None)
        if argmax_str is not None and committed_str != argmax_str:
            flips.append({"pos": i, "committed": committed_str, "committed_rank": crank,
                          "committed_lp": clp, "clean_argmax": argmax_str, "clean_argmax_lp": argmax_lp})

    print(f"\n=== {len(flips)} wrong-accept positions (tree committed != clean argmax) ===", flush=True)
    for f in flips:
        gap = (f["clean_argmax_lp"] - f["committed_lp"]) if (f["committed_lp"] is not None) else None
        print(f"  pos {f['pos']:4d}: committed={f['committed']!r:16} rank={f['committed_rank']} "
              f"lp={f['committed_lp']} | clean_argmax={f['clean_argmax']!r:16} lp={f['clean_argmax_lp']:.2f} "
              f"| gap(nats)={'NA' if gap is None else round(gap,2)}", flush=True)

    # 3. verdict
    ident_flips = [f for f in flips if any(c.isalpha() or c == '_' for c in (f["committed"] or ""))]
    outrank = [f for f in ident_flips if f["committed_rank"] is None]  # committed not even in clean top-20
    small = [f for f in ident_flips if f["committed_rank"] is not None and f["committed_rank"] <= 3]
    print(f"\n=== VERDICT ===", flush=True)
    print(f"  identifier-like wrong-accepts: {len(ident_flips)}; "
          f"committed OUT of clean top-20 (near-impossible => drift insufficient): {len(outrank)}; "
          f"committed in top-3 (small-gap flip => drift plausibly sufficient): {len(small)}", flush=True)
    print("  If most garble tokens are OUT of top-20 under clean teacher-force, the ~9e-4-scale drift is",
          flush=True)
    print("  NOT sufficient (something bigger / spec-accept-logic is implicated). If they're rank 2-3 with",
          flush=True)
    print("  a small gap, the drift plausibly flips them -> whole-GDN-native fix should kill garble.", flush=True)
    out = "/tmp/garble_probe_result.json"
    json.dump({"garbled": garbled["name"], "undef": garbled["undef"], "flips": flips}, open(out, "w"))
    print(f"  wrote {out}", flush=True)


if __name__ == "__main__":
    main()
