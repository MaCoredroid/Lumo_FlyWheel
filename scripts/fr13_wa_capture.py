#!/usr/bin/env python3
"""Single-seed WRONG-ACCEPT pin: capture, for each committed near-neighbor (undefined-name) token,
the TREE's OWN logprob p_tree for that token. High p_tree on a wrong token => tree-verify thought it
likely (drift-inflated prob, accept correctly honored it). Low p_tree => correct-but-unlucky temp-0.6
tail (no bug). No rate stats — per-token deterministic readout on garbling samples."""
import sys, json, ast, urllib.request, concurrent.futures
sys.path.insert(0, "scripts")
from fr13_garble_gate import PROMPTS, extract_code, undefined_names

ENDPOINT = "http://127.0.0.1:9950/v1/chat/completions"

def gen(prompt, seed):
    body = json.dumps({"model": "qwen3.6-27b", "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0.6, "max_tokens": 700, "seed": seed,
                       "logprobs": True, "top_logprobs": 1,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    r = urllib.request.Request(ENDPOINT, data=body, headers={"Content-Type": "application/json"})
    j = json.loads(urllib.request.urlopen(r, timeout=300).read())
    ch = j["choices"][0]
    toks = [(t["token"], t["logprob"]) for t in (ch.get("logprobs", {}) or {}).get("content", [])]
    return ch["message"].get("content") or "", toks

def spans_for_identifier(toks, ident):
    """Find token index ranges whose concatenation contains `ident`; return (start_idx,end_idx,tokens)."""
    text = "".join(t for t, _ in toks)
    # char offset -> token idx map
    bounds = []; off = 0
    for i, (t, _) in enumerate(toks):
        bounds.append((off, off + len(t), i)); off += len(t)
    hits = []
    start = 0
    while True:
        k = text.find(ident, start)
        if k < 0: break
        end = k + len(ident)
        idxs = [i for (a, b, i) in bounds if a < end and b > k]
        if idxs: hits.append(idxs)
        start = k + 1
    return hits

N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
tasks = [(name, p, s) for (name, p) in PROMPTS for s in range(N)]
results = []
with concurrent.futures.ThreadPoolExecutor(4) as ex:
    futs = {ex.submit(gen, p, s): (name, s) for (name, p, s) in tasks}
    for f in concurrent.futures.as_completed(futs):
        name, s = futs[f]
        try: text, toks = f.result()
        except Exception as e: print(f"  [{name}#{s}] ERR {e}"); continue
        results.append((name, s, text, toks))

print(f"=== captured {len(results)} samples; scanning committed near-neighbors + their p_tree ===")
n_garble = 0
allmins = []
for name, s, text, toks in results:
    code = extract_code(text)
    loads, nundef, undef = undefined_names(code)
    if nundef == 0 or undef == ["<syntaxerror>"]: continue
    n_garble += 1
    for ident in sorted(set(undef)):
        for idxs in spans_for_identifier(toks, ident)[:1]:  # first occurrence
            sub = [(toks[i][0], round(toks[i][1], 3)) for i in idxs]
            # the divergence token = the min-logprob token in the identifier span
            mn = min((toks[i][1] for i in idxs), default=0.0)
            allmins.append(mn)
            print(f"  [{name}#{s}] undefined '{ident}'  min_lp={round(mn,3)}  tokens={sub}")
if allmins:
    import statistics as st
    hi = sum(1 for m in allmins if m > -2.3)   # prob > 0.10
    lo = sum(1 for m in allmins if m < -4.6)    # prob < 0.01
    print(f"\n=== VERDICT over {len(allmins)} near-neighbor divergence tokens ===")
    print(f"  median divergence-token p_tree logprob = {round(st.median(allmins),3)} (prob {round(2.718**st.median(allmins),4)})")
    print(f"  HIGH p_tree (>0.10, tree CONFIDENT in the wrong token => DRIFT-inflated) : {hi}/{len(allmins)}")
    print(f"  LOW  p_tree (<0.01, correct-but-unlucky temp-0.6 tail)                   : {lo}/{len(allmins)}")
    print(f"  -> DRIFT if mostly HIGH; MODEL-TAIL if mostly LOW")
else:
    print("  (no garbling samples captured this run — boot-variance; rerun or raise N)")
