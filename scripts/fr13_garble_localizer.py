#!/usr/bin/env python3
"""FR13 garble LOCALIZER — within-boot, teacher-forced classifier for the tree spec-decode
near-neighbor "misspell" garble. Per garbled token, decides:

  WRONG-ACCEPT : the model's OWN next-token prefill distribution (no-spec reference) strongly
                 prefers the CORRECT continuation, yet the tree committed the near-neighbor.
                 => the tree spec-decode ACCEPTED a token the model itself would not emit.
  MODEL-TAIL   : the prefill distribution itself ranks the garble token comparably to the
                 correct token => the model would plausibly sample it at temp 0.6; not spec-specific.

METHOD (fully exact, no retokenization ambiguity):
  1. GENERATE via /v1/completions with prompt = /tokenize(messages) token-ids, return_token_ids=True,
     logprobs=1. This goes through the SAME engine-level tree spec-decode as the reproducer's chat
     path (identical input tokens) and returns the EXACT generated token-ids + text_offset.
  2. DETECT near-neighbor garbles: parse the emitted code (reuse fr13_garble_gate.undefined_names),
     match each undefined-name use to a CANONICAL id the prompt required (or a correctly-spelled
     defined name) at edit-distance 1-2. The literal misspelled string is the garble.
  3. Locate the divergence token: the FIRST generated token that carries the misspelled char.
     The teacher-force prefix = prompt-ids + generated-token-ids[:that index] (an exact token cut).
  4. TEACHER-FORCE: /v1/completions {prompt: <that exact token prefix>, max_tokens:1,
     temperature:1.0, logprobs:20, seed:FIXED} after /reset_prefix_cache. temp=1.0 => the ranking
     and nat-margins are temperature-invariant. This single prefill step is NOT spec-decoded => it
     is the model's own (no-spec) next-token distribution at that position.
     ANTI-TRAP: we read ONLY this teacher-forced distribution, never the spec-decode-surfaced
     placeholder logprob of the accepted draft (the -12.422 sentinel). We assert the distribution
     is smooth (not a constant sentinel) before trusting it.
  5. CLASSIFY by comparing, in the teacher-forced dist, the correct-continuation token C vs the
     committed garble token X (probability + rank + nat-margin).

Usage:
  python3 scripts/fr13_garble_localizer.py --endpoint http://127.0.0.1:9950 --model qwen3.6-27b \
      --n 40 --out output/fr13_garble_loc/loc.jsonl [--reset-cache] [--repeat-check]

Only MEASURES. Never edits sampler/model/routing. Native is not called.
"""
import sys, os, json, argparse, urllib.request, math, re, ast, builtins, difflib, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fr13_garble_gate import PROMPTS, extract_code, undefined_names, BUILTINS  # reuse the reproducer

MARGIN_NATS = 1.0          # "strongly prefers" threshold (task spec: > ~1 nat)
LOGPROB_FLOOR = -30.0      # prob assigned to a token absent from the returned top-20

# ------------------------------------------------------------------ http helpers
def _post(endpoint, path, body, timeout=600):
    r = urllib.request.Request(endpoint.rstrip("/") + path, data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=timeout).read())

def _post_raw(endpoint, path, timeout=60):
    r = urllib.request.Request(endpoint.rstrip("/") + path, data=b"",
                               headers={"Content-Type": "application/json"}, method="POST")
    return urllib.request.urlopen(r, timeout=timeout).read()

def tokenize_messages(endpoint, model, messages):
    j = _post(endpoint, "/tokenize", {"model": model, "messages": messages,
              "add_generation_prompt": True, "chat_template_kwargs": {"enable_thinking": False}})
    return j["tokens"]

def reset_cache(endpoint):
    try:
        _post_raw(endpoint, "/reset_prefix_cache"); return True
    except Exception:
        return False

# ------------------------------------------------------------------ code analysis
def analyze(code):
    """Return (loads_undef_list, defined_set). Mirrors fr13_garble_gate.undefined_names but also
    exposes the DEFINED set so we can match a used-misspelling to a correctly-defined name."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return [], set()
    defined = set(BUILTINS); imported = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined.add(node.name)
            for a in node.args.args + node.args.kwonlyargs + getattr(node.args, "posonlyargs", []):
                defined.add(a.arg)
            if node.args.vararg: defined.add(node.args.vararg.arg)
            if node.args.kwarg: defined.add(node.args.kwarg.arg)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            for t in ([node.target] if hasattr(node, "target") else node.targets):
                for n in ast.walk(t):
                    if isinstance(n, ast.Name): defined.add(n.id)
        elif isinstance(node, (ast.For, ast.comprehension, ast.With, ast.withitem)):
            tgt = getattr(node, "target", None) or getattr(node, "optional_vars", None)
            if tgt:
                for n in ast.walk(tgt):
                    if isinstance(n, ast.Name): defined.add(n.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names: imported.add((a.asname or a.name).split(".")[0])
    known = defined | imported
    undef = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in known:
            undef.append(node.id)
    return undef, (defined - BUILTINS)

def parse_required(prompt_text):
    """Canonical identifiers the prompt demanded: function name + params + the 'EXACT variables' list."""
    ids = set()
    m = re.search(r"`(\w+)\(([^)]*)\)`", prompt_text)
    if m:
        ids.add(m.group(1))
        for p in m.group(2).split(","):
            p = p.strip()
            if p: ids.add(p)
    m = re.search(r"EXACT variables:\s*(.+?)\.", prompt_text, re.S)
    if m:
        for v in m.group(1).split(","):
            v = v.strip()
            if re.fullmatch(r"[A-Za-z_]\w*", v): ids.add(v)
    return ids

def edist(a, b):
    return _lev(a, b)

def _lev(a, b):
    if a == b: return 0
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i-1] != b[j-1]))
        prev = cur
    return prev[lb]

def common_prefix_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y: break
        n += 1
    return n

def find_garbles(text, required):
    """Return list of (canonical, misspelled) near-neighbor garble pairs found in `text`.
    canonical = the correctly-spelled id the model was supposed to use; misspelled = the literal
    wrong spelling that actually appears in the generated code (edit-distance 1-2, shared prefix>=3)."""
    code = extract_code(text)
    undef, defined = analyze(code)
    out = []
    seen = set()
    for u in undef:
        # Case 1: u itself is a misspelling of a required canonical id (use-site garble)
        cand = [(edist(u, c), c) for c in required if c != u]
        cand = [(d, c) for d, c in cand if 1 <= d <= 2 and common_prefix_len(u, c) >= 3
                and abs(len(u) - len(c)) <= 3]
        if cand:
            d, c = min(cand)
            key = (c, u)
            if key not in seen and u in text:
                seen.add(key); out.append({"canonical": c, "misspelled": u, "edist": d, "site": "use"})
            continue
        # Case 2: u is (essentially) a required id that is undefined because a DEFINED name
        # misspells it (def-site garble). The misspelled string is the defined near-neighbor.
        if u in required:
            cand2 = [(edist(u, dname), dname) for dname in defined if dname != u]
            cand2 = [(d, dn) for d, dn in cand2 if 1 <= d <= 2 and common_prefix_len(u, dn) >= 3
                     and abs(len(u) - len(dn)) <= 3]
            if cand2:
                d, dn = min(cand2)
                key = (u, dn)
                if key not in seen and dn in text:
                    seen.add(key); out.append({"canonical": u, "misspelled": dn, "edist": d, "site": "def"})
    return out

# ------------------------------------------------------------------ generation + teacher-force
def generate(endpoint, model, prompt_ids, seed, max_tokens=700, temperature=0.6):
    j = _post(endpoint, "/v1/completions", {"model": model, "prompt": prompt_ids,
              "max_tokens": max_tokens, "temperature": temperature, "seed": seed,
              "logprobs": 1, "return_token_ids": True})
    ch = j["choices"][0]
    lp = ch.get("logprobs") or {}
    return {"text": ch.get("text") or "", "token_ids": ch.get("token_ids") or [],
            "tok_strs": lp.get("tokens") or [], "text_offset": lp.get("text_offset") or [],
            "finish": ch.get("finish_reason")}

def teacher_force(endpoint, model, prefix_ids, seed, reset=True):
    """Single prefill step (NOT spec-decoded): the model's own next-token distribution after
    prefix_ids. Returns sorted [(token_str, logprob), ...] top-20 and the raw dict."""
    if reset:
        reset_cache(endpoint)
    j = _post(endpoint, "/v1/completions", {"model": model, "prompt": prefix_ids,
              "max_tokens": 1, "temperature": 1.0, "seed": seed, "logprobs": 20})
    ch = j["choices"][0]
    lp = ch.get("logprobs") or {}
    tl = (lp.get("top_logprobs") or [None])[0] or {}
    dist = sorted(tl.items(), key=lambda kv: -kv[1])
    return dist

def dist_is_smooth(dist):
    """Reject the -12.422 constant sentinel / degenerate distributions (anti-placeholder guard)."""
    if len(dist) < 3: return False
    lps = [v for _, v in dist]
    if max(lps) < -11.0: return False           # no real argmax => suspicious
    if len(set(round(v, 3) for v in lps)) < 3:  # ~constant => sentinel
        return False
    return True

# ------------------------------------------------------------------ per-garble localize
def localize_one(endpoint, model, prompt_ids, gen, canonical, misspelled, seed, reset=True):
    """Teacher-force at the first divergent token of `misspelled` and classify."""
    text = gen["text"]; toks = gen["tok_strs"]; ids = gen["token_ids"]; offs = gen["text_offset"]
    if not (toks and ids and offs) or len(toks) != len(ids) or len(offs) != len(ids):
        return {"error": "gen token/offset arrays missing or misaligned"}
    # word-boundary search: a trailing-drop garble (e.g. 'sliced_pixel_bound') is a SUBSTRING of the
    # correct id ('sliced_pixel_bounds'); a plain find() would lock onto the correct occurrence and
    # measure the wrong position. Require identifier-char boundaries on both sides.
    m = re.search(r"(?<![A-Za-z0-9_])" + re.escape(misspelled) + r"(?![A-Za-z0-9_])", text)
    if not m:
        return {"error": "misspelled string not found as a standalone identifier in text"}
    off = m.start()
    div_in_id = common_prefix_len(misspelled, canonical)   # first char where spellings differ
    div_char = off + div_in_id                              # char offset in text where they diverge
    # cut token = the generated token that CONTAINS div_char (it carries the divergent char);
    # everything strictly before it is the shared prefix.
    cut = None
    for t in range(len(offs)):
        end = offs[t] + len(toks[t])
        if offs[t] <= div_char < end:
            cut = t; break
    if cut is None:
        return {"error": f"could not map divergence char {div_char} to a token"}
    cut_start = offs[cut]                                   # char where the committed cut token begins
    prefix_ids = list(prompt_ids) + list(ids[:cut])         # EXACT token cut, no retokenization
    dist = teacher_force(endpoint, model, prefix_ids, seed, reset=reset)
    if not dist_is_smooth(dist):
        return {"error": "teacher-forced distribution failed smoothness/anti-sentinel check",
                "dist_head": dist[:5]}
    lp_map = {k: v for k, v in dist}
    # Build the expected text continuation from cut_start, robust to a cut token that carries a
    # leading space / other non-identifier chars (e.g. ' world'). prelead = non-id chars in the cut
    # token before the identifier; both spellings share prelead, then diverge inside the id.
    consumed_id = max(0, cut_start - off)                   # identifier chars already in the prefix
    prelead = text[cut_start:off] if cut_start < off else ""
    exp_correct = prelead + canonical[consumed_id:]         # correct continuation text from cut_start
    exp_garble = prelead + misspelled[consumed_id:]         # == actual committed continuation text
    X_str = toks[cut]                                       # exact committed garble token (ground truth)

    # correct-continuation token C = highest-lp token that KEEPS the correct spelling reachable
    # (exp_correct.startswith(k)) and is NOT the committed garble token. A token like '_row' that is
    # an ambiguous prefix of the garble token '_rows' still counts as correct-consistent: emitting it
    # does NOT commit to the garble, whereas the tree committed '_rows' in one token.
    C = None
    for k, v in dist:
        if len(k) > 0 and exp_correct.startswith(k) and k != X_str:
            C = (k, v); break
    # committed garble token X = the exact token the tree emitted; its prob in the (no-spec) prefill.
    X = (X_str, lp_map[X_str]) if X_str in lp_map else None
    lp_C = C[1] if C else LOGPROB_FLOOR
    lp_X = X[1] if X else LOGPROB_FLOOR
    rank_of = lambda tok: next((i for i, (k, _) in enumerate(dist) if k == tok), None)
    rank_C = rank_of(C[0]) if C else None
    rank_X = rank_of(X[0]) if X else None
    argmax_tok, argmax_lp = dist[0]
    argmax_is_correct = exp_correct.startswith(argmax_tok) and argmax_tok != X_str
    margin = lp_C - lp_X

    # classification (task spec): WRONG_ACCEPT = prefill strongly prefers correct (near-top) with the
    # committed garble > ~1 nat below it; MODEL_TAIL = the garble is within ~1 nat of correct (model
    # itself would plausibly emit it at temp 0.6).
    if C is None:
        label = "OTHER"
        reason = "no correct-consistent token in teacher-forced top-20"
    elif (rank_C is not None and rank_C <= 3) and margin > MARGIN_NATS:
        label = "WRONG_ACCEPT"
        reason = (f"prefill prefers correct {C[0]!r} (rank {rank_C}) over committed garble "
                  f"{X_str!r} by {margin:.2f} nats")
    elif margin <= MARGIN_NATS:
        label = "MODEL_TAIL"
        reason = (f"prefill ranks committed garble {X_str!r} comparably to correct "
                  f"(margin {margin:.2f} nats <= {MARGIN_NATS})")
    else:
        label = "OTHER"
        reason = f"correct {C[0]!r} rank {rank_C}, margin {margin:.2f}"

    return {
        "canonical": canonical, "misspelled": misspelled,
        "consumed_prefix_chars": consumed_id, "cut_token_index": cut,
        "committed_garble_token": X_str,
        "correct_cont_token": C[0] if C else None, "correct_cont_logprob": (C[1] if C else None),
        "correct_cont_prob": (math.exp(C[1]) if C else None), "correct_cont_rank": rank_C,
        "garble_token_logprob": (X[1] if X else None),
        "garble_token_prob": (math.exp(X[1]) if X else None), "garble_token_rank": rank_X,
        "argmax_token": argmax_tok, "argmax_prob": math.exp(argmax_lp),
        "argmax_is_correct": argmax_is_correct,
        "margin_nats": margin, "label": label, "reason": reason,
        "teacher_forced_top20": [{"tok": k, "logprob": v, "prob": math.exp(v)} for k, v in dist],
    }

# ------------------------------------------------------------------ driver
def run(args):
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    # cache prompt token-ids per prompt (identical across seeds)
    ptoks, required = {}, {}
    for name, ptext in PROMPTS:
        ptoks[name] = tokenize_messages(args.endpoint, args.model, [{"role": "user", "content": ptext}])
        required[name] = parse_required(ptext)
        print(f"[prompt {name}] {len(ptoks[name])} prompt-tokens; required ids: {sorted(required[name])}")

    # ---- Phase 1: generate ALL (prompt,seed) concurrently (matches reproducer's conc-4 regime; the
    # garble is batch-dependent). Teacher-forcing (Phase 2) is serial + cache-reset = deterministic. ----
    import concurrent.futures
    tasks = [(name, seed) for (name, _) in PROMPTS for seed in range(args.n)]
    gens = {}
    def _do_gen(name, seed):
        return (name, seed), generate(args.endpoint, args.model, ptoks[name], seed)
    with concurrent.futures.ThreadPoolExecutor(args.gen_concurrency) as ex:
        futs = [ex.submit(_do_gen, name, seed) for (name, seed) in tasks]
        done = 0
        for f in concurrent.futures.as_completed(futs):
            k, g = f.result(); gens[k] = g; done += 1
            if done % 30 == 0: print(f"  ... generated {done}/{len(tasks)}")
    print(f"[gen] {len(gens)} generations done (concurrency {args.gen_concurrency})")
    # capture raw generations (within-boot audit + offline re-classification without re-gen)
    cap = args.out + ".gens"
    with open(cap, "w") as cf:
        for (name, seed), g in gens.items():
            cf.write(json.dumps({"prompt": name, "seed": seed, "finish": g["finish"],
                                 "text": g["text"], "token_ids": g["token_ids"],
                                 "tok_strs": g["tok_strs"], "text_offset": g["text_offset"]}) + "\n")
    print(f"[gen] captured raw generations -> {cap}")

    # ---- Phase 2: detect near-neighbor garbles + teacher-force each (serial, deterministic) ----
    records = []
    n_gen = len(gens); n_gen_with_garble = 0
    fh = open(args.out, "w")
    for name, ptext in PROMPTS:
        for seed in range(args.n):
            gen = gens[(name, seed)]
            garbles = find_garbles(gen["text"], required[name])
            if garbles: n_gen_with_garble += 1
            for g in garbles:
                res = localize_one(args.endpoint, args.model, ptoks[name], gen,
                                   g["canonical"], g["misspelled"], seed, reset=args.reset_cache)
                rec = {"prompt": name, "seed": seed, "site": g["site"], "edist": g["edist"], **res}
                records.append(rec); fh.write(json.dumps(rec) + "\n"); fh.flush()
                lbl = rec.get("label", "ERR")
                if "error" in rec:
                    print(f"  [{name} s{seed}] {g['misspelled']!r} -> ERROR: {rec['error']}")
                else:
                    print(f"  [{name} s{seed}] {g['canonical']!r} vs garble {g['misspelled']!r} "
                          f"({g['site']}) -> {lbl}  Pcorrect={rec['correct_cont_prob']!s:.6} "
                          f"(rank {rec['correct_cont_rank']}) Pgarble={rec['garble_token_prob']!s:.6} "
                          f"(rank {rec['garble_token_rank']}) margin={rec['margin_nats']:.2f}n")
    fh.close()

    # ------ optional determinism repeat on the teacher-force measurement ------
    if args.repeat_check and records:
        good = [r for r in records if "error" not in r]
        if good:
            r0 = good[0]
            name = r0["prompt"]; seed = r0["seed"]
            gen = generate(args.endpoint, args.model, ptoks[name], seed)
            gs = find_garbles(gen["text"], required[name])
            if gs:
                g = gs[0]
                a = localize_one(args.endpoint, args.model, ptoks[name], gen, g["canonical"],
                                 g["misspelled"], seed, reset=True)
                b = localize_one(args.endpoint, args.model, ptoks[name], gen, g["canonical"],
                                 g["misspelled"], seed, reset=True)
                same = (a.get("teacher_forced_top20") == b.get("teacher_forced_top20")
                        and a.get("label") == b.get("label"))
                print(f"\n[REPEAT-CHECK] same-boot re-gen+teacher-force of ({name} s{seed}): "
                      f"identical top-20 & label = {same}")

    # ------ summary ------
    good = [r for r in records if "error" not in r]
    errs = [r for r in records if "error" in r]
    from collections import Counter
    labels = Counter(r["label"] for r in good)
    print("\n" + "=" * 78)
    print(f"SUMMARY  gens={n_gen}  gens-with-near-neighbor-garble={n_gen_with_garble}  "
          f"garble-tokens-localized={len(good)}  localize-errors={len(errs)}")
    print(f"  WRONG_ACCEPT={labels.get('WRONG_ACCEPT',0)}  MODEL_TAIL={labels.get('MODEL_TAIL',0)}  "
          f"OTHER={labels.get('OTHER',0)}")
    print("\n  per-garble table (correct vs garble, teacher-forced prefill dist):")
    print(f"    {'label':13} {'canonical->garble':42} {'P_correct':>10} {'rk':>3} "
          f"{'P_garble':>10} {'rk':>3} {'margin_n':>8}")
    for r in good:
        pc = r['correct_cont_prob']; pg = r['garble_token_prob']
        print(f"    {r['label']:13} {(r['canonical']+' -> '+r['misspelled'])[:42]:42} "
              f"{(pc if pc is not None else 0):10.6f} {str(r['correct_cont_rank']):>3} "
              f"{(pg if pg is not None else 0):10.6f} {str(r['garble_token_rank']):>3} "
              f"{r['margin_nats']:8.2f}")
    if errs:
        print(f"\n  {len(errs)} localize errors (e.g.): "
              f"{[e.get('error') for e in errs[:4]]}")
    print("=" * 78)
    print(f"records -> {args.out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=40, help="seeds per prompt (3 prompts)")
    ap.add_argument("--gen-concurrency", type=int, default=4, help="parallel generation (Phase 1)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reset-cache", action="store_true", default=True,
                    help="reset prefix cache before each teacher-force (default on)")
    ap.add_argument("--no-reset-cache", dest="reset_cache", action="store_false")
    ap.add_argument("--repeat-check", action="store_true")
    args = ap.parse_args()
    run(args)
