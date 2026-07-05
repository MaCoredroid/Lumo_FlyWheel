#!/usr/bin/env python3
"""route_probe_reduce.py — reduce the FR13 turn-1 cold-prefill route-flip replay.

Reads the per-arm sample_*.json completions produced by route_probe.sh and reports,
PER BOOT (arm):
  * the distribution of the turn-1 ROUTE choice
        explore_subagent (`agent` tool, esp. the Explore subagent)
        read_file
        tool:<other>
        NO_TOOL(finish=<reason>)          <- the give-up phenotype (no tool call)
  * the first-divergent-token position ACROSS the N samples (are the 8 draws
    identical => deterministic-per-config, or do they split, and where), reported
    at whitespace-token, character, and — if a tokenizer is loadable — BPE-token
    granularity, on both the full generated stream and the reasoning (think) channel
    where the autopsy's drift lives.
  * a coherent-off-task DRIFT tag (web-dev attractor / CJK), the give-up signature.
Plus a CROSS-ARM summary: the modal route per arm and whether cat8_cache flips vs
cat8_nocache (the autopsy's H1 cold-prefill route split), including the first token
at which the cache vs no-cache modal streams diverge.

Read-only. No GPU. Usage:
    .venv/bin/python route_probe_reduce.py --run-root output/fr13_route_probe
    [--tokenizer /models/qwen3.6-27b-fp8]   # optional BPE-token granularity
"""
import argparse, glob, json, os, re, sys
from collections import Counter, OrderedDict

MODEL_EXPECT = "qwen3.6-27b"
ARM_ORDER = ["cat8_cache", "cat8_nocache", "native_exseed"]

# coherent off-task attractors the autopsy verified in the give-ups (web-dev n-gram
# + CJK), used only as a phenotype TAG, never as the primary route classifier.
DRIFT_RE = re.compile(
    r"PostgreSQL|PostgreSQLDriver|getRecords|document\.write|fwrite|<\s*html|"
    r"</\s*html|XSS|Node\.js|require\(|\bPHP\b|SELECT .*FROM|CREATE TABLE",
    re.IGNORECASE)
CJK_RE = re.compile(r"[㐀-鿿぀-ヿ가-힯]")


def load_samples(cdir):
    """Return a flat list of (sample_file, choice_idx, choice_dict) across sample_*.json.
    Handles n>1 responses (multiple choices per file) as independent draws."""
    out = []
    for pf in sorted(glob.glob(os.path.join(cdir, "sample_*.json")),
                     key=lambda p: int(re.search(r"sample_(\d+)", p).group(1))):
        try:
            d = json.load(open(pf, errors="ignore"))
        except Exception as e:
            out.append((os.path.basename(pf), 0, {"_parse_error": str(e)[:160]}))
            continue
        chs = d.get("choices")
        if not chs:
            out.append((os.path.basename(pf), 0, {"_no_choices": json.dumps(d)[:200]}))
            continue
        for ci, ch in enumerate(chs):
            out.append((os.path.basename(pf), ci, ch))
    return out


def sample_view(ch):
    """Extract the fields we classify on from a single choice."""
    if "_parse_error" in ch or "_no_choices" in ch:
        return {"ok": False, "err": ch.get("_parse_error") or ch.get("_no_choices"),
                "route": "REQ_ERR", "reasoning": "", "content": "", "tool_calls": [],
                "finish": None}
    m = ch.get("message") or {}
    reasoning = m.get("reasoning_content") or ""
    content = m.get("content") or ""
    tcs = m.get("tool_calls") or []
    tools = []
    for t in tcs:
        fn = t.get("function") or {}
        tools.append({"name": fn.get("name"), "arguments": fn.get("arguments") or ""})
    # qwen sometimes leaves <think> in content if the reasoning parser did not split
    if not reasoning and "<think>" in content:
        mth = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        if mth:
            reasoning = mth.group(1)
            content = content.replace(mth.group(0), "")
    first = tools[0]["name"] if tools else None
    return {"ok": True, "reasoning": reasoning, "content": content, "tool_calls": tools,
            "first_tool": first, "finish": ch.get("finish_reason")}


def route_class(v):
    if not v["ok"]:
        return "REQ_ERR"
    tools = v["tool_calls"]
    if not tools:
        return f"NO_TOOL(finish={v['finish']})"
    name = tools[0]["name"]
    if name == "agent":
        args = (tools[0]["arguments"] or "")
        sub = ""
        try:
            aj = json.loads(args)
            sub = str(aj.get("subagent_type") or aj.get("name") or aj.get("agent") or "")
        except Exception:
            m = re.search(r"[Ee]xplore", args)
            sub = "Explore" if m else ""
        return f"explore_subagent[{sub}]" if sub else "agent_subagent"
    if name == "read_file":
        return "read_file"
    return f"tool:{name}"


def drift_tag(v):
    blob = (v["reasoning"] or "") + "\n" + (v["content"] or "")
    tags = []
    if DRIFT_RE.search(blob):
        tags.append("webdev")
    if CJK_RE.search(blob):
        tags.append("cjk")
    return tags


def tool_sig(v):
    return " ".join(f"<tool_call name={t['name']}>{(t['arguments'] or '')[:400]}"
                    for t in v["tool_calls"])


def full_stream(v):
    # generated stream in emission order: think -> answer -> tool call
    return (v["reasoning"] or "") + "\n␞<answer>\n" + (v["content"] or "") \
           + "\n␞<toolcall>\n" + tool_sig(v)


def _load_tokenizer(path):
    if not path:
        return None
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    except Exception as e:
        print(f"[warn] tokenizer '{path}' not loadable ({type(e).__name__}: {str(e)[:80]}); "
              "falling back to whitespace/char granularity", file=sys.stderr)
        return None


def first_divergence(streams, tok=None):
    """Given a list of strings, find the first position at which they are NOT all
    equal, at char / whitespace-token / (optional) BPE-token granularity. Returns a
    dict with indices (None = all identical to the shortest length) and a context
    preview of the distinct branches at the split."""
    res = {"n_streams": len(streams), "n_distinct": len(set(streams)),
           "all_identical": len(set(streams)) == 1}

    def _fd(seqs):
        if len(set(map(tuple, seqs))) == 1:
            return None
        L = min(len(s) for s in seqs)
        for i in range(L):
            col = {tuple([s[i]]) for s in seqs}
            if len(col) > 1:
                return i
        return L  # identical up to the shortest, then a length split

    # char
    res["char_index"] = _fd([list(s) for s in streams])
    # whitespace token
    ws = [s.split() for s in streams]
    res["ws_token_index"] = _fd(ws)
    # bpe
    if tok is not None:
        try:
            ids = [tok.encode(s, add_special_tokens=False) for s in streams]
            res["bpe_token_index"] = _fd(ids)
        except Exception as e:
            res["bpe_token_index"] = f"tok_err:{str(e)[:60]}"
    # context preview at the whitespace split
    idx = res["ws_token_index"]
    if isinstance(idx, int):
        prefix = " ".join(ws[0][:idx])
        res["common_prefix_tail"] = ("…" + prefix[-160:]) if len(prefix) > 160 else prefix
        branches = []
        seen = set()
        for w in ws:
            cont = " ".join(w[idx:idx + 12])
            if cont not in seen:
                seen.add(cont)
                branches.append(cont[:160])
        res["branches_at_split"] = branches[:8]
    return res


def reduce_arm(arm, cdir, tok):
    meta = {}
    mp = os.path.join(cdir, "arm_meta.json")
    if os.path.exists(mp):
        meta = json.load(open(mp))
    raw = load_samples(cdir)
    views = [sample_view(ch) for (_pf, _ci, ch) in raw]

    # ---- fail-loud asserts ----
    errs = []
    if not views:
        errs.append("NO SAMPLES (route_probe.sh did not run or produced nothing)")
    if meta.get("model_served") and meta["model_served"] != MODEL_EXPECT:
        errs.append(f"model_served={meta['model_served']} != {MODEL_EXPECT}")
    n_ok = sum(1 for v in views if v["ok"] and (v["tool_calls"] or v["content"].strip() or v["reasoning"].strip()))
    if views and n_ok == 0:
        errs.append("ALL completions empty/errored")

    routes = Counter(route_class(v) for v in views)
    finishes = Counter(v["finish"] for v in views)
    drift = {}
    for i, v in enumerate(views, 1):
        t = drift_tag(v)
        if t:
            drift[i] = t
    full = first_divergence([full_stream(v) for v in views], tok) if views else {}
    think = first_divergence([(v["reasoning"] or "") for v in views], tok) if views else {}

    per_sample = []
    for i, v in enumerate(views, 1):
        per_sample.append({
            "i": i, "route": route_class(v), "finish": v["finish"],
            "n_tool": len(v["tool_calls"]),
            "rsn_chars": len(v["reasoning"] or ""), "content_chars": len(v["content"] or ""),
            "drift": drift.get(i, []),
            "reasoning_head": (v["reasoning"] or "")[:120],
        })

    return {
        "arm": arm, "meta": meta, "n_samples": len(views), "n_nonempty": n_ok,
        "route_distribution": OrderedDict(routes.most_common()),
        "route_mode": (routes.most_common(1)[0][0] if routes else None),
        "finish_reasons": dict(finishes),
        "drift_samples": drift,
        "full_stream_divergence": full,
        "think_channel_divergence": think,
        "per_sample": per_sample,
        "_errors": errs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default="output/fr13_route_probe")
    ap.add_argument("--tokenizer", default=os.environ.get("ROUTE_PROBE_TOKENIZER", ""),
                    help="optional HF tokenizer path for BPE-token granularity "
                         "(e.g. /models/qwen3.6-27b-fp8)")
    ap.add_argument("--out", default="", help="optional JSON dump path")
    a = ap.parse_args()

    if not os.path.isdir(a.run_root):
        print(f"FAIL: run-root not found: {a.run_root} (run route_probe.sh first)", file=sys.stderr)
        sys.exit(2)
    tok = _load_tokenizer(a.tokenizer)

    arms = [x for x in ARM_ORDER if os.path.isdir(os.path.join(a.run_root, x))]
    arms += [os.path.basename(p) for p in sorted(glob.glob(os.path.join(a.run_root, "*")))
             if os.path.isdir(p) and os.path.basename(p) not in arms]
    if not arms:
        print(f"FAIL: no arm dirs under {a.run_root}", file=sys.stderr)
        sys.exit(2)

    report = OrderedDict()
    any_err = False
    for arm in arms:
        r = reduce_arm(arm, os.path.join(a.run_root, arm), tok)
        report[arm] = r
        if r["_errors"]:
            any_err = True

    # ---- human-readable ----
    print("=" * 78)
    print("FR13 turn-1 COLD-PREFILL ROUTE-FLIP replay  —  reduce")
    print("=" * 78)
    for arm in arms:
        r = report[arm]
        m = r["meta"]
        print(f"\n### {arm}   (cache={m.get('cache_boot_log')}  "
              f"exact_seed_eng={m.get('exact_seed_eng_log')}  "
              f"nonempty={r['n_nonempty']}/{r['n_samples']})")
        if r["_errors"]:
            print("  !!! ERRORS:", "; ".join(r["_errors"]))
        print("  route distribution:")
        for route, c in r["route_distribution"].items():
            print(f"      {c:>2}x  {route}")
        print(f"  route mode        : {r['route_mode']}")
        print(f"  finish_reasons    : {r['finish_reasons']}")
        if r["drift_samples"]:
            print(f"  DRIFT (off-task)  : {r['drift_samples']}")
        fd = r["full_stream_divergence"]
        td = r["think_channel_divergence"]
        print(f"  full-stream split : identical={fd.get('all_identical')} "
              f"distinct={fd.get('n_distinct')}/{fd.get('n_streams')} "
              f"char={fd.get('char_index')} ws_tok={fd.get('ws_token_index')} "
              f"bpe_tok={fd.get('bpe_token_index','n/a')}")
        print(f"  think-chan split  : identical={td.get('all_identical')} "
              f"distinct={td.get('n_distinct')}/{td.get('n_streams')} "
              f"char={td.get('char_index')} ws_tok={td.get('ws_token_index')} "
              f"bpe_tok={td.get('bpe_token_index','n/a')}")
        if fd.get("branches_at_split"):
            print(f"  split prefix tail : {fd.get('common_prefix_tail','')!r}")
            for b in fd["branches_at_split"]:
                print(f"      ->  {b!r}")

    # ---- cross-arm: the route flip ----
    print("\n" + "=" * 78)
    print("CROSS-ARM  (autopsy H1: cold-prefill route split cache vs no-cache)")
    print("=" * 78)
    modes = {arm: report[arm]["route_mode"] for arm in arms}
    for arm in arms:
        print(f"  {arm:<16} route_mode = {modes[arm]}")
    if "cat8_cache" in report and "cat8_nocache" in report:
        a1 = report["cat8_cache"]["route_mode"]
        b1 = report["cat8_nocache"]["route_mode"]
        print(f"\n  cat8_cache vs cat8_nocache modal route: "
              f"{'FLIP' if a1 != b1 else 'SAME'}  ({a1}  vs  {b1})")
        # first token where the two modal streams diverge (sample 1 of each)
        try:
            s_c = full_stream(sample_view(load_samples(os.path.join(a.run_root, 'cat8_cache'))[0][2]))
            s_n = full_stream(sample_view(load_samples(os.path.join(a.run_root, 'cat8_nocache'))[0][2]))
            xd = first_divergence([s_c, s_n], tok)
            print(f"  cache-vs-nocache stream split: char={xd.get('char_index')} "
                  f"ws_tok={xd.get('ws_token_index')} bpe_tok={xd.get('bpe_token_index','n/a')}")
            if xd.get("common_prefix_tail") is not None:
                print(f"    common prefix tail: {xd['common_prefix_tail']!r}")
                for b in xd.get("branches_at_split", []):
                    print(f"      ->  {b!r}")
        except Exception as e:
            print(f"  (cross-stream split unavailable: {type(e).__name__}: {str(e)[:80]})")

    if a.out:
        json.dump(report, open(a.out, "w"), ensure_ascii=False, indent=1)
        print(f"\n[json] {a.out}")

    if any_err:
        print("\nFAIL: one or more arms had errors (see !!! above).", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
