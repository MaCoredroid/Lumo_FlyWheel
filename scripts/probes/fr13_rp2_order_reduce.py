#!/usr/bin/env python3
"""fr13_rp2_order_reduce.py — reduce the §47 order-vs-seed probe (fr13_rp2_order_probe.sh).

Per sample (in send order): route, finish, completion_tokens, token-1 chosen logprob,
token-1 top-5 alternatives, prefix-cache query/hit deltas (cold verification).
Verdicts:
  ORDER_DEPENDENT   — P1 (fixed seed, reset, cold) varies across positions
                      (tok1 logprob spread > 0.02 nats OR >1 distinct route)
  SEED_EFFECT_EXTRA — P3 spread exceeds P1 spread by >0.1 nats (a genuine seed term
                      beyond order would show here; expected NO)
  CONTROL_CLEAN     — C1+C2 flat (spread <= 0.02 nats, 1 route)
Reads message.reasoning (NOT reasoning_content — that field is null on this server and
was the §47 reducer-bug trap).
"""
import argparse, json, os, re, sys

def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None

def metrics_val(path, needle):
    if not os.path.exists(path):
        return None
    tot = 0.0
    found = False
    for ln in open(path, errors="ignore"):
        if ln.startswith("#") or needle not in ln:
            continue
        m = re.search(r"\s([0-9.e+-]+)\s*$", ln)
        if m:
            tot += float(m.group(1)); found = True
    return tot if found else None

def sample_row(cdir, tag):
    d = load(os.path.join(cdir, f"sample_{tag}.json"))
    if not d or not d.get("choices"):
        return {"tag": tag, "error": "missing/unparseable"}
    c0 = d["choices"][0]
    m = c0.get("message") or {}
    tc = m.get("tool_calls") or []
    route = tc[0].get("function", {}).get("name") if tc else "NO_TOOL"
    lp = (c0.get("logprobs") or {}).get("content") or []
    tok1 = lp[0] if lp else None
    row = {
        "tag": tag,
        "route": route,
        "finish": c0.get("finish_reason"),
        "ct": (d.get("usage") or {}).get("completion_tokens"),
        "tok1": (tok1 or {}).get("token"),
        "tok1_lp": (tok1 or {}).get("logprob"),
        "tok1_top5": [(t.get("token"), round(t.get("logprob"), 4))
                      for t in ((tok1 or {}).get("top_logprobs") or [])[:5]],
        "reasoning_head": (m.get("reasoning") or "")[:60].replace("\n", "\\n"),
    }
    q_pre = metrics_val(os.path.join(cdir, f"metrics_{tag}_pre.txt"), "prefix_cache_queries")
    q_post = metrics_val(os.path.join(cdir, f"metrics_{tag}_post.txt"), "prefix_cache_queries")
    h_pre = metrics_val(os.path.join(cdir, f"metrics_{tag}_pre.txt"), "prefix_cache_hits")
    h_post = metrics_val(os.path.join(cdir, f"metrics_{tag}_post.txt"), "prefix_cache_hits")
    if None not in (q_pre, q_post):
        row["q_delta"] = q_post - q_pre
    if None not in (h_pre, h_post):
        row["hit_delta"] = h_post - h_pre
    return row

def phase_of(tag):
    return tag.split("_")[0]

def spread(rows):
    lps = [r["tok1_lp"] for r in rows if r.get("tok1_lp") is not None]
    if not lps:
        return None
    return max(lps) - min(lps)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    args = ap.parse_args()

    verdicts = {}
    for arm in sorted(os.listdir(args.run_root)):
        cdir = os.path.join(args.run_root, arm)
        if not os.path.isdir(cdir):
            continue
        tags = []
        for f in os.listdir(cdir):
            m = re.match(r"send_((?:P|C)\d+_[A-Za-z0-9]+)\.json$", f)
            if m:
                tags.append(m.group(1))
        def order_key(t):
            ph, sub = t.split("_", 1)
            return (ph, int(sub.lstrip("s")))
        tags.sort(key=order_key)
        print(f"\n===== ARM {arm} ({len(tags)} samples, send order) =====")
        rows = [sample_row(cdir, t) for t in tags]
        for r in rows:
            if "error" in r:
                print(f"  {r['tag']:<8} ERROR {r['error']}")
                continue
            lp = f"{r['tok1_lp']:.4f}" if r.get("tok1_lp") is not None else "None"
            cold = ""
            if r.get("hit_delta") is not None:
                cold = f" hitΔ={r['hit_delta']:.0f}"
            print(f"  {r['tag']:<8} route={r['route']:<12} finish={str(r['finish']):<7} "
                  f"ct={str(r['ct']):<5} tok1={str(r['tok1'])[:8]:<8}@{lp}{cold}  «{r['reasoning_head'][:45]}»")
        by = {}
        for r in rows:
            if "error" not in r:
                by.setdefault(phase_of(r["tag"]), []).append(r)
        arm_v = {}
        for ph, prs in sorted(by.items()):
            sp = spread(prs)
            routes = sorted({r["route"] for r in prs})
            arm_v[ph] = {"n": len(prs), "tok1_lp_spread": sp, "routes": routes,
                         "tok1_lps": [round(r["tok1_lp"], 4) for r in prs if r.get("tok1_lp") is not None]}
            print(f"  -- {ph}: n={len(prs)} spread={sp if sp is None else round(sp,4)} routes={routes}")
        if "P1" in arm_v:
            v = arm_v["P1"]
            order_dep = (v["tok1_lp_spread"] or 0) > 0.02 or len(v["routes"]) > 1
            print(f"  VERDICT[{arm}] ORDER_DEPENDENT={order_dep} "
                  f"(P1 fixed-seed spread={v['tok1_lp_spread']}, routes={v['routes']})")
            arm_v["ORDER_DEPENDENT"] = order_dep
            if "P3" in arm_v and v["tok1_lp_spread"] is not None and arm_v["P3"]["tok1_lp_spread"] is not None:
                extra = arm_v["P3"]["tok1_lp_spread"] - v["tok1_lp_spread"]
                arm_v["SEED_EFFECT_EXTRA"] = extra > 0.1
                print(f"  VERDICT[{arm}] SEED_EFFECT_EXTRA={extra > 0.1} (P3-P1 spread delta={round(extra,4)})")
        if "C1" in arm_v:
            clean = all((arm_v[ph]["tok1_lp_spread"] or 0) <= 0.02 and len(arm_v[ph]["routes"]) == 1
                        for ph in ("C1", "C2") if ph in arm_v)
            arm_v["CONTROL_CLEAN"] = clean
            print(f"  VERDICT[{arm}] CONTROL_CLEAN={clean}")
        verdicts[arm] = arm_v

    out = os.path.join(args.run_root, "rp2_reduce.json")
    json.dump(verdicts, open(out, "w"), indent=1, default=str)
    print(f"\n[reduce] wrote {out}")

if __name__ == "__main__":
    main()
