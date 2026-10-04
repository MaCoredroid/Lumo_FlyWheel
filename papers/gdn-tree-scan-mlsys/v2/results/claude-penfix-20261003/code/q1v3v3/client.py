#!/usr/bin/env python3
"""q1v3 host client: drives every predeclared observation of one arm against the running engine (stdlib only).

  client.py --arm A|B|V|P|CAND --run RUN_ROOT [--ready READY.json] [--plan-only]

For each observation: write RUN/<arm>/control.json (atomic), POST one chat completion (same body in every arm:
the prefix's frozen messages/tools, T=1.0, top_p=1.0, no seed, max_tokens = forced outputs + 8), then wait for
the hooks' sealed record RUN/<arm>/cases/<obs_id>.json. The first observation of every prefix carries a
per-arm cache_salt (cold prefill; natural-O0 digest) and, in arm A, captures the common O0.
CAND requests are signed exactly like scripts/v2exp/run_tree_arm.sh (fixed32_auth.headers + READY.json).
"""
from __future__ import annotations

import argparse, http.client, importlib.util, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import q1v3_common as Q  # noqa: E402

NC_ORDER = ["NC_CONV", "NC_GDN", "NC_KV", "NC_STALE", "NC_SIB"]   # least to most invasive; NC_SIB last


def plan(doc, arm, run_id):
    order = list(doc["prefixes"])
    cases = sorted(doc["cases"], key=lambda c: (order.index(c["prefix"]), c["case_id"]))
    obs, seen = [], set()
    for c in cases:
        first = c["prefix"] not in seen
        seen.add(c["prefix"])
        obs.append({"obs_id": f"{arm}__{c['case_id']}__r0", "case_id": c["case_id"], "arm": arm, "mutation": None,
                    "o0_mode": "capture" if (arm == "A" and first) else c.get("o0_mode", "import"), "natural_digest": first or c.get("o0_mode") == "none",
                    "cache_salt": f"q1v3-cold-{run_id}-{arm}-{c['prefix']}" if first else None})
    if arm == "CAND":
        for cid in doc["candidate_repeats"]:
            obs.append({"obs_id": f"{arm}__{cid}__r1", "case_id": cid, "arm": arm, "mutation": None, "o0_mode": "import",
                        "natural_digest": False, "cache_salt": None})
        ncs = sorted(doc["negative_controls"], key=lambda n: (NC_ORDER.index(n["mutation"]), n["nc_id"]))
        for n in ncs:
            obs.append({"obs_id": f"{arm}__{n['base_case']}__{n['mutation']}", "case_id": n["base_case"], "arm": arm, "nc_id": n["nc_id"],
                        "mutation": {"mutation": n["mutation"], "cycle": n["cycle"], "sibling_nodes": n.get("sibling_nodes")},
                        "o0_mode": "import", "natural_digest": False, "cache_salt": None})
    return obs


def load_auth(ready):
    os.environ["V2EXP_READY_FILE"] = ready
    spec = importlib.util.spec_from_file_location("fixed32_auth", os.path.join(os.path.dirname(HERE), "fixed32_auth.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.headers


def post(body, headers, timeout):
    c = http.client.HTTPConnection("127.0.0.1", 9950, timeout=timeout)
    h = {"Content-Type": "application/json"}; h.update(headers or {})
    t0 = time.time()
    c.request("POST", "/v1/chat/completions", body=json.dumps(body).encode(), headers=h)
    r = c.getresponse(); raw = r.read()
    out = {"status": r.status, "seconds": round(time.time() - t0, 3)}
    try:
        d = json.loads(raw)
        ch = (d.get("choices") or [{}])[0]
        out.update(usage=d.get("usage"), finish_reason=ch.get("finish_reason"), error=d.get("error"))
    except Exception:
        out["raw_head"] = raw[:500].decode(errors="replace")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, choices=["A", "B", "V", "P", "CAND"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--ready")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--http-timeout", type=int, default=1800)
    ap.add_argument("--seal-timeout", type=int, default=180)
    a = ap.parse_args()
    import cases as CS
    doc = CS.load()
    run_id = os.path.basename(os.path.abspath(a.run))
    obs = plan(doc, a.arm, run_id)
    out = os.path.join(a.run, a.arm)
    os.makedirs(os.path.join(out, "responses"), exist_ok=True)
    Q.write_json(os.path.join(out, "PLAN.json"), {"arm": a.arm, "observations": obs, "cases_sha256": Q.sha256_file(CS.CASES_FILE)})
    if a.plan_only:
        print(json.dumps({"arm": a.arm, "observations": len(obs)})); return 0
    headers_fn = load_auth(a.ready) if a.arm == "CAND" else None
    by_id = {c["case_id"]: c for c in doc["cases"]}
    log = open(os.path.join(out, "client.log"), "a")
    consecutive_fail, results = 0, []
    for n, o in enumerate(obs):
        c = by_id[o["case_id"]]
        req = json.load(open(os.path.join(HERE, doc["prefixes"][c["prefix"]]["request_file"])))
        body = {"model": Q.SERVED_MODEL, "messages": req["messages"], "max_tokens": c["max_tokens"], "temperature": 1.0, "top_p": 1.0,
                "stream": False, "ignore_eos": False, "chat_template_kwargs": req.get("chat_template_kwargs") or {}}
        if req.get("tools"):
            body["tools"] = req["tools"]
        if o["cache_salt"]:
            body["cache_salt"] = o["cache_salt"]
        ctrl = dict(o, seq=n, written_utc=Q.utc())
        Q.write_json(os.path.join(out, "control.json"), ctrl)
        seal = os.path.join(out, "cases", f"{o['obs_id']}.json")
        try:
            resp = post(body, headers_fn(n, o["obs_id"], body) if headers_fn else None, a.http_timeout)
        except Exception as e:
            resp = {"status": None, "error": f"{type(e).__name__}: {e}"}
        t0 = time.time()
        while not os.path.exists(seal) and time.time() - t0 < (a.seal_timeout if resp.get("status") == 200 else 30):
            time.sleep(1)
        rec = {"obs_id": o["obs_id"], "response": resp, "sealed": os.path.exists(seal), "utc": Q.utc()}
        if rec["sealed"]:
            d = json.load(open(seal)); rec["valid"] = d["valid"]; rec["problems"] = d["problems"][:5]
        Q.write_json(os.path.join(out, "responses", f"{o['obs_id']}.json"), rec)
        results.append(rec)
        line = f"[{a.arm} {n+1}/{len(obs)}] {o['obs_id']} http={resp.get('status')} {resp.get('seconds')}s sealed={rec['sealed']} valid={rec.get('valid')} {rec.get('problems') or ''}"
        print(line, flush=True); log.write(line + "\n"); log.flush()
        ok = resp.get("status") == 200 and rec.get("valid")
        consecutive_fail = 0 if ok else consecutive_fail + 1
        if resp.get("status") is None or consecutive_fail >= 3:
            print(f"[{a.arm}] stopping: engine unreachable or 3 consecutive failures", flush=True)
            break
    summary = {"arm": a.arm, "planned": len(obs), "attempted": len(results), "sealed": sum(r["sealed"] for r in results),
               "valid": sum(bool(r.get("valid")) for r in results), "utc": Q.utc()}
    Q.write_json(os.path.join(out, "CLIENT-SUMMARY.json"), summary)
    print(json.dumps(summary))
    return 0 if summary["valid"] == summary["planned"] else 1


if __name__ == "__main__":
    sys.exit(main())
