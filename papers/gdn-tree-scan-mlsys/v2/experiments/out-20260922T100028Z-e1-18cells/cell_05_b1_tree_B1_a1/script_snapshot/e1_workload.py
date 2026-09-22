#!/usr/bin/env python3
"""E1 cell workload client (frozen; e1/E1_FREEZE.md). Phases: WARM-UP then TIMED, each request recorded verbatim with
`return_tokens_as_token_ids=True` (MANDATORY token-id contract), logprobs 1, temperature 0, fixed seed.
  B1: warm-up = 2 sequential single requests of pilot prompt 1 (32 tokens); timed = the 8 pilot prompts sequentially (128).
  B4: warm-up = ONE actual four-request cohort of pilot prompts 1-4 (32 tokens each, submitted together);
      timed = cohort A (pilot 1-4) then cohort B (pilot 5-8), each submitted together (128 tokens each).
Outputs under <run>/cohort/req_<slot>_<prefix>/capture_request.json (same schema as cohort_request.v2) for EVERY request,
<run>/e1_manifest.json {phases: {warmup: [api ids], timed: [api ids]}, requests: [...]} and <run>/workload_summary.json.
Any request error, missing response id, or missing token ids -> exit 6 (the cell is invalid).
Usage: e1_workload.py <base_url> <pool.json> <frozen_prefixes.json> <run_dir> --batch {1,4} [--model qwen3.6-27b] [--seed 20260921]"""
import argparse, hashlib, json, os, sys, threading, time, urllib.request
def one(base, model, prefix, out_dir, max_tokens, seed, slot, results, phase):
    body = {"model": model, "prompt": prefix["text"], "max_tokens": max_tokens, "temperature": 0, "seed": seed, "logprobs": 1, "echo": False, "return_tokens_as_token_ids": True}
    data = json.dumps(body).encode(); t0 = time.time(); t0p = time.perf_counter()
    req = urllib.request.Request(base + "/v1/completions", data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1800) as r: resp = json.loads(r.read().decode())
        err = None
    except Exception as e:  # noqa: BLE001
        resp, err = None, repr(e)
    t1 = time.time(); t1p = time.perf_counter()
    rec = {"phase": phase, "slot": slot, "prefix_id": prefix["id"], "prefix_sha256": prefix.get("prefix_sha256") or hashlib.sha256(prefix["text"].encode()).hexdigest(), "prompt_sha256": hashlib.sha256(prefix["text"].encode()).hexdigest(),
           "request_body_sha256": hashlib.sha256(data).hexdigest(), "request": {k: v for k, v in body.items() if k != "prompt"}, "max_tokens": max_tokens, "seed": seed, "temperature": 0,
           "request_start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)), "request_end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t1)), "t_start_perf": t0p, "t_end_perf": t1p, "elapsed_s": t1p - t0p, "error": err}
    if resp is not None:
        ch = (resp.get("choices") or [{}])[0]; lp = (ch.get("logprobs") or {})
        rec.update({"response_id": resp.get("id"), "usage": resp.get("usage"), "finish_reason": ch.get("finish_reason"), "text": ch.get("text"), "response_logprobs_tokens": lp.get("tokens"), "response_logprobs_token_logprobs": lp.get("token_logprobs"),
                    "response_raw_sha256": hashlib.sha256(json.dumps(resp, sort_keys=True).encode()).hexdigest()})
    d = os.path.join(out_dir, f"req_{slot}_{prefix['id']}"); os.makedirs(d, exist_ok=True); json.dump(rec, open(os.path.join(d, "capture_request.json"), "w"), indent=1); results[slot] = rec
def cohort(base, model, prefixes, out_dir, max_tokens, seed, slot0, phase):
    res = {}; th = [threading.Thread(target=one, args=(base, model, p, out_dir, max_tokens, seed, f"{slot0}{i}", res, phase)) for i, p in enumerate(prefixes)]
    t0 = time.perf_counter()
    for t in th: t.start()
    for t in th: t.join()
    return [res[f"{slot0}{i}"] for i in range(len(prefixes))], time.perf_counter() - t0
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("base"); ap.add_argument("pool"); ap.add_argument("frozen"); ap.add_argument("run"); ap.add_argument("--batch", type=int, required=True, choices=[1, 4]); ap.add_argument("--model", default="qwen3.6-27b"); ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--warmup-tokens", type=int, default=32); ap.add_argument("--timed-tokens", type=int, default=128)
    ap.add_argument("--phase", default="all", choices=["all", "preflight", "main"], help="preflight = the frozen UNTIMED native preflight segment only (B1: the 8 pilot prompts sequentially, 32 tokens; B4: cohorts 1-4 and 5-8, 32 tokens each), written to e1_preflight_manifest.json; main = warm-up + timed (merges a preflight manifest if present); all = main without preflight"); a = ap.parse_args()
    pool = {p["id"]: p for p in json.load(open(a.pool))["prefixes"]}; fro = json.load(open(a.frozen)); pil = [pool[r["id"]] for r in fro["pilot"]]
    for r in fro["pilot"]:
        if hashlib.sha256(pool[r["id"]]["text"].encode()).hexdigest() != r["prefix_sha256"]: print("REFUSING: pilot prefix hash mismatch", r["id"]); sys.exit(6)
    if len(pil) != 8: print("REFUSING: need exactly 8 pilot prefixes"); sys.exit(6)
    out = os.path.join(a.run, "cohort"); os.makedirs(out, exist_ok=True); man = {"phases": {"preflight": [], "warmup": [], "timed": []}, "requests": [], "batch": a.batch, "submission": []}
    t_all = time.perf_counter()
    if a.phase == "preflight":
        if a.batch == 1:
            for i, p in enumerate(pil):
                r = {}; one(a.base, a.model, p, out, a.warmup_tokens, a.seed, f"f{i}", r, "preflight"); rec = r[f"f{i}"]; man["requests"].append(rec); man["submission"].append({"phase": "preflight", "slots": [f"f{i}"], "utc": rec["request_start_utc"]})
        else:
            for name, grp, s0 in (("PA(1-4)", pil[:4], "fa"), ("PB(5-8)", pil[4:], "fb")):
                recs, el = cohort(a.base, a.model, grp, out, a.warmup_tokens, a.seed, s0, "preflight"); man["requests"] += recs; man["submission"].append({"phase": "preflight", "cohort": name, "slots": [x["slot"] for x in recs], "elapsed_s": el})
        bad = [r["slot"] for r in man["requests"] if r.get("error") or not r.get("response_id") or not any(str(t).startswith("token_id:") for t in (r.get("response_logprobs_tokens") or []))]
        for r in man["requests"]: man["phases"]["preflight"].append(r.get("response_id"))
        man["requests"] = [{k: v for k, v in r.items() if k not in ("text", "response_logprobs_token_logprobs", "response_logprobs_tokens")} | {"n_api_tokens": len([t for t in (r.get("response_logprobs_tokens") or []) if str(t).startswith("token_id:")])} for r in man["requests"]]
        man["elapsed_s_total"] = time.perf_counter() - t_all; man["invalid"] = bad or None
        json.dump(man, open(os.path.join(a.run, "e1_preflight_manifest.json"), "w"), indent=1); print("preflight segment done:", len(man["requests"]), "requests; invalid", bad or None); sys.exit(6 if bad else 0)
    pre = json.load(open(os.path.join(a.run, "e1_preflight_manifest.json"))) if (a.phase == "main" and os.path.exists(os.path.join(a.run, "e1_preflight_manifest.json"))) else None
    if pre: man["phases"]["preflight"] = pre["phases"]["preflight"]; man["submission"] += pre["submission"]
    if a.batch == 1:
        for i in range(2):
            r = {}; one(a.base, a.model, pil[0], out, a.warmup_tokens, a.seed, f"w{i}", r, "warmup"); rec = r[f"w{i}"]; man["requests"].append(rec); man["submission"].append({"phase": "warmup", "slots": [f"w{i}"], "utc": rec["request_start_utc"]})
        for i, p in enumerate(pil):
            r = {}; one(a.base, a.model, p, out, a.timed_tokens, a.seed, f"t{i}", r, "timed"); rec = r[f"t{i}"]; man["requests"].append(rec); man["submission"].append({"phase": "timed", "slots": [f"t{i}"], "utc": rec["request_start_utc"]})
    else:
        recs, el = cohort(a.base, a.model, pil[:4], out, a.warmup_tokens, a.seed, "w", "warmup"); man["requests"] += recs; man["submission"].append({"phase": "warmup", "cohort": "warm(1-4)", "slots": [x["slot"] for x in recs], "elapsed_s": el})
        for name, grp, s0 in (("A(1-4)", pil[:4], "ta"), ("B(5-8)", pil[4:], "tb")):
            recs, el = cohort(a.base, a.model, grp, out, a.timed_tokens, a.seed, s0, "timed"); man["requests"] += recs; man["submission"].append({"phase": "timed", "cohort": name, "slots": [x["slot"] for x in recs], "elapsed_s": el})
    bad = [r["slot"] for r in man["requests"] if r.get("error") or not r.get("response_id") or not any(str(t).startswith("token_id:") for t in (r.get("response_logprobs_tokens") or []))]
    for r in man["requests"]: man["phases"][r["phase"]].append(r.get("response_id"))
    man["requests"] = [{k: v for k, v in r.items() if k not in ("text", "response_logprobs_token_logprobs", "response_logprobs_tokens")} | {"n_api_tokens": len([t for t in (r.get("response_logprobs_tokens") or []) if str(t).startswith("token_id:")])} for r in man["requests"]]
    if pre: man["requests"] = pre["requests"] + man["requests"]; man["preflight_elapsed_s"] = pre.get("elapsed_s_total")
    man["elapsed_s_total"] = time.perf_counter() - t_all; man["invalid"] = bad or None
    json.dump(man, open(os.path.join(a.run, "e1_manifest.json"), "w"), indent=1)
    json.dump({"batch": a.batch, "n_requests": len(man["requests"]), "preflight": len(man["phases"]["preflight"]), "warmup": len(man["phases"]["warmup"]), "timed": len(man["phases"]["timed"]), "invalid": bad or None, "finish": {r["slot"]: r.get("finish_reason") for r in man["requests"]}, "api_tokens": {r["slot"]: r["n_api_tokens"] for r in man["requests"]}}, open(os.path.join(a.run, "workload_summary.json"), "w"), indent=1)
    print("workload done:", "batch", a.batch, "requests", len(man["requests"]), "invalid", bad or None)
    sys.exit(6 if bad else 0)
if __name__ == "__main__":
    main()
