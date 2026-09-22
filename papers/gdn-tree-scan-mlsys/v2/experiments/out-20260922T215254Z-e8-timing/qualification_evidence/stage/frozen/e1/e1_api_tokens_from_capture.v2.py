#!/usr/bin/env python3
"""API token map v2 (2026-09-22): builds the joiner's --api-tokens map {recorder request id: [API emitted token ids]}.
Domains: (a) TOKEN IDS — the response carries `token_id:<id>` logprob tokens (return_tokens_as_token_ids=True; the FIXED
contract for every E1 cell and cohort client v2); (b) DECODED STRINGS — PROVISIONAL recovery when a response carries only
decoded token strings (the B4 pilot's cohort client v1 omitted return_tokens_as_token_ids): each API string must have
EXACTLY ONE preimage id under the pinned checkpoint tokenizer's per-id decode over the FULL vocabulary; the recovered ids
are the API side (the ledger is NOT consulted for recovery). The report records the tokenizer identity, the decoded-map
hash, the domain per request and every ambiguity; any ambiguity or unmapped/absent evidence -> REFUSED (exit 2).
Usage: e1_api_tokens_from_capture.v2.py <run_dir> <events.jsonl> <out_map.json> [--tokenizer /models/qwen3.6-27b-fp8] [--report out.json]"""
import argparse, glob, hashlib, json, os, sys
def fsha(p):
    h = hashlib.sha256(); h.update(open(p, "rb").read()); return h.hexdigest()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("events"); ap.add_argument("out"); ap.add_argument("--tokenizer", default="/models/qwen3.6-27b-fp8"); ap.add_argument("--report", default=None); a = ap.parse_args()
    files = [os.path.join(a.run, "capture_request.json")] if os.path.exists(os.path.join(a.run, "capture_request.json")) else []
    files += sorted(glob.glob(os.path.join(a.run, "cohort", "req_*", "capture_request.json")))
    rep = {"domains": {}, "tokenizer": None, "refusals": []}
    def refuse(m): rep["refusals"].append(m); print("REFUSING:", m)
    inv = None
    def ensure_inv():
        nonlocal inv
        if inv is not None: return
        from transformers import AutoTokenizer  # type: ignore
        tk = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True); full = [tk.decode([i]) for i in range(len(tk))]
        inv = {}
        for i, s in enumerate(full): inv.setdefault(s, []).append(i)
        tj = os.path.join(a.tokenizer, "tokenizer.json")
        rep["tokenizer"] = {"path": a.tokenizer, "tokenizer_json_sha256": (fsha(tj) if os.path.exists(tj) else None), "vocab_size": len(tk), "decoder": "transformers AutoTokenizer.decode([id]) per id over the full vocabulary",
                            "full_decoded_map_sha256_json_list": hashlib.sha256(json.dumps(full, ensure_ascii=False).encode()).hexdigest(), "n_strings_with_multiple_preimages": sum(1 for v in inv.values() if len(v) > 1),
                            "status": "PROVISIONAL (string-domain recovery; pending independent certification that the served vLLM logprob-token formatter equals this decoder)"}
    api = {}
    for f in files:
        rec = json.load(open(f)); rid = rec.get("response_id") or rec.get("id")
        if not rid: refuse(f"capture_request without a response id: {f}"); continue
        toks = rec.get("response_logprobs_tokens") or []
        ids = [int(x[9:]) for x in toks if isinstance(x, str) and x.startswith("token_id:")]
        if ids and len(ids) == len(toks): api[str(rid)] = ids; rep["domains"][str(rid)] = "token_ids"; continue
        if not toks: refuse(f"{rid}: no API token evidence"); continue
        try: ensure_inv()
        except Exception as e:  # noqa: BLE001
            refuse(f"{rid}: tokenizer unavailable for string-domain recovery: {e!r}"); continue
        pre = [inv.get(t, []) for t in toks]; amb = [(i, toks[i], pre[i]) for i in range(len(toks)) if len(pre[i]) != 1]
        if amb: refuse(f"{rid}: {len(amb)} API strings without a unique vocabulary preimage: {amb[:3]}"); continue
        api[str(rid)] = [p[0] for p in pre]; rep["domains"][str(rid)] = "decoded_strings_recovered_unique_preimage"
    rec_ids = set()
    for l in open(a.events):
        if not l.strip(): continue
        e = json.loads(l)
        if e.get("event") == "output_rows":
            for r in e["rows"]: rec_ids.add(r["request_id"])
    out = {}
    for k, v in api.items():
        m = [r for r in rec_ids if r == k or r.startswith(k + "-")]
        if len(m) != 1: refuse(f"API id {k}: {len(m)} recorder matches {sorted(m)}"); continue
        out[m[0]] = v
    if len(set(out)) != len(api) or len(out) != len(files): refuse(f"{len(out)} mapped of {len(files)} responses")
    unmatched = [r for r in rec_ids if not any(r == k or r.startswith(k + "-") for k in api)]
    if unmatched: refuse(f"recorder requests without an API response: {sorted(unmatched)[:4]}")
    rep["mapped"] = {k: len(v) for k, v in out.items()}
    if a.report: json.dump(rep, open(a.report, "w"), indent=1)
    if rep["refusals"]: sys.exit(2)
    json.dump(out, open(a.out, "w"), indent=1); print("api-tokens map:", {k: len(v) for k, v in out.items()}, "domains:", rep["domains"])
if __name__ == "__main__":
    main()
