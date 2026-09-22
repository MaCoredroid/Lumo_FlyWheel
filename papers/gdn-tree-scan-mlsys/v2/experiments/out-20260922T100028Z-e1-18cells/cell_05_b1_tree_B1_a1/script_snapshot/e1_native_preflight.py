#!/usr/bin/env python3
"""E1 NATIVE UNTIMED PREFLIGHT v2 (pass/fail ONLY against the FROZEN rules of e1/E1_FREEZE.md; no threshold tuning).
Two modes on the cell's own artifacts:
  --live  (inside the boot, BEFORE any warm-up/timed request): evaluates the frozen preflight segment
          (e1_preflight_manifest.json; B1: the 8 pilot prompts sequentially, B4: cohorts 1-4 and 5-8; 32 tokens each) on
          docker_inspect.json (written at boot), the LIVE recorder ledger PREFIX (no seal is claimed: contiguous `n`
          counters, no error events, output rows for every preflight request) and the preflight API responses;
  --final (after shutdown): the same rules on the complete sealed ledger plus the seal check.
Rules: P1 route/config (SPEC_CONFIG num_speculative_tokens == requested, mtp method, no speculative_token_tree;
  FR10_DECODE_MODE_DEFAULT == naive_mtp, FR10_ENABLE_TREE_GDN == 0; Cmd has --no-async-scheduling, --enforce-eager,
  --no-enable-prefix-caching, --max-num-seqs N, FLASH_ATTN; no 'ENGAGED' line in the server log so far);
  P2 occupancy (active sets never exceed N; every preflight request (B1) / cohort (B4) has >= 1 pure physical step with
  exactly its declared active set); P3 complete output + termination for every preflight request (finish in {length,stop},
  >= 1 token id, finish == length -> n == max_tokens); P4 greedy decisions on matched prefixes: the FIRST API token id
  (decoded with the checkpoint tokenizer) equals the unpatched native reference `native_top1` (E7a native-score boot) for
  every preflight prefix whose reference top-1/top-2 margin is >= 0.5 nats (FROZEN from the reference scoring, before any
  native E1 data: p015 6.75, p021 3.62, p058 2.87, p083 3.12 qualify; p072 0.38, p017 0.00 (a tie), p085 0.12, p095 0.12
  are reported, not gated — a tie cannot define a greedy decision); P5 ledger validity (live: prefix; final: sealed, 0
  errors, 0 sink failures). Writes <cell>/native_preflight_<mode>.json; exit 0 PASS / 1 FAIL.
Usage: e1_native_preflight.py <cell_dir> --arm ARM --batch N --nspec K (--live | --final) [--scores ...] [--tokenizer ...]"""
import argparse, json, os, sys
MARGIN_GATE_NATS = 0.5
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cell"); ap.add_argument("--arm", required=True); ap.add_argument("--batch", type=int, required=True); ap.add_argument("--nspec", type=int, required=True)
    g = ap.add_mutually_exclusive_group(required=True); g.add_argument("--live", action="store_true"); g.add_argument("--final", action="store_true")
    ap.add_argument("--scores", required=True, help="F2: the SNAPSHOTTED native reference prefix_scores.json (campaign_snapshot/prefix_scores.json); no live default"); ap.add_argument("--tokenizer", default="/models/qwen3.6-27b-fp8"); a = ap.parse_args()
    import hashlib
    mode = "live" if a.live else "final"; R = {"cell": a.cell, "arm": a.arm, "batch": a.batch, "mode": mode, "margin_gate_nats": MARGIN_GATE_NATS, "reference_scores": {"path": a.scores, "sha256": hashlib.sha256(open(a.scores, "rb").read()).hexdigest()}, "checks": []}
    def chk(n, ok, d=None): R["checks"].append({"check": n, "pass": bool(ok), "detail": d}); print(("PASS " if ok else "FAIL ") + n, "" if d is None else str(d)[:220])
    ins = json.load(open(os.path.join(a.cell, "docker_inspect.json")))[0]; cmd = " ".join(ins["Config"]["Cmd"]); env = dict(e.split("=", 1) for e in ins["Config"]["Env"] if "=" in e)
    sc = json.loads(env.get("SPEC_CONFIG", "{}")) if env.get("SPEC_CONFIG") else {}
    chk("P1_spec_config_requested_length_mtp_no_tree", int(sc.get("num_speculative_tokens", -1)) == a.nspec and "mtp" in str(sc.get("method", "")) and "speculative_token_tree" not in json.dumps(sc), sc)
    chk("P1_decode_mode_naive_mtp_tree_gdn_off", env.get("FR10_DECODE_MODE_DEFAULT") == "naive_mtp" and env.get("FR10_ENABLE_TREE_GDN") == "0")
    chk("P1_cmd_flags", all(x in cmd for x in ("--no-async-scheduling", "--enforce-eager", "--no-enable-prefix-caching", "FLASH_ATTN")) and (f"--max-num-seqs '{a.batch}'" in cmd or f"--max-num-seqs {a.batch}" in cmd))
    logp = os.path.join(a.cell, "docker_logs_preflight.txt" if a.live else "docker_logs.txt"); log = open(logp, errors="replace").read() if os.path.exists(logp) else ""
    chk("P1_no_fr13_engaged_in_server_log", os.path.exists(logp) and "ENGAGED" not in log)
    man = json.load(open(os.path.join(a.cell, "e1_preflight_manifest.json" if a.live else "e1_manifest.json")))
    pre = [x for x in man["requests"] if x["phase"] == "preflight"]
    ev = [json.loads(l) for l in open(os.path.join(a.cell, "logs", "e1_events.jsonl")) if l.strip()]
    ns = [e.get("n") for e in ev]; contiguous = ns == list(range(1, len(ns) + 1)); errs = [e for e in ev if e.get("event") == "error"]
    rec_ids = {r["request_id"] for e in ev if e.get("event") == "output_rows" for r in e["rows"]}
    rec_of = {}
    for x in pre:
        m = [r for r in rec_ids if r == x.get("response_id") or r.startswith(str(x.get("response_id")) + "-")]; rec_of[x["slot"]] = m[0] if len(m) == 1 else None
    ps = [e for e in ev if e.get("event") == "physical_step"]; occ = [len(p["request_ids"]) for p in ps]
    chk("P2_occupancy_never_exceeds_declared", bool(occ) and max(occ) <= a.batch, {"max_active": max(occ) if occ else None, "n_physical_steps": len(ps)})
    if a.batch == 1: chk("P2_every_preflight_request_has_a_single_active_pure_step", len(pre) == 8 and all(any(p["request_ids"] == [rec_of[x["slot"]]] for p in ps) for x in pre))
    else:
        cohorts = [set(rec_of[s] for s in sub["slots"]) for sub in man["submission"] if sub["phase"] == "preflight"]
        chk("P2_every_preflight_cohort_has_a_full_four_active_pure_step", len(cohorts) == 2 and all(any(set(p["request_ids"]) == c for p in ps) for c in cohorts))
    bad = [x["slot"] for x in pre if x.get("finish_reason") not in ("length", "stop") or x.get("n_api_tokens", 0) < 1 or (x.get("finish_reason") == "length" and x.get("n_api_tokens") != x.get("max_tokens"))]
    chk("P3_complete_output_and_termination_every_preflight_request", len(pre) == 8 and not bad, {"bad_slots": bad, "finish": {x["slot"]: x.get("finish_reason") for x in pre}})
    try:
        from transformers import AutoTokenizer  # type: ignore
        tk = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True); ref = {p["id"]: p for p in json.load(open(a.scores))["prefixes"]}
        gated = []; reported = []
        for x in pre:
            cr = json.load(open(os.path.join(a.cell, "cohort", f"req_{x['slot']}_{x['prefix_id']}", "capture_request.json"))); toks = cr.get("response_logprobs_tokens") or []
            ids = [int(t[9:]) for t in toks if isinstance(t, str) and t.startswith("token_id:")]; first = tk.decode([ids[0]]) if ids else None
            rf = ref.get(x["prefix_id"], {}); want = rf.get("native_top1"); mg = rf.get("native_margin_nats"); item = {"slot": x["slot"], "prefix": x["prefix_id"], "first": first, "reference_native_top1": want, "reference_margin_nats": mg, "equal": first == want}
            (gated if (mg is not None and mg >= MARGIN_GATE_NATS) else reported).append(item)
        chk("P4_first_greedy_token_equals_unpatched_native_reference_on_margin_gated_prefixes", len(gated) >= 4 and all(i["equal"] for i in gated), {"gated": gated, "reported_low_margin_not_gated": reported})
    except Exception as e:  # noqa: BLE001
        chk("P4_first_greedy_token_equals_unpatched_native_reference_on_margin_gated_prefixes", False, repr(e))
    if a.live: chk("P5_live_ledger_prefix_valid_no_seal_claimed", contiguous and not errs and all(v is not None for v in rec_of.values()) and len(ev) > 0, {"n_events": len(ev), "unmapped_slots": [k for k, v in rec_of.items() if v is None]})
    else:
        seal = [e for e in ev if e.get("event") == "run_close"]
        chk("P5_final_ledger_sealed_no_errors_rows_for_every_request", contiguous and bool(seal) and not errs and not seal[-1].get("sink_failed") and all(v is not None for v in rec_of.values()), {"n_events": len(ev), "seal": (seal[-1] if seal else None)})
    R["all_pass"] = all(c["pass"] for c in R["checks"]); R["verdict"] = f"NATIVE PREFLIGHT ({mode}) PASS" if R["all_pass"] else f"NATIVE PREFLIGHT ({mode}) FAIL (cell invalid; arm x batch stays unqualified)"
    json.dump(R, open(os.path.join(a.cell, f"native_preflight_{mode}.json"), "w"), indent=1); print(R["verdict"]); sys.exit(0 if R["all_pass"] else 1)
if __name__ == "__main__":
    main()
