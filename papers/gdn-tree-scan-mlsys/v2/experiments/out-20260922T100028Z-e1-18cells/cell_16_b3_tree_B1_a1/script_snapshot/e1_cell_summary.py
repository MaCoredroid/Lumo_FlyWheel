#!/usr/bin/env python3
"""E1 cell post-processing (frozen rules, e1/E1_FREEZE.md): API token map (token-id domain ONLY for cells), joiner with the
phase manifest and the declared occupancy, then per-prompt (B1) / per-cohort (B4) accounting and the STRUCTURAL COVERAGE
FLOORS (B1: all 8 prompts >= 1 retained interval; B4: both cohorts >= 1 complete four-request interval). Writes
<cell>/cell_result.preliminary.json with status VALID / INVALID (instrumentation loss, refused map, join invalid) /
INSUFFICIENT_SUPPORT (floor missed; EOS/budget outcome retained; no replacement). The TERMINAL cell_result.json (with
terminal_seal) is written only by the runner after all gates pass (F1). Never selects or trims: the primary rate keeps every valid
retained interval; the >1.5 s count/durations and any trimmed rate are diagnostics only.
Usage: e1_cell_summary.py <cell_dir> --batch {1,4} [--snapshot DIR]"""
import argparse, json, os, subprocess, sys
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cell"); ap.add_argument("--batch", type=int, required=True); ap.add_argument("--snapshot", default=None); a = ap.parse_args()
    S = a.snapshot or os.path.join(a.cell, "script_snapshot"); R = {"cell": a.cell, "batch": a.batch, "status": None, "reasons": []}
    ev = os.path.join(a.cell, "logs", "e1_events.jsonl"); man_p = os.path.join(a.cell, "e1_manifest.json")
    if not os.path.exists(ev): R["status"] = "INVALID"; R["reasons"].append("no recorder ledger"); return finish(a, R)
    if not os.path.exists(man_p): R["status"] = "INVALID"; R["reasons"].append("no workload manifest"); return finish(a, R)
    man = json.load(open(man_p))
    if man.get("invalid"): R["status"] = "INVALID"; R["reasons"].append(f"workload invalid requests {man['invalid']}"); return finish(a, R)
    r = subprocess.run([sys.executable, os.path.join(S, "e1_api_tokens_from_capture.v2.py"), a.cell, ev, os.path.join(a.cell, "e1_api_tokens.json"), "--report", os.path.join(a.cell, "e1_api_tokens.report.json")], capture_output=True, text=True)
    open(os.path.join(a.cell, "e1_api_tokens.log"), "w").write(r.stdout + r.stderr)
    rep = json.load(open(os.path.join(a.cell, "e1_api_tokens.report.json"))) if os.path.exists(os.path.join(a.cell, "e1_api_tokens.report.json")) else {}
    if r.returncode != 0: R["status"] = "INVALID"; R["reasons"].append("API token map refused: " + "; ".join(rep.get("refusals", [])[:3])); return finish(a, R)
    if any(v != "token_ids" for v in rep.get("domains", {}).values()): R["status"] = "INVALID"; R["reasons"].append("API token evidence not in the token-id domain (cells require return_tokens_as_token_ids)"); return finish(a, R)
    # translate EVERY phase's API response ids to engine (recorder) ids with the SAME unique rule as the token map (exact or
    # unique 'api-' prefix); a missing or ambiguous translation invalidates the cell
    amap0 = json.load(open(os.path.join(a.cell, "e1_api_tokens.json"))); rec_ids = list(amap0)
    def to_engine(api_id):
        m = [r for r in rec_ids if r == api_id or r.startswith(str(api_id) + "-")]
        return m[0] if len(m) == 1 else None
    phases_e = {"warmup": [], "timed": []}; untranslated = []
    for ph, ids in man["phases"].items():
        tgt = "timed" if ph == "timed" else "warmup"   # preflight + warm-up = the joiner's prespecified 'warmup' exclusion
        for api_id in ids:
            e_id = to_engine(api_id)
            if e_id is None: untranslated.append((ph, api_id))
            else: phases_e[tgt].append(e_id)
    if untranslated: R["status"] = "INVALID"; R["reasons"].append(f"phase requests without a unique engine id: {untranslated[:4]}"); return finish(a, R)
    jm = {"phases": phases_e, "cell_phases_api_ids": man["phases"], "api_to_engine_rule": "exact or unique '<api id>-' prefix (same as the token map)", "note": "preflight and warm-up requests are the joiner's prespecified 'warmup' exclusion"}; json.dump(jm, open(os.path.join(a.cell, "e1_join_manifest.json"), "w"))
    r = subprocess.run([sys.executable, os.path.join(S, "e1_join.py"), ev, "--api-tokens", os.path.join(a.cell, "e1_api_tokens.json"), "--manifest", os.path.join(a.cell, "e1_join_manifest.json"), "--expect-reqs", str(a.batch), "--json", os.path.join(a.cell, "e1_join.json")], capture_output=True, text=True)
    open(os.path.join(a.cell, "e1_join.log"), "w").write(r.stdout + r.stderr)
    if not os.path.exists(os.path.join(a.cell, "e1_join.json")): R["status"] = "INVALID"; R["reasons"].append(f"joiner produced no report (rc {r.returncode})"); return finish(a, R)
    J = json.load(open(os.path.join(a.cell, "e1_join.json"))); R["join"] = {k: v for k, v in J.items() if k not in ("steps", "excluded")}
    if J.get("invalid") or J.get("refused"): R["status"] = "INVALID"; R["reasons"].append(f"join invalid/refused: {J.get('invalid') or J.get('refused')}"); return finish(a, R)
    if r.returncode == 3 and not (J.get("steps") or []):
        # launch red-team C6: a COMPLETE, reconciled run with zero usable intervals (EOS/budget outcome) is INSUFFICIENT SUPPORT, not
        # instrumentation loss; the EOS/budget outcome is retained and the campaign continues
        R["status"] = "INSUFFICIENT_SUPPORT"; R["reasons"].append("zero retained intervals with complete API reconciliation (joiner rc 3; EOS/budget outcome retained)")
        R["floor"] = {"rule": "coverage floor", "missing": "all units (zero retained intervals)", "pass": False}; R["api_totals"] = {x["slot"]: x.get("n_api_tokens") for x in man["requests"]}; R["finish"] = {x["slot"]: x.get("finish_reason") for x in man["requests"]}
        R["primary"] = {"tokens_per_wall_second": None, "n_usable": 0}; return finish(a, R)
    if r.returncode != 0: R["status"] = "INVALID"; R["reasons"].append(f"joiner rc {r.returncode}"); return finish(a, R)
    # per-prompt / per-cohort accounting from the joiner's retained steps (each retained step carries its active request ids and bound tokens)
    steps = J.get("steps") or []; ret = [dict(s, request_ids=[x["request_id"] for x in s["rows"]], tokens=sum(len(x["emitted_ids"]) for x in s["rows"])) for s in steps]   # joiner: `steps` = the USABLE (retained) physical intervals; tokens here = ledger emitted counts (the primary uses API-bound counts)
    req_prompt = {x.get("response_id"): (x["prefix_id"], x["phase"], x["slot"]) for x in man["requests"]}
    amap = json.load(open(os.path.join(a.cell, "e1_api_tokens.json")))   # recorder id -> ids
    api2rec = {}
    for rec_id in amap:
        for api_id in req_prompt:
            if rec_id == api_id or rec_id.startswith(api_id + "-"): api2rec[api_id] = rec_id
    rec2prompt = {api2rec[k]: v for k, v in req_prompt.items() if k in api2rec}
    per = {}
    if a.batch == 1:
        for s in ret:
            rid = s["request_ids"][0]; pid, ph, slot = rec2prompt.get(rid, ("?", "?", "?"))
            if ph != "timed": continue
            d = per.setdefault(pid, {"intervals": 0, "wall_s": 0.0, "tokens": 0}); d["intervals"] += 1; d["wall_s"] += s["wall_s"]; d["tokens"] += s.get("tokens", 0)
        prompts = [x["prefix_id"] for x in man["requests"] if x["phase"] == "timed"]
        missing = [p for p in prompts if per.get(p, {}).get("intervals", 0) < 1]
        R["per_prompt"] = per; R["floor"] = {"rule": "every one of the 8 timed prompts >= 1 retained interval", "prompts": prompts, "missing": missing, "pass": (len(prompts) == 8 and not missing)}
    else:
        cohorts = {}
        for sub in man["submission"]:
            if sub["phase"] == "timed": cohorts[sub["cohort"]] = set(api2rec.get(x.get("response_id")) for x in man["requests"] if x["slot"] in sub["slots"])
        for s in ret:
            act = set(s["request_ids"]); name = next((n for n, c in cohorts.items() if c == act), None)
            if name is None: continue
            d = per.setdefault(name, {"intervals": 0, "wall_s": 0.0, "tokens": 0}); d["intervals"] += 1; d["wall_s"] += s["wall_s"]; d["tokens"] += s.get("tokens", 0)
        missing = [n for n in cohorts if per.get(n, {}).get("intervals", 0) < 1]
        R["per_cohort"] = per; R["floor"] = {"rule": "each of the 2 timed cohorts >= 1 complete four-request same-cohort interval", "cohorts": sorted(cohorts), "missing": missing, "pass": (len(cohorts) == 2 and not missing)}
    R["api_totals"] = {req_prompt.get(k, ("?",))[0] + "/" + req_prompt.get(k, ("?", "?", "?"))[2]: n for k, n in ((x.get("response_id"), x.get("n_api_tokens")) for x in man["requests"])}
    R["finish"] = {x["slot"]: x.get("finish_reason") for x in man["requests"]}
    R["primary"] = {"tokens_per_wall_second": J.get("tokens_per_wall_second"), "sum_wall_s_unique_physical_steps": J.get("sum_wall_s_unique_physical_steps"), "sum_emitted_tokens_api_bound_pure_support": J.get("sum_emitted_tokens_api_bound_pure_support"), "n_usable": J.get("n_usable"), "n_excluded": J.get("n_excluded")}
    R["diagnostic_over_cap"] = J.get("over_cap_diagnostic")
    R["status"] = "VALID" if R["floor"]["pass"] else "INSUFFICIENT_SUPPORT"
    if not R["floor"]["pass"]: R["reasons"].append(f"coverage floor missed: present {len(R['floor'].get('prompts', R['floor'].get('cohorts', [])))} units, zero-interval units {R['floor']['missing']}")
    return finish(a, R)
def finish(a, R):
    # F1 (final reviewer recheck 2026-09-22): this is a PRELIMINARY summary. The durable terminal result `cell_result.json`
    # (with `terminal_seal`) is written ONLY by the campaign runner after every gate of the cell has passed (verify, this
    # summary, and — for a native preflight boot — the FINAL sealed-ledger audit). A cell without a terminal seal is a failed
    # attempt for the aggregate and for resume, whatever this preliminary file says.
    R["preliminary"] = True; R["note"] = "PRELIMINARY — not eligible until the runner writes cell_result.json with terminal_seal"
    json.dump(R, open(os.path.join(a.cell, "cell_result.preliminary.json"), "w"), indent=1, default=str); print("cell_result (PRELIMINARY):", R["status"], R["reasons"], (R.get("primary") or {}).get("tokens_per_wall_second")); return 0 if R["status"] in ("VALID", "INSUFFICIENT_SUPPORT") else 1
if __name__ == "__main__":
    sys.exit(main())
