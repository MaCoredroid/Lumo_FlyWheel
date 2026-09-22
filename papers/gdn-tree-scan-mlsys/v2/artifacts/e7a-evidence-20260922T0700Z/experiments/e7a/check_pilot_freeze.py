#!/usr/bin/env python3
"""Fail-closed validator for the structured pilot freeze record (handoff item 2; review 10).
PASS (rc 0) only if ALL hold:
  status == "FROZEN" with an ISO frozen_utc;
  pilot_evidence has exactly the 8 frozen pilot ids (unique, same set as frozen_prefixes.json["pilot"]), each with a run
  dir that exists and whose capture_provenance.json (a) hashes to the recorded provenance_sha256, (b) has all_pass true and
  no unverified/failed checks, (c) records payload_sha256 == sha256(logs/tree_gdn_capture_payload.pt) == recorded payload
  sha, (d) records the recorded inspector_sha256, (e) names the same prefix id; and every recorded inspector sha is the
  same single version;
  pilot_harness_summary: path exists, sha256 matches, and its operand_sets list the 8 payload shas as VERIFIED-FRESH with
  n_verified_fresh == 8 and no invalid cells;
  criteria: candidate_selection.candidates non-empty with a rationale; tolerances non-empty, every class has FINITE
  non-negative pilot_max and tolerance (>= pilot_max) and either tolerance == pilot_max*margin exactly (margin >= 1) or an
  explicit per-class rule string (the pinned bitwise class: 0 / 0 / "0 ULP integer view"); unacceptable_decision_changes
  non-empty; timing_precision_target finite relative_half_width_max (0 < x <= 0.5) and probe_drift_max (0 < x <= 0.05)
  with a basis; cost_variance finite non-negative with finite metric_variance leaves; resampling_unit mentions "prefix".
  Infinity/NaN anywhere numeric is rejected (review 11).
Usage: check_pilot_freeze.py <PILOT_FREEZE.json> <frozen_prefixes.json> [--json out]"""
import argparse, datetime as dt, hashlib, json, os, sys

def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()

def validate(freeze_p, frozen_p):
    out = {"freeze_path": freeze_p, "freeze_sha256": sha(freeze_p) if os.path.exists(freeze_p) else None, "checks": {}}
    def check(n, ok, d=None): out["checks"][n] = {"pass": bool(ok), "detail": d}
    try:
        f = json.load(open(freeze_p))
    except Exception as exc:
        check("freeze_record_parses", False, repr(exc)[:200]); out["all_pass"] = False; return out
    check("freeze_record_parses", True); check("schema", f.get("schema") == "e7a.pilot_freeze.v1", f.get("schema"))
    check("status_frozen", f.get("status") == "FROZEN", f.get("status"))
    try:
        dt.datetime.fromisoformat(str(f.get("frozen_utc")).replace("Z", "+00:00")); check("frozen_utc_iso", True, f.get("frozen_utc"))
    except Exception:
        check("frozen_utc_iso", False, f.get("frozen_utc"))
    fr = json.load(open(frozen_p)); pilot_ids = [r["id"] for r in fr["pilot"]]
    ev = f.get("pilot_evidence") or []
    ids = [e.get("prefix_id") for e in ev]
    check("eight_unique_pilot_ids_match_frozen_set", len(ev) == 8 and len(set(ids)) == 8 and set(ids) == set(pilot_ids), {"ids": ids, "frozen": pilot_ids})
    insp = set(); shas = []
    for e in ev:
        pid = e.get("prefix_id"); d = e.get("run_dir") or ""
        prov_p = os.path.join(d, "capture_provenance.json"); pay_p = os.path.join(d, "logs", "tree_gdn_capture_payload.pt")
        if not (os.path.isdir(d) and os.path.exists(prov_p) and os.path.exists(pay_p)):
            check(f"evidence_{pid}", False, {"run_dir": d, "reason": "run dir / provenance / payload missing"}); continue
        pv = json.load(open(prov_p)); ph = sha(pay_p)
        ok = (sha(prov_p) == e.get("provenance_sha256") and pv.get("all_pass") is True and not pv.get("unverified") and not pv.get("failed")
              and pv.get("payload_sha256") == ph == e.get("payload_sha256") and pv.get("inspector_sha256") == e.get("inspector_sha256")
              and ((pv.get("request") or {}).get("prefix_id") == pid))
        check(f"evidence_{pid}", ok, {"provenance_sha_match": sha(prov_p) == e.get("provenance_sha256"), "all_pass": pv.get("all_pass"), "payload_sha_match": pv.get("payload_sha256") == ph == e.get("payload_sha256"),
                                  "inspector_match": pv.get("inspector_sha256") == e.get("inspector_sha256"), "prefix_match": (pv.get("request") or {}).get("prefix_id") == pid})
        insp.add(e.get("inspector_sha256")); shas.append(ph)
    check("single_inspector_version_across_pilot", len(insp) == 1 and None not in insp, sorted(str(x)[:16] for x in insp))
    hs = f.get("pilot_harness_summary") or {}
    sp = hs.get("path")
    if sp and os.path.exists(sp):
        ok_sha = sha(sp) == hs.get("sha256")
        try:
            sm = json.load(open(sp)); sets = sm.get("operand_sets") or []
            fresh = {s_.get("payload_sha256") for s_ in sets if str(s_.get("status", "")).startswith("VERIFIED-FRESH")}
            ok_cov = len(shas) == 8 and set(shas) <= fresh and sm.get("n_verified_fresh") == 8 and sm.get("n_invalid_cells") == 0
            check("harness_summary_hash_bound_and_covers_pilot", ok_sha and ok_cov, {"sha_match": ok_sha, "fresh_in_summary": len(fresh), "pilot_covered": len(set(shas) & fresh), "n_verified_fresh": sm.get("n_verified_fresh"), "n_invalid_cells": sm.get("n_invalid_cells")})
        except Exception as exc:
            check("harness_summary_hash_bound_and_covers_pilot", False, repr(exc)[:200])
    else:
        check("harness_summary_hash_bound_and_covers_pilot", False, {"path": sp, "reason": "missing"})
    c = f.get("criteria") or {}
    cs = c.get("candidate_selection") or {}
    check("candidate_selection_filled", isinstance(cs.get("candidates"), list) and len(cs["candidates"]) > 0 and bool(cs.get("rationale")), cs)
    tol = c.get("tolerances") or {}
    import math
    def num(x): return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)   # finite only (review 11)
    def nonneg(x): return num(x) and x >= 0
    def finite_tree(x):  # every numeric leaf finite; non-numeric leaves rejected
        if isinstance(x, dict): return len(x) > 0 and all(finite_tree(v) for v in x.values())
        if isinstance(x, list): return len(x) > 0 and all(finite_tree(v) for v in x)
        return num(x)
    def tol_class_ok(v):
        # errors are non-negative and finite; tolerance is either EXACTLY pilot_max*margin (margin >= 1) or governed by an
        # explicit per-class rule string (e.g. the pinned bitwise class: pilot_max 0, tolerance 0, rule "0 ULP integer view")
        if not (isinstance(v, dict) and nonneg(v.get("pilot_max")) and nonneg(v.get("tolerance")) and v["tolerance"] >= v["pilot_max"]):
            return False
        rule = v.get("rule")
        if isinstance(rule, str) and rule.strip():
            return True
        return num(v.get("margin")) and v["margin"] >= 1 and math.isclose(v["tolerance"], v["pilot_max"] * v["margin"], rel_tol=1e-9, abs_tol=0.0)
    tol_ok = bool(tol) and all(tol_class_ok(v) for v in tol.values())
    check("tolerances_filled_finite_nonneg_rule_enforced", tol_ok, {k: v for k, v in list(tol.items())[:6]})
    udc = c.get("unacceptable_decision_changes") or []
    check("decision_change_criteria_filled", isinstance(udc, list) and len(udc) > 0 and all(isinstance(x, str) and x.strip() for x in udc), udc[:5])
    tp = c.get("timing_precision_target") or {}
    check("timing_precision_filled", num(tp.get("relative_half_width_max")) and 0 < tp["relative_half_width_max"] <= 0.5 and num(tp.get("probe_drift_max")) and 0 < tp["probe_drift_max"] <= 0.05 and bool(tp.get("basis")), tp)
    cv = c.get("cost_variance") or {}
    check("cost_variance_filled_finite_nonneg", bool(cv) and all(nonneg(cv.get(k)) for k in ("boot_health_s", "request_s", "driver_cycle_s", "payload_bytes")) and finite_tree(cv.get("metric_variance")),
          {k: cv.get(k) for k in ("boot_health_s", "request_s", "driver_cycle_s", "payload_bytes")})
    check("resampling_unit_is_prefix", "prefix" in str(f.get("resampling_unit", "")).lower(), f.get("resampling_unit"))
    out["all_pass"] = all(v["pass"] for v in out["checks"].values())
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("freeze"); ap.add_argument("frozen"); ap.add_argument("--json"); a = ap.parse_args()
    r = validate(a.freeze, a.frozen)
    if a.json: json.dump(r, open(a.json, "w"), indent=2)
    print(json.dumps({k: v["pass"] for k, v in r["checks"].items()}, indent=1)); print("PILOT FREEZE:", "VALID" if r["all_pass"] else "INVALID")
    sys.exit(0 if r["all_pass"] else 40)
