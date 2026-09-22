#!/usr/bin/env python3
"""Evaluate a harness run (pilot or confirmation) against the FROZEN criteria in PILOT_FREEZE.json (handoff item 2).
For every frozen tolerance class, compute the run's worst case over its VERIFIED-FRESH operand sets from the result_*.json
files (same extraction as pilot_maxima) and report PASS (<= tolerance) / FAIL, plus every pre-declared decision-change
criterion that can be evaluated mechanically (bitwise classes, non-finite cells, controls, provenance coverage, ordering).
Nothing is dropped or re-run: a FAIL is a reportable numerical finding. Usage:
  check_confirmation_vs_freeze.py <PILOT_FREEZE.json> <harness_run_dir> <frozen_prefixes.json> <out_json> [--set confirmation|pilot]"""
import argparse, glob, json, math, os, statistics as st, sys
ap = argparse.ArgumentParser(); ap.add_argument("freeze"); ap.add_argument("run"); ap.add_argument("frozen"); ap.add_argument("out"); ap.add_argument("--set", default="confirmation"); a = ap.parse_args()
F = json.load(open(a.freeze)); tol = F["criteria"]["tolerances"]
rs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(a.run, "result_*.json")))]
summ = json.load(open(os.path.join(a.run, "summary.json")))
frozen = json.load(open(a.frozen)); want_ids = [r["id"] for r in frozen[a.set]]
def col(path):
    out = []
    for r in rs:
        d = r
        for k in path: d = d[k]
        out.append(d)
    return out
def mx(vals): return max(vals) if vals else float("nan")
def mn(vals): return min(vals) if vals else float("nan")
S1 = lambda k, f="max_abs": col(("stage1_verify", k, f)); S3 = lambda k, f="max_abs": col(("stage3_own_factors_own_commit", k, f))
observed = {
 "scan_fp32_out_vs_native_specupdate_bitwise": 1.0 - mn(col(("stage1_verify", "A_prod_out32_vs_native_sg32_bitwise_frac"))),
 "scan_bf16_out_vs_native_specupdate_bitwise": 1.0 - mn(col(("stage1_verify", "A_prod_out16_vs_native_sg16_bitwise_frac"))),
 "served_bf16_out_bytes_equal_production_scan_out16": 1.0 - mn([1.0 if r["stage1_verify"].get("A_prod_out16_bytes_equal_payload_serving_out") is True else 0.0 for r in rs]),
 "fp32_ieee_realizations_out_vs_fp64_oracle_max_abs": mx(S1("A_prod_out32_vs_oracle") + S1("B_fs[ieee]_out32_vs_oracle") + S1("C_nm[ieee]_out32_vs_oracle")),
 "compact_commit_state_vs_fp64_oracle_max_abs": mx(S3("B_fs[ieee]_compact_own_vs_oracle") + S3("C_nm[ieee]_compact_own_vs_oracle")),
 "compact_commit_state_vs_native_specupdate_max_abs": mx(S3("B_fs[ieee]_compact_own_vs_native_sg") + S3("C_nm[ieee]_compact_own_vs_native_sg")),
 "replay_state_vs_fp64_oracle_max_abs": mx(S3("A_prod_replay_vs_oracle")),
 "replay_state_vs_native_specupdate_max_abs": mx(S3("A_prod_replay_vs_native_sg_state")),
 "bf16_store_realizations_disagreement_with_native (1 - bitwise frac)": 1.0 - mn(col(("stage1_verify", "B_fs[ieee]_out16_vs_native_sg16_bitwise_frac")) + col(("stage1_verify", "C_nm[ieee]_out16_vs_native_sg16_bitwise_frac"))),
 "native_specupdate_bf16_out_vs_fp64_oracle_max_abs": mx(col(("references", "native_sg_out16_vs_oracle", "max_abs"))),
}
# served bf16 out vs oracle: from the per-run CPU self-tests if present next to the payloads (paths recorded in provenance)
served_abs, served_ulp = [], []
for r in rs:
    pth = r["provenance"].get("path", ""); d = os.path.dirname(os.path.dirname(pth.replace("/work/", "/home/mark/lumo-paper-v2-20260921/", 1)))
    pid = (r["provenance"].get("request_binding") or {}).get("prefix_id")
    f = os.path.join(d, f"cpu_selftest_fresh_{pid}.json") if pid else None
    if f and os.path.exists(f):
        v = json.load(open(f))
        def find(dd, key):
            if isinstance(dd, dict):
                if key in dd: return dd[key]
                for x in dd.values():
                    y = find(x, key)
                    if y is not None: return y
        so = find(v, "served_out_bf16_vs_oracle")
        if isinstance(so, dict): served_abs.append(so["max_abs"]); served_ulp.append(so.get("ulp16_max_sig", math.inf))
observed["served_bf16_out_vs_fp64_oracle_max_abs (CPU self-test on the payload's serving_out)"] = mx(served_abs)
observed["served_bf16_out_vs_fp64_oracle_ulp16_on_significant"] = mx(served_ulp)
classes = {}
for k, t in tol.items():
    o = observed.get(k); lim = t["tolerance"]
    classes[k] = {"observed_worst": o, "tolerance": lim, "pilot_max": t.get("pilot_max"), "rule": t.get("rule"),
                  "pass": (o is not None and not (isinstance(o, float) and math.isnan(o)) and o <= lim), "n_sets": len(rs),
                  "note": None if k in observed else "class not evaluable from this run's files"}
# decision-change criteria (mechanical part)
ids = [(r["provenance"].get("request_binding") or {}).get("prefix_id") for r in rs]
statuses = [str(r["provenance"].get("status", "")) for r in rs]
dc = {
 "all_sets_verified_fresh": all(s.startswith("VERIFIED-FRESH") for s in statuses),
 "set_coverage_equals_frozen_set": sorted(x for x in ids if x) == sorted(want_ids),
 "no_duplicate_prefix": len(set(x for x in ids if x)) == len([x for x in ids if x]),
 "no_invalid_cells": summ.get("n_invalid_cells", 1) == 0,
 "sibling_reorder_bitwise_invariant_all_sets": all(r["controls"][c]["out_bitwise_frac"] == 1.0 for r in rs for c in ("A_prod_sibling_reverse", "B_fs_sibling_reverse", "C_nm_sibling_reverse")),
 "replay_repeat_bitwise_equal_all_sets": all(r["stage3_own_factors_own_commit"]["A_prod_replay_repeat_bitwise_equal_all_paths"] is True for r in rs),
 "ordering_A_le_B_eq_C_out_vs_oracle": all(r["stage1_verify"]["A_prod_out32_vs_oracle"]["max_abs"] <= r["stage1_verify"]["B_fs[ieee]_out32_vs_oracle"]["max_abs"] + 1e-12 and abs(r["stage1_verify"]["B_fs[ieee]_out32_vs_oracle"]["max_abs"] - r["stage1_verify"]["C_nm[ieee]_out32_vs_oracle"]["max_abs"]) <= 1e-12 for r in rs),
 "source_hashes_start_end_match": bool(summ.get("source_hashes_start_end_match")),
 "probe_drift_all_pass": all(v.get("pass") for v in summ.get("probe_checks", {}).values()),
}
out = {"set": a.set, "run_dir": a.run, "freeze": a.freeze, "n_sets": len(rs), "prefix_ids": ids, "classes": classes, "decision_change_checks": dc,
       "attribution": summ.get("attribution", {}).get("label"), "all_classes_pass": all(c["pass"] for c in classes.values()), "all_decision_checks_pass": all(dc.values())}
out["verdict"] = "WITHIN FROZEN CRITERIA" if (out["all_classes_pass"] and out["all_decision_checks_pass"]) else "OUTSIDE FROZEN CRITERIA (reportable finding; nothing dropped)"
json.dump(out, open(a.out, "w"), indent=2, default=str)
for k, c in classes.items(): print(f"{'PASS' if c['pass'] else 'FAIL'} {k}: observed {c['observed_worst']!r} <= tol {c['tolerance']!r}")
for k, v in dc.items(): print(f"{'PASS' if v else 'FAIL'} decision:{k}")
print("VERDICT:", out["verdict"]); sys.exit(0 if out["verdict"].startswith("WITHIN") else 1)
