#!/usr/bin/env python3
"""RETROSPECTIVE COMPLETENESS AUDIT of a harness run against EVERY machine-evaluable rule declared in PILOT_FREEZE.json
(review 14: the v1 checker omitted the padding-control bitwise rule, the compact-state B==C ordering rule and the
verify-kernel timing half-width target). This is a re-analysis of EXISTING data; it does not change the frozen record, the
thresholds, or any earlier verdict file, and it does not imply a corrected preregistration.
Rules: T* = frozen tolerance classes (worst case over sets AND per set); D* = pre-declared unacceptable decision changes;
P* = timing precision target (half-width = (p90-p10)/(2*median) per set, VERIFY-class kernels gated, COMMIT-class reported;
probe drift per set); I* = harness integrity. Strict equality where the freeze says so. Missing data or a non-finite value
FAILS CLOSED (rule reported as FAIL with reason 'missing'/'non-finite').
Usage: check_confirmation_vs_freeze.v2.py <PILOT_FREEZE.json> <harness_run_dir> <frozen_prefixes.json> <out_json>
       [--set confirmation|pilot] [--capture-roots r1,r2,...]"""
import argparse, glob, hashlib, json, math, os, sys
ap = argparse.ArgumentParser(); ap.add_argument("freeze"); ap.add_argument("run"); ap.add_argument("frozen"); ap.add_argument("out")
ap.add_argument("--set", default="confirmation"); ap.add_argument("--capture-roots", default=""); a = ap.parse_args()
F = json.load(open(a.freeze)); tol = F["criteria"]["tolerances"]; TP = F["criteria"]["timing_precision_target"]
freeze_sha = hashlib.sha256(open(a.freeze, "rb").read()).hexdigest()
files = sorted(glob.glob(os.path.join(a.run, "result_*.json"))); rs = [json.load(open(f)) for f in files]
summ = json.load(open(os.path.join(a.run, "summary.json"))) if os.path.exists(os.path.join(a.run, "summary.json")) else {}
frozen = json.load(open(a.frozen)); want_ids = [r["id"] for r in frozen[a.set]]
ids = [((r.get("provenance") or {}).get("request_binding") or {}).get("prefix_id") for r in rs]
MISSING = object()
def get(d, path):
    for k in path:
        if not isinstance(d, dict) or k not in d: return MISSING
        d = d[k]
    return d
def fin(x): return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
def per_set(path):
    out = []
    for pid, r in zip(ids, rs):
        v = get(r, path); out.append((pid, v))
    return out
def rule(rid, text, rows, ok_fn, *, extra=None):
    """rows: list of (pid, value-or-MISSING). ok_fn(value)->bool. Missing/non-finite -> FAIL closed."""
    per = []; allok = True
    for pid, v in rows:
        if v is MISSING: per.append({"prefix": pid, "value": None, "pass": False, "reason": "missing"}); allok = False; continue
        if isinstance(v, float) and not math.isfinite(v): per.append({"prefix": pid, "value": v, "pass": False, "reason": "non-finite"}); allok = False; continue
        ok = bool(ok_fn(v)); per.append({"prefix": pid, "value": v, "pass": ok}); allok = allok and ok
    rec = {"id": rid, "rule": text, "pass": allok and len(rows) > 0, "n_sets": len(rows), "failing_prefixes": [p["prefix"] for p in per if not p["pass"]], "per_set": per}
    if len(rows) == 0: rec["reason"] = "no sets"
    if extra: rec.update(extra)
    return rec
R = []
# ---------------- T: frozen tolerance classes (worst case over sets, and per set)
S1 = lambda k, f="max_abs": ("stage1_verify", k, f); S3 = lambda k, f="max_abs": ("stage3_own_factors_own_commit", k, f)
def maxrows(paths):
    rows = []
    for pid, r in zip(ids, rs):
        vals = [get(r, p) for p in paths]
        if any(v is MISSING for v in vals): rows.append((pid, MISSING)); continue
        if any(not fin(v) for v in vals): rows.append((pid, float("nan"))); continue
        rows.append((pid, max(vals)))
    return rows
def one_minus_min(paths):
    rows = maxrows([]) if False else []
    for pid, r in zip(ids, rs):
        vals = [get(r, p) for p in paths]
        if any(v is MISSING for v in vals): rows.append((pid, MISSING)); continue
        if any(not fin(v) for v in vals): rows.append((pid, float("nan"))); continue
        rows.append((pid, 1.0 - min(vals)))
    return rows
served_rows_abs, served_rows_ulp = [], []
for pid, r in zip(ids, rs):
    pth = (r.get("provenance") or {}).get("path", ""); d = os.path.dirname(os.path.dirname(pth.replace("/work/", "/home/mark/lumo-paper-v2-20260921/", 1)))
    f = os.path.join(d, f"cpu_selftest_fresh_{pid}.json") if pid else None
    so = MISSING
    if f and os.path.exists(f):
        v = json.load(open(f))
        def find(dd, key):
            if isinstance(dd, dict):
                if key in dd: return dd[key]
                for x in dd.values():
                    y = find(x, key)
                    if y is not None: return y
        so = find(v, "served_out_bf16_vs_oracle") or MISSING
    served_rows_abs.append((pid, so["max_abs"] if so is not MISSING else MISSING)); served_rows_ulp.append((pid, so.get("ulp16_max_sig", MISSING) if so is not MISSING else MISSING))
TROWS = {
 "scan_fp32_out_vs_native_specupdate_bitwise": one_minus_min([("stage1_verify", "A_prod_out32_vs_native_sg32_bitwise_frac")]),
 "scan_bf16_out_vs_native_specupdate_bitwise": one_minus_min([("stage1_verify", "A_prod_out16_vs_native_sg16_bitwise_frac")]),
 "served_bf16_out_bytes_equal_production_scan_out16": [(pid, (0.0 if get(r, ("stage1_verify", "A_prod_out16_bytes_equal_payload_serving_out")) is True else 1.0) if get(r, ("stage1_verify", "A_prod_out16_bytes_equal_payload_serving_out")) is not MISSING else MISSING) for pid, r in zip(ids, rs)],
 "fp32_ieee_realizations_out_vs_fp64_oracle_max_abs": maxrows([S1("A_prod_out32_vs_oracle"), S1("B_fs[ieee]_out32_vs_oracle"), S1("C_nm[ieee]_out32_vs_oracle")]),
 "compact_commit_state_vs_fp64_oracle_max_abs": maxrows([S3("B_fs[ieee]_compact_own_vs_oracle"), S3("C_nm[ieee]_compact_own_vs_oracle")]),
 "compact_commit_state_vs_native_specupdate_max_abs": maxrows([S3("B_fs[ieee]_compact_own_vs_native_sg"), S3("C_nm[ieee]_compact_own_vs_native_sg")]),
 "replay_state_vs_fp64_oracle_max_abs": maxrows([S3("A_prod_replay_vs_oracle")]),
 "replay_state_vs_native_specupdate_max_abs": maxrows([S3("A_prod_replay_vs_native_sg_state")]),
 "bf16_store_realizations_disagreement_with_native (1 - bitwise frac)": one_minus_min([("stage1_verify", "B_fs[ieee]_out16_vs_native_sg16_bitwise_frac"), ("stage1_verify", "C_nm[ieee]_out16_vs_native_sg16_bitwise_frac")]),
 "native_specupdate_bf16_out_vs_fp64_oracle_max_abs": maxrows([("references", "native_sg_out16_vs_oracle", "max_abs")]),
 "served_bf16_out_vs_fp64_oracle_max_abs (CPU self-test on the payload's serving_out)": served_rows_abs,
 "served_bf16_out_vs_fp64_oracle_ulp16_on_significant": served_rows_ulp,
}
for i, (k, t) in enumerate(tol.items(), 1):
    rows = TROWS.get(k)
    if rows is None: R.append({"id": f"T{i}", "rule": k, "pass": False, "reason": "class not evaluable (no extraction)", "tolerance": t["tolerance"]}); continue
    lim = t["tolerance"]; rec = rule(f"T{i}", k, rows, lambda v, lim=lim: v <= lim, extra={"tolerance": lim, "pilot_max": t.get("pilot_max"), "frozen_rule_text": t.get("rule")})
    vals = [p["value"] for p in rec["per_set"] if fin(p["value"])]; rec["observed_worst"] = max(vals) if vals else None; R.append(rec)
# ---------------- D: unacceptable decision changes (machine-evaluable parts)
D = F["criteria"]["unacceptable_decision_changes"]
R.append(rule("D1", D[0], [(pid, (get(r, ("stage1_verify", "A_prod_out32_vs_native_sg32_bitwise_frac")), get(r, ("stage1_verify", "A_prod_out16_vs_native_sg16_bitwise_frac")))) for pid, r in zip(ids, rs)],
              lambda v: v[0] is not MISSING and v[1] is not MISSING and v[0] == 1.0 and v[1] == 1.0))
R.append(rule("D2", D[1], [(pid, (get(r, ("stage1_verify", "A_prod_out16_bytes_equal_payload_serving_out")), get(r, ("stage1_verify", "native_sg_out16_bytes_equal_payload_serving_out")))) for pid, r in zip(ids, rs)],
              lambda v: v[0] is True and v[1] is True))
cand_classes = ["fp32_ieee_realizations_out_vs_fp64_oracle_max_abs", "compact_commit_state_vs_fp64_oracle_max_abs", "compact_commit_state_vs_native_specupdate_max_abs", "replay_state_vs_fp64_oracle_max_abs", "replay_state_vs_native_specupdate_max_abs"]
d3rows = []
for j, pid in enumerate(ids):
    vs = []
    for c in cand_classes:
        v = TROWS[c][j][1]; vs.append((v, tol[c]["tolerance"]))
    d3rows.append((pid, MISSING if any(v is MISSING for v, _ in vs) else (float("nan") if any(not fin(v) for v, _ in vs) else max(v / t for v, t in vs))))
R.append(rule("D3", D[2] + " [value = max over candidate classes of observed/tolerance]", d3rows, lambda v: v <= 1.0))
R.append(rule("D4", D[3] + " [value = min bf16 agreement of B_fs[ieee]/C_nm[ieee] out16 vs native]", [(pid, MISSING if any(get(r, p) is MISSING for p in (("stage1_verify", "B_fs[ieee]_out16_vs_native_sg16_bitwise_frac"), ("stage1_verify", "C_nm[ieee]_out16_vs_native_sg16_bitwise_frac"))) else min(get(r, ("stage1_verify", "B_fs[ieee]_out16_vs_native_sg16_bitwise_frac")), get(r, ("stage1_verify", "C_nm[ieee]_out16_vs_native_sg16_bitwise_frac")))) for pid, r in zip(ids, rs)],
              lambda v: v >= 0.999683, extra={"floor": 0.999683}))
R.append(rule("D5", D[4] + " [value = (ulp16_max_sig, max_abs)]", [(pid, MISSING if served_rows_ulp[j][1] is MISSING or served_rows_abs[j][1] is MISSING else (served_rows_ulp[j][1], served_rows_abs[j][1])) for j, pid in enumerate(ids)],
              lambda v: v[0] <= 1 and v[1] <= tol["served_bf16_out_vs_fp64_oracle_max_abs (CPU self-test on the payload's serving_out)"]["tolerance"]))
def nonfinite_count(o):
    if isinstance(o, bool): return 0
    if isinstance(o, float): return 0 if math.isfinite(o) else 1
    if isinstance(o, dict): return sum(nonfinite_count(v) for v in o.values())
    if isinstance(o, list): return sum(nonfinite_count(v) for v in o)
    return 0
R.append(rule("D6a", "any non-finite comparison cell [value = non-finite numeric leaves in the result file; summary n_invalid_cells also required == 0]",
              [(pid, nonfinite_count(r)) for pid, r in zip(ids, rs)], lambda v: v == 0, extra={"summary_n_invalid_cells": summ.get("n_invalid_cells", "missing"), "summary_pass": summ.get("n_invalid_cells") == 0}))
R[-1]["pass"] = R[-1]["pass"] and summ.get("n_invalid_cells") == 0
def ctrl_rows(names, fields):
    rows = []
    for pid, r in zip(ids, rs):
        vals = []
        for nme in names:
            c = get(r, ("controls", nme))
            if c is MISSING: vals = MISSING; break
            for f in fields:
                if f in c: vals.append(c[f])
                elif f == "U_bitwise_frac" and nme.startswith("A_prod"): continue   # A has no factor U
                else: vals = MISSING; break
            if vals is MISSING: break
        rows.append((pid, vals if vals is MISSING else min(vals)))
    return rows
R.append(rule("D6b", "any sibling-reorder control not bitwise-invariant [value = min bitwise frac over A/B/C out (+U for B/C)]", ctrl_rows(["A_prod_sibling_reverse", "B_fs_sibling_reverse", "C_nm_sibling_reverse"], ["out_bitwise_frac", "U_bitwise_frac"]), lambda v: v == 1.0))
R.append(rule("D6c", "any PADDING control not bitwise-invariant (OMITTED BY THE v1 CHECKER) [value = min bitwise frac over B_fs/C_nm npad32-vs-npad16 out and U]", ctrl_rows(["B_fs_npad32_vs_npad16", "C_nm_npad32_vs_npad16"], ["out_bitwise_frac", "U_bitwise_frac"]), lambda v: v == 1.0,
              extra={"per_set_detail": [{"prefix": pid, **{n: get(r, ("controls", n)) if get(r, ("controls", n)) is not MISSING else None for n in ("B_fs_npad32_vs_npad16", "C_nm_npad32_vs_npad16")}} for pid, r in zip(ids, rs)]}))
R.append(rule("D6d", "any replay repeat-launch not bitwise-equal", [(pid, get(r, ("stage3_own_factors_own_commit", "A_prod_replay_repeat_bitwise_equal_all_paths"))) for pid, r in zip(ids, rs)], lambda v: v is True))
def ord_out(r):
    A, B, C = get(r, S1("A_prod_out32_vs_oracle")), get(r, S1("B_fs[ieee]_out32_vs_oracle")), get(r, S1("C_nm[ieee]_out32_vs_oracle"))
    if any(x is MISSING for x in (A, B, C)): return MISSING
    return (A, B, C)
R.append(rule("D7a", D[6] + " — out vs oracle: A <= B and B == C (strict float equality) [value = (A,B,C) max_abs]", [(pid, ord_out(r)) for pid, r in zip(ids, rs)], lambda v: fin(v[0]) and fin(v[1]) and fin(v[2]) and v[0] <= v[1] and v[1] == v[2]))
def ord_state(r):
    B, C = get(r, S3("B_fs[ieee]_compact_own_vs_oracle")), get(r, S3("C_nm[ieee]_compact_own_vs_oracle"))
    if B is MISSING or C is MISSING: return MISSING
    return (B, C)
R.append(rule("D7b", D[6] + " — COMPACT STATE: B == C (strict float equality; OMITTED BY THE v1 CHECKER) [value = (B,C) compact_own_vs_oracle max_abs]", [(pid, ord_state(r)) for pid, r in zip(ids, rs)], lambda v: fin(v[0]) and fin(v[1]) and v[0] == v[1],
              extra={"vs_native_sg_equal_per_set": [{"prefix": pid, "B": get(r, S3("B_fs[ieee]_compact_own_vs_native_sg")), "C": get(r, S3("C_nm[ieee]_compact_own_vs_native_sg"))} for pid, r in zip(ids, rs)]}))
statuses = [str((r.get("provenance") or {}).get("status", "")) for r in rs]
R.append(rule("D8a", "every harness set is VERIFIED-FRESH (provenance PASS bound to the payload sha256)", [(pid, s) for pid, s in zip(ids, statuses)], lambda v: v.startswith("VERIFIED-FRESH")))
present = set(x for x in ids if x)
R.append({"id": "D8b", "rule": D[7] + " — coverage: every frozen prefix has a VERIFIED-FRESH harness set", "pass": sorted(present) == sorted(want_ids), "missing_prefixes": sorted(set(want_ids) - present), "unexpected_prefixes": sorted(present - set(want_ids)), "n_frozen": len(want_ids), "n_present": len(present)})
R.append({"id": "D8c", "rule": D[7] + " — no second run for the same prefix", "pass": len([x for x in ids if x]) == len(present), "duplicates": sorted(x for x in present if ids.count(x) > 1)})
prov = {}
if a.capture_roots:
    for root in a.capture_roots.split(","):
        for f in glob.glob(os.path.join(root, "capture_*", "capture_provenance.json")):
            p = json.load(open(f)); pid = ((p.get("request_binding") or {}).get("prefix_id")) or os.path.basename(os.path.dirname(f)).split("_", 2)[-1]
            prov.setdefault(pid, []).append({"file": f, "all_pass": p.get("all_pass"), "inspector_sha256": p.get("inspector_sha256") or p.get("source_sha256")})
        for d in glob.glob(os.path.join(root, "capture_*")):
            if os.path.isdir(d) and not os.path.exists(os.path.join(d, "capture_provenance.json")):
                pid = os.path.basename(d).split("_", 2)[-1]; prov.setdefault(pid, []).append({"file": d, "all_pass": None, "note": "no provenance verdict (boot without a payload)"})
    rows = [(pid, prov.get(pid, MISSING)) for pid in want_ids]
    R.append(rule("D8d", D[7] + " — provenance verdict of record per frozen prefix from the capture roots (FAIL/missing counted as a failure)", rows, lambda v: any(x.get("all_pass") is True for x in v), extra={"capture_roots": a.capture_roots.split(",")}))
R.append({"id": "D9", "rule": D[8], "pass": all("tf32" not in k for k in tol.keys()), "evaluation": "static: no frozen tolerance class references a tf32 realization; tf32 rows are reported as ablation only"})
# ---------------- P: timing precision target
VERIFY_KERNELS = ["A_prod_scan_verify_out16", "B_fs[ieee]_verify_out16", "C_nm[ieee]_verify_out16", "native_sg_chain_depth5_verify(context)"]
COMMIT_KERNELS = ["B_fs[ieee]_commit_deepest", "C_nm[ieee]_commit_deepest", "A_prod_replay_commit_deepest", "A_prod_replay_commit_zero_accept", "native_pk_one_token(context)"]
def hw(t):
    if t is MISSING or not isinstance(t, dict): return MISSING
    m, lo, hi = t.get("sync_median_us"), t.get("sync_p10_us"), t.get("sync_p90_us")
    if not (fin(m) and fin(lo) and fin(hi)) or m <= 0: return float("nan")
    return (hi - lo) / (2.0 * m)
hw_rows = []; hw_cells = []
for pid, r in zip(ids, rs):
    vals = []
    for kname in VERIFY_KERNELS:
        h = hw(get(r, ("timing_us", kname)))
        if h is MISSING: vals = MISSING; break
        vals.append(h); hw_cells.append({"prefix": pid, "kernel": kname, "relative_half_width": h, "pass": fin(h) and h <= TP["relative_half_width_max"]})
    hw_rows.append((pid, vals if vals is MISSING else (float("nan") if any(not fin(v) for v in vals) else max(vals))))
R.append(rule("P1", f"verify-class kernel timing half-width (p90-p10)/(2*median) <= {TP['relative_half_width_max']} per set (OMITTED BY THE v1 CHECKER) [value = max over {VERIFY_KERNELS}]", hw_rows, lambda v: v <= TP["relative_half_width_max"],
              extra={"cells_over_target": [c for c in hw_cells if not c["pass"]], "n_cells": len(hw_cells), "basis": TP.get("basis")}))
cm_cells = []
for pid, r in zip(ids, rs):
    for kname in COMMIT_KERNELS:
        h = hw(get(r, ("timing_us", kname))); cm_cells.append({"prefix": pid, "kernel": kname, "relative_half_width": None if h is MISSING else h, "label": "imprecise (> 0.05)" if (h is not MISSING and fin(h) and h > 0.05) else ("missing" if h is MISSING else "ok")})
R.append({"id": "P2", "rule": "commit-class kernels REPORTED (not gated): rows above 0.05 labeled imprecise", "pass": True, "reported_only": True, "n_imprecise": sum(1 for c in cm_cells if c["label"].startswith("imprecise")), "cells": cm_cells})
pc = summ.get("probe_checks", {})
def drift_of(r, j):
    tb = get(r, ("timing_us", "probe_matmul4096_fp16_us_before")); ta = get(r, ("timing_us", "probe_matmul4096_fp16_us_after"))
    if tb is MISSING or ta is MISSING or not fin(tb) or not fin(ta) or tb <= 0: return MISSING
    return abs(ta - tb) / tb
R.append(rule("P3", f"probe drift (|after-before|/before of the matmul probe) <= {TP['probe_drift_max']} per set [recomputed from each result's timing block; summary probe_checks cross-checked]",
              [(pid, drift_of(r, j)) for j, (pid, r) in enumerate(zip(ids, rs))], lambda v: v <= TP["probe_drift_max"], extra={"summary_probe_checks_all_pass": all(v.get("pass") for v in pc.values()) if pc else None, "summary_probe_fail_sets": [k for k, v in pc.items() if not v.get("pass")]}))
# ---------------- I: harness integrity
R.append({"id": "I1", "rule": "harness source hashes identical at start and end of the run", "pass": summ.get("source_hashes_start_end_match") is True, "value": summ.get("source_hashes_start_end_match")})
R.append({"id": "I2", "rule": "attribution label recorded (shared-device timing diagnostic)", "pass": bool((summ.get("attribution") or {}).get("label")), "value": (summ.get("attribution") or {}).get("label")})
out = {"kind": "RETROSPECTIVE COMPLETENESS AUDIT (review 14): existing data re-evaluated against every machine-evaluable frozen rule; no threshold changed; no earlier verdict replaced",
       "set": a.set, "run_dir": a.run, "freeze": a.freeze, "freeze_sha256": freeze_sha, "frozen_prefixes": a.frozen, "n_sets": len(rs), "prefix_ids": ids,
       "rules": R, "n_rules": len(R), "n_fail": sum(1 for x in R if not x["pass"]), "failing_rule_ids": [x["id"] for x in R if not x["pass"]],
       "v1_omitted_rule_ids": ["D6c", "D7b", "P1"], "v1_omitted_rules_fail": [x["id"] for x in R if x["id"] in ("D6c", "D7b", "P1") and not x["pass"]]}
out["verdict"] = "WITHIN FROZEN CRITERIA" if out["n_fail"] == 0 else f"OUTSIDE FROZEN CRITERIA on {out['n_fail']} rule(s) (reportable finding; nothing dropped; no promotion)"
json.dump(out, open(a.out, "w"), indent=1, default=str)
print(out["kind"]); print(f"set={a.set} n_sets={len(rs)} freeze={freeze_sha[:16]}")
for x in R: print(f"{'PASS' if x['pass'] else 'FAIL'} {x['id']:4s} {x['rule'][:110]}" + (f"  failing={x.get('failing_prefixes')}" if x.get("failing_prefixes") else "") + ("  [REPORTED ONLY]" if x.get("reported_only") else ""))
print("VERDICT:", out["verdict"]); sys.exit(0 if out["n_fail"] == 0 else 1)
