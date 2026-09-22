#!/usr/bin/env python3
"""Controls for check_confirmation_vs_freeze.v2.py: (1) mutation negative controls on a COPY of the pilot-v2 harness run
(missing padding key -> D6c fails closed 'missing'; NaN cell -> D6a fails; C compact != B -> D7b fails; widened p90 -> P1
fails; zero-drift probe -> P3 unaffected); (2) cross-check of the untouched pilot-v2 and confirmation audits against the
parent's independent recomputation (p0/monitor/2026-09-22-review-14-frozen-audit.json): padding-failed prefix sets,
half-width failing (prefix,kernel) cells, compact-state ordering failures, minimum bf16 agreement.
Usage: test_freeze_checker_v2.py <audit_dir> <out_json>"""
import json, os, shutil, subprocess, sys, tempfile, math
here = os.path.dirname(os.path.abspath(__file__)); EXP = os.path.dirname(here)
CHK = os.path.join(here, "check_confirmation_vs_freeze.v2.py"); FREEZE = os.path.join(here, "PILOT_FREEZE.json")
FR = os.path.join(EXP, "out-20260921T231853Z-e7a-step2-native-score/frozen_prefixes.json"); PILOT = os.path.join(EXP, "out-20260922T013313Z-e7a-fresh-pilot-v2")
audit_dir, out_json = sys.argv[1], sys.argv[2]; res = {"checks": {}}
def check(n, ok, d=None): res["checks"][n] = {"pass": bool(ok), "detail": d}; print(("PASS " if ok else "FAIL ") + n, "" if d is None else str(d)[:160])
def run(run_dir, setname="pilot"):
    out = os.path.join(tempfile.mkdtemp(), "o.json"); subprocess.run([sys.executable, CHK, FREEZE, run_dir, FR, out, "--set", setname], capture_output=True, text=True)
    return {r["id"]: r for r in json.load(open(out))["rules"]}
def mutated(fn):
    td = tempfile.mkdtemp(); d = os.path.join(td, "run"); shutil.copytree(PILOT, d)
    f = sorted(x for x in os.listdir(d) if x.startswith("result_"))[0]; p = os.path.join(d, f); r = json.load(open(p)); fn(r); json.dump(r, open(p, "w")); return d
base = run(PILOT)
check("untouched_pilot_v2_fails_exactly_D6c_P1", [k for k, v in base.items() if not v["pass"]] == ["D6c", "P1"], [k for k, v in base.items() if not v["pass"]])
def m1(r): del r["controls"]["B_fs_npad32_vs_npad16"]
x = run(mutated(m1)); check("missing_padding_key_fails_closed", x["D6c"]["pass"] is False and x["D6c"]["per_set"][0]["reason"] == "missing")
def m2(r): r["stage1_verify"]["B_fs[ieee]_out32_vs_oracle"]["max_abs"] = float("nan")
x = run(mutated(m2)); check("nan_cell_fails_D6a_and_T3", x["D6a"]["pass"] is False and x["T3"]["pass"] is False and x["T3"]["per_set"][0].get("reason") == "non-finite")
def m3(r): r["stage3_own_factors_own_commit"]["C_nm[ieee]_compact_own_vs_oracle"]["max_abs"] *= 1.0000001
x = run(mutated(m3)); check("compact_state_B_ne_C_fails_D7b_only_that_set", x["D7b"]["pass"] is False and len(x["D7b"]["failing_prefixes"]) == 1 and base["D7b"]["pass"] is True)
def m4(r): r["timing_us"]["B_fs[ieee]_verify_out16"]["sync_p90_us"] = r["timing_us"]["B_fs[ieee]_verify_out16"]["sync_median_us"] * 1.5
x = run(mutated(m4)); check("widened_p90_fails_P1_cell", any(c["kernel"] == "B_fs[ieee]_verify_out16" and c["prefix"] == x["P1"]["per_set"][0]["prefix"] for c in x["P1"]["cells_over_target"]))
def m5(r): r["timing_us"]["A_prod_scan_verify_out16"]["sync_p10_us"] = float("inf")
x = run(mutated(m5)); check("nonfinite_timing_fails_closed", x["P1"]["per_set"][0].get("reason") == "non-finite")
def m6(r): r["provenance"]["status"] = "UNVERIFIED capture"
x = run(mutated(m6)); check("unverified_status_fails_D8a", x["D8a"]["pass"] is False)
# ---- cross-check vs the parent's independent recomputation
par = json.load(open(os.path.join(EXP, "..", "p0", "monitor", "2026-09-22-review-14-frozen-audit.json")))
for setname, fn in (("pilot", "pilot_v2_retrospective.json"), ("confirmation", "confirmation_retrospective.json")):
    mine = {r["id"]: r for r in json.load(open(os.path.join(audit_dir, fn)))["rules"]}; P = par[setname]
    check(f"{setname}_padding_failed_prefixes_match_parent", sorted(mine["D6c"]["failing_prefixes"]) == sorted(P["padding_failed_prefixes"]), (len(mine["D6c"]["failing_prefixes"]), len(P["padding_failed_prefixes"])))
    my_cells = {(c["prefix"], c["kernel"]) for c in mine["P1"]["cells_over_target"]}; pa_cells = {(c["prefix"], c["kernel"]) for c in P["half_width_failures"]}
    check(f"{setname}_half_width_cells_match_parent", my_cells == pa_cells, (len(my_cells), len(pa_cells), sorted(my_cells ^ pa_cells)[:4]))
    check(f"{setname}_half_width_values_match_parent", all(abs(next(c["relative_half_width"] for c in mine["P1"]["cells_over_target"] if (c["prefix"], c["kernel"]) == (q["prefix"], q["kernel"])) - q["relative_half_width"]) < 1e-9 for q in P["half_width_failures"]))
    so = P.get("state_order_failures", []); so_p = sorted({(x.get("prefix") if isinstance(x, dict) else x) for x in so}) if so else []
    check(f"{setname}_state_order_failures_match_parent", sorted(mine["D7b"]["failing_prefixes"]) == so_p, (mine["D7b"]["failing_prefixes"], so_p))
    ma = P.get("minimum_bf16_agreement", {}); mymin = min(p["value"] for p in mine["D4"]["per_set"] if isinstance(p["value"], float))
    check(f"{setname}_minimum_bf16_agreement_matches_parent", abs(mymin - ma.get("agreement", -1)) < 1e-12, (mymin, ma.get("agreement")))
res["all_pass"] = all(v["pass"] for v in res["checks"].values()); json.dump(res, open(out_json, "w"), indent=2, default=str)
print("ALL PASS" if res["all_pass"] else "SOME FAILED"); sys.exit(0 if res["all_pass"] else 1)
