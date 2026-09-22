#!/usr/bin/env python3
"""CPU controls for the confirmation freeze gate (review 10). Builds a synthetic but fully CONSISTENT freeze fixture
(8 fake pilot run dirs with hash-consistent provenance/payload files, a hash-bound harness summary, filled criteria) and
verifies check_pilot_freeze.py returns VALID; then each single-fault mutation must be INVALID:
  status_only_template (the real template with status flipped), stale_provenance_hash, stale_summary_hash,
  summary_missing_one_payload, seven_entries, duplicate_id, wrong_id, payload_sha_mismatch, inspector_mismatch,
  provenance_not_all_pass, empty_tolerances, tolerance_below_pilot_max, empty_candidates, empty_decision_changes,
  timing_precision_missing, cost_variance_missing, frozen_utc_missing, infinite/NaN/negative numeric criteria,
  tolerance not equal to pilot_max*margin without an explicit rule, non-finite metric-variance leaf (review 11);
  positive: a 0-ULP bitwise class with an explicit rule.
Also drives capture_loop.v5.sh (SET=confirmation) with FREEZE_JSON/FREEZE_MD overrides: status-only .md + template json
-> refused rc 30 before any boot; valid json + .md not FROZEN -> rc 32; valid json + FROZEN md -> gate passes (the run is
then aborted by a bogus pool path BEFORE any docker call; the loop's validation stage rejects it).
Usage: test_pilot_freeze_gate.py <frozen_prefixes.json> <out_json>"""
import copy, hashlib, json, os, subprocess, sys, tempfile
from pathlib import Path
here = Path(__file__).resolve().parent; frozen_p, out_p = sys.argv[1:3]
sys.path.insert(0, str(here)); import check_pilot_freeze as G
frozen = json.load(open(frozen_p)); pilot_ids = [r["id"] for r in frozen["pilot"]]
INSP = "e" * 64
def sha_b(b): return hashlib.sha256(b).hexdigest()
res = {"checks": {}, "check_pilot_freeze_sha256": sha_b(open(here / "check_pilot_freeze.py", "rb").read()), "loop_v5_sha256": sha_b(open(here / "capture_loop.v5.sh", "rb").read())}
def rec(n, ok, d=None): res["checks"][n] = {"pass": bool(ok), "detail": d}; print(("OK   " if ok else "BAD  ") + n, "" if d is None else str(d)[:150])
with tempfile.TemporaryDirectory(prefix="e7a-freeze-") as td:
    td = Path(td); ev = []; sets = []
    for i, pid in enumerate(pilot_ids):
        d = td / f"capture_{i+1:02d}_{pid}"; (d / "logs").mkdir(parents=True)
        pay = os.urandom(256) + pid.encode(); (d / "logs" / "tree_gdn_capture_payload.pt").write_bytes(pay); ph = sha_b(pay)
        prov = {"all_pass": True, "failed": [], "unverified": [], "payload_sha256": ph, "inspector_sha256": INSP, "request": {"prefix_id": pid}, "checks": {"x": {"pass": True}}}
        (d / "capture_provenance.json").write_text(json.dumps(prov)); prsha = sha_b((d / "capture_provenance.json").read_bytes())
        ev.append({"prefix_id": pid, "run_dir": str(d), "payload_sha256": ph, "provenance_sha256": prsha, "inspector_sha256": INSP})
        sets.append({"label": pid, "status": "VERIFIED-FRESH (provenance PASS bound to this payload sha256; prefix %s)" % pid, "payload_sha256": ph, "prefix_id": pid})
    summ = td / "summary.json"; summ.write_text(json.dumps({"operand_sets": sets, "n_verified_fresh": 8, "n_invalid_cells": 0, "independent_prefixes_established": 8}))
    good = {"schema": "e7a.pilot_freeze.v1", "status": "FROZEN", "frozen_utc": "2026-09-22T09:00:00Z",
            "resampling_unit": "frozen prefix (one boot, one request, one payload per prefix)", "pilot_evidence": ev,
            "pilot_harness_summary": {"path": str(summ), "sha256": sha_b(summ.read_bytes())},
            "criteria": {"candidate_selection": {"candidates": ["scan-fp32", "treewy-fs-fp32", "bole-neumann-fp32"], "rationale": "fixture"},
                         "tolerances": {"fp32_vs_oracle_out": {"pilot_max": 2e-8, "margin": 2.0, "tolerance": 4e-8}, "bf16_store_agreement_floor": {"pilot_max": 1.3e-4, "margin": 1.5, "tolerance": 1.3e-4 * 1.5}},
                         "unacceptable_decision_changes": ["any fp32 scan row not bit-identical to native spec-update", "any non-finite cell"],
                         "timing_precision_target": {"relative_half_width_max": 0.05, "probe_drift_max": 0.05, "basis": "fixture"},
                         "cost_variance": {"boot_health_s": 328.0, "request_s": 13.5, "driver_cycle_s": 357.0, "payload_bytes": 3607409, "metric_variance": {"bf16_max_abs": 1e-6}}}}
    def run(fz, tag):
        fp = td / f"{tag}.json"; fp.write_text(json.dumps(fz)); r = G.validate(str(fp), frozen_p); return r["all_pass"], [k for k, v in r["checks"].items() if not v["pass"]]
    ok, failed = run(good, "good"); rec("positive_consistent_fixture_VALID", ok, failed)
    # negatives (each a single fault on a deep copy)
    def neg(name, mut):
        fz = copy.deepcopy(good); mut(fz); ok, failed = run(fz, name); rec(f"neg_{name}_INVALID", (not ok) and len(failed) > 0, failed)
    tmpl = json.load(open(here / "PILOT_FREEZE.json")); tmpl["status"] = "FROZEN"; tmpl["frozen_utc"] = "2026-09-22T09:00:00Z"
    fp = td / "status_only.json"; fp.write_text(json.dumps(tmpl)); r = G.validate(str(fp), frozen_p); rec("neg_status_only_template_INVALID", not r["all_pass"], [k for k, v in r["checks"].items() if not v["pass"]][:6])
    neg("stale_provenance_hash", lambda f: f["pilot_evidence"][3].update(provenance_sha256="0" * 64))
    neg("stale_summary_hash", lambda f: f["pilot_harness_summary"].update(sha256="0" * 64))
    def summ_missing(f):
        s2 = td / "summary_missing.json"; sm = json.loads(summ.read_text()); sm["operand_sets"] = sm["operand_sets"][:7]; sm["n_verified_fresh"] = 7; s2.write_text(json.dumps(sm)); f["pilot_harness_summary"] = {"path": str(s2), "sha256": sha_b(s2.read_bytes())}
    neg("summary_missing_one_payload", summ_missing)
    neg("seven_entries", lambda f: f["pilot_evidence"].pop())
    neg("duplicate_id", lambda f: f["pilot_evidence"].__setitem__(7, copy.deepcopy(f["pilot_evidence"][0])))
    neg("wrong_id", lambda f: f["pilot_evidence"][2].update(prefix_id="p999"))
    neg("payload_sha_mismatch", lambda f: f["pilot_evidence"][1].update(payload_sha256="1" * 64))
    neg("inspector_mismatch", lambda f: f["pilot_evidence"][5].update(inspector_sha256="f" * 64))
    def not_all_pass(f):
        d = Path(f["pilot_evidence"][4]["run_dir"]); pv = json.loads((d / "capture_provenance.json").read_text()); pv["all_pass"] = False; (d / "capture_provenance.json").write_text(json.dumps(pv)); f["pilot_evidence"][4]["provenance_sha256"] = sha_b((d / "capture_provenance.json").read_bytes())
    neg("provenance_not_all_pass", not_all_pass)
    # restore the mutated provenance for the remaining cases
    d4 = Path(good["pilot_evidence"][4]["run_dir"]); pv = json.loads((d4 / "capture_provenance.json").read_text()); pv["all_pass"] = True; (d4 / "capture_provenance.json").write_text(json.dumps(pv))
    assert sha_b((d4 / "capture_provenance.json").read_bytes()) == good["pilot_evidence"][4]["provenance_sha256"]
    neg("empty_tolerances", lambda f: f["criteria"].update(tolerances={}))
    neg("tolerance_below_pilot_max", lambda f: f["criteria"]["tolerances"]["fp32_vs_oracle_out"].update(tolerance=1e-9))
    neg("empty_candidates", lambda f: f["criteria"]["candidate_selection"].update(candidates=[]))
    neg("empty_decision_changes", lambda f: f["criteria"].update(unacceptable_decision_changes=[]))
    neg("timing_precision_missing", lambda f: f["criteria"].update(timing_precision_target={}))
    neg("cost_variance_missing", lambda f: f["criteria"].update(cost_variance={}))
    neg("frozen_utc_missing", lambda f: f.update(frozen_utc=None))
    INF, NAN = float("inf"), float("nan")
    neg("infinite_tolerance_class", lambda f: f["criteria"]["tolerances"]["fp32_vs_oracle_out"].update(pilot_max=INF, margin=INF, tolerance=INF))
    neg("nan_pilot_max", lambda f: f["criteria"]["tolerances"]["fp32_vs_oracle_out"].update(pilot_max=NAN))
    neg("negative_pilot_max", lambda f: f["criteria"]["tolerances"]["fp32_vs_oracle_out"].update(pilot_max=-1e-8, tolerance=4e-8))
    neg("tolerance_not_pilot_max_times_margin_without_rule", lambda f: f["criteria"]["tolerances"]["fp32_vs_oracle_out"].update(tolerance=5e-8))
    neg("infinite_timing_precision", lambda f: f["criteria"]["timing_precision_target"].update(relative_half_width_max=INF))
    neg("infinite_cost", lambda f: f["criteria"]["cost_variance"].update(boot_health_s=INF))
    neg("negative_cost", lambda f: f["criteria"]["cost_variance"].update(request_s=-1.0))
    neg("nonfinite_metric_variance_leaf", lambda f: f["criteria"]["cost_variance"].update(metric_variance={"bf16_max_abs": NAN}))
    fz = copy.deepcopy(good); fz["criteria"]["tolerances"]["scan_fp32_vs_native_specupdate_bitwise"] = {"pilot_max": 0, "tolerance": 0, "rule": "0 ULP, integer view (bit-identical) — pinned native spec-update kernel"}
    ok, failed = run(fz, "bitwise_pos"); rec("positive_bitwise_zero_ulp_class_with_rule_VALID", ok, failed)
    # loop-level gate (SET=confirmation) with overrides; the pool path is bogus so nothing past the gate can boot
    goodp = td / "good.json"; md_frozen = td / "FROZEN.md"; md_frozen.write_text("# x\n\nStatus: FROZEN 2026-09-22T09:00:00Z\n"); md_not = td / "NOT.md"; md_not.write_text("Status: NOT FROZEN\n")
    pool_real = str(Path(frozen_p).resolve().parent.parent / "out-20260921T230815Z-e7a-step2-prefixes" / "prefix_pool.json")
    def loop(fz, md, tag):
        # START_INDEX=99 skips every confirmation prefix, so a passing gate can never reach a boot
        env = dict(os.environ, SET="confirmation", FREEZE_JSON=str(fz), FREEZE_MD=str(md), SCORES="", START_INDEX="99")
        r = subprocess.run(["bash", str(here / "capture_loop.v5.sh"), frozen_p, pool_real, "language_model.model.layers.62.linear_attn", str(td / f"root_{tag}")], env=env, capture_output=True, text=True, timeout=120)
        return r.returncode, (r.stderr + r.stdout)[-400:]
    rc, o = loop(td / "status_only.json", md_frozen, "a"); rec("loop_refuses_status_only_json_rc30", rc == 30, o[-120:])
    rc, o = loop(goodp, md_not, "b"); rec("loop_refuses_md_not_frozen_rc32", rc == 32, o[-120:])
    rc, o = loop(goodp, md_frozen, "c"); rec("loop_passes_gate_no_boot_all_prefixes_skipped", rc == 0 and "CAPTURE LOOP DONE" in o and "docker" not in o.lower() and (td / "root_c" / "PILOT_FREEZE.frozen.json").exists(), {"rc": rc, "tail": o[-160:]})
res["all_pass"] = all(v["pass"] for v in res["checks"].values()); json.dump(res, open(out_p, "w"), indent=2)
print("ALL GATE CONTROLS OK" if res["all_pass"] else "GATE CONTROLS FAILED"); sys.exit(0 if res["all_pass"] else 1)
