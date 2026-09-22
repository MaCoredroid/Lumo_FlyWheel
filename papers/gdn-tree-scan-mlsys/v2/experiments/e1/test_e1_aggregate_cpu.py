#!/usr/bin/env python3
"""CPU controls for e1_aggregate.py on synthetic campaign roots: (1) all 18 cells VALID with known rates -> per-arm means,
paired differences, bootstrap interval bracketing the mean difference, precision computed against the native-5 mean;
(2) one block's tree/B4 cell INSUFFICIENT_SUPPORT -> every B4 contrast involving tree 'not estimable', B1 unaffected, the
cell listed; (3) an INVALID attempt followed by a VALID attempt -> the VALID one is used; (4) deterministic (same seed ->
identical interval). Usage: <out_json>"""
import json, os, subprocess, sys, tempfile
here = os.path.dirname(os.path.abspath(__file__)); res = {"checks": {}}
def check(n, ok, d=None): res["checks"][n] = {"pass": bool(ok), "detail": d}; print(("PASS " if ok else "FAIL ") + n, "" if d is None else str(d)[:160])
cells = json.load(open(os.path.join(here, "e1_cells.v1.json")))["cells"]
RATE = {"native-5": {"B1": 3.0, "B4": 9.0}, "native-11": {"B1": 3.3, "B4": 9.9}, "tree": {"B1": 3.6, "B4": 12.0}}
SEAL = {"sealed": True, "utc": "x", "gates": {"verify": "PASS"}}
def make(root, insufficient=(), invalid_first=(), unsealed=(), final_fail=()):
    os.makedirs(os.path.join(root, "campaign_snapshot"), exist_ok=True); json.dump({"cells": cells}, open(os.path.join(root, "campaign_snapshot", "e1_cells.json"), "w"))
    for c in cells:
        base = "cell_%02d_b%d_%s_%s" % (c["index"], c["block"], c["arm"], c["batch"]); k = 1
        if c["index"] in invalid_first:
            os.makedirs(os.path.join(root, base + "_a1")); json.dump({"status": "INVALID", "reasons": ["x"], "terminal_seal": SEAL}, open(os.path.join(root, base + "_a1", "cell_result.json"), "w")); k = 2
        d = os.path.join(root, base + "_a%d" % k); os.makedirs(d)
        if c["index"] in insufficient: json.dump({"status": "INSUFFICIENT_SUPPORT", "floor": {"pass": False}, "terminal_seal": SEAL}, open(os.path.join(d, "cell_result.json"), "w")); continue
        rate = RATE[c["arm"]][c["batch"]] * (1 + 0.02 * (c["block"] - 2))
        if c["index"] in unsealed:   # a PRELIMINARY VALID summary (or a cell_result.json without terminal_seal) must be REFUSED
            json.dump({"status": "VALID", "primary": {"tokens_per_wall_second": 123.0, "n_usable": 50}, "preliminary": True}, open(os.path.join(d, "cell_result.preliminary.json"), "w")); json.dump({"status": "VALID", "primary": {"tokens_per_wall_second": 123.0, "n_usable": 50}}, open(os.path.join(d, "cell_result.json"), "w")); continue
        if c["index"] in final_fail:  # sealed INVALID after a failed final audit (runner F1 path): preliminary VALID rate preserved separately, never counted
            json.dump({"status": "VALID", "primary": {"tokens_per_wall_second": 123.0}, "preliminary": True}, open(os.path.join(d, "cell_result.preliminary.json"), "w")); json.dump({"status": "INVALID", "reasons": ["native FINAL audit FAILED"], "rate_not_eligible": True, "terminal_seal": dict(SEAL, gates={"verify": "PASS", "summary": "VALID", "preflight_final": "FAIL"})}, open(os.path.join(d, "cell_result.json"), "w")); continue
        json.dump({"status": "VALID", "primary": {"tokens_per_wall_second": rate, "n_usable": 50}, "diagnostic_over_cap": {"n_intervals_over_cap": 0}, "terminal_seal": SEAL}, open(os.path.join(d, "cell_result.json"), "w"))
def run(root): r = subprocess.run([sys.executable, os.path.join(here, "e1_aggregate.py"), root], capture_output=True, text=True); return r.returncode, json.load(open(os.path.join(root, "aggregate.json"))), r.stdout
with tempfile.TemporaryDirectory() as td:
    r1 = os.path.join(td, "full"); make(r1); rc, A, o = run(r1); b1 = A["by_batch"]["B1"]; c = b1["contrasts"]["tree_minus_native5"]
    check("full_campaign_means_and_paired_bootstrap", rc == 0 and abs(b1["per_arm_mean_over_blocks"]["native-5"] - 3.0) < 1e-9 and abs(c["mean_difference"] - 0.6) < 1e-9 and c["bootstrap_95pct_interval"][0] <= 0.6 <= c["bootstrap_95pct_interval"][1] and abs(c["precision_half_width_over_baseline"] - (c["bootstrap_95pct_interval"][1] - c["bootstrap_95pct_interval"][0]) / 6.0) < 1e-9, c)
    check("all_18_cells_listed", len(A["cells"]) == 18 and all(x["status"] == "VALID" for x in A["cells"]))
    tree_b4_block2 = [x["index"] for x in cells if x["arm"] == "tree" and x["batch"] == "B4" and x["block"] == 2][0]
    r2 = os.path.join(td, "insuff"); make(r2, insufficient=(tree_b4_block2,)); rc, A2, o = run(r2); b4 = A2["by_batch"]["B4"]["contrasts"]
    check("insufficient_block_makes_tree_B4_contrasts_not_estimable_B1_unaffected", "not estimable" in b4["tree_minus_native5"]["status"] and "not estimable" in b4["tree_minus_native11"]["status"] and "paired_block_differences" in b4["native11_minus_native5"] and "paired_block_differences" in A2["by_batch"]["B1"]["contrasts"]["tree_minus_native5"] and any(x["status"] == "INSUFFICIENT_SUPPORT" for x in A2["cells"]))
    r3 = os.path.join(td, "retry"); make(r3, invalid_first=(1,)); rc, A3, o = run(r3); cell1 = [x for x in A3["cells"] if x["index"] == 1][0]
    check("valid_second_attempt_used_after_invalid_first", cell1["status"] == "VALID" and cell1["attempt"].endswith("_a2") and len(cell1["attempts"]) == 2)
    rc, A4, o = run(r1); check("deterministic_same_seed", A4["by_batch"]["B1"]["contrasts"]["tree_minus_native5"]["bootstrap_95pct_interval"] == c["bootstrap_95pct_interval"])
    r5 = os.path.join(td, "unsealed"); make(r5, unsealed=(1,)); rc, A5, o = run(r5); c1 = [x for x in A5["cells"] if x["index"] == 1][0]
    check("unsealed_valid_123_refused_not_counted", c1["rate"] is None and "UNSEALED" in c1["status"] and "not estimable" in A5["by_batch"]["B1"]["contrasts"]["tree_minus_native5"]["status"], c1)
    r6 = os.path.join(td, "finalfail"); make(r6, final_fail=(1,)); rc, A6, o = run(r6); c1 = [x for x in A6["cells"] if x["index"] == 1][0]
    check("sealed_invalid_after_final_audit_not_counted_preliminary_preserved", c1["rate"] is None and c1["status"] == "INVALID" and "not estimable" in A6["by_batch"]["B1"]["contrasts"]["tree_minus_native5"]["status"] and os.path.exists(os.path.join(r6, c1["attempt"], "cell_result.preliminary.json")), c1)
res["all_pass"] = all(v["pass"] for v in res["checks"].values()); json.dump(res, open(sys.argv[1], "w"), indent=2); print("ALL PASS" if res["all_pass"] else "SOME FAILED"); sys.exit(0 if res["all_pass"] else 1)
