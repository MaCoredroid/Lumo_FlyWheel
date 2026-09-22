#!/usr/bin/env bash
# F1 EXACT-RUNNER CONTROL (NO GPU): the REAL runner v2 with a mirror whose e1 sources for the driver / verifier / summary /
# native preflight are STUBS: driver "succeeds" (creates the cell with driver_trace preflight_verdict=PASS and a script_snapshot
# of the stubs), verifier passes, summary writes a PRELIMINARY VALID result with rate 123, the preflight passes --live and
# FAILS --final when FAKE_FINAL_FAIL=1. Expected: runner exit 12, NO qualification.json, terminal cell_result.json sealed
# INVALID (rate not eligible), cell_result.preliminary.json preserved with VALID 123, aggregate refuses it (rate None), resume
# treats the attempt as failed (exit 46 without E1_RETRY_FAILED); with FAKE_FINAL_FAIL=0 and the retry flag a NEW attempt is
# sealed VALID 123, qualification.json carries the key and the aggregate counts exactly that attempt. A tree (qualified) cell
# is sealed after verify+summary. Usage: <out_json>
set -uo pipefail
OUT=$1; WT=/home/mark/lumo-paper-v2-20260921; EXP=$WT/papers/gdn-tree-scan-mlsys/v2/experiments
S=$(mktemp -d /tmp/claude-1000/-home-mark-lumo-paper-v2-20260921/c8955ff9-f3ae-4551-869b-1a98b38ca2c3/scratchpad/e1f1.XXXX)
mkdir -p "$S/papers/gdn-tree-scan-mlsys/v2/experiments" "$S/scripts" "$S/src/lumo_flywheel_serving"
for d in e1 e7a; do cp -r "$EXP/$d" "$S/papers/gdn-tree-scan-mlsys/v2/experiments/$d"; done
cp "$WT/scripts/gpu_oom_guard.sh" "$WT/scripts/fr10_phase4_patch_vllm_tree_gdn.py" "$WT/scripts/fr13_device_multidraft_kernel.py" "$S/scripts/"; cp "$WT/src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py" "$WT/src/lumo_flywheel_serving/fr10_decode_modes.py" "$WT/src/lumo_flywheel_serving/fr13_tree_conv_fused.py" "$S/src/lumo_flywheel_serving/"
( cd "$S" && git init -q && git add -A >/dev/null 2>&1 && git -c user.email=t@t -c user.name=t commit -qm stub >/dev/null 2>&1 )
SE="$S/papers/gdn-tree-scan-mlsys/v2/experiments"
sed -i "s#^WT=/home/mark/lumo-paper-v2-20260921; EXP=#WT=$S; EXP=#" "$SE/e1/e1_run_cells.v2.sh"
cat > "$SE/e1/e1_cell_driver.v1.sh" <<'STUB'
#!/usr/bin/env bash
# STUB driver: pretend the boot + preflight-live + workload succeeded; copy the (stub) gate scripts into the cell snapshot
set -euo pipefail
RUN="$3/${E1_RUN_NAME:?}"; [[ -e "$RUN" ]] && { echo "REFUSING: exists"; exit 44; }; mkdir -p "$RUN/script_snapshot" "$RUN/logs"
for f in e1_cell_verify.py e1_cell_summary.py e1_native_preflight.py; do cp "${E1_SOURCE_DIR:?}/$f" "$RUN/script_snapshot/$f"; done
{ echo "start_utc=x cell=$2"; echo "driver_exec=stub from_snapshot=1"; [[ "${E1_PREFLIGHT:-0}" == "1" ]] && echo "preflight_verdict=PASS utc=x"; echo "exit_utc=x rc=0"; } > "$RUN/driver_trace.txt"
exit 0
STUB
cat > "$SE/e1/e1_cell_verify.py" <<'STUB'
import sys; print("VERIFY PASS (stub)"); sys.exit(0)
STUB
cat > "$SE/e1/e1_cell_summary.py" <<'STUB'
import json, os, sys
cell = sys.argv[1]; json.dump({"status": "VALID", "preliminary": True, "primary": {"tokens_per_wall_second": 123.0, "n_usable": 50}, "reasons": []}, open(os.path.join(cell, "cell_result.preliminary.json"), "w")); print("cell_result (PRELIMINARY): VALID [] 123.0"); sys.exit(0)
STUB
cat > "$SE/e1/e1_native_preflight.py" <<'STUB'
import os, sys
mode = "live" if "--live" in sys.argv else "final"
if mode == "final" and os.environ.get("FAKE_FINAL_FAIL") == "1": print("FAIL P5_final (stub)"); print("NATIVE PREFLIGHT (final) FAIL"); sys.exit(1)
print(f"NATIVE PREFLIGHT ({mode}) PASS (stub)"); sys.exit(0)
STUB
export E1_POOL=$EXP/out-20260921T230815Z-e7a-step2-prefixes/prefix_pool.json E1_FROZEN=$EXP/out-20260921T231853Z-e7a-step2-native-score/frozen_prefixes.json E1_SCORES=$EXP/out-20260921T231853Z-e7a-step2-native-score/prefix_scores.json E1_QUAL_MANIFEST=$SE/e1/e1_qualification_manifest.v1.json
ROOT=$S/campaign; mkdir -p "$ROOT"
python3 - "$OUT" "$S" "$SE" "$ROOT" "$EXP" <<'PY'
import json, os, subprocess, sys
out, S, SE, ROOT, EXP = sys.argv[1:6]; res = {"checks": {}, "scratch": S}
def check(n, ok, d=None): res["checks"][n] = {"pass": bool(ok), "detail": d}; print(("PASS " if ok else "FAIL ") + n, "" if d is None else str(d)[:200])
def runner(a, b, extra=None): return subprocess.run(["bash", f"{SE}/e1/e1_run_cells.v2.sh", f"{SE}/e1/e1_cells.v1.json", ROOT, str(a), str(b)], env=dict(os.environ, **(extra or {})), capture_output=True, text=True, timeout=900)
def agg(): subprocess.run([sys.executable, f"{EXP}/e1/e1_aggregate.py", ROOT], capture_output=True, text=True); return json.load(open(f"{ROOT}/aggregate.json"))
# 1) native cell 1: everything passes, FINAL audit fails
r = runner(1, 1, {"FAKE_FINAL_FAIL": "1"}); a1 = f"{ROOT}/cell_01_b1_native-5_B1_a1"
term = json.load(open(f"{a1}/cell_result.json")) if os.path.exists(f"{a1}/cell_result.json") else {}; pre = json.load(open(f"{a1}/cell_result.preliminary.json")) if os.path.exists(f"{a1}/cell_result.preliminary.json") else {}
check("final_audit_fail_runner_exit12_no_qualification", r.returncode == 12 and not os.path.exists(f"{ROOT}/qualification.json"), (r.returncode, r.stdout[-200:]))
check("terminal_result_sealed_INVALID_rate_not_eligible", term.get("status") == "INVALID" and (term.get("terminal_seal") or {}).get("sealed") is True and term.get("rate_not_eligible") is True and "primary" not in term and (term["terminal_seal"]["gates"].get("preflight_final") == "FAIL"), {k: term.get(k) for k in ("status", "rate_not_eligible", "terminal_seal")})
check("preliminary_VALID_123_preserved_separately", pre.get("status") == "VALID" and pre.get("primary", {}).get("tokens_per_wall_second") == 123.0 and pre.get("preliminary") is True)
A = agg(); c1 = [x for x in A["cells"] if x["index"] == 1][0]
check("aggregate_refuses_unqualified_123", c1["rate"] is None and c1["status"] == "INVALID" and "not estimable" in A["by_batch"]["B1"]["contrasts"]["tree_minus_native5"]["status"], c1)
r2 = runner(1, 1, {"FAKE_FINAL_FAIL": "1"}); check("resume_treats_failed_final_as_failed_attempt_exit46", r2.returncode == 46 and not os.path.exists(f"{ROOT}/cell_01_b1_native-5_B1_a2"), r2.returncode)
# 2) retry with a passing final audit -> new attempt sealed VALID, qualified, counted
r3 = runner(1, 1, {"FAKE_FINAL_FAIL": "0", "E1_RETRY_FAILED": "1"}); a2 = f"{ROOT}/cell_01_b1_native-5_B1_a2"; t2 = json.load(open(f"{a2}/cell_result.json")) if os.path.exists(f"{a2}/cell_result.json") else {}
q = json.load(open(f"{ROOT}/qualification.json")) if os.path.exists(f"{ROOT}/qualification.json") else {}
check("retry_pass_sealed_VALID_qualified", r3.returncode == 0 and t2.get("status") == "VALID" and (t2.get("terminal_seal") or {}).get("sealed") is True and t2["terminal_seal"]["gates"].get("preflight_final") == "PASS" and "native-5/B1" in q.get("qualified", {}), (r3.returncode, t2.get("status"), list(q.get("qualified", {}))))
A2 = agg(); c1 = [x for x in A2["cells"] if x["index"] == 1][0]; check("aggregate_counts_only_the_sealed_valid_attempt", c1["rate"] == 123.0 and c1["attempt"].endswith("_a2"), c1)
r4 = runner(1, 1); check("measured_cell_never_rerun", r4.returncode == 0 and "never re-run" in open(f"{ROOT}/campaign.log").read() and not os.path.exists(f"{ROOT}/cell_01_b1_native-5_B1_a3"), r4.returncode)
# 3) qualified tree cell: sealed after verify+summary (no preflight)
r5 = runner(5, 5); a5 = f"{ROOT}/cell_05_b1_tree_B1_a1"; t5 = json.load(open(f"{a5}/cell_result.json")) if os.path.exists(f"{a5}/cell_result.json") else {}
check("tree_cell_sealed_after_verify_and_summary", r5.returncode == 0 and t5.get("status") == "VALID" and (t5.get("terminal_seal") or {}).get("sealed") is True and "preflight_final" not in t5["terminal_seal"]["gates"], (r5.returncode, t5.get("terminal_seal")))
res["all_pass"] = all(v["pass"] for v in res["checks"].values()); json.dump(res, open(out, "w"), indent=2); print("ALL PASS" if res["all_pass"] else "SOME FAILED"); sys.exit(0 if res["all_pass"] else 1)
PY
rc=$?; echo "scratch: $S"; exit $rc
