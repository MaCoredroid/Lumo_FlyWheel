#!/usr/bin/env bash
# E1 CAMPAIGN STUB PREFLIGHT (NO GPU, no container): drives the REAL runner v2 -> campaign snapshot -> re-exec -> cell driver
# snapshot/re-exec -> STUB launchers (dump env, exit 1) in a scratch mirror. Checks: campaign snapshot + SHA256SUMS +
# campaign_identity; runner and driver executed from their snapshots; attempt dirs (_a1, refusal without E1_RETRY_FAILED
# = exit 46, _a2 with it); tamper -> exit 45; driver refuses an existing cell dir (exit 44); frozen env per arm incl. the
# EXACT SPEC_CONFIG/TREE, native decode mode/tree-GDN, policy B, no heavy capture; NOT_QUALIFIED cells listed and skipped;
# no docker_inspect.json anywhere. Usage: <out_json>
set -uo pipefail
OUT=$1; WT=/home/mark/lumo-paper-v2-20260921; EXP=$WT/papers/gdn-tree-scan-mlsys/v2/experiments
S=$(mktemp -d /tmp/claude-1000/-home-mark-lumo-paper-v2-20260921/c8955ff9-f3ae-4551-869b-1a98b38ca2c3/scratchpad/e1camp.XXXX)
mkdir -p "$S/papers/gdn-tree-scan-mlsys/v2/experiments" "$S/scripts" "$S/src/lumo_flywheel_serving"
for d in e1 e7a e2; do cp -r "$EXP/$d" "$S/papers/gdn-tree-scan-mlsys/v2/experiments/$d"; done
cp "$WT/scripts/gpu_oom_guard.sh" "$WT/scripts/fr10_phase4_patch_vllm_tree_gdn.py" "$WT/scripts/fr13_device_multidraft_kernel.py" "$S/scripts/"; cp "$WT/src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py" "$WT/src/lumo_flywheel_serving/fr10_decode_modes.py" "$WT/src/lumo_flywheel_serving/fr13_tree_conv_fused.py" "$S/src/lumo_flywheel_serving/"
( cd "$S" && git init -q && git add -A >/dev/null 2>&1 && git -c user.email=t@t -c user.name=t commit -qm stub >/dev/null 2>&1 )
SE="$S/papers/gdn-tree-scan-mlsys/v2/experiments"
for L in e1/e1_native_launch.v2.sh e7a/e7a_capture_launch.v7.sh; do cat > "$SE/$L" <<'STUB'
#!/usr/bin/env bash
env | grep -E '^(MAX_NUM_SEQS|VLLM_SYNC_SCHED|ENFORCE_EAGER|FR13_ENABLE_APC|E1_SHIM_PATH|E1_RECORD|E1_RUNTIME_DIR|FR13_SFWD_GPU_TIMER|FR13_ATTN_KV_REMAP|FR13_SLOT_REORDER|FR13_KV_REMAP_SYNCFREE|FR13_REPLAY_ROUTE|FR13_EAGER_PACK|FR13_TREE_CONV_FUSED|FR13_TREE_RUNROW_INIT|ATTENTION_BACKEND|NUM_SPECULATIVE_TOKENS|FR10_DECODE_MODE_DEFAULT|FR10_ENABLE_TREE_GDN|FR13_FINAL_LOGIT_CAPTURE|FR10_LAYER_HIDDEN_CAPTURE|FR10_TREE_GDN_CAPTURE_PAYLOAD|E7B_SUBSTITUTE|E7A_CAPTURE_SHIM|CONTAINER|LOG_DIR|SEED|FR10_METRICS|LUMO_MTP_DRAFT_TRACE_FILE|LUMO_TREE_PATH_LCP_LOG|LUMO_TREE_SAMPLER_DEBUG_LOG|SPEC_CONFIG|TREE|GPU_UTIL|MAX_MODEL_LEN)=' | sort > "${STUB_RECORD:?}"; exit 1
STUB
chmod +x "$SE/$L"; done
sed -i "s#^WT=/home/mark/lumo-paper-v2-20260921\$#WT=$S#" "$SE/e1/e1_cell_driver.v1.sh"
sed -i "s#^WT=/home/mark/lumo-paper-v2-20260921; EXP=#WT=$S; EXP=#" "$SE/e1/e1_run_cells.v2.sh"
export E1_POOL=$EXP/out-20260921T230815Z-e7a-step2-prefixes/prefix_pool.json E1_FROZEN=$EXP/out-20260921T231853Z-e7a-step2-native-score/frozen_prefixes.json E1_SCORES=$EXP/out-20260921T231853Z-e7a-step2-native-score/prefix_scores.json E1_QUAL_MANIFEST=$SE/e1/e1_qualification_manifest.v1.json
export GPU_UTIL=0.9 MAX_MODEL_LEN=8192   # F3: ambient values in a resume shell must NOT reach the launcher
ROOT=$S/campaign; mkdir -p "$ROOT"
run_runner() { STUB_RECORD=$1 bash "$SE/e1/e1_run_cells.v2.sh" "$SE/e1/e1_cells.v1.json" "$ROOT" "$2" "$3" > "$4" 2>&1; echo $?; }
python3 - "$OUT" "$S" "$SE" "$ROOT" <<'PY'
import json, os, subprocess, sys, shutil
out, S, SE, ROOT = sys.argv[1:5]; res = {"checks": {}, "scratch": S}
def check(n, ok, d=None): res["checks"][n] = {"pass": bool(ok), "detail": d}; print(("PASS " if ok else "FAIL ") + n, "" if d is None else str(d)[:180])
def runner(rec, a, b, env_extra=None):
    env = dict(os.environ, STUB_RECORD=rec, **(env_extra or {})); r = subprocess.run(["bash", f"{SE}/e1/e1_run_cells.v2.sh", f"{SE}/e1/e1_cells.v1.json", ROOT, str(a), str(b)], env=env, capture_output=True, text=True, timeout=900); return r
def envof(rec): return dict(l.split("=", 1) for l in open(rec).read().splitlines() if "=" in l) if os.path.exists(rec) else {}
cells = json.load(open(f"{SE}/e1/e1_cells.v1.json"))["cells"]; spec = {c["index"]: c["spec_config"] for c in cells}
# 1) first run: cell 1 (native-5 B1, preflight key) -> stub fails -> exit 10
r = runner(f"{S}/stub_c1.txt", 1, 1); snap = f"{ROOT}/campaign_snapshot"
check("campaign_snapshot_created_with_sums_and_identity", os.path.exists(f"{snap}/SHA256SUMS") and os.path.exists(f"{snap}/campaign_identity.txt") and all(os.path.exists(f"{snap}/{f}") for f in ("e1_cell_driver.v1.sh", "e1_run_cells.v2.sh", "e1_cells.json", "e1_qualification_manifest.json", "E1_FREEZE.md", "e7a_capture_launch.v7.sh", "e1_native_launch.v2.sh", "prefix_pool.json", "frozen_prefixes.json", "prefix_scores.json")) and "prefix_scores.json" in open(f"{snap}/SHA256SUMS").read(), sorted(os.listdir(snap))[:20])
log = open(f"{ROOT}/campaign.log").read()
check("runner_exit_10_on_launcher_failure_cell1", r.returncode == 10 and "cell 1 block 1 native-5 B1 state=PREFLIGHT attempts=0" in log, (r.returncode, log[-200:]))
a1 = f"{ROOT}/cell_01_b1_native-5_B1_a1"; tr = open(f"{a1}/driver_trace.txt").read() if os.path.exists(f"{a1}/driver_trace.txt") else ""
check("attempt_a1_driver_from_its_snapshot_and_campaign_source", "from_snapshot=1" in tr and f"snapshot_dir={a1}/script_snapshot" in tr and open(f"{a1}/script_snapshot/SHA256SUMS").read().count("e1_native_launch.v2.sh") == 1, tr.splitlines()[:2])
e = envof(f"{S}/stub_c1.txt")
check("cell1_frozen_gpu_util_max_model_len_not_ambient", e.get("GPU_UTIL") == "0.6" and e.get("MAX_MODEL_LEN") == "16384", {k: e.get(k) for k in ("GPU_UTIL", "MAX_MODEL_LEN")})
check("cell1_native_frozen_env_exact_spec", e.get("SPEC_CONFIG") == spec[1] and e.get("NUM_SPECULATIVE_TOKENS") == "5" and e.get("FR10_DECODE_MODE_DEFAULT") == "naive_mtp" and e.get("FR10_ENABLE_TREE_GDN") == "0" and e.get("ATTENTION_BACKEND") == "FLASH_ATTN" and e.get("MAX_NUM_SEQS") == "1" and e.get("VLLM_SYNC_SCHED") == "1" and e.get("ENFORCE_EAGER") == "1" and e.get("FR13_ENABLE_APC") == "0" and e.get("FR10_METRICS") == "0" and not e.get("LUMO_MTP_DRAFT_TRACE_FILE") and e.get("E1_RECORD") == "/logs/e1_events.jsonl" and e.get("FR13_SFWD_GPU_TIMER") == "1" and "campaign_snapshot" not in e.get("E1_SHIM_PATH", "") and "script_snapshot/e1_event_recorder_shim.py" in e.get("E1_SHIM_PATH", ""), {k: e.get(k) for k in ("SPEC_CONFIG", "FR10_DECODE_MODE_DEFAULT", "FR10_METRICS", "MAX_NUM_SEQS")})
# 2) rerun without retry flag -> exit 46 (failed attempt retained); with flag -> _a2 created
r2 = runner(f"{S}/stub_c1b.txt", 1, 1); check("rerun_refused_without_retry_flag_exit46", r2.returncode == 46 and not os.path.exists(f"{ROOT}/cell_01_b1_native-5_B1_a2"), r2.returncode)
r3 = runner(f"{S}/stub_c1c.txt", 1, 1, {"E1_RETRY_FAILED": "1"}); check("retry_creates_new_attempt_a2_failure_retained", r3.returncode == 10 and os.path.exists(f"{ROOT}/cell_01_b1_native-5_B1_a2") and os.path.exists(f"{a1}/driver_trace.txt"), r3.returncode)
# 3) tree cells 5/6 via the runner (tree qualified in the manifest)
r5 = runner(f"{S}/stub_c5.txt", 5, 5); e5 = envof(f"{S}/stub_c5.txt")
check("cell5_tree_frozen_env_exact_spec_policyB_no_capture", r5.returncode == 10 and e5.get("SPEC_CONFIG") == spec[5] and "(0, 0, 0, 0, 1)" in e5.get("TREE", "") and e5.get("FR13_ATTN_KV_REMAP") == "1" and e5.get("FR13_SLOT_REORDER") == "0" and e5.get("FR13_KV_REMAP_SYNCFREE") == "1" and e5.get("ATTENTION_BACKEND") == "TREE_ATTN" and e5.get("FR13_REPLAY_ROUTE") == "1" and e5.get("FR10_METRICS") == "0" and not e5.get("LUMO_MTP_DRAFT_TRACE_FILE") and not e5.get("LUMO_TREE_PATH_LCP_LOG") and not e5.get("FR13_FINAL_LOGIT_CAPTURE") and not e5.get("FR10_LAYER_HIDDEN_CAPTURE") and not e5.get("E7B_SUBSTITUTE") and e5.get("E7A_CAPTURE_SHIM") == "0" and e5.get("MAX_NUM_SEQS") == "1", {k: e5.get(k) for k in ("SPEC_CONFIG", "FR13_ATTN_KV_REMAP", "FR10_METRICS", "MAX_NUM_SEQS")})
r6 = runner(f"{S}/stub_c6.txt", 6, 6); e6 = envof(f"{S}/stub_c6.txt"); check("cell6_tree_B4_max_num_seqs_4", r6.returncode == 10 and e6.get("MAX_NUM_SEQS") == "4" and e6.get("SPEC_CONFIG") == spec[6], e6.get("MAX_NUM_SEQS"))
# 4) not-qualified cell is listed and skipped: make a manifest without tree, fresh root
q = json.load(open(f"{SE}/e1/e1_qualification_manifest.v1.json")); q["qualified"] = {}; q2 = f"{S}/manifest_none.json"; json.dump(q, open(q2, "w")); ROOT2 = f"{S}/campaign2"; os.makedirs(ROOT2)
r7 = subprocess.run(["bash", f"{SE}/e1/e1_run_cells.v2.sh", f"{SE}/e1/e1_cells.v1.json", ROOT2, "5", "5"], env=dict(os.environ, STUB_RECORD=f"{S}/stub_x.txt", E1_QUAL_MANIFEST=q2), capture_output=True, text=True, timeout=600)
cr = json.load(open(f"{ROOT2}/cell_05_b1_tree_B1_a1/cell_result.json")) if os.path.exists(f"{ROOT2}/cell_05_b1_tree_B1_a1/cell_result.json") else {}
check("not_qualified_cell_listed_and_skipped", r7.returncode == 0 and cr.get("status") == "NOT_QUALIFIED_NOT_TIMED" and not os.path.exists(f"{S}/stub_x.txt"), (r7.returncode, cr))
# 5a) route-dependency guard: mutate each frozen dependency ALONE (incl. the two dynamically loaded modules) -> exit 45; restore -> the runner proceeds past the guard
ident = open(f"{snap}/campaign_identity.txt").read()
check("identity_lists_dynamic_deps", "scripts/fr13_device_multidraft_kernel.py" in ident and "src/lumo_flywheel_serving/fr13_tree_conv_fused.py" in ident and "src/lumo_flywheel_serving/fr10_decode_modes.py" in ident, [l.split("  ")[1] for l in ident.splitlines() if "  " in l])
for dep in ("scripts/fr13_device_multidraft_kernel.py", "src/lumo_flywheel_serving/fr13_tree_conv_fused.py", "scripts/fr10_phase4_patch_vllm_tree_gdn.py", "src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py", "src/lumo_flywheel_serving/fr10_decode_modes.py"):
    fp = f"{S}/{dep}"; orig = open(fp, "rb").read(); open(fp, "ab").write(b"\n# mutated\n")
    rm = runner(f"{S}/stub_m.txt", 7, 7); open(fp, "wb").write(orig)
    check(f"mutated_{dep.split('/')[-1]}_refused_exit45", rm.returncode == 45 and dep in (rm.stderr + rm.stdout), (rm.returncode, (rm.stderr + rm.stdout)[-160:]))
rr = runner(f"{S}/stub_r.txt", 7, 7); check("restored_deps_pass_guard_again", rr.returncode in (10, 0) and "REFUSING" not in (rr.stderr + rr.stdout), rr.returncode)
# 5b) tamper the campaign snapshot -> exit 45
open(f"{snap}/e1_workload.py", "a").write("\n# tampered\n"); r8 = runner(f"{S}/stub_t.txt", 7, 7); check("tampered_snapshot_refused_exit45", r8.returncode == 45, r8.returncode)
# 6) driver refuses an existing cell dir (exit 44)
r9 = subprocess.run(["bash", f"{SE}/e1/e1_cell_driver.v1.sh", f"{SE}/e1/e1_cells.v1.json", "1", ROOT], env=dict(os.environ, STUB_RECORD=f"{S}/stub_d.txt", E1_RUN_NAME="cell_01_b1_native-5_B1_a1"), capture_output=True, text=True, timeout=300)
check("driver_refuses_existing_cell_dir_exit44", r9.returncode == 44, r9.returncode)
check("no_container_inspect_anywhere", not any("docker_inspect.json" in fs for _, _, fs in os.walk(ROOT)))
res["all_pass"] = all(v["pass"] for v in res["checks"].values()); json.dump(res, open(out, "w"), indent=2); print("ALL PASS" if res["all_pass"] else "SOME FAILED"); sys.exit(0 if res["all_pass"] else 1)
PY
rc=$?; echo "scratch: $S"; exit $rc
