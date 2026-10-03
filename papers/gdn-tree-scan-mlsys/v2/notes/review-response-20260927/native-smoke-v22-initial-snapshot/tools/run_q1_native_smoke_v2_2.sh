#!/usr/bin/env bash
# Q1 full-model NATIVE SMOKE launcher v2.2 = v2.1 + the preliminary review's two lifecycle gaps: (1) ONE owner for cleanup/finalization (finalize() once;
# the EXIT trap never re-runs cleanup or rewrites sealed receipts after finalization), (2) Docker enumeration failure refused (never read as 'absent'), and
# (3) the cleanup library is sourced only AFTER the gate has validated its hash.  v2.1 = v2 + the reviewer's cleanup-contract fix (v2 stop_engine ignored a failed docker stop/rm, wrote engine_stopped_utc
# anyway and the terminal exit used only the driver rc): stop_engine/finish_run now live in the sourced q1_native_smoke_cleanup_v2_1.sh (gate-bound):
# terminal state is VERIFIED after stop, logs/inspect and cleanup failures are preserved, the stopped timestamp is written only after a verified stopped
# state, an owned container that cannot be stopped/removed makes the terminal status non-zero, and the trap never masks a driver failure.
# (Q1-NATIVE-SMOKE-DESIGN.json, sha 8e5f96be...): ONE fresh spec-off reference engine (aligned_nonpacked
# primary arm, patched FA2 fork through the ordinary one-token causal FLASH_ATTN path, mamba_cache_mode align), ONE case
# calibration-short_available__c0__root-only (shortest frozen prefix, |P| = 13487; chain [r, z] = [11352, 25559] at positions [13487, 13488]),
# R = 2 requests driven by q1_reference_driver_v2 with hash+length authentication of every sealed observation and raw object.
# NOT executed by anyone but the parent-gated launch: refuses unless GATE-Q1-NATIVE-SMOKE.json carries gate "Q1-NATIVE-SMOKE", approved:true,
# approved_run_id == RUN_ID, the exact reviewed scope and reviewed_hashes equal to every executable dependency listed below.
# Any failure (gate, contention, boot, install/patch receipt, health, driver stop) stops the run, preserves everything and exits non-zero. No retry.
set -uo pipefail
REPO=/home/mark/lumotree-review-20260927
C=papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927
MON=$REPO/papers/gdn-tree-scan-mlsys/v2/p0/monitor/review-response-20260927
GATE=${SMOKE_GATE:-$MON/GATE-Q1-NATIVE-SMOKE.json}
DESIGN=$MON/Q1-NATIVE-SMOKE-DESIGN.json
IMAGE=sha256:ffa30d66ff5c9346c6389507cc529827fc9934a6d2ee37855934f94fe1061cdc
IMAGE_DIGEST=vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776
FORK_SO=/home/mark/fr14_splitk_build_20260818/_vllm_fa2_qrow32_gqa_pair_splitk_b1_sm121a.abi3.so
FORK_SHA=28570f835ea72c99d03aab9fb03c494388bbb9c264ee4dc96eec047f50d7f857; FORK_SIZE=300123792
ARM=aligned_nonpacked; PROCESS=A; REPEATS=2; CASE=calibration-short_available__c0__root-only; EXPECTED_REQUESTS=2
HEALTH_TIMEOUT_S=${HEALTH_TIMEOUT_S:-2400}; URL=http://127.0.0.1:9950
DRY=0; [[ "${1:-}" == "--dry-run" ]] && DRY=1
RUN_ID=${RUN_ID:-UNSET}
OUT=$REPO/$C/runs/q1-native-smoke/$RUN_ID
TOOLS=$REPO/$C/tools
FIX=$REPO/$C/fullmodel/fixtures/token-fixtures.v1.json
PREFIX=$REPO/$C/prefix-plan/selected/calibration-short_available/token-ids.u32le
sha() { sha256sum "$1" | cut -d' ' -f1; }
CONTAINER=lumotree-q1-ref-$ARM-$RUN_ID
CLEANUP_LIB=$TOOLS/q1_native_smoke_cleanup_v2_2.sh            # sourced only after the gate validates its hash (below)
echo "RUN_ID=$RUN_ID"; echo "OUT=$OUT"; echo "ARM=$ARM PROCESS=$PROCESS CASE=$CASE REPEATS=$REPEATS EXPECTED_REQUESTS=$EXPECTED_REQUESTS"; echo "CONTAINER=$CONTAINER (port 9950)"
echo "STEPS: gate+hash check -> contention/port/image/fork/prefix checks -> job (q1_reference_job.py --smoke) -> config render (q1_spec_off_engine_config_v2.py) -> LAUNCH-BINDING -> docker run -d (fa2 install + runner patch + vllm serve) -> /health wait -> q1_reference_driver_v2.py (R=2) -> docker stop + logs -> RUN-RECEIPT"
if [[ $DRY == 1 ]]; then echo "(dry-run: nothing launched, nothing written)"; exit 0; fi
if [[ "$RUN_ID" == "UNSET" ]]; then echo "RUN_ID must be set to the gate's approved_run_id"; exit 3; fi
if [[ -e "$OUT" ]]; then echo "output run directory already exists: $OUT"; exit 7; fi
python3 - "$GATE" "$DESIGN" "$IMAGE" "$FORK_SO" "$FORK_SHA" "$FORK_SIZE" "$TOOLS" "$FIX" "$PREFIX" "$0" "$RUN_ID" "$REPO" "$C" "$ARM" "$REPEATS" "$CASE" "$EXPECTED_REQUESTS" "$CLEANUP_LIB" <<'PY'
import json, sys, hashlib, os
gate, design, image, fork, fork_sha, fork_size, tools, fix, prefix, launcher, run_id, repo, c, arm, repeats, case, expected_requests, cleanup_lib = sys.argv[1:19]
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for ch in iter(lambda: f.read(1 << 22), b""): h.update(ch)
    return h.hexdigest()
if not os.path.exists(gate): print("GATE-Q1-NATIVE-SMOKE REFUSES LAUNCH: gate file missing", gate); sys.exit(3)
g = json.load(open(gate)); problems = []
if g.get("gate") != "Q1-NATIVE-SMOKE": problems.append("gate file is not GATE-Q1-NATIVE-SMOKE")
if g.get("approved") is not True: problems.append("approved is not true")
if g.get("approved_run_id") != run_id: problems.append(f"approved_run_id {g.get('approved_run_id')} != RUN_ID {run_id}")
sc = g.get("reviewed_scope") or {}
want_scope = {"arm": arm, "process": "A", "repeats": int(repeats), "fresh_processes": 1, "case_ids": [case], "expected_requests": int(expected_requests), "save_kv_bytes": True, "block": "calibration"}
for k, v in want_scope.items():
    if sc.get(k) != v: problems.append(f"reviewed_scope.{k} = {sc.get(k)!r} != required {v!r}")
rh = g.get("reviewed_hashes") or {}
camp = os.path.join(repo, c)
expect = {"image_id": image, "smoke_design": sha(design), "hooks_v2": sha(os.path.join(tools, "q1_reference_hooks_v2.py")), "driver_v2": sha(os.path.join(tools, "q1_reference_driver_v2.py")),
          "patcher_v2": sha(os.path.join(tools, "q1_patch_reference_runner_v2.py")), "config_v2": sha(os.path.join(tools, "q1_spec_off_engine_config_v2.py")),
          "fa2_install": sha(os.path.join(tools, "q1_reference_fa2_install.py")), "job_builder": sha(os.path.join(tools, "q1_reference_job.py")),
          "fa2_source_index": sha(os.path.join(camp, "identity", "native_source", "fa2_source_index.json")), "token_fixtures": sha(fix), "prefix_token_ids": sha(prefix),
          "stock_runner_copy": sha(os.path.join(camp, "identity", "native_source", "vllm__v1__worker__gpu_model_runner.py")),
          "fa2_patcher": sha(os.path.join(repo, "scripts", "fr13_patch_fa2_tree_bias.py")), "lm_head_shim": sha(os.path.join(repo, "scripts", "fr14_patch_nvfp4_lmhead.py")),
          "fork_binary": sha(fork) if os.path.exists(fork) else None, "launcher": sha(launcher), "cleanup_lib": sha(cleanup_lib)}
for k, v in expect.items():
    if rh.get(k) != v: problems.append(f"reviewed_hashes.{k} = {rh.get(k)} != current {v}")
if expect["fork_binary"] != fork_sha or (os.path.exists(fork) and os.path.getsize(fork) != int(fork_size)): problems.append("fork binary sha/size != pinned")
fx = json.load(open(fix)); pre = [p for p in fx["prefixes"] if p["prefix_id"] == "calibration-short_available"][0]
if sha(prefix) != pre["token_ids_sha256"] or os.path.getsize(prefix) != 4 * int(pre["prefix_len"]): problems.append("frozen prefix token bytes != fixture identity")
if problems:
    print("GATE-Q1-NATIVE-SMOKE REFUSES LAUNCH:"); [print(" -", p) for p in problems]; sys.exit(3)
expect["gate"] = sha(gate)
json.dump({"expect": expect, "scope": want_scope, "run_id": run_id}, open("/tmp/q1_smoke_binding_stage.json", "w"))
print("GATE-Q1-NATIVE-SMOKE approved with matching reviewed hashes, scope and run id")
PY
rc=$?; [[ $rc -ne 0 ]] && exit $rc
. "$CLEANUP_LIB"                                              # hash validated by the gate above; provides owned_container_state / stop_engine / finalize / on_exit_trap
if ! nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > /tmp/q1_smoke_gpu_before.txt 2>&1; then echo "nvidia-smi contention check FAILED to run"; exit 4; fi
if [[ $(wc -l < /tmp/q1_smoke_gpu_before.txt) -gt 1 ]]; then echo "GPU busy:"; cat /tmp/q1_smoke_gpu_before.txt; exit 4; fi
RUNNING_NAMES=$(docker ps --format '{{.Names}}') || { echo "docker ps enumeration FAILED (rc=$?); refusing (a query failure is not an empty result)"; exit 4; }
if grep -qE '^lumotree-' <<<"$RUNNING_NAMES"; then echo "a lumotree container is already running:"; echo "$RUNNING_NAMES"; exit 4; fi
ALL_NAMES=$(docker ps -a --format '{{.Names}}') || { echo "docker ps -a enumeration FAILED (rc=$?); refusing"; exit 4; }
if grep -qx "$CONTAINER" <<<"$ALL_NAMES"; then echo "container name already exists: $CONTAINER"; exit 4; fi
if (ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | grep -q ":9950 "; then echo "port 9950 in use"; exit 4; fi
docker image inspect "$IMAGE" > /tmp/q1_smoke_image.json || { echo "image inspect failed"; exit 5; }
grep -q "$IMAGE_DIGEST" /tmp/q1_smoke_image.json || { echo "image digest mismatch"; exit 5; }
mkdir -p "$OUT/q1_ref" "$OUT/driver"
mv /tmp/q1_smoke_gpu_before.txt "$OUT/gpu_contention_before.txt"; mv /tmp/q1_smoke_image.json "$OUT/image_inspect.json"
cp "$GATE" "$OUT/GATE-Q1-NATIVE-SMOKE.snapshot.json"; cp "$DESIGN" "$OUT/Q1-NATIVE-SMOKE-DESIGN.snapshot.json"
nvidia-smi --query-gpu=name,driver_version,clocks.sm,temperature.gpu,memory.used --format=csv > "$OUT/gpu_state_before.txt"
RECEIPT_WRITTEN=0
write_receipt() {
  python3 - "$OUT" "$1" <<'PY'
import json, os, sys, hashlib, datetime
out, status = sys.argv[1], sys.argv[2]
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for ch in iter(lambda: f.read(1 << 22), b""): h.update(ch)
    return h.hexdigest()
files = {}; objs = 0; obytes = 0
for root, dirs, fns in os.walk(out):
    for fn in fns:
        p = os.path.join(root, fn); rel = os.path.relpath(p, out)
        if rel.startswith("q1_ref/objects/"):
            objs += 1; obytes += os.path.getsize(p); continue
        if rel == "RUN-RECEIPT.json": continue
        files[rel] = {"sha256": sha(p), "bytes": os.path.getsize(p)}
def rd(rel):
    p = os.path.join(out, rel); return open(p).read().strip() if os.path.exists(p) else None
rc = {"schema": "lumo.q1.fullmodel.native-smoke-run-receipt.v2", "run": out, "status": status, "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
      "files_recursive_excluding_object_store": files, "object_store": {"objects": objs, "bytes": obytes},
      "engine": {"started_utc": rd("engine_started_utc.txt"), "healthy_utc": rd("engine_healthy_utc.txt"), "stopped_utc": rd("engine_stopped_utc.txt"), "exit": rd("engine_exit.txt"),
                 "cleanup_state": rd("engine_cleanup_state.txt"), "cleanup_err": rd("engine_cleanup.err"), "finalized": rd("FINALIZED.txt"), "cleanup_contract": "single finalization owner; stopped_utc only after verified stopped state; docker enumeration failure never read as absent; non-zero terminal status if the owned container could not be stopped/removed"},
      "driver": {"started_utc": rd("driver_started_utc.txt"), "ended_utc": rd("driver_ended_utc.txt"), "exit": rd("driver_exit.txt")},
      "note": "native instrumentation/calibration smoke only: no candidate, no held-out, no timing claim, no qualification"}
json.dump(rc, open(os.path.join(out, "RUN-RECEIPT.json"), "w"), indent=1); print("receipt:", status, "files:", len(files), "objects:", objs)
PY
}
trap 'rc=$?; on_exit_trap $rc; frc=$?; if [[ $rc -eq 0 && $frc -ne 0 ]]; then exit $frc; fi' EXIT   # single owner: no-op once finalized; an unexpected abort is finalized exactly once; never masks a failure
# ---- job (hash-bound; the engine refuses to boot without it) ----
python3 "$TOOLS/q1_reference_job.py" --fixtures "$FIX" --run-id "$RUN_ID" --arm "$ARM" --process "$PROCESS" --repeats "$REPEATS" --smoke --out-dir-in-container /logs/q1_ref --control-path-in-container /logs/q1_ref_control.json --out "$OUT/q1_ref_hooks_job.json" || { echo "job build failed"; exit 5; }
JOB_SHA=$(sha "$OUT/q1_ref_hooks_job.json")
# ---- engine configuration rendered for THIS run/log dir/job (validate() must report no problems) ----
python3 "$TOOLS/q1_spec_off_engine_config_v2.py" --run-id "$RUN_ID" --log-dir "$OUT" --job-sha256 "$JOB_SHA" --out "$OUT/engine-config.json" || { echo "engine config render reported problems"; exit 5; }
python3 - "$OUT" "$ARM" "$JOB_SHA" "$0" <<'PY'
import json, sys, hashlib, os, datetime
out, arm, job_sha, launcher = sys.argv[1:5]
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for ch in iter(lambda: f.read(1 << 22), b""): h.update(ch)
    return h.hexdigest()
st = json.load(open("/tmp/q1_smoke_binding_stage.json")); cfg = json.load(open(os.path.join(out, "engine-config.json"))); a = cfg["arms"][arm]
assert cfg["problems"] == [], cfg["problems"]
assert a["env"]["Q1_REF_HOOKS_JOB_SHA256"] == job_sha and a["env"]["VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE"] == "0" and f"{out}:/logs" in " ".join(a["docker_argv_without_script"])
st.update({"schema": "lumo.q1.fullmodel.native-smoke-launch-binding.v2", "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "arm": arm, "job_sha256": job_sha,
           "engine_config_sha256": sha(os.path.join(out, "engine-config.json")), "rendered_command_sha256": hashlib.sha256(a["rendered_command_NOT_EXECUTED"].encode()).hexdigest(),
           "container_name": a["container_name"], "snapshots": {"gate": "GATE-Q1-NATIVE-SMOKE.snapshot.json", "design": "Q1-NATIVE-SMOKE-DESIGN.snapshot.json", "job": "q1_ref_hooks_job.json", "engine_config": "engine-config.json"}})
json.dump(st, open(os.path.join(out, "LAUNCH-BINDING.json"), "x"), indent=1)
json.dump({"argv": a["docker_argv_without_script"], "script": a["in_container_script"]}, open(os.path.join(out, "engine_launch_argv.json"), "x"), indent=1)
os.remove("/tmp/q1_smoke_binding_stage.json")
PY
[[ $? -ne 0 ]] && { echo "launch binding failed"; exit 5; }
# ---- boot the reference engine exactly as rendered (docker run -d ... bash -lc "<install fa2; patch runner; exec vllm serve>") ----
date -u +%FT%TZ > "$OUT/engine_started_utc.txt"
python3 - "$OUT" <<'PY'
import json, subprocess, sys, os
out = sys.argv[1]; la = json.load(open(os.path.join(out, "engine_launch_argv.json")))
r = subprocess.run(la["argv"] + [la["script"]], capture_output=True, text=True)
open(os.path.join(out, "docker_run.out"), "w").write(r.stdout); open(os.path.join(out, "docker_run.err"), "w").write(r.stderr)
sys.exit(r.returncode)
PY
[[ $? -ne 0 ]] && { echo "docker run failed"; cat "$OUT/docker_run.err"; exit 6; }
echo "engine container started; waiting for $URL/health (timeout ${HEALTH_TIMEOUT_S}s)"
t0=$(date +%s); healthy=0
while :; do
  if curl -fsS --max-time 5 "$URL/health" > /dev/null 2>&1; then healthy=1; break; fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then echo "engine container exited before becoming healthy"; break; fi
  if (( $(date +%s) - t0 > HEALTH_TIMEOUT_S )); then echo "health timeout"; break; fi
  sleep 10
done
if [[ $healthy -ne 1 ]]; then finalize "ENGINE_NOT_HEALTHY" 6; exit 6; fi
date -u +%FT%TZ > "$OUT/engine_healthy_utc.txt"
# install + patch receipts must exist and be clean before any request
python3 - "$OUT" <<'PY'
import json, sys, os
out = sys.argv[1]; pr = []
for rel, key in (("q1_ref_fa2_install_receipt.json", "problems"), ("q1_ref_patch_receipt.json", None)):
    p = os.path.join(out, rel)
    if not os.path.exists(p): pr.append(f"{rel} missing"); continue
    d = json.load(open(p))
    if key and d.get(key): pr.append(f"{rel}: {d[key]}")
if pr: print("BOOT RECEIPTS NOT CLEAN:", pr); sys.exit(6)
print("boot receipts present and clean")
PY
[[ $? -ne 0 ]] && { finalize "BOOT_RECEIPTS_NOT_CLEAN" 6; exit 6; }
# ---- drive R=2 forced requests; the driver authenticates every seal + raw object and stops on the first failure ----
date -u +%FT%TZ > "$OUT/driver_started_utc.txt"
python3 "$TOOLS/q1_reference_driver_v2.py" --job "$OUT/q1_ref_hooks_job.json" --prefix-plan-dir "$REPO/$C/prefix-plan" --out-dir "$OUT/driver" --control-path "$OUT/q1_ref_control.json" --hooks-out-dir "$OUT/q1_ref" --url "$URL/v1/completions" > "$OUT/driver.log" 2>&1
DRV_RC=$?
echo "exit=$DRV_RC" > "$OUT/driver_exit.txt"; date -u +%FT%TZ > "$OUT/driver_ended_utc.txt"; tail -n 20 "$OUT/driver.log"
finalize "COMPLETED_driver_rc=${DRV_RC}" "$DRV_RC"; exit $?
