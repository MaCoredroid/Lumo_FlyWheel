#!/usr/bin/env bash
# E1 CELL DRIVER v1 (2026-09-22): runs ONE frozen E1 cell (e1/e1_cells.v1.json) as a fresh, uniquely owned server boot with
# the E1 event recorder, then the frozen workload (warm-up + timed phases, mandatory API token ids), then shutdown (recorder
# seal), post-boot verification, API map + joiner + per-prompt/per-cohort summary. Immutable: every script it will execute
# is copied into <cell>/script_snapshot/ (SHA256SUMS) and the driver re-execs its own snapshot copy (trace line).
# Helpers (precheck/port_free/wait_health/record_boot/cleanup/arm_oom_guard) are verbatim from e7a/serve_drivers.v12.sh.
# NO-HEAVY-CAPTURE RULE (parent 2026-09-22): timed arms carry only the E1 recorder + forward timer; FR10_METRICS=0 and no
# draft/sampler/path traces (they clone full logits / copy drafts to the CPU); engagement proof comes from the server log.
# Arms: tree -> e7a_capture_launch.v7.sh (KV policy B explicit, no capture shims, no E7b runtime); native-5/native-11 ->
# e1_native_launch.v2.sh (stock MTP method in the patched runner with every FR13 feature OFF — label "patched-runner native
# baseline"; same recorder/timer instrumentation). Nothing here is a qualification: the runner decides eligibility.
# Usage: [E1_SOURCE_DIR=<campaign_snapshot>] [E1_RUN_NAME=cell_..._aK] [E1_PREFLIGHT=1] e1_cell_driver.v1.sh <cells.json> <cell_index> <campaign_root>
# E1_PREFLIGHT=1 (native first planned boot per arm x batch): frozen untimed preflight segment -> live pass/fail -> fixed warm-up
# -> timed, all in the same boot; a FAIL issues no warm-up/timed request (exit 7). Existing cell dirs are REFUSED (exit 44).
set -euo pipefail
WT=/home/mark/lumo-paper-v2-20260921
EXP=$WT/papers/gdn-tree-scan-mlsys/v2/experiments
E=${E1_SOURCE_DIR:-$EXP/e1}   # the campaign runner passes its frozen campaign_snapshot here
IMAGE="vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776"
POOL=${E1_POOL:-$EXP/out-20260921T230815Z-e7a-step2-prefixes/prefix_pool.json}
FROZEN=${E1_FROZEN:-$EXP/out-20260921T231853Z-e7a-step2-native-score/frozen_prefixes.json}
RUN=""; C=""
precheck() {  # $1 out file: fail-closed GPU inventory, memory/swap guard  (v1 fix: `apps`/`_rc` are LOCAL — a global `rc=$?`
  # here would overwrite cleanup()'s `local rc` through bash's dynamic scoping and turn every abort into exit 0)
  local d=$1 apps _rc
  { echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; free -g; swapon --show || true
    echo "--- containers ---"; docker ps --format '{{.Names}} {{.Image}} {{.Status}}' || true
    echo "--- vllm/EngineCore procs ---"; { ps -eo pid,etime,rss,args | grep -i "vllm\|EngineCore" | grep -v grep | cut -c1-200; } || echo "(none)"
  } > "$d" 2>&1
  set +e; apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1); _rc=$?; set -e
  echo "--- compute apps (nvidia-smi rc=$_rc) ---" >> "$d"; echo "$apps" >> "$d"
  if (( _rc != 0 )); then echo "REFUSING: nvidia-smi compute-apps query FAILED (rc=$_rc): $apps" >&2; return 3; fi
  if [[ -n "$(echo "$apps" | tr -d '[:space:]')" ]]; then echo "REFUSING: a compute process is present on the GPU (see $d)" >&2; return 3; fi
  local avail_kib swap_used_kib
  avail_kib=$(awk '/MemAvailable/ {print $2}' /proc/meminfo); swap_used_kib=$(awk '/SwapTotal/ {t=$2} /SwapFree/ {f=$2} END {print t-f}' /proc/meminfo)
  echo "MemAvailable_GiB=$((avail_kib/1048576)) SwapUsed_GiB=$((swap_used_kib/1048576))" >> "$d"
  if (( swap_used_kib > 0 && avail_kib < swap_used_kib + 8388608 )); then echo "REFUSING: not enough RAM to swap in (avail=$((avail_kib/1048576))GiB, swap used=$((swap_used_kib/1048576))GiB)" >&2; return 3; fi
}
port_free() { ! (ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]$1$"); }
wait_health() {  # $1 port, $2 container, $3 timeout s ; health must belong to $2
  local port=$1 c=$2 t=$3 i=0 mapped
  while (( i < t )); do
    if [[ -z "$(docker ps -q -f name="^${c}$")" ]]; then echo "container $c exited"; docker logs "$c" 2>&1 | tail -40; return 4; fi
    if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      mapped=$(docker port "$c" 9950/tcp 2>/dev/null | head -1 || true)
      if [[ "$mapped" != *":$port" ]]; then echo "health answered on :$port but container $c maps '$mapped'"; return 7; fi
      echo "healthy after ${i}s (container $c maps $mapped)"; return 0
    fi
    sleep 5; i=$((i+5))
  done
  echo "health timeout"; docker logs "$c" 2>&1 | tail -40; return 5
}
record_boot() {  # $1 run_dir $2 container $3 launcher
  local d=$1 c=$2 l=$3
  docker inspect "$c" > "$d/docker_inspect.json" 2>&1 || true
  sha256sum "$l" > "$d/launcher.sha256"
  git -C "$WT" rev-parse HEAD > "$d/worktree_head.txt"
  { echo "worktree_sha256=$(sha256sum "$WT/scripts/fr10_phase4_patch_vllm_tree_gdn.py" | cut -d' ' -f1)";
    echo "worktree_blob=$(git -C "$WT" hash-object "$WT/scripts/fr10_phase4_patch_vllm_tree_gdn.py")";
    echo "HEAD_blob=$(git -C "$WT" rev-parse HEAD:scripts/fr10_phase4_patch_vllm_tree_gdn.py)";
    echo "differs_from_HEAD=$([[ "$(git -C "$WT" hash-object "$WT/scripts/fr10_phase4_patch_vllm_tree_gdn.py")" == "$(git -C "$WT" rev-parse HEAD:scripts/fr10_phase4_patch_vllm_tree_gdn.py)" ]] && echo no || echo yes)"; } > "$d/patcher_identity.txt"
  git -C "$WT" diff -- scripts/fr10_phase4_patch_vllm_tree_gdn.py > "$d/patcher_worktree_vs_HEAD.diff" 2>/dev/null || true
  { echo "image_digest=$IMAGE"; docker image inspect "$IMAGE" --format '{{.Id}} {{.Architecture}}'; } > "$d/image_identity.txt" 2>&1 || true
  docker inspect "$c" --format '{{range .Mounts}}{{.Source}}->{{.Destination}} rw={{.RW}}{{"\n"}}{{end}}' > "$d/mounts.txt" 2>&1 || true
  if grep -q "^/models->/models rw=true" "$d/mounts.txt"; then echo "REFUSING: /models mounted read-write" >&2; return 8; fi
}
cleanup() {  # EXIT/ERR: save logs/identity of OUR container only, then stop it gracefully (recorder seal at atexit) and remove it
  local rc=$?
  if [[ -n "$C" ]] && docker ps -a -q -f name="^${C}$" | grep -q .; then
    docker logs "$C" > "$RUN/docker_logs.txt" 2>&1 || true
    sha256sum "$WT/scripts/fr10_phase4_patch_vllm_tree_gdn.py" > "$RUN/patcher.sha256.final" 2>/dev/null || true
    docker inspect "$C" > "$RUN/docker_inspect_final.json" 2>&1 || true
    docker stop -t 30 "$C" >/dev/null 2>&1 || true
    docker rm -f "$C" >/dev/null 2>&1 || true
    echo "cleanup: removed owned container $C (rc=$rc)" >> "$RUN/driver_trace.txt"
  fi
  [[ -n "$RUN" ]] && { echo "exit_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) rc=$rc" >> "$RUN/driver_trace.txt"; precheck "$RUN/precheck_after.txt" >/dev/null 2>&1 || true; }
  exit $rc
}
trap cleanup EXIT
arm_oom_guard() {  # $1 container name, $2 run dir: start the REPOSITORY guard scoped to OUR container and VERIFY it is alive
  local c=$1 d=$2 cid
  cid=$(docker inspect -f '{{.Id}}' "$c" 2>/dev/null || true)
  [[ "$cid" =~ ^[0-9a-f]{64}$ ]] || { echo "REFUSING: cannot resolve immutable container id for $c" | tee -a "$d/driver_trace.txt" >&2; return 12; }
  GPU_GUARD_CONTAINER_ID="$cid" GPU_GUARD_EXPECTED_NAME="$c" GPU_GUARD_LOG="$d/gpu_oom_guard.log" \
    setsid bash "$SNAP/gpu_oom_guard.sh" >> "$d/gpu_oom_guard.stdout" 2>&1 </dev/null &
  local gp=$!
  disown 2>/dev/null || true
  sleep 4
  if kill -0 "$gp" 2>/dev/null && grep -q "gpu_oom_guard START .*id=$cid expected_name=$c" "$d/gpu_oom_guard.log" 2>/dev/null; then
    echo "guard_pid=$gp container_id=$cid container_name=$c alive=true utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$d/driver_trace.txt"
  else
    echo "REFUSING: OOM guard did not stay alive (pid $gp) — not proceeding unguarded" | tee -a "$d/driver_trace.txt" >&2
    return 12
  fi
}
# ---------------------------------------------------------------- snapshot + re-exec (immutability)
CELLS=$1; IDX=$2; ROOT=$3
if [[ -z "${E1_FROM_SNAPSHOT:-}" ]]; then
  CELL=${E1_RUN_NAME:-$(python3 - "$CELLS" "$IDX" <<'PYC'
import json, sys
c = [x for x in json.load(open(sys.argv[1]))["cells"] if x["index"] == int(sys.argv[2])][0]
print("cell_%02d_b%d_%s_%s_a1" % (c["index"], c["block"], c["arm"], c["batch"]))
PYC
)}
  RUN="$ROOT/$CELL"
  [[ -e "$RUN" ]] && { echo "REFUSING: cell artifacts already exist at $RUN (attempts are never overwritten; use a new attempt dir)" >&2; exit 44; }
  mkdir -p "$RUN/script_snapshot" "$RUN/logs"
  for f in e1_cell_driver.v1.sh e1_native_launch.v2.sh e1_workload.py e1_cell_verify.py e1_cell_summary.py e1_native_preflight.py e1_recorder.py e1_event_recorder_shim.py e1_join.py e1_api_tokens_from_capture.v2.py e7a_capture_launch.v7.sh; do
    if [[ -f "$E/$f" ]]; then cp "$E/$f" "$RUN/script_snapshot/$f"; elif [[ -f "$E/../e7a/$f" ]]; then cp "$E/../e7a/$f" "$RUN/script_snapshot/$f"; else echo "REFUSING: source $f not found under $E" >&2; exit 43; fi
  done
  cp "$CELLS" "$RUN/script_snapshot/e1_cells.json"; cp "$WT/scripts/gpu_oom_guard.sh" "$RUN/script_snapshot/gpu_oom_guard.sh"
  cp "$FROZEN" "$RUN/script_snapshot/frozen_prefixes.json"; sha256sum "$POOL" > "$RUN/pool.sha256"
  SCORES=${E1_SCORES:-$E/prefix_scores.json}   # F2: the native reference scores used by the preflight greedy rule come from the campaign snapshot
  if [[ "${E1_PREFLIGHT:-0}" == "1" ]]; then [[ -f "$SCORES" ]] || { echo "REFUSING: native reference prefix_scores.json not found at $SCORES (E1_SCORES / campaign snapshot)" >&2; exit 43; }; cp "$SCORES" "$RUN/script_snapshot/prefix_scores.json"; fi
  ( cd "$RUN/script_snapshot" && sha256sum * > SHA256SUMS ) || true
  export E1_FROM_SNAPSHOT=1 E1_SNAPSHOT_DIR="$RUN/script_snapshot" E1_RUN="$RUN"
  exec bash "$RUN/script_snapshot/e1_cell_driver.v1.sh" "$@"
fi
SNAP=$E1_SNAPSHOT_DIR; RUN=$E1_RUN; CELLS=$SNAP/e1_cells.json
eval "$(python3 - "$CELLS" "$IDX" <<'PYC'
import json, sys
c = [x for x in json.load(open(sys.argv[1]))["cells"] if x["index"] == int(sys.argv[2])][0]
import shlex
fs = json.load(open(sys.argv[1]))["frozen_settings"]
print("ARM=%s; BATCH=%s; NSEQ=%d; NSPEC=%d; BLOCK=%d; SPEC_CONFIG_FROZEN=%s; GPU_UTIL_FROZEN=%s; MAX_MODEL_LEN_FROZEN=%d" % (c["arm"], c["batch"], c["max_num_seqs"], c["num_speculative_tokens"], c["block"], shlex.quote(c["spec_config"]), fs["gpu_util"], int(fs["max_model_len"])))
PYC
)"
TREE_FROZEN=$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); print(d.get("speculative_token_tree",""))' "$SPEC_CONFIG_FROZEN")
PORT=9950
echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) cell=$IDX block=$BLOCK arm=$ARM batch=$BATCH max_num_seqs=$NSEQ gpu_util_frozen=$GPU_UTIL_FROZEN max_model_len_frozen=$MAX_MODEL_LEN_FROZEN spec_config_frozen=$SPEC_CONFIG_FROZEN" > "$RUN/driver_trace.txt"
echo "driver_exec=$(basename "$0") sha256=$(sha256sum "$0" | cut -c1-16) from_snapshot=${E1_FROM_SNAPSHOT:-0} snapshot_dir=${E1_SNAPSHOT_DIR:-}" >> "$RUN/driver_trace.txt"
precheck "$RUN/precheck_before.txt"
port_free "$PORT" || { echo "REFUSING: port $PORT in use" >&2; exit 9; }
C="e1-cell$(printf '%02d' "$IDX")-$(date -u +%H%M%S)"
RUNREL=${RUN#"$WT"/}
COMMON=(REPO="$WT" CONTAINER="$C" PORT="$PORT" GPU_UTIL="$GPU_UTIL_FROZEN" MAX_MODEL_LEN="$MAX_MODEL_LEN_FROZEN" MAX_NUM_SEQS="$NSEQ" SEED=20260921 ENFORCE_EAGER=1 GPU_OOM_GUARD=0 LOG_DIR="$RUN/logs" SPEC_CONFIG="$SPEC_CONFIG_FROZEN" TREE="$TREE_FROZEN" NUM_SPECULATIVE_TOKENS="$NSPEC"
        E1_SHIM_PATH="/workspace/$RUNREL/script_snapshot/e1_event_recorder_shim.py" E1_RECORD=/logs/e1_events.jsonl E1_RUNTIME_DIR="/workspace/$RUNREL/script_snapshot" FR13_SFWD_GPU_TIMER=1 VLLM_SYNC_SCHED=1)
case "$ARM" in
  tree)
    LAUNCHER=$SNAP/e7a_capture_launch.v7.sh
    env "${COMMON[@]}" ATTENTION_BACKEND=TREE_ATTN FR13_REPLAY_ROUTE=1 FR13_EAGER_PACK=1 FR13_TREE_CONV_FUSED=1 FR13_TREE_RUNROW_INIT=1 FR13_ENABLE_APC=0 E7A_CAPTURE_SHIM=0 FR10_METRICS=0 \
        FR13_ATTN_KV_REMAP=1 FR13_SLOT_REORDER=0 FR13_KV_REMAP_SYNCFREE=1 \
        FR13_FORCE_SPINE_COMMIT=0 FR13_COMMIT_ARGMAX_GATE=0 FR13_FINAL_LOGIT_CAPTURE="" FR10_LAYER_HIDDEN_CAPTURE="" FR13_DECODE_GDN_CAPTURE="" FR10_TREE_GDN_CAPTURE_PAYLOAD="" \
        LUMO_TREE_PATH_LCP_LOG="" LUMO_MTP_DRAFT_TRACE_FILE="" LUMO_TREE_SAMPLER_DEBUG_LOG="" \
        bash "$LAUNCHER" > "$RUN/launcher_stdout.txt" 2>&1 ;;   # no-heavy-capture rule: no metrics/draft/sampler traces in timed arms (only the recorder)
  native-5|native-11)
    LAUNCHER=$SNAP/e1_native_launch.v2.sh
    env "${COMMON[@]}" ATTENTION_BACKEND=FLASH_ATTN FR13_ENABLE_APC=0 FR10_DECODE_MODE_DEFAULT=naive_mtp FR10_ENABLE_TREE_GDN=0 FR10_METRICS=0 LUMO_MTP_DRAFT_TRACE_FILE="" \
        bash "$LAUNCHER" > "$RUN/launcher_stdout.txt" 2>&1 ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac
record_boot "$RUN" "$C" "$LAUNCHER"
arm_oom_guard "$C" "$RUN"
wait_health "$PORT" "$C" 1500 | tee "$RUN/health.txt"
mkdir -p "$RUN/loaded_backend"
for f in vllm/v1/attention/backends/tree_attn.py vllm/v1/worker/gpu_model_runner.py vllm/v1/sample/rejection_sampler.py vllm/model_executor/layers/mamba/gdn_linear_attn.py vllm/v1/spec_decode/eagle.py; do
  docker cp "$C:/usr/local/lib/python3.12/dist-packages/$f" "$RUN/loaded_backend/$(basename "$f")" 2>/dev/null || echo "loaded_backend copy failed: $f" >> "$RUN/driver_trace.txt"
done
( cd "$RUN/loaded_backend" && sha256sum * > SHA256SUMS ) 2>/dev/null || true
curl -s "http://127.0.0.1:$PORT/v1/models" > "$RUN/models.json" || true
if [[ "${E1_PREFLIGHT:-0}" == "1" ]]; then
  # NATIVE UNTIMED PREFLIGHT (frozen segment; pass/fail BEFORE any warm-up/timed request; same boot; no threshold tuning)
  echo "preflight_start_utc=$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" >> "$RUN/driver_trace.txt"
  python3 "$SNAP/e1_workload.py" "http://127.0.0.1:$PORT" "$POOL" "$SNAP/frozen_prefixes.json" "$RUN" --batch "$NSEQ" --phase preflight > "$RUN/workload_preflight_stdout.txt" 2>&1 || { echo "PREFLIGHT WORKLOAD FAILED (cell invalid)" >&2; tail -5 "$RUN/workload_preflight_stdout.txt"; exit 6; }
  docker logs "$C" > "$RUN/docker_logs_preflight.txt" 2>&1 || true
  if ! python3 "$SNAP/e1_native_preflight.py" "$RUN" --arm "$ARM" --batch "$NSEQ" --nspec "$NSPEC" --live --scores "$SNAP/prefix_scores.json" > "$RUN/native_preflight_live.log" 2>&1; then
    echo "preflight_verdict=FAIL utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$RUN/driver_trace.txt"; echo "NATIVE PREFLIGHT FAILED — no warm-up/timed requests issued; cell INVALID" >&2; grep '^FAIL' "$RUN/native_preflight_live.log"; exit 7
  fi
  echo "preflight_verdict=PASS utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$RUN/driver_trace.txt"
fi
echo "workload_start_utc=$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" >> "$RUN/driver_trace.txt"
python3 "$SNAP/e1_workload.py" "http://127.0.0.1:$PORT" "$POOL" "$SNAP/frozen_prefixes.json" "$RUN" --batch "$NSEQ" --phase main > "$RUN/workload_stdout.txt" 2>&1 || { echo "WORKLOAD FAILED (cell invalid)" >&2; tail -5 "$RUN/workload_stdout.txt"; exit 6; }
echo "workload_end_utc=$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" >> "$RUN/driver_trace.txt"
curl -s "http://127.0.0.1:$PORT/metrics" > "$RUN/metrics_after.txt" 2>/dev/null || true
# graceful stop -> recorder seal (atexit); then archive (cleanup trap does the rest)
docker logs "$C" > "$RUN/docker_logs.txt" 2>&1 || true
docker stop -t 60 "$C" >/dev/null 2>&1 || true; sleep 2
docker logs "$C" > "$RUN/docker_logs.txt" 2>&1 || true
docker inspect "$C" > "$RUN/docker_inspect_final.json" 2>&1 || true
docker rm -f "$C" >/dev/null 2>&1 || true; echo "container_stopped_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$RUN/driver_trace.txt"; C=""
echo "cell boot done: $RUN"
