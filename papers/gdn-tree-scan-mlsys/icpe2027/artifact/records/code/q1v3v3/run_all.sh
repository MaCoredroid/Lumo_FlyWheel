#!/usr/bin/env bash
# q1v3 end-to-end driver: native A -> TreeHost candidate -> native B -> native V -> native P -> reducer.
#
#   bash run_all.sh [--wait-idle] [--stages A,CAND,B,V,P] [--run-id ID] [--dry-run]
#
# One GPU engine at a time. Each stage: preflight (no container running, no v2exp queue/pipeline process),
# boot, drive every predeclared observation (client.py), save engine logs, stop + remove ONLY its own container.
# Re-running with the same --run-id skips stages whose STAGE-<arm>.json says complete (resume after a failure).
#   A    native, KV pinned to KV_PIN_BYTES (40 GiB), captures the common O0 of every prefix
#   CAND deployed TreeHost serve-only vehicle (generated q1v3 copy), KV from GPU_UTIL=0.70 (cannot be pinned at B1)
#   B    native, same pin as A, fresh process                       (process repeatability)
#   V    native, KV bytes = the candidate's observed KV tensor bytes (KV geometry matched to the candidate)
#   P    native, pin as A, stock packed recurrent GDN decode         (equivalent-implementation variant)
set -uo pipefail
WT=/home/user/shared/treehost-v2exp-20260930
Q1V3=$WT/scripts/v2exp/q1v3v3
RUNS=${Q1V3_RUNS:-/home/user/shared/treehost-v2exp-runs/q1v3v3/runs}
PY=/home/user/shared/projectwheel-nvfp4-port-20260816/.venv/bin/python
KV_PIN_BYTES=${KV_PIN_BYTES:-42949672960}          # 40 GiB: >= the 34.1 GiB a 131072-token request needs; ~150 blocks
KV_V_FALLBACK_BYTES=${KV_V_FALLBACK_BYTES:-56543355208}  # 52.66 GiB (Codex candidate stage-1 KV) if no candidate boot record
STAGES=A,CAND,B,V,P; WAIT_IDLE=0; DRY=0; RUN_ID=${RUN_ID:-q1v3-$(date -u +%Y%m%dT%H%M%SZ)}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --wait-idle) WAIT_IDLE=1 ;;
    --stages) STAGES=$2; shift ;;
    --run-id) RUN_ID=$2; shift ;;
    --dry-run) DRY=1 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac; shift
done
RUN=$RUNS/$RUN_ID
mkdir -p "$RUN"
exec > >(tee -a "$RUN/driver.log") 2>&1
log(){ echo "[q1v3 $(date -u +%FT%TZ)] $*"; }
log "run=$RUN stages=$STAGES commit=$(git -C "$WT" rev-parse HEAD) dirty=$(git -C "$WT" status --porcelain -- scripts/v2exp/q1v3v3 | wc -l)"

# ---- frozen inputs: protocol, cases and every q1v3 source must match FREEZE.json -------------------------
(cd "$Q1V3" && PYTHONDONTWRITEBYTECODE=1 python3 freeze.py --verify) || { log "FREEZE verification failed; refusing"; exit 2; }
(cd "$Q1V3" && PYTHONDONTWRITEBYTECODE=1 python3 cases.py validate) || { log "case validation failed"; exit 2; }
[[ -e "$RUN/FREEZE.snapshot.json" ]] || cp "$Q1V3/FREEZE.json" "$RUN/FREEZE.snapshot.json"

idle(){ [[ -z "$(docker ps -q)" ]] && [[ $(ps -eo args | grep -E "(v2exp_pipeline[a-z_]*|queue_v2|queue|run_native_arm|run_tree_arm|run_swe_native_arm|run_swe_tree_arm|promoab_[a-z0-9_]*)\.sh" | grep -v grep | wc -l) == 0 ]]; }
preflight(){
  if ! idle; then
    [[ $WAIT_IDLE == 1 ]] || { log "GPU/queue not idle (container or v2exp queue running); refusing (use --wait-idle)"; docker ps; exit 3; }
    log "waiting for the GPU to become idle"; while ! idle; do sleep 60; done; sleep 30
  fi
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader > "$RUN/gpu_apps_$1.txt" 2>&1
  [[ ! -s "$RUN/gpu_apps_$1.txt" ]] || { log "GPU compute processes present before $1"; cat "$RUN/gpu_apps_$1.txt"; exit 3; }
  df -BG --output=avail /home/user/shared | tail -1 | tr -dc 0-9 > "$RUN/disk_$1.txt"
  (( $(cat "$RUN/disk_$1.txt") >= 300 )) || { log "less than 300 GiB free on /home/user/shared"; exit 3; }
}
stage_done(){ [[ -f "$RUN/STAGE-$1.json" ]] && python3 -c "import json,sys; sys.exit(0 if json.load(open('$RUN/STAGE-$1.json')).get('complete') else 1)"; }
write_stage(){ python3 - "$RUN" "$1" "$2" <<'PY'
import json, os, sys
run, arm, rc = sys.argv[1], sys.argv[2], int(sys.argv[3])
s = os.path.join(run, arm, "CLIENT-SUMMARY.json")
summ = json.load(open(s)) if os.path.exists(s) else {}
doc = {"arm": arm, "client_rc": rc, "summary": summ, "complete": rc == 0 and summ.get("valid") == summ.get("planned")}
json.dump(doc, open(os.path.join(run, f"STAGE-{arm}.json"), "w"), indent=1); print(json.dumps(doc))
PY
}

run_native(){  # arm kv_bytes packed
  local ARM=$1 KVB=$2 PACKED=$3 NAME=q1v3-native-$1
  stage_done "$ARM" && { log "stage $ARM already complete; skipping"; return 0; }
  [[ -e "$RUN/$ARM/cases" ]] && { log "stage $ARM has partial output; move $RUN/$ARM aside to rerun it"; return 1; }
  log "stage $ARM: kv_bytes=$KVB packed=$PACKED"
  [[ $DRY == 1 ]] && { (cd "$Q1V3" && PYTHONDONTWRITEBYTECODE=1 python3 client.py --arm "$ARM" --run "$RUN" --plan-only); return 0; }
  preflight "$ARM"
  PYTHONPATH=$WT/src $PY -c "from project_wheel_serving.model_server import recover_host_memory; recover_host_memory()" || log "recover_host_memory failed"
  free -g > "$RUN/free_before_$ARM.txt"
  awk '/^MemFree:/{exit ($2/1048576 < 85)}' /proc/meminfo || { log "MemFree < 85 GiB after recovery"; return 1; }
  mkdir -p "$RUN/$ARM"
  bash "$Q1V3/serve_native_q1v3.sh" "$ARM" "$RUN" "$KVB" "$PACKED" || return 1
  local t0; t0=$(date +%s)
  until curl -sf http://127.0.0.1:9950/health >/dev/null; do
    sleep 10
    docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null | grep -q true || { log "$NAME exited during boot"; docker logs "$NAME" > "$RUN/$ARM/engine.log" 2>&1; docker rm "$NAME" >/dev/null; return 1; }
    (( $(date +%s) - t0 > 1500 )) && { log "$NAME boot timeout"; break; }
  done
  log "$NAME healthy after $(( $(date +%s) - t0 ))s"
  (cd "$Q1V3" && PYTHONDONTWRITEBYTECODE=1 python3 client.py --arm "$ARM" --run "$RUN"); local rc=$?
  docker logs "$NAME" > "$RUN/$ARM/engine.log" 2>&1
  docker stop -t 30 "$NAME" >/dev/null 2>&1; docker rm "$NAME" >/dev/null 2>&1 && log "removed $NAME"
  write_stage "$ARM" "$rc"
  return $rc
}

run_cand(){
  local ARM=CAND
  stage_done "$ARM" && { log "stage CAND already complete; skipping"; return 0; }
  [[ -e "$RUN/$ARM/cases" ]] && { log "stage CAND has partial output; move $RUN/$ARM aside to rerun it"; return 1; }
  mkdir -p "$RUN/$ARM" "$RUN/CAND/serve"
  PYTHONDONTWRITEBYTECODE=1 python3 "$Q1V3/make_cand_launch.py" --out "$RUN/CAND/gen" > "$RUN/CAND/gen.log" || { log "launch-chain generation refused"; cat "$RUN/CAND/gen.log"; return 1; }
  for f in "$RUN"/CAND/gen/*.sh; do bash -n "$f" || { log "generated $f fails bash -n"; return 1; }; done
  [[ $DRY == 1 ]] && { (cd "$Q1V3" && PYTHONDONTWRITEBYTECODE=1 python3 client.py --arm CAND --run "$RUN" --plan-only); return 0; }
  preflight "$ARM"
  local TS; TS=$(date -u +%Y%m%dT%H%M%SZ)
  # Env copied from scripts/v2exp/run_tree_arm.sh (Cqc10 ten-task deployment vehicle, hydra27_fixed32, promoted stack).
  PROMOAB_EXTRA_ENV="FR13_B1_CREDENTIAL_POINTER=/nonexistent
FR13_FA2_QROW32_B1_TIERB_WORKLOAD=exact16_qc_remainder_10
FR13_FA2_QROW32_B1_TIERB_TASK_IDS=astropy__astropy-13977,astropy__astropy-14096,astropy__astropy-14182,astropy__astropy-14309,astropy__astropy-14365,astropy__astropy-14369,astropy__astropy-14508,astropy__astropy-14539,astropy__astropy-14598,astropy__astropy-14995
FR13_FA2_QROW32_B1_TIERB_SUBSET_SHA256=716503a46a991e3b187e14777f96f074c1a3359d9f8b7928f4453a6b9da1ee9b" \
  ARM_KIND=C PROMOAB_FA2=default PROMOAB_TOPK=promoted PROMOAB_KIND=hydra27_fixed32 PROMOAB_SUBSET=exact16_qc_remainder_10 \
  PROMOAB_ARM_SUFFIX=_q1v3$TS Q1V3_SRC="$Q1V3" Q1V3_RUN="$RUN" Q1V3_SERVE_ROOT="$RUN/CAND/serve" \
  Q1V3_VARIANT="$RUN/CAND/gen/variant_q1v3.sh" Q1V3_LAUNCHER="$RUN/CAND/gen/launcher_q1v3.sh" \
    setsid bash "$RUN/CAND/gen/promoab_q1v3.sh" > "$RUN/CAND/promoab.out" 2>&1 < /dev/null &
  local DRV=$! RR="" t0; t0=$(date +%s)
  log "candidate vehicle pid=$DRV"
  while :; do
    sleep 15
    RR=$(ls -d "$RUN"/CAND/serve/fr14_promoab_C_q1v3${TS}_* 2>/dev/null | tail -1)
    [[ -n "$RR" && -f "$RR/READY.json" ]] && break
    kill -0 $DRV 2>/dev/null || { log "vehicle exited before READY"; tail -40 "$RUN/CAND/promoab.out"; write_stage CAND 1; return 1; }
    (( $(date +%s) - t0 > 3000 )) && { log "READY timeout"; [[ -n "$RR" ]] && touch "$RR/STOP"; wait $DRV; write_stage CAND 1; return 1; }
  done
  log "candidate READY after $(( $(date +%s) - t0 ))s: $RR"
  python3 - "$RUN/CAND/patch_receipt.json" <<'PY' || { log "candidate hook patch receipt missing/unclean"; touch "$RR/STOP"; wait $DRV; write_stage CAND 1; return 1; }
import json, sys
d = json.load(open(sys.argv[1])); print(json.dumps({k: d[k] for k in ("applied", "problems")}))
sys.exit(0 if d["applied"] and not d["problems"] else 1)
PY
  grep -q "gqa_pair_splitk" "$(ls "$RR"/*/launch.log | head -1)" || log "WARNING: deployed split-K attention arm not visible in launch.log"
  (cd "$Q1V3" && PYTHONDONTWRITEBYTECODE=1 python3 client.py --arm CAND --run "$RUN" --ready "$RR/READY.json"); local rc=$?
  touch "$RR/STOP"; log "stop requested; waiting for vehicle teardown"; wait $DRV; log "vehicle rc=$?"
  local SUF; SUF=hydra27_fixed32_promoab_C_q1v3$TS
  for c in $(docker ps -aq --filter "name=fr13-bigdenom-$SUF"); do
    docker logs "$c" > "$RUN/CAND/container_after_teardown.log" 2>&1
    docker stop -t 30 "$c" >/dev/null 2>&1; docker rm "$c" >/dev/null && log "removed own container $c"
  done
  write_stage CAND "$rc"
  return $rc
}

kv_for_v(){ python3 - "$RUN/CAND" "$KV_V_FALLBACK_BYTES" <<'PY'
import glob, json, sys
vals = [json.load(open(p)).get("kv_cache", {}).get("tensor_bytes_total") for p in glob.glob(sys.argv[1] + "/boot.*.json")]
vals = [v for v in vals if isinstance(v, int) and v > 0]
print(vals[0] if vals else sys.argv[2])
PY
}

FAILED=0
IFS=',' read -ra ST <<< "$STAGES"
for s in "${ST[@]}"; do
  case "$s" in
    A) run_native A "$KV_PIN_BYTES" 0 || { FAILED=1; log "stage A failed; later stages need its O0 -> stop"; break; } ;;
    CAND) run_cand || { FAILED=1; log "stage CAND failed (continuing with native stages; reducer will report)"; } ;;
    B) run_native B "$KV_PIN_BYTES" 0 || FAILED=1 ;;
    V) run_native V "$(kv_for_v)" 0 || FAILED=1 ;;
    P) run_native P "$KV_PIN_BYTES" 1 || FAILED=1 ;;
    *) log "unknown stage $s"; exit 2 ;;
  esac
done
[[ $DRY == 1 ]] && { log "dry run complete"; exit 0; }
log "reducing"
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 python3 "$Q1V3/reduce.py" --run "$RUN" --out "$RUN/VERDICT.json"; RRC=$?
log "done failed_stages=$FAILED reducer_rc=$RRC verdict=$(python3 -c "import json;print(json.load(open('$RUN/VERDICT.json'))['verdict'])" 2>/dev/null)"
exit $(( FAILED || RRC ))
