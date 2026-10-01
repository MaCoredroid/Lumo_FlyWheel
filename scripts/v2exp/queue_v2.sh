#!/usr/bin/env bash
# Sequential GPU queue for the v2 replay benchmark. One server at a time.
set -uo pipefail
ROOT=/home/mark/shared/lumotree-v2exp-runs
V=/home/mark/shared/lumotree-v2exp-20260930/scripts/v2exp
QLOG=$ROOT/queue.log
log(){ echo "[queue $(date -u +%FT%TZ)] $*" | tee -a "$QLOG"; }
idle(){ [[ -z "$(docker ps -q)" ]] && [[ $(ps -eo args | awk '$1=="bash" && $2 ~ /(run_native_arm|run_tree_arm|run_swe_native_arm|run_swe_tree_arm|promoab_[a-z0-9_]*)\.sh$/' | wc -l) == 0 ]]; }
wait_idle(){ while ! idle; do sleep 30; done; sleep 20; }
[[ -e $ROOT/QUEUE-STOP ]] && rm -f $ROOT/QUEUE-STOP
if [[ -n "${QUEUE_STEPS:-}" ]]; then IFS=';' read -ra STEPS <<< "$QUEUE_STEPS"
else STEPS=("tree sampled" "native mtp3 greedy" "native ar sampled"); fi
for step in "${STEPS[@]}"; do
  wait_idle
  [[ -e $ROOT/QUEUE-STOP ]] && { log "QUEUE-STOP present; halting before: $step"; exit 0; }
  set -- $step
  log "start: $step"
  if [[ $1 == tree ]]; then mode=$2; shift 2; env "$@" bash $V/run_tree_arm.sh "$mode" 1024; rc=$?
  elif [[ $1 == swe-tree ]]; then shift 1; env "$@" bash $V/run_swe_tree_arm.sh; rc=$?
  elif [[ $1 == swe-native ]]; then arm=$2; shift 2; env "$@" bash $V/run_swe_native_arm.sh "$arm"; rc=$?
  else bash $V/run_native_arm.sh "$2" "$3" 1024; rc=$?; fi
  log "end: $step rc=$rc"
  # leave no own container behind before the next step
  for c in $(docker ps -aq --filter "name=v2exp"); do docker stop -t 30 "$c" >/dev/null 2>&1; docker rm "$c" >/dev/null 2>&1 && log "cleaned own container $c"; done
  if [[ $rc != 0 && "${QUEUE_CONTINUE_ON_FAIL:-0}" != 1 ]]; then log "step failed; halting queue for inspection"; exit 1; fi
done
log "queue complete"
