#!/usr/bin/env bash
# Sequential GPU queue for the v2 replay benchmark. One server at a time.
set -uo pipefail
ROOT=/home/mark/shared/lumotree-v2exp-runs
V=/home/mark/shared/lumotree-v2exp-20260930/scripts/v2exp
QLOG=$ROOT/queue.log
log(){ echo "[queue $(date -u +%FT%TZ)] $*" | tee -a "$QLOG"; }
idle(){ [[ -z "$(docker ps -q)" ]] && ! pgrep -f "run_native_arm.sh|run_tree_arm.sh|promoab_serve_only.sh" >/dev/null; }
wait_idle(){ while ! idle; do sleep 30; done; sleep 20; }
[[ -e $ROOT/QUEUE-STOP ]] && rm -f $ROOT/QUEUE-STOP
if [[ -n "${QUEUE_STEPS:-}" ]]; then IFS=';' read -ra STEPS <<< "$QUEUE_STEPS"
else STEPS=("tree sampled" "native mtp3 greedy" "native ar sampled"); fi
for step in "${STEPS[@]}"; do
  wait_idle
  [[ -e $ROOT/QUEUE-STOP ]] && { log "QUEUE-STOP present; halting before: $step"; exit 0; }
  set -- $step
  log "start: $step"
  if [[ $1 == tree ]]; then bash $V/run_tree_arm.sh "$2" 1024; rc=$?
  else bash $V/run_native_arm.sh "$2" "$3" 1024; rc=$?; fi
  log "end: $step rc=$rc"
  if [[ $rc != 0 ]]; then log "step failed; halting queue for inspection"; exit 1; fi
done
log "queue complete"
