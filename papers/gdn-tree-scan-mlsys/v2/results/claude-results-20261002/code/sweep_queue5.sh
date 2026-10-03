#!/usr/bin/env bash
# Confirmation-set capture (retry with tilde-safe dump-dir check), after queue4 completes.
set -uo pipefail
ROOT=/home/mark/shared/lumotree-v2exp-runs; V=/home/mark/shared/lumotree-v2exp-20260930/scripts/v2exp
log(){ echo "[sweep $(date -u +%FT%TZ)] $*" | tee -a "$ROOT/sweep.log"; }
while pgrep -f "sweep_queue4[.]sh" >/dev/null; do sleep 30; done
for i in $(seq 1 120); do [[ -z "$(docker ps -q)" ]] && break; sleep 10; done
[[ -z "$(docker ps -q)" ]] || { log "queue5: not idle; abort"; exit 3; }
log "start: capture retry (native mtp5, dumps on, 4 tasks)"
bash $V/run_swe_capture.sh mtp5 $ROOT/capture/subset_capture4.json 1500; log "end: capture retry rc=$?"
