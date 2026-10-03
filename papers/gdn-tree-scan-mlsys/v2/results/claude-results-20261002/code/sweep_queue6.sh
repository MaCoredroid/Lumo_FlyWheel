#!/usr/bin/env bash
# After queue4 (T2, M1): gate-fix arms (runtime fix for duplicate padding), then the capture retry.
set -uo pipefail
ROOT=/home/mark/shared/lumotree-v2exp-runs; V=/home/mark/shared/lumotree-v2exp-20260930/scripts/v2exp
log(){ echo "[sweep $(date -u +%FT%TZ)] $*" | tee -a "$ROOT/sweep.log"; }
idle(){ for i in $(seq 1 720); do
  if [[ -z "$(docker ps -q)" ]] && ! pgrep -f "run_tree_arm|run_native_arm|run_swe_|gatediag.sh|gatefix.sh|replay.py" >/dev/null; then return 0; fi; sleep 10; done; return 1; }
step(){ local name=$1; shift; idle || { log "not idle; abort before $name"; exit 3; }
  log "start: $name"; env "$@"; local rc=$?; log "end: $name rc=$rc"; }
while pgrep -f "sweep_queue4[.]sh" >/dev/null; do sleep 30; done
log "queue6 start $(git -C /home/mark/shared/lumotree-v2exp-20260930 rev-parse --short HEAD)"
step "AFdiag fixed always-3, passive trace, 8 req" GF_TAG=-swAFdiag GF_NGRAM=1 GF_AGREE=0 GF_HIST=1 GF_HOOKS=1 GF_LIMIT=8 GF_MAXTOK=512 bash $V/gatefix/gatefix.sh
step "GF1 fixed gate (pre-registered predicate)" GF_TAG=-swGF1 bash $V/gatefix/gatefix.sh
step "AF1 fixed always-3 (permissive)" GF_TAG=-swAF1 GF_NGRAM=1 GF_AGREE=0 GF_HIST=1 bash $V/gatefix/gatefix.sh
step "capture retry (native mtp5, dumps on, 4 tasks)" bash $V/run_swe_capture.sh mtp5 $ROOT/capture/subset_capture4.json 1500
log "queue6 complete"
