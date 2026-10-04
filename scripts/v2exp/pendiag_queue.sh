#!/usr/bin/env bash
# Penalty-history capture after the q1v3 v3 harvest.
set -uo pipefail
ROOT=/home/mark/shared/lumotree-v2exp-runs; V=/home/mark/shared/lumotree-v2exp-20260930/scripts/v2exp
log(){ echo "[pendiag $(date -u +%FT%TZ)] $*" | tee -a "$ROOT/sweep.log"; }
while pgrep -f "q1v3v3_harvest_queue[.]sh|confirm_reps[.]sh" >/dev/null; do sleep 30; done
for i in $(seq 1 120); do [[ -z "$(docker ps -q)" ]] && break; sleep 10; done
log "start: penalty-history capture"
PD_TAG=-pd1 PD_HOOKS=1 PD_LIMIT=4 PD_MAXTOK=256 bash $V/pendiag/pendiag.sh
log "end: penalty-history capture rc=$?"
