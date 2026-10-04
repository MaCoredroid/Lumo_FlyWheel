#!/usr/bin/env bash
# q1v3 v3 natural-path harvest: deployed LumoTree (ARM_KIND=C, gate off) with record-only hooks on 6 fresh
# prefixes from unseen SWE tasks. Runs after the confirmation replicates.
set -uo pipefail
ROOT=/home/mark/shared/lumotree-v2exp-runs; V=/home/mark/shared/lumotree-v2exp-20260930/scripts/v2exp
log(){ echo "[q1v3v3 $(date -u +%FT%TZ)] $*" | tee -a "$ROOT/sweep.log"; }
while pgrep -f "confirm_reps[.]sh" >/dev/null; do sleep 30; done
for i in $(seq 1 120); do [[ -z "$(docker ps -q)" ]] && break; sleep 10; done
log "start: natural-path harvest (6 fresh prefixes)"
GD_ARM_KIND=C GD_LIMIT=6 GD_MAXTOK=160 GD_REQUESTS=$ROOT/q1v3v3/harvest_requests bash $V/gatediag/gatediag.sh
log "end: natural-path harvest rc=$?"
