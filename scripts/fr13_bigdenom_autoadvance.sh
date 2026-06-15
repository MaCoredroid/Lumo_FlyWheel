#!/usr/bin/env bash
# FR13 big-denominator AUTO-ADVANCE orchestrator (GPU-SERIALIZED).
#
# Drives the remaining campaign stages WITHOUT overlapping the serialized GPU:
#   wait native_a teardown -> recover -> launch cat9_a (full arm) -> wait its
#   teardown -> recover -> run Phase 3 rescore (both arms) -> consolidate.
#
# native_a is assumed ALREADY RUNNING (launched separately). This script never
# launches a second concurrent vLLM; it waits for each arm's terminal markers
# (arm_ended_at.txt + pairdump_stats.json + no fr13-bigdenom/swe-codex docker)
# before starting the next GPU stage. Idempotent-ish: skips cat9_a if it already
# has terminal markers.
#
# Emits one progress line per stage transition to stdout (Monitor-friendly:
# STAGE=/DONE=/FAIL= prefixes).
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
REPO=/home/mark/shared/lumoFlyWheel
RUNROOT=output/fr13_bigdenom_swe
SUBSET=subset_astropy12907.json

log() { echo "$(date -u +%H:%M:%S) $*"; }

recover() {
  PYTHONPATH="$REPO/src" .venv/bin/python - <<'PY' 2>/dev/null || true
from lumo_flywheel_serving.model_server import recover_host_memory
recover_host_memory()
PY
}

# An arm is DONE when its teardown trap wrote arm_ended_at.txt AND no live
# fr13-bigdenom / swe-codex container remains for this campaign.
arm_done() {  # <arm_dir_name>
  local a="$1"
  [[ -f "$RUNROOT/$a/arm_ended_at.txt" ]] || return 1
  # no campaign serving/agent container still up
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -qE "fr13-bigdenom|swe-codex"; then
    return 1
  fi
  return 0
}

wait_arm_done() {  # <arm_dir_name>
  local a="$1"
  log "STAGE=WAIT_$a (polling teardown markers)"
  while ! arm_done "$a"; do sleep 20; done
  log "DONE=$a teardown observed; n_pairdumps=$(ls "$RUNROOT/$a/proxy_pair_dumps"/pair_*.json 2>/dev/null | wc -l)"
}

# ---- 1. wait for native_a (already running) to finish + recover ----
if arm_done native_a; then
  log "native_a already done"
else
  wait_arm_done native_a
fi
recover
sleep 5

# ---- 2. cat9_a full arm (the SECOND GPU stage) ----
if arm_done cat9_a; then
  log "cat9_a already done; skipping launch"
else
  log "STAGE=LAUNCH_cat9_a"
  bash scripts/fr13_bigdenom_swe_serve.sh cat9_a cat9 "$SUBSET" \
    > "$RUNROOT/cat9_a.serve.log" 2>&1 &
  CAT9_PID=$!
  log "cat9_a serve launched pid=$CAT9_PID"
  wait "$CAT9_PID" || log "WARN cat9_a serve exited nonzero (rc=$?); checking teardown markers anyway"
fi
# confirm terminal markers (teardown trap may have run on early exit)
if ! arm_done cat9_a; then
  log "FAIL=cat9_a did not reach terminal markers (no arm_ended_at.txt or container still up)"
  exit 3
fi
recover
sleep 5

# ---- 3. NON-VACUITY pre-check: both arms have non-empty pair-dumps ----
for a in native_a cat9_a; do
  n=$(ls "$RUNROOT/$a/proxy_pair_dumps"/pair_*.json 2>/dev/null | wc -l)
  if (( n == 0 )); then log "FAIL=$a pair-dump empty (vacuous #9)"; exit 4; fi
  log "DONE=$a pairdumps=$n"
done

# ---- 4. Phase 3 rescore (third GPU stage: both arms, recurrent oracle) ----
log "STAGE=PHASE3_RESCORE"
bash scripts/fr13_bigdenom_phase3_rescore.sh > "$RUNROOT/phase3.log" 2>&1
RC=$?
if (( RC != 0 )); then
  log "FAIL=phase3 rescore rc=$RC (see $RUNROOT/phase3.log)"
  exit 5
fi
recover
log "DONE=PHASE3 consolidated=output/fr13_bigdenom_rescore/consolidated.json"
log "CAMPAIGN_COMPLETE"
