#!/usr/bin/env bash
# Generalization check on the frozen disjoint confirmation set (4 unseen SWE tasks, 43 requests):
# deployed LumoTree, native MTP-5, SGLang EAGLE s7k1d8 — same replay harness and sampling as the tuning corpus.
set -uo pipefail
ROOT=/home/mark/shared/lumotree-v2exp-runs; V=/home/mark/shared/lumotree-v2exp-20260930/scripts/v2exp
REQ=$ROOT/confirm/requests
log(){ echo "[confirm $(date -u +%FT%TZ)] $*" | tee -a "$ROOT/sweep.log"; }
idle(){ for i in $(seq 1 360); do
  if [[ -z "$(docker ps -q)" ]] && ! pgrep -f "run_tree_arm|run_native_arm|run_swe_|gatediag.sh|gatefix.sh|replay.py" >/dev/null; then return 0; fi; sleep 10; done; return 1; }
step(){ local name=$1; shift; idle || { log "not idle; abort before $name"; exit 3; }
  log "start: $name"; env "$@"; local rc=$?; log "end: $name rc=$rc"; }
log "confirm queue start $(git -C /home/mark/shared/lumotree-v2exp-20260930 rev-parse --short HEAD) set=$(sha256sum $ROOT/confirm/MANIFEST.json | cut -c1-12)"
step "CT1 deployed LumoTree on confirm set" V2_ARM_KIND=C V2_TAG=-cfT1 V2_REQUESTS=$REQ bash $V/run_tree_arm.sh sampled
step "CM1 native mtp5 on confirm set" V2_TAG=-cfM1 V2_REQUESTS=$REQ bash $V/run_native_arm.sh mtp5 sampled
step "CS1 sglang s7k1d8 on confirm set" V2_TAG=-cfS1 V2_REQUESTS=$REQ bash $V/run_native_arm.sh sglang-s7k1d8 sampled
log "confirm queue complete"
