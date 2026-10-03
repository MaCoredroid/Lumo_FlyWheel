#!/usr/bin/env bash
# Sweep continuation (2026-10-02): the gated 3-pass arms (G, A3) are held after G1/A1 showed a
# suffix-chain collapse after the handoff on gated steps (suspected implementation defect).
# Remaining: confirmation-set capture, deployed replicate, native MTP-5 drift control.
set -uo pipefail
ROOT=/home/mark/shared/lumotree-v2exp-runs; V=/home/mark/shared/lumotree-v2exp-20260930/scripts/v2exp
LOG=$ROOT/sweep.log
log(){ echo "[sweep $(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
idle(){ for i in $(seq 1 720); do
  if [[ -z "$(docker ps -q)" ]] && ! pgrep -f "run_all.sh|v2exp_pipeline|dist_sample.py|replay.py|run_swe_|run_tree_arm|run_native_arm|gatediag.sh" >/dev/null; then return 0; fi; sleep 10; done; return 1; }
step(){ local name=$1; shift; idle || { log "not idle; abort before $name"; exit 3; }
  log "start: $name"; env "$@"; local rc=$?; log "end: $name rc=$rc"; }
log "queue3 start $(git -C /home/mark/shared/lumotree-v2exp-20260930 rev-parse --short HEAD) (gated arms held)"
step "gatediag (passive trace, gate permissive)" bash $V/gatediag/gatediag.sh
step "capture (native mtp5, dumps on, 4 tasks)" bash $V/run_swe_capture.sh mtp5 $ROOT/capture/subset_capture4.json 1500
step "T2 deployed hydra27"         V2_ARM_KIND=C V2_TAG=-swT2 bash $V/run_tree_arm.sh sampled
step "M1 native mtp5 (drift control)" V2_TAG=-swM1 bash $V/run_native_arm.sh mtp5 sampled
log "queue3 complete"
