#!/usr/bin/env bash
# FR13_TREE_PENALTY_HISTORY re-qualification, stage 2 (2026-10-03): continuation v3 then the SWE ten-task arm.
# Split from penfix_queue.sh because q1v3v3/run_all.sh's idle regex ((...|queue|...)\.sh) matched the queue's own
# file name and waited on its parent forever. This file name matches none of run_all.sh's patterns.
set -uo pipefail
ROOT=/home/mark/shared/lumotree-v2exp-runs; WT=/home/mark/shared/lumotree-v2exp-20260930; V=$WT/scripts/v2exp
log(){ echo "[penfix $(date -u +%FT%TZ)] $*" | tee -a "$ROOT/sweep.log"; }
idle(){ for i in $(seq 1 720); do
  if [[ -z "$(docker ps -q)" ]] && ! pgrep -f "run_tree_arm|run_native_arm|run_swe_|gatediag.sh|gatefix.sh|pendiag.sh|replay.py|q1v3v3/run_all" >/dev/null; then return 0; fi; sleep 10; done; return 1; }
step(){ local name=$1; shift; idle || { log "not idle; abort before $name"; exit 3; }
  log "start: $name"; env "$@"; local rc=$?; log "end: $name rc=$rc"; return $rc; }
cmp -s $WT/scripts/fr10_phase4_patch_vllm_tree_gdn.py /home/mark/shared/lumoFlyWheel-nvfp4-port-20260816/scripts/fr10_phase4_patch_vllm_tree_gdn.py || { log "patcher copies differ; refusing"; exit 2; }
log "penfix stage 2 start commit=$(git -C $WT rev-parse --short HEAD) patcher_sha256=$(sha256sum $WT/scripts/fr10_phase4_patch_vllm_tree_gdn.py | cut -c1-16)"
step "q1v3 v3 continuation, fixed patcher" bash $V/q1v3v3/run_all.sh --wait-idle --run-id q1v3v3-penfix-$(date -u +%Y%m%dT%H%M%SZ)
step "SWE ten-task LumoTree arm, fixed patcher" bash $V/run_swe_tree_arm.sh
log "penfix stage 2 complete"
