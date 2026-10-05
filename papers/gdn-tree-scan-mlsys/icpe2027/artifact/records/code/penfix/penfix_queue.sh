#!/usr/bin/env bash
# FR13_TREE_PENALTY_HISTORY re-qualification and re-measurement queue (2026-10-03).
#   PF0   capture: deployed TreeHost with the fixed patcher + record-only pendiag wrapper (4 requests x 256 tokens);
#         HISTORY GATE: every captured tree row must carry its per-path penalty history, or the queue stops.
#         The full Monte-Carlo sampler test then runs on CPU concurrently with the GPU steps.
#   PF1-3 tuning-corpus replays (43 requests, sampled, cap 1024)      cfPF1-3 confirmation-set replays (43 requests)
#   q1v3 v3 continuation (frozen protocol, fresh run id, all five arms)  SWE ten-task TreeHost arm
# Native arms are untouched (chain histories are correct for chains) and are not repeated.
set -uo pipefail
ROOT=/home/user/shared/treehost-v2exp-runs; WT=/home/user/shared/treehost-v2exp-20260930; V=$WT/scripts/v2exp
PY=/home/user/shared/projectwheel-nvfp4-port-20260816/.venv/bin/python
RES=$V/results/sampling_20261003; mkdir -p "$RES"
log(){ echo "[penfix $(date -u +%FT%TZ)] $*" | tee -a "$ROOT/sweep.log"; }
idle(){ for i in $(seq 1 720); do
  if [[ -z "$(docker ps -q)" ]] && ! pgrep -f "run_tree_arm|run_native_arm|run_swe_|gatediag.sh|gatefix.sh|pendiag.sh|replay.py|q1v3v3/run_all" >/dev/null; then return 0; fi; sleep 10; done; return 1; }
step(){ local name=$1; shift; idle || { log "not idle; abort before $name"; exit 3; }
  log "start: $name"; env "$@"; local rc=$?; log "end: $name rc=$rc"; return $rc; }
PSHA=$(sha256sum $WT/scripts/fr10_phase4_patch_vllm_tree_gdn.py | cut -c1-16)
cmp -s $WT/scripts/fr10_phase4_patch_vllm_tree_gdn.py /home/user/shared/projectwheel-nvfp4-port-20260816/scripts/fr10_phase4_patch_vllm_tree_gdn.py || { log "patcher copies differ; refusing"; exit 2; }
log "penfix queue start commit=$(git -C $WT rev-parse --short HEAD) patcher_sha256=$PSHA"
step "PF0 capture (fixed patcher, record-only pendiag, 4 req x 256 tok)" PD_TAG=-pf0 PD_HOOKS=0 PD_LIMIT=4 PD_MAXTOK=256 bash $V/pendiag/pendiag.sh || { log "capture failed"; exit 4; }
TRACE=$(ls -t $ROOT/replay/tree-pf0-sampled-*/pen_trace.jsonl | head -1); [[ -s "$TRACE" ]] || { log "no pen_trace"; exit 4; }
echo "$TRACE" > "$RES/pf0_trace_path.txt"
PYTHONDONTWRITEBYTECODE=1 $PY $V/pendiag/analyze_pen.py "$TRACE" "$RES/pen_analysis_pf0_history.json" --steps 0 > "$RES/pen_analysis_pf0_history.log" 2>&1
python3 - "$RES/pen_analysis_pf0_history.json" <<'PY' || { log "HISTORY GATE FAILED (see $RES/pen_analysis_pf0_history.json); queue stopped"; exit 5; }
import json, sys
h = json.load(open(sys.argv[1]))["history"]; print("history", h)
sys.exit(0 if h["rows"] > 0 and h["correct_match"] == h["rows"] else 1)
PY
log "HISTORY GATE PASSED: $(python3 -c "import json;print(json.load(open('$RES/pen_analysis_pf0_history.json'))['history'])")"
( PYTHONDONTWRITEBYTECODE=1 nice -n 10 $PY $V/pendiag/analyze_pen.py "$TRACE" "$RES/pen_analysis_pf0_mc.json" --steps 6 --walks 200000 > "$RES/pen_analysis_pf0_mc.log" 2>&1; log "MC sampler analysis rc=$? -> $RES/pen_analysis_pf0_mc.json" ) &
for i in 1 2 3; do step "PF$i TreeHost + penalty fix, tuning corpus" V2_ARM_KIND=C V2_TAG=-pfT$i bash $V/run_tree_arm.sh sampled; done
for i in 1 2 3; do step "cfPF$i TreeHost + penalty fix, confirmation set" V2_ARM_KIND=C V2_TAG=-cfPF$i V2_REQUESTS=$ROOT/confirm/requests bash $V/run_tree_arm.sh sampled; done
step "q1v3 v3 continuation, fixed patcher" bash $V/q1v3v3/run_all.sh --wait-idle --run-id q1v3v3-penfix-$(date -u +%Y%m%dT%H%M%SZ)
step "SWE ten-task TreeHost arm, fixed patcher" bash $V/run_swe_tree_arm.sh
wait
log "penfix queue complete"
