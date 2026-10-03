#!/usr/bin/env bash
# Runs after v2exp_pipeline.sh (incl. the SWE study) finishes: distribution diagnostic v3, then the full-model
# correctness check (q1v3) if its runner exists. Logs to pipeline.log.
set -uo pipefail
ROOT=/home/mark/shared/lumotree-v2exp-runs
V=/home/mark/shared/lumotree-v2exp-20260930/scripts/v2exp
LOG=$ROOT/pipeline.log
log(){ echo "[pipeline-after $(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
cd "$ROOT"
log "waiting for v2exp_pipeline.sh to finish"
while ps -eo args | awk '$1=="bash" && $2 ~ /v2exp_pipeline\.sh$/' | grep -q .; do sleep 60; done
sleep 30
log "stage 5: distribution diagnostic v3 (frozen DIST-PROTOCOL-v3.md)"
mkdir -p "$ROOT/dist3"
S5="native ar sampled V2_SEED=21 V2_TAG=-v3s21;tree sampled V2_TAG=-v3a;native mtp5 sampled V2_SEED=31 V2_TAG=-v3s31;native ar sampled V2_SEED=22 V2_TAG=-v3s22;tree sampled V2_TAG=-v3b;native ar sampled V2_SEED=41 V2_DIST_TEMP=0.65 V2_TAG=-v3neg;native ar sampled V2_SEED=23 V2_TAG=-v3s23;native mtp5 sampled V2_SEED=32 V2_TAG=-v3s32;tree sampled V2_TAG=-v3c;native ar sampled V2_SEED=24 V2_TAG=-v3s24"
V2_DIST=1 V2_DIST_SAMPLES=40 V2_DIST_MAXTOK=24 V2_DIST_ORDER=request-outer QUEUE_CONTINUE_ON_FAIL=1 QUEUE_STEPS="$S5" \
  bash "$V/queue_v2.sh" > "$ROOT/queue_dist3.out" 2>&1
log "stage 5 done rc=$?"
if [[ -x "$V/q1v3/run_q1v3.sh" || -f "$V/q1v3/run_q1v3.sh" ]]; then
  sleep 30; log "stage 6: full-model correctness (q1v3)"
  bash "$V/q1v3/run_q1v3.sh" > "$ROOT/q1v3_run.out" 2>&1
  log "stage 6 done rc=$?"
else
  log "stage 6 skipped: q1v3/run_q1v3.sh not present"
fi
log "pipeline-after complete"
