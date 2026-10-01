#!/usr/bin/env bash
# Runs the remaining v2exp GPU work end to end, one stage at a time, independent of any chat session.
# Stages: wait for old queue to halt -> corrected distribution queue (frozen DIST-PROTOCOL.md) ->
# kernel benchmark (smoke, then full if smoke passes) -> identical-token SGLang x3 + stock-attention LumoTree x2
# -> same-build SWE study (LumoTree, MTP-5, SGLang 7/8). Each stage logs to pipeline.log.
set -uo pipefail
ROOT=/home/mark/shared/lumotree-v2exp-runs
V=/home/mark/shared/lumotree-v2exp-20260930/scripts/v2exp
LOG=$ROOT/pipeline.log
log(){ echo "[pipeline $(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
qrun(){ ps -eo args | awk '$1=="bash" && $2 ~ /(queue|queue_v2)\.sh$/' | wc -l; }
cd "$ROOT"
log "start; waiting for the old distribution queue to halt"
while [[ $(qrun) != 0 ]]; do sleep 30; done
rm -f "$ROOT/QUEUE-STOP"
log "old queue halted; stage 1: corrected distribution queue"
V2_DIST=1 V2_DIST_SAMPLES=40 V2_DIST_MAXTOK=24 QUEUE_CONTINUE_ON_FAIL=1 \
QUEUE_STEPS="tree sampled;native ar sampled V2_SEED=1 V2_TAG=-seed1;native ar sampled V2_SEED=2 V2_DIST_TEMP=0.65 V2_TAG=-seed2t065" \
  bash "$V/queue_v2.sh" > "$ROOT/queue_dist2.out" 2>&1
log "stage 1 done rc=$?"
sleep 30
log "stage 2: kernel benchmark smoke"
mkdir -p "$ROOT/kernel"
bash "$V/kernel/run_kernel_bench.sh" --smoke > "$ROOT/kernel/smoke-$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1; SRC=$?
log "kernel smoke rc=$SRC"
if [[ $SRC == 0 ]]; then
  sleep 20
  bash "$V/kernel/run_kernel_bench.sh" > "$ROOT/kernel/full-$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1
  log "kernel full rc=$?"
else
  log "kernel smoke failed; full benchmark skipped (needs fixing)"
fi
for c in $(docker ps -aq --filter name=lumo-kbench); do docker stop -t 20 "$c" >/dev/null 2>&1; docker rm "$c" >/dev/null 2>&1; done
sleep 30
log "stage 3: identical-token SGLang x3 + stock-attention LumoTree x2"
QUEUE_CONTINUE_ON_FAIL=1 \
QUEUE_STEPS="native sglang-s7k1d8 sampled V2_IDS=1 V2_TAG=-ids;tree sampled V2_FA2=stock V2_TAG=-fa2stock;native sglang-s7k1d8 sampled V2_IDS=1 V2_TAG=-ids;tree sampled V2_FA2=stock V2_TAG=-fa2stock;native sglang-s7k1d8 sampled V2_IDS=1 V2_TAG=-ids" \
  bash "$V/queue_v2.sh" > "$ROOT/queue_stage3.out" 2>&1
log "stage 3 done rc=$?"
sleep 30
log "stage 4: same-build SWE study"
QUEUE_CONTINUE_ON_FAIL=1 QUEUE_STEPS="swe-tree;swe-native mtp5;swe-native sglang-s7k1d8" \
  bash "$V/queue_v2.sh" > "$ROOT/queue_swe.out" 2>&1
log "stage 4 done rc=$?"
log "pipeline complete"
