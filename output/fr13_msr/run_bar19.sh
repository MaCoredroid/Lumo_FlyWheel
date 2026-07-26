#!/usr/bin/env bash
# DEPTHSYNC live arm (CLEAN, speed-of-record, THE gate per user call): composed
# stack (baked gather-64k + CG) + FR13_DM_DEPTHSYNC=1 — one lever vs dvkcg
# (41.29 tps @ eps 2.51, accept 5.973, step 353.7). B=4, 4-task, offloaded.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
while docker ps --format '{{.Names}}' | grep -q fr13; do sleep 60; done
sleep 30
SEQF=output/fr13_msr/seq_bar19.sh
cat > "$SEQF" <<'SEQ'
export FR13_ENABLE_APC=1
export MAMBA_BLOCK_SIZE=1024
export APC_BLOCK_SIZE=1024
export MAMBA_SSM_CACHE_DTYPE=float32
export FR13_COMMITTER_BATCHED=1
export FR13_PARENT_GATHER=1
export FR13_CONV_PREGATHER=1
export FR13_FLAGS_INKERNEL=1
export FR13_HC_INTERNAL=0
export FR13_SUBTREE_PARALLEL=1
export FR13_SUBTREE_PARALLEL_SELFCHECK=0
export FR13_DRAFTER_GRAPH=1
export FR13_COMMITTER_GRAPH=1
export FR13_DM_DEPTHSYNC=1
run_variant bar19 tail6 21 1
SEQ
RUNROOT=output/fr13_msr TAG=msr WALL=0 HEALTH_TIMEOUT_S=3600 \
  SUBSET=output/fr13_b1_gold_swe/subset_b4_sixteen.json \
  BSIZE=4 CONC=4 GPU_UTIL=0.70 DEPLOY_FORCE_TEMP=0.6 \
  SEQUENCE_FILE="$PWD/$SEQF" \
  bash scripts/fr13_b4_campaign_driver.sh
echo "BAR19_DONE"
