#!/usr/bin/env bash
# R4 live arm (CLEAN, speed-of-record): lean stack + FR13_DRAFTER_GRAPH=1,
# B=4, 4-task, offloaded. Gate vs lean_b4 29.02@eps1.78 (eps-matched) —
# floor-frame verdict = delta step_wall_ms (R4 budget: drafter −75 toward
# floor). Waits for if_lean (attribution arm) to finish first.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
until grep -q "IF_LEAN_DONE" output/fr13_msr/if_lean_console.log 2>/dev/null; do sleep 120; done
while docker ps --format '{{.Names}}' | grep -q fr13; do sleep 60; done
sleep 30

SEQF=output/fr13_msr/seq_r4live.sh
cat > "$SEQF" <<'EOF'
export FR13_ENABLE_APC=1
export MAMBA_BLOCK_SIZE=1024
export APC_BLOCK_SIZE=1024
export MAMBA_SSM_CACHE_DTYPE=float32
export FR13_COMMITTER_BATCHED=1
export FR13_PARENT_GATHER=1
export FR13_CONV_PREGATHER=1
export FR13_FLAGS_INKERNEL=1
export FR13_HC_INTERNAL=0
export FR13_CONV_WB_BATCHED=0
export FR13_CONV_NODEBANK=0
export FR13_SPEC_BLOCKS_CAP=0
export FR13_SUBTREE_PARALLEL=1
export FR13_SUBTREE_PARALLEL_SELFCHECK=0
export FR13_DRAFTER_GRAPH=1
run_variant r4live tail6 21 1
EOF

RUNROOT=output/fr13_msr TAG=msr WALL=0 HEALTH_TIMEOUT_S=3600 \
  SUBSET=output/fr13_b1_gold_swe/subset_b4_four.json \
  BSIZE=4 CONC=4 GPU_UTIL=0.70 DEPLOY_FORCE_TEMP=0.6 \
  SEQUENCE_FILE="$PWD/$SEQF" \
  bash scripts/fr13_b4_campaign_driver.sh
echo "R4LIVE_DONE"
