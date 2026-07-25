#!/usr/bin/env bash
# r4t1 (CLEAN, speed-of-record): lean + R4 + tail-sync-batched fix + cleanup
# pass 1 (VN/NPR deleted). Gate vs r4live 344ms/step @ eps 1.77 (eps-matched).
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
if docker ps --format '{{.Names}}' | grep -q fr13; then echo "REFUSING"; exit 2; fi
SEQF=output/fr13_msr/seq_r4t1.sh
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
export FR13_CONV_WB_BATCHED=0
export FR13_CONV_NODEBANK=0
export FR13_SPEC_BLOCKS_CAP=0
export FR13_SUBTREE_PARALLEL=1
export FR13_SUBTREE_PARALLEL_SELFCHECK=0
export FR13_DRAFTER_GRAPH=1
run_variant r4t1 tail6 21 1
SEQ
RUNROOT=output/fr13_msr TAG=msr WALL=0 HEALTH_TIMEOUT_S=3600 \
  SUBSET=output/fr13_b1_gold_swe/subset_b4_four.json \
  BSIZE=4 CONC=4 GPU_UTIL=0.70 DEPLOY_FORCE_TEMP=0.6 \
  SEQUENCE_FILE="$PWD/$SEQF" \
  bash scripts/fr13_b4_campaign_driver.sh
echo "R4T1_DONE"
