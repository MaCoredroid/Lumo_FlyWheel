#!/usr/bin/env bash
# INSTRUMENTED run (attribution, never speed-of-record): lean+R4, FULL head
# (DVK off), draft-id dump ON. One 4-task arm -> exact offline K/subset accept
# curve on the same draft distribution (kills the composition confound).
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
while docker ps --format '{{.Names}}' | grep -q fr13; do sleep 60; done
sleep 30
SEQF=output/fr13_msr/seq_dvkdump.sh
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
export FR13_DVK_DRAFTID_DUMP=/logs/draftids.jsonl
run_variant dvkdump tail6 21 1
SEQ
RUNROOT=output/fr13_msr TAG=msr WALL=0 HEALTH_TIMEOUT_S=3600 \
  SUBSET=output/fr13_b1_gold_swe/subset_b4_four.json \
  BSIZE=4 CONC=4 GPU_UTIL=0.70 DEPLOY_FORCE_TEMP=0.6 \
  SEQUENCE_FILE="$PWD/$SEQF" \
  bash scripts/fr13_b4_campaign_driver.sh
echo "DVKDUMP_DONE"
