#!/usr/bin/env bash
# DVK live arm (CLEAN, speed-of-record): lean stack + R4 + gather-64k drafter
# head. B=4, 4-task, offloaded. Reads: accept comb + per-position vs lean/r4live
# band (suffix-luck caveat: compare comb positions 1-5), garble eyeball,
# measured_tps_fullstep_wall + eps. Probe accept is distribution-confounded for
# the measured subset; THIS arm is the real DVK gate.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
while docker ps --format '{{.Names}}' | grep -q fr13; do sleep 60; done
sleep 30

SEQF=output/fr13_msr/seq_dvkg64L.sh
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
export FR13_DRAFT_VOCAB_K=65536
export FR13_DRAFT_VOCAB_BLOCKS=/workspace/scripts/fr13_dvk_subset_blocks.json
run_variant dvkg64L tail6 21 1
SEQ

RUNROOT=output/fr13_msr TAG=msr WALL=0 HEALTH_TIMEOUT_S=3600 \
  SUBSET=output/fr13_b1_gold_swe/subset_b4_four.json \
  BSIZE=4 CONC=4 GPU_UTIL=0.70 DEPLOY_FORCE_TEMP=0.6 \
  SEQUENCE_FILE="$PWD/$SEQF" \
  bash scripts/fr13_b4_campaign_driver.sh
echo "DVKG64L_DONE"
