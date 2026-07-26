#!/usr/bin/env bash
# DVK accept-recovery arm: lean stack + R4 + CONTIG-128k drafter head (no
# BLOCKS: corpus-committed coverage overestimated draft-time coverage on the
# 64k gather arm — per-pos 1-4 fell 5-18%; BPE-id order covers the fat draft
# tail better). Probe-proven band accept 3.249 at loop 41.5ms (-28.5).
# Reads: per-position 1-4 recovery vs r4live 0.897/0.768/0.643/0.550 +
# measured_tps_fullstep_wall + eps. Waits for the attn bench (last GPU user).
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
until grep -q "ATTN_BENCH_DONE" output/fr13_verify_profile/attn_bench_console.log 2>/dev/null; do sleep 120; done
while docker ps --format '{{.Names}}' | grep -q fr13; do sleep 60; done
sleep 30

SEQF=output/fr13_msr/seq_dvkg128L.sh
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
export FR13_DRAFT_VOCAB_K=131072
run_variant dvkg128L tail6 21 1
SEQ

RUNROOT=output/fr13_msr TAG=msr WALL=0 HEALTH_TIMEOUT_S=3600 \
  SUBSET=output/fr13_b1_gold_swe/subset_b4_four.json \
  BSIZE=4 CONC=4 GPU_UTIL=0.70 DEPLOY_FORCE_TEMP=0.6 \
  SEQUENCE_FILE="$PWD/$SEQF" \
  bash scripts/fr13_b4_campaign_driver.sh
echo "DVKG128L_DONE"
