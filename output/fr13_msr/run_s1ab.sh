#!/usr/bin/env bash
# SAME-SESSION A/B: =3 arm then =2 arm, one boot each, same subset, back to
# back — shares workload phase / host state / cache warmth, so the SLOPE
# comparison ((accept+1)/step_wall) is within-session and phase-controlled.
# Answers: does the one-graph =2 region actually cost per-event vs =3, or was
# the cross-boot gap (14.8 vs 18.7) phase? (boot-54 sidecars: the largest
# delta was sfwd = the verify forward, which =2 does not touch.)
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
while docker ps --format '{{.Names}}' | grep -q fr13; do sleep 60; done
sleep 30
SEQF=output/fr13_msr/seq_s1ab.sh
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
export FR13_TAW=1
export FR13_CAPDBG=1
# ARM A: =3 (walk+products+conv+committer; sampler OUTSIDE the graph)
export FR13_STEP_GRAPH=3
run_variant s1ab_m3 tail6 21 1
# ARM B: =2 (S1-full one graph; sampler INSIDE)
export FR13_STEP_GRAPH=2
run_variant s1ab_m2 tail6 21 1
SEQ
(
  for _i in $(seq 1 480); do
    if docker ps --format '{{.Names}}' | grep -qE '^fr13-bigdenom-s1ab_(m3|m2)$'; then
      docker logs -f $(docker ps --format '{{.Names}}' | grep -E '^fr13-bigdenom-s1ab_(m3|m2)$' | head -1) \
        >> output/fr13_msr/s1ab_stream.log 2>&1
    fi
    sleep 5
  done
) &
RUNROOT=output/fr13_msr TAG=msr WALL=0 HEALTH_TIMEOUT_S=3600 \
  SUBSET=output/fr13_b1_gold_swe/subset_b4_four.json \
  BSIZE=4 CONC=4 GPU_UTIL=0.70 DEPLOY_FORCE_TEMP=0.6 \
  SEQUENCE_FILE="$PWD/$SEQF" \
  bash scripts/fr13_b4_campaign_driver.sh
echo "S1AB_DONE"
