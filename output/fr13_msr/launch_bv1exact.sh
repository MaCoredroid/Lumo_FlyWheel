#!/usr/bin/env bash
# bv1-EXACT replay: boots from the worktree pinned at bv1's commit (04b4abdc8)
# so /workspace = the exact tree bv1 ran (incl. kernel signatures). Same seq
# content as bv1. Discriminator: loops here = ENVIRONMENTAL; clean = today's code.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel/.claude/worktrees/bv1exact
if docker ps --format '{{.Names}}' | grep -q fr13; then echo "REFUSING"; exit 2; fi
mkdir -p output/fr13_msr output/fr13_sfwd_sidecar
cp /home/mark/shared/lumoFlyWheel/output/fr13_msr/seq_bv1.sh output/fr13_msr/seq_bv1x.sh
sed -i 's/bv1_fixbatch/bv1x_replay/; s|/bv1_|/bv1x_|g' output/fr13_msr/seq_bv1x.sh
RUNROOT=output/fr13_msr TAG=msr WALL=0 HEALTH_TIMEOUT_S=3600 \
  SUBSET=output/fr13_b1_gold_swe/subset_b4_four.json \
  BSIZE=4 CONC=4 GPU_UTIL=0.70 DEPLOY_FORCE_TEMP=0.6 \
  SEQUENCE_FILE="$PWD/output/fr13_msr/seq_bv1x.sh" \
  bash scripts/fr13_b4_campaign_driver.sh
