#!/usr/bin/env bash
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
if docker ps --format '{{.Names}}' | grep -q fr13; then echo "REFUSING"; exit 2; fi
RUNROOT=output/fr13_msr TAG=msr WALL=0 HEALTH_TIMEOUT_S=3600 \
  SUBSET=output/fr13_b1_gold_swe/subset_b4_four.json \
  BSIZE=4 CONC=4 GPU_UTIL=0.70 DEPLOY_FORCE_TEMP=0.6 \
  SEQUENCE_FILE="$PWD/output/fr13_msr/seq_g3_stack4.sh" \
  bash scripts/fr13_b4_campaign_driver.sh
