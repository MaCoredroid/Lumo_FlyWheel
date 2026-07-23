#!/usr/bin/env bash
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
RUNROOT=output/fr13_ovl16_gate TAG=ov1 WALL=0 HEALTH_TIMEOUT_S=3600 \
  SUBSET=output/fr13_b1_gold_swe/subset_b4_sixteen.json \
  BSIZE=4 CONC=4 GPU_UTIL=0.60 DEPLOY_FORCE_TEMP=0.6 \
  SEQUENCE_FILE=/tmp/smoke_seq_ovl16.sh \
  bash scripts/fr13_b4_campaign_driver.sh
