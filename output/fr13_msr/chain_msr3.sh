#!/usr/bin/env bash
# waits for chain_nsys (and everything before it) to fully finish, then runs M3.
cd /home/mark/shared/lumoFlyWheel
while pgrep -f chain_nsys_after_msr >/dev/null || pgrep -f fr13_b4_campaign_driver >/dev/null || docker ps --format '{{.Names}}' | grep -q fr13; do
  sleep 180
done
sleep 60
echo "[chain2] launching M3 $(date -u +%H:%M:%SZ)"
RUNROOT=output/fr13_msr TAG=msr WALL=0 HEALTH_TIMEOUT_S=3600 \
  SUBSET=output/fr13_b1_gold_swe/subset_b4_four.json \
  BSIZE=4 CONC=4 GPU_UTIL=0.70 DEPLOY_FORCE_TEMP=0.6 \
  SEQUENCE_FILE="$PWD/output/fr13_msr/seq_msr3.sh" \
  bash scripts/fr13_b4_campaign_driver.sh
