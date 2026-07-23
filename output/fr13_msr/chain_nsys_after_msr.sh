#!/usr/bin/env bash
# waits for the msr driver to exit cleanly, then launches the 0.70 nsys
# verifier-breakdown arm (SIGSTOP-protected capture, marathon-task subset).
cd /home/mark/shared/lumoFlyWheel
while pgrep -f fr13_b4_campaign_driver >/dev/null || docker ps --format '{{.Names}}' | grep -q fr13; do
  sleep 120
done
sleep 60
echo "[chain] msr campaign done $(date -u +%H:%M:%SZ) — launching nsys 0.70 arm"
bash output/fr13_msr/run_nsys_070.sh
