#!/usr/bin/env bash
# FR13 GPU_UTIL raise probe (cache-hit lever 1, project_fr13_tail6_cache_hitrate_gap).
# Question: at GPU_UTIL=0.70 (vs campaign 0.60), does (a) the KV pool grow as
# expected (~150-160k tokens; 0.60->129,024 / 0.82->230,400 measured), (b) host
# available memory stay >12GB with no oom-guard kill under B=4 + subagents,
# (c) the windowed prefix-cache hit rate move up materially vs the same config
# at 0.60 (ovl16 run: cumulative ~40%)?
# 4-task slice (subset_b4_four), same sequence-file env as ovl16 (overlap armed)
# so the ONLY delta vs ovl16 is GPU_UTIL. Speed numbers from this probe are
# DIAGNOSTIC ONLY (4-task, not the 16-task basis).
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel

if docker ps --format '{{.Names}}' | grep -q fr13; then
  echo "REFUSING: an fr13 container is running (gate in flight?)"; exit 2
fi

RUNROOT=output/fr13_utilprobe
UTIL=${UTIL:-0.70}
TAG=up$(echo "$UTIL" | tr -d '0.')
export FR13_PROBE_TAG="$TAG"

# host-mem + hit-rate watcher (self-terminates with the driver)
(
  while sleep 60; do
    ts=$(date -u +%H:%M:%SZ)
    avail=$(free -m | awk '/^Mem:/ {print $7}')
    hit=$(curl -fsS -m 3 http://127.0.0.1:9950/metrics 2>/dev/null | awk '
      /^vllm:prefix_cache_hits_total/ {h+=$2} /^vllm:prefix_cache_queries_total/ {q+=$2}
      END {if (q>0) printf "%.1f", 100*h/q; else printf "NA"}')
    echo "$ts avail_mb=$avail cum_hit=$hit" >> "$RUNROOT/watcher_$TAG.csv"
    pgrep -f fr13_b4_campaign_driver >/dev/null || break
  done
) &
WATCHER=$!

RUNROOT="$RUNROOT" TAG="$TAG" WALL=0 HEALTH_TIMEOUT_S=3600 \
  SUBSET=output/fr13_b1_gold_swe/subset_b4_four.json \
  BSIZE=4 CONC=4 GPU_UTIL="$UTIL" DEPLOY_FORCE_TEMP=0.6 \
  SEQUENCE_FILE="$PWD/$RUNROOT/smoke_seq_utilprobe.sh" \
  bash scripts/fr13_b4_campaign_driver.sh
RC=$?
kill "$WATCHER" 2>/dev/null
echo "PROBE_DONE rc=$RC $(date -u +%H:%M:%SZ)"
echo "boot pool line:"
grep -h "GPU KV cache size" "$RUNROOT"/*"$TAG"*/docker_full.log 2>/dev/null | tail -1
tail -5 "$RUNROOT/watcher_$TAG.csv" 2>/dev/null
