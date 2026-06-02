#!/bin/bash
ALIEN=${LUMO_TUNNEL_HOST:-alienware}
SRC=${LUMO_TRACK_B_REQUEST_METRICS_OUT:-/tmp/track_b_e2e_proxy_capture/request_metrics.jsonl}
DST=${LUMO_REMOTE_TRACK_B_REQUEST_METRICS_OUT:-$SRC}
# Do not truncate the x86 mirror here. This script is run under a respawn loop;
# truncating on restart invalidates byte offsets already sampled by live tasks.
ssh -o ConnectTimeout=10 "$ALIEN" "mkdir -p $(dirname "$DST") && touch $DST"
echo "[stream] $(date -u +%FT%TZ) following $SRC -> $ALIEN:$DST"
while true; do
  # -n 0: only rows appended from now on (this run); -F: follow across trunccorotate
  tail -n 0 -F "$SRC" 2>/dev/null | ssh -o ServerAliveInterval=20 -o ServerAliveCountMax=3 -o ConnectTimeout=10 "$ALIEN" "cat >> $DST"
  echo "[stream] $(date -u +%FT%TZ) pipe dropped; reconnecting in 2s" 
  sleep 2
done
