#!/bin/bash
ALIEN=${LUMO_TUNNEL_HOST:-alienware}
SRC=${LUMO_TRACK_B_REQUEST_METRICS_OUT:-/tmp/track_b_e2e_proxy_capture/request_metrics.jsonl}
DST=${LUMO_REMOTE_TRACK_B_REQUEST_METRICS_OUT:-$SRC}
# fresh mirror on alienware so per-task byte offsets align with this run
ssh -o ConnectTimeout=10 "$ALIEN" "mkdir -p $(dirname "$DST") && : > $DST"
echo "[stream] $(date -u +%FT%TZ) mirror reset; following $SRC -> $ALIEN:$DST"
while true; do
  # -n 0: only rows appended from now on (this run); -F: follow across trunccorotate
  tail -n 0 -F "$SRC" 2>/dev/null | ssh -o ServerAliveInterval=20 -o ServerAliveCountMax=3 -o ConnectTimeout=10 "$ALIEN" "cat >> $DST"
  echo "[stream] $(date -u +%FT%TZ) pipe dropped; reconnecting in 2s" 
  sleep 2
done
