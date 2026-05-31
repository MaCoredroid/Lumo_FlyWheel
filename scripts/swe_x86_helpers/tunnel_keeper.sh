#!/bin/bash
# Actively keep the DGX->alienware reverse tunnel forwarding healthy.
# The failure mode we hit: ssh -R process stays alive but the forward dies
# silently. So we TEST the actual forward and restart on failure.
ALIEN=${LUMO_TUNNEL_HOST:-alienware}
REMOTE_PROXY_PORT=${LUMO_TUNNEL_REMOTE_PROXY_PORT:-8022}
LOCAL_PROXY_PORT=${LUMO_TUNNEL_LOCAL_PROXY_PORT:-8022}
REMOTE_METRICS_PORT=${LUMO_TUNNEL_REMOTE_METRICS_PORT:-9950}
LOCAL_METRICS_PORT=${LUMO_TUNNEL_LOCAL_METRICS_PORT:-9950}
start_tunnel(){
  ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
      -o TCPKeepAlive=yes \
      -R "${REMOTE_PROXY_PORT}:127.0.0.1:${LOCAL_PROXY_PORT}" \
      -R "${REMOTE_METRICS_PORT}:127.0.0.1:${LOCAL_METRICS_PORT}" \
      "$ALIEN" &
  TPID=$!
  echo "[keeper] $(date -u +%FT%TZ) started tunnel pid=$TPID"
}
pkill -f "ssh -N.*-R ${REMOTE_PROXY_PORT}:127.0.0.1:${LOCAL_PROXY_PORT}" 2>/dev/null; sleep 1
start_tunnel
while true; do
  sleep 30
  code=$(timeout 12 ssh -o ConnectTimeout=8 "$ALIEN" "curl -s --max-time 6 -o /dev/null -w '%{http_code}' http://127.0.0.1:${REMOTE_PROXY_PORT}/v1/models" 2>/dev/null)
  if [ "$code" != "403" ] && [ "$code" != "200" ]; then
    echo "[keeper] $(date -u +%FT%TZ) forward DEAD (code=$code); restarting"
    kill $TPID 2>/dev/null; pkill -f "ssh -N.*-R ${REMOTE_PROXY_PORT}:127.0.0.1:${LOCAL_PROXY_PORT}" 2>/dev/null; sleep 2
    start_tunnel
  fi
done
