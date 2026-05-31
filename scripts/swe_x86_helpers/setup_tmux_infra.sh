#!/bin/bash
# Fault-tolerant infra for the SWE-Bench concurrency probe: run the three
# helpers (reverse-tunnel keeper, capture streamer, vLLM step-trace sampler)
# inside a detached tmux session so they survive Claude-session compaction /
# restarts. Each window wraps its helper in a respawn loop for crash recovery.
set -u
SESS=swe_infra
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REMOTE_PROXY_PORT=${LUMO_TUNNEL_REMOTE_PROXY_PORT:-8022}
LOCAL_PROXY_PORT=${LUMO_TUNNEL_LOCAL_PROXY_PORT:-8022}

# 1) clean any existing instances (script-file context => pkill patterns don't
#    match this shell's own cmdline, so no self-kill).
pkill -9 -f tunnel_keeper.sh 2>/dev/null || true
pkill -9 -f stream_capture_to_alienware.sh 2>/dev/null || true
pkill -9 -f swe_dgx_steptrace.py 2>/dev/null || true
pkill -9 -f "ssh -N.*-R ${REMOTE_PROXY_PORT}:127.0.0.1:${LOCAL_PROXY_PORT}" 2>/dev/null || true
tmux kill-session -t "$SESS" 2>/dev/null || true
sleep 2

# 2) (re)create the tmux session with one respawn-looped window per helper.
tmux new-session -d -s "$SESS" -n keeper \
  "while true; do bash '$SCRIPT_DIR/tunnel_keeper.sh'; echo \"[keeper exited rc=\\$? \$(date -u +%FT%TZ)] restarting\"; sleep 2; done"
tmux new-window -t "$SESS" -n streamer \
  "while true; do bash '$SCRIPT_DIR/stream_capture_to_alienware.sh'; echo \"[streamer exited rc=\\$? \$(date -u +%FT%TZ)] restarting\"; sleep 2; done"
tmux new-window -t "$SESS" -n steptrace \
  "while true; do python3 '$SCRIPT_DIR/swe_dgx_steptrace.py'; echo \"[steptrace exited rc=\\$? \$(date -u +%FT%TZ)] restarting\"; sleep 2; done"

sleep 5
echo "=== tmux sessions ==="; tmux ls
echo "=== windows ==="; tmux list-windows -t "$SESS"
echo "=== procs ==="; echo "keeper=$(pgrep -fc tunnel_keeper.sh||echo 0) streamer=$(pgrep -fc stream_capture_to_alienware.sh||echo 0) steptrace=$(pgrep -fc swe_dgx_steptrace.py||echo 0)"
