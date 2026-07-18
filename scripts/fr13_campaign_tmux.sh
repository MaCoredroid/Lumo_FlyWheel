#!/usr/bin/env bash
# fr13_campaign_tmux.sh — run a B=4 campaign in a DETACHED TMUX session so it survives Claude-harness
# background-task reaping (observed 2026-07-18: run_in_background tasks killed mid-run; &+disown drivers
# reaped intermittently -> orphaned fr13-bigdenom-* vLLM containers holding ~90GB unified mem). The tmux
# server is oom-protected (oom_protect_session) and independent of the harness lifecycle.
#
# Usage: RUNROOT=... TAG=... SUBSET=... [WALL BSIZE CONC SEQUENCE_FILE GPU_UTIL DEPLOY_FORCE_TEMP FR13_*...] \
#          bash scripts/fr13_campaign_tmux.sh <session-name>
# Writes: $RUNROOT/campaign.<name>.env (env snapshot sourced inside tmux), $RUNROOT/driver.<name>.pid
#         (the driver pid, for pidfile-based stop — NEVER stop campaigns via pkill patterns, they can
#         match the caller's own command line and self-kill the shell), $RUNROOT/driver.<name>.log.
set -euo pipefail
name=${1:?usage: fr13_campaign_tmux.sh <session-name>}
: "${RUNROOT:?RUNROOT required}"; : "${TAG:?TAG required}"
mkdir -p "$RUNROOT"
envfile="$RUNROOT/campaign.$name.env"
: > "$envfile"
for v in RUNROOT TAG SUBSET WALL BSIZE CONC SEQUENCE_FILE GPU_UTIL DEPLOY_FORCE_TEMP; do
  [[ -n "${!v:-}" ]] && printf '%s=%q\n' "$v" "${!v}" >> "$envfile"
done
for k in $(compgen -e | grep '^FR13_' || true); do
  printf '%s=%q\n' "$k" "${!k:-}" >> "$envfile"
done
repo=$(cd "$(dirname "$0")/.." && pwd)
tmux kill-session -t "$name" 2>/dev/null || true
tmux new-session -d -s "$name" \
  "cd '$repo' && set -a && source '$envfile' && set +a && echo \$\$ > '$RUNROOT/driver.$name.pid' && exec bash scripts/fr13_b4_campaign_driver.sh > '$RUNROOT/driver.$name.log' 2>&1"
sleep 2
if tmux has-session -t "$name" 2>/dev/null; then
  echo "[tmux-campaign] '$name' RUNNING (driver pid $(cat "$RUNROOT/driver.$name.pid" 2>/dev/null || echo '?'), log $RUNROOT/driver.$name.log)"
else
  echo "[tmux-campaign] FAILED to start '$name' (check $RUNROOT/driver.$name.log)"; exit 1
fi
