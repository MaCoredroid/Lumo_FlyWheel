#!/bin/bash
# 5-minute supervisor for a running codex experiment (run_codex_experiment.py).
# Auto-fixes the SAFE, idempotent things (reverse tunnel / swe_infra tmux /
# proxy) and LOUDLY ALERTS on the risky ones (vLLM engine down, runner process
# gone) rather than blindly restarting them -- relaunching vLLM is a heavy
# config reload and relaunching the runner would spawn a duplicate suite, so
# those need a human/agent decision.
#
# Usage: experiment_supervisor.sh <runner_pattern> <proxy_force_temp> [interval_s]
# Runs forever; intended to live in a tmux window. Logs to /tmp/exp_supervisor.log
set -u
RUNNER_PAT="${1:-run_codex_experiment.py}"
TEMP="${2:-1.0}"
INTERVAL="${3:-300}"
LOG=/tmp/exp_supervisor.log
REPO=/home/mark/shared/lumoFlyWheel
REMOTE_PROXY_PORT=${LUMO_TUNNEL_REMOTE_PROXY_PORT:-8022}
PROXY_LOCAL=${LUMO_CODEX_PROXY_MODELS_URL:-http://127.0.0.1:8022/v1/models}
VLLM_METRICS=${LUMO_VLLM_METRICS_URL:-http://127.0.0.1:9950/metrics}

log(){ echo "[supervisor $(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

log "START pat='$RUNNER_PAT' temp=$TEMP interval=${INTERVAL}s"
while true; do
  # 1) swe_infra tmux (tunnel keeper / streamer / steptrace) -- self-healing
  if ! tmux has-session -t swe_infra 2>/dev/null; then
    log "FIX: swe_infra tmux gone -> rebuilding"
    bash "$REPO/scripts/swe_x86_helpers/setup_tmux_infra.sh" >>"$LOG" 2>&1
  fi
  # 2) reverse tunnel forward: curl the proxy THROUGH the tunnel from alienware
  fwd=$(timeout 12 ssh -n -o ConnectTimeout=6 alienware \
        "curl -s -m5 -o /dev/null -w '%{http_code}' http://127.0.0.1:${REMOTE_PROXY_PORT}/v1/models 2>/dev/null" 2>/dev/null)
  if [ "$fwd" != "200" ] && [ "$fwd" != "403" ]; then
    log "WARN: tunnel forward unhealthy (alienware->$REMOTE_PROXY_PORT http='$fwd'); tunnel_keeper should self-heal within 30s"
  fi
  # 3) proxy (local) -- safe to restart (preserves the run's forced temperature)
  ph=$(curl -s -m5 -o /dev/null -w "%{http_code}" "$PROXY_LOCAL" 2>/dev/null)
  if [ "$ph" != "200" ] && [ "$ph" != "403" ]; then
    log "FIX: proxy down (http='$ph') -> restarting with FORCE_TEMPERATURE=$TEMP"
    LUMO_PROXY_FORCE_TEMPERATURE="$TEMP" bash "$REPO/scripts/swe_x86_helpers/relaunch_proxy.sh" >>"$LOG" 2>&1
  fi
  # 4) vLLM engine -- RISKY to auto-relaunch (config reload); alert only
  if ! curl -s -m5 "$VLLM_METRICS" 2>/dev/null | grep -q "^vllm:"; then
    log "ALERT: vLLM /metrics not serving -- engine may be down. NOT auto-relaunching (heavy/risky); needs intervention."
  fi
  # 5) runner process -- alert only (restart would spawn a duplicate suite)
  if ! pgrep -f "$RUNNER_PAT" >/dev/null 2>&1; then
    log "ALERT: runner ('$RUNNER_PAT') not running -- per-task commit loop is down. Check if it finished or crashed."
  fi
  sleep "$INTERVAL"
done
