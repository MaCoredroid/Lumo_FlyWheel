#!/bin/bash
# FR13 CODEX-OFFLOAD: launch the inference_proxy ON ALIENWARE (x86), forwarding
# to the GB10 vLLM over tailscale, capturing pair-dumps LOCALLY on alienware.
#
# This is relaunch_proxy.sh adapted for the offloaded codex path (OFFLOAD_CODEX=1
# in fr13_bigdenom_swe_serve.sh). It runs on the alienware box (rsynced there with
# the repo src/ + model_registry.yaml). The GB10 runs ONLY vLLM (no proxy, no codex
# docker) so the timing-sensitive deploy-speed numbers are uncontended by the
# unified-memory bandwidth that the proxy + codex docker would otherwise steal.
#
# Network: --upstream-base-url MUST point at the GB10's tailscale IP:9950 (the
# vLLM container publishes -p 9950:9950 to 0.0.0.0, reachable over tailscale).
# Listen: 127.0.0.1:8022 so the codex-runner docker (--network=host on alienware)
# hits it exactly like the legacy on-GB10 path.
#
# Required env (set by the SSH caller from the GB10):
#   LUMO_PROXY_OFFLOAD_REPO   = remote repo root on alienware (e.g. ~/lumo_proxy_offload/repo)
#   LUMO_PROXY_OFFLOAD_VENV   = python interpreter on alienware (e.g. ~/swe_eval_offload/venv/bin/python)
#   LUMO_PROXY_UPSTREAM_BASE_URL = http://<GB10_TS_IP>:9950
#   LUMO_PROXY_PAIR_DUMP_DIR  = local alienware dir for pair dumps
#   LUMO_PROXY_REQUEST_DUMP_DIR = local alienware dir for request dumps
#   LUMO_PROXY_FORCE_TEMPERATURE = 0.0 (the canonical deploy temp pin)
set -u

REPO=${LUMO_PROXY_OFFLOAD_REPO:?LUMO_PROXY_OFFLOAD_REPO required}
VENV_PY=${LUMO_PROXY_OFFLOAD_VENV:?LUMO_PROXY_OFFLOAD_VENV required}
cd "$REPO" || { echo "FAIL: cd $REPO"; exit 2; }

# Preserve the exact proxy semantics of the on-GB10 gold-gate path.
export LUMO_PROXY_RETRY_UPSTREAM_400=1
export LUMO_PROXY_AUTO_CONTINUE=1
export LUMO_PROXY_AUTO_CONTINUE_MESSAGE="${LUMO_PROXY_AUTO_CONTINUE_MESSAGE:-Continue working on this task. Do not stop until you have left a concrete source edit that makes the tests pass. If your previous attempt did not pass, read the failure and try a different approach. Do not spend time on environment/pip/conda setup.}"
export LUMO_PROXY_AUTO_CONTINUE_MAX_RETRIES=${LUMO_PROXY_AUTO_CONTINUE_MAX_RETRIES:-10}
export LUMO_PROXY_MAX_OUTPUT_TOKENS=${LUMO_PROXY_MAX_OUTPUT_TOKENS:-80000}
export LUMO_PROXY_NONSTREAM_BYPASS=1
export LUMO_PROXY_REQUEST_DUMP_DIR=${LUMO_PROXY_REQUEST_DUMP_DIR:-/tmp/lumo_proxy_request_dumps}
export LUMO_PROXY_FORCE_TEMPERATURE=${LUMO_PROXY_FORCE_TEMPERATURE:-0.0}
# pair-dump dir is read from the env by inference_proxy (LUMO_PROXY_PAIR_DUMP_DIR)
[ -n "${LUMO_PROXY_PAIR_DUMP_DIR:-}" ] && export LUMO_PROXY_PAIR_DUMP_DIR
[ -n "${LUMO_PROXY_FORCE_TOP_P:-}" ] && export LUMO_PROXY_FORCE_TOP_P || unset LUMO_PROXY_FORCE_TOP_P
[ -n "${LUMO_PROXY_FR10_DECODE_MODE:-}" ] && export LUMO_PROXY_FR10_DECODE_MODE || unset LUMO_PROXY_FR10_DECODE_MODE
export LUMO_TRACK_B_REQUEST_METRICS_OUT=${LUMO_TRACK_B_REQUEST_METRICS_OUT:-/tmp/track_b_e2e_proxy_capture/request_metrics.jsonl}

LISTEN_HOST=${LUMO_PROXY_LISTEN_HOST:-127.0.0.1}
LISTEN_PORT=${LUMO_PROXY_LISTEN_PORT:-8022}
UPSTREAM_BASE_URL=${LUMO_PROXY_UPSTREAM_BASE_URL:?LUMO_PROXY_UPSTREAM_BASE_URL required (http://<GB10_TS_IP>:9950)}
PID_FILE=${LUMO_PROXY_PID_FILE:-/tmp/lumo_offload_proxy_${LISTEN_PORT}.pid}
LOG_PATH=${LUMO_PROXY_LOG_PATH:-/tmp/lumo_offload_proxy_${LISTEN_PORT}.log}
STATE_ROOT=${LUMO_PROXY_STATE_ROOT:-/tmp/lumo_offload_proxy_${LISTEN_PORT}_state}
NOHUP_PATH=${LUMO_PROXY_NOHUP_PATH:-/tmp/lumo_offload_proxy_${LISTEN_PORT}.nohup}

mkdir -p "$(dirname "$LUMO_TRACK_B_REQUEST_METRICS_OUT")" "$STATE_ROOT" \
         "${LUMO_PROXY_PAIR_DUMP_DIR:-/tmp/lumo_offload_pair_dumps}" \
         "$LUMO_PROXY_REQUEST_DUMP_DIR" 2>/dev/null

# kill any old offload proxy (pid-file then port)
OLD=$(cat "$PID_FILE" 2>/dev/null || true)
[ -n "$OLD" ] && kill "$OLD" 2>/dev/null
pkill -f "lumo_flywheel_serving.inference_proxy" 2>/dev/null
sleep 2

REG_ARG=()
[ -f "$REPO/model_registry.yaml" ] && REG_ARG=(--registry-path "$REPO/model_registry.yaml")

PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}" \
setsid -f "$VENV_PY" -m lumo_flywheel_serving.inference_proxy \
  --listen-host "$LISTEN_HOST" --listen-port "$LISTEN_PORT" \
  --upstream-base-url "$UPSTREAM_BASE_URL" \
  --pid-file "$PID_FILE" \
  --log-path "$LOG_PATH" \
  "${REG_ARG[@]}" \
  --state-root "$STATE_ROOT" \
  > "$NOHUP_PATH" 2>&1 &
NEWPID=$!
echo "offload-proxy launched pid=$NEWPID listen=$LISTEN_HOST:$LISTEN_PORT upstream=$UPSTREAM_BASE_URL"
echo "$NEWPID"
