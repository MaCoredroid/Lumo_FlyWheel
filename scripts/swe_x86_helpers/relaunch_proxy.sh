#!/bin/bash
set -u
cd /home/mark/shared/lumoFlyWheel
# preserve exact env (bypass kept ON; auto-continue + forced temp intact)
export LUMO_PROXY_RETRY_UPSTREAM_400=1
export LUMO_PROXY_AUTO_CONTINUE=1
export LUMO_PROXY_AUTO_CONTINUE_MESSAGE="Continue working on this task. Do not stop until you have left a concrete source edit that makes the tests pass. If your previous attempt did not pass, read the failure and try a different approach. Do not spend time on environment/pip/conda setup."
# 10 was a "doom loop" antipattern (research wcf7colyp: OpenAI/Anthropic/harnesses do
# NOT retry the whole turn 10x; finer-grained single-request retry is the norm). Combined
# with the 16384 max-token cap each retry is a ~12min capped runaway, so 10x = ~2hr.
# Revert to the proxy's own default (3): bounds the flavor-2 runaway while still recovering
# a model that planned a tool call in text but didn't emit it (1-2 nudges).
export LUMO_PROXY_AUTO_CONTINUE_MAX_RETRIES=${LUMO_PROXY_AUTO_CONTINUE_MAX_RETRIES:-3}
# Cap max_output_tokens to bound the qwen tool-call runaway (flavor-2 endless-reasoning
# would grind to 80000 tok ~= 83min). 16384 is ABOVE the observed legit-tool-call max
# (10710 tok; legit p99.9=8592) so it truncates ZERO legit turns, while cutting a runaway
# to ~17min. Data: 4000-dump analysis 2026-06-25. Override via env if needed.
export LUMO_PROXY_MAX_OUTPUT_TOKENS=${LUMO_PROXY_MAX_OUTPUT_TOKENS:-16384}
export LUMO_PROXY_NONSTREAM_BYPASS=1
export LUMO_PROXY_REQUEST_DUMP_DIR=/tmp/lumo_proxy_request_dumps
# Temperature is overridable by the caller's env (e.g. the experiment runner
# switching temp 1.0<->0.6). Leave top_p unforced unless the caller explicitly
# sets LUMO_PROXY_FORCE_TOP_P; the agentic B4 baseline uses the model default.
if [ -n "${LUMO_PROXY_FORCE_TOP_P:-}" ]; then
  export LUMO_PROXY_FORCE_TOP_P
else
  unset LUMO_PROXY_FORCE_TOP_P
fi
if [ -n "${LUMO_PROXY_FR10_DECODE_MODE:-}" ]; then
  export LUMO_PROXY_FR10_DECODE_MODE
else
  unset LUMO_PROXY_FR10_DECODE_MODE
fi
export LUMO_PROXY_FORCE_TEMPERATURE=${LUMO_PROXY_FORCE_TEMPERATURE:-1.0}
export LUMO_TRACK_B_REQUEST_METRICS_OUT=${LUMO_TRACK_B_REQUEST_METRICS_OUT:-/tmp/track_b_e2e_proxy_capture/request_metrics.jsonl}
LISTEN_HOST=${LUMO_PROXY_LISTEN_HOST:-127.0.0.1}
LISTEN_PORT=${LUMO_PROXY_LISTEN_PORT:-8022}
UPSTREAM_BASE_URL=${LUMO_PROXY_UPSTREAM_BASE_URL:-http://127.0.0.1:9950}
PID_FILE=${LUMO_PROXY_PID_FILE:-/tmp/track_b_e2e_proxy_${LISTEN_PORT}.pid}
LOG_PATH=${LUMO_PROXY_LOG_PATH:-/tmp/track_b_e2e_proxy_${LISTEN_PORT}.log}
STATE_ROOT=${LUMO_PROXY_STATE_ROOT:-/tmp/track_b_e2e_proxy_${LISTEN_PORT}_state}
NOHUP_PATH=${LUMO_PROXY_NOHUP_PATH:-/tmp/track_b_e2e_proxy_${LISTEN_PORT}.nohup}
# kill old proxy by pid-file (clean), fallback to port
OLD=$(cat "$PID_FILE" 2>/dev/null || true)
[ -n "$OLD" ] && kill "$OLD" 2>/dev/null
sleep 2
setsid -f .venv/bin/python -m lumo_flywheel_serving.inference_proxy \
  --listen-host "$LISTEN_HOST" --listen-port "$LISTEN_PORT" \
  --upstream-base-url "$UPSTREAM_BASE_URL" \
  --pid-file "$PID_FILE" \
  --log-path "$LOG_PATH" \
  --registry-path /home/mark/shared/lumoFlyWheel/model_registry.yaml \
  --state-root "$STATE_ROOT" \
  > "$NOHUP_PATH" 2>&1 &
echo "new proxy pid=$!"
