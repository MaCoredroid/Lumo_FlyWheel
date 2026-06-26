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
#   LUMO_PROXY_FORCE_TEMPERATURE = 0.6 (the REAL deployment temp; 0.0 ONLY for the temp-0 argmax-flip gate)
set -u

REPO=${LUMO_PROXY_OFFLOAD_REPO:?LUMO_PROXY_OFFLOAD_REPO required}
VENV_PY=${LUMO_PROXY_OFFLOAD_VENV:?LUMO_PROXY_OFFLOAD_VENV required}
cd "$REPO" || { echo "FAIL: cd $REPO"; exit 2; }

# Preserve the exact proxy semantics of the on-GB10 gold-gate path.
export LUMO_PROXY_RETRY_UPSTREAM_400=1
export LUMO_PROXY_AUTO_CONTINUE=1
export LUMO_PROXY_AUTO_CONTINUE_MESSAGE="${LUMO_PROXY_AUTO_CONTINUE_MESSAGE:-Continue working on this task. Do not stop until you have left a concrete source edit that makes the tests pass. If your previous attempt did not pass, read the failure and try a different approach. Do not spend time on environment/pip/conda setup.}"
# 10 -> 3 (proxy default): the 10x whole-turn auto-continue is the "doom loop" antipattern
# (research wcf7colyp); with the 16384 cap each retry is a ~12min capped runaway so 10x = ~2hr.
export LUMO_PROXY_AUTO_CONTINUE_MAX_RETRIES=${LUMO_PROXY_AUTO_CONTINUE_MAX_RETRIES:-3}
# Cap max_output_tokens to bound the qwen tool-call runaway (flavor-2 endless-reasoning
# grinds to 80000 tok ~= 83min). 16384 is ABOVE the observed legit-tool-call max (10710
# tok; legit p99.9=8592) so it truncates ZERO legit turns, cutting a runaway to ~17min.
export LUMO_PROXY_MAX_OUTPUT_TOKENS=${LUMO_PROXY_MAX_OUTPUT_TOKENS:-16384}
export LUMO_PROXY_NONSTREAM_BYPASS=1
export LUMO_PROXY_REQUEST_DUMP_DIR=${LUMO_PROXY_REQUEST_DUMP_DIR:-/tmp/lumo_proxy_request_dumps}
export LUMO_PROXY_FORCE_TEMPERATURE=${LUMO_PROXY_FORCE_TEMPERATURE:-0.6}
# pair-dump dir is read from the env by inference_proxy (LUMO_PROXY_PAIR_DUMP_DIR)
[ -n "${LUMO_PROXY_PAIR_DUMP_DIR:-}" ] && export LUMO_PROXY_PAIR_DUMP_DIR
# Qwen3/Qwen3-Next thinking-mode sampling is the GENERAL default for ALL agentic
# serving (no-spec / spine / tree, cache on|off) -- the degenerate temperature-only
# regime causes <think> + tool-call argument runaway regardless of inference path.
# Opt OUT with LUMO_PROXY_QWEN_SAMPLING=0 ONLY for the lossless A/B gate (presence_penalty
# shifts the argmax, which would skew the within-floor / argmax-flip comparison).
if [ "${LUMO_PROXY_QWEN_SAMPLING:-1}" = "1" ]; then
  export LUMO_PROXY_FORCE_TOP_P=${LUMO_PROXY_FORCE_TOP_P:-0.95}
  export LUMO_PROXY_FORCE_TOP_K=${LUMO_PROXY_FORCE_TOP_K:-20}
  export LUMO_PROXY_FORCE_PRESENCE_PENALTY=${LUMO_PROXY_FORCE_PRESENCE_PENALTY:-1.0}
  export LUMO_PROXY_FORCE_MIN_P=${LUMO_PROXY_FORCE_MIN_P:-0}
else
  unset LUMO_PROXY_FORCE_TOP_P LUMO_PROXY_FORCE_TOP_K LUMO_PROXY_FORCE_PRESENCE_PENALTY LUMO_PROXY_FORCE_MIN_P
fi
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
