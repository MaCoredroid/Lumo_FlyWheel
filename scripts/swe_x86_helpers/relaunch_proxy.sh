#!/bin/bash
set -u
cd /home/mark/shared/lumoFlyWheel
# preserve exact env (bypass kept ON; auto-continue + forced temp/top_p intact)
export LUMO_PROXY_RETRY_UPSTREAM_400=1
export LUMO_PROXY_AUTO_CONTINUE=1
export LUMO_PROXY_AUTO_CONTINUE_MESSAGE="Continue working on this task. Do not stop until you have left a concrete source edit that makes the tests pass. If your previous attempt did not pass, read the failure and try a different approach. Do not spend time on environment/pip/conda setup."
export LUMO_PROXY_AUTO_CONTINUE_MAX_RETRIES=10
export LUMO_PROXY_MAX_OUTPUT_TOKENS=80000
export LUMO_PROXY_NONSTREAM_BYPASS=1
export LUMO_PROXY_REQUEST_DUMP_DIR=/tmp/lumo_proxy_request_dumps
export LUMO_PROXY_FORCE_TOP_P=0.95
export LUMO_PROXY_FORCE_TEMPERATURE=1.0
export LUMO_TRACK_B_REQUEST_METRICS_OUT=/tmp/track_b_e2e_proxy_capture/request_metrics.jsonl
# kill old proxy on 8022 by pid-file (clean), fallback to port
OLD=$(cat /tmp/track_b_e2e_proxy_8022.pid 2>/dev/null || true)
[ -n "$OLD" ] && kill "$OLD" 2>/dev/null
sleep 2
nohup .venv/bin/python -m lumo_flywheel_serving.inference_proxy \
  --listen-host 127.0.0.1 --listen-port 8022 \
  --upstream-base-url http://127.0.0.1:9950 \
  --pid-file /tmp/track_b_e2e_proxy_8022.pid \
  --log-path /tmp/track_b_e2e_proxy_8022.log \
  --registry-path /home/mark/shared/lumoFlyWheel/model_registry.yaml \
  --state-root /tmp/track_b_e2e_proxy_8022_state \
  > /tmp/track_b_e2e_proxy_8022.nohup 2>&1 &
echo "new proxy pid=$!"
