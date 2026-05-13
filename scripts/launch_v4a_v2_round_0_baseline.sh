#!/usr/bin/env bash
# v4a_v2 Round 0 — D-point re-baseline on 11 active tasks under the
# §13-§17 proxy stack.
#
# Per-task budget: 1800s. Repeat: 4. Total worst-case wallclock:
#   11 × 4 × 1800 s = 79,200 s (22 hr) — typically much less.
#
# Output: output/track_b_e2e_v4a_v2/round_0/
# Capture: /tmp/track_b_e2e_proxy_capture/request_metrics.jsonl
#          (proxy restarted with LUMO_TRACK_B_REQUEST_METRICS_OUT set)
set -euo pipefail

REPO_ROOT="/home/mark/shared/lumoFlyWheel"
cd "$REPO_ROOT"

RUNTIME_CONFIG_HASH="sha256:5ae88ac4e10201f83a617e2bda3f1c07da4c7217c80db5482d317a79dd93b43a"
WARMUP_SP_JSON="output/track_b_e2e_v4a/round_0/codex_system_prompt.json"
ENDPOINT="http://127.0.0.1:8022/v1"
NOW_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Each codex attempt runs inside a `codex-runner:v1` container. The
# attempt's workspace is bind-mounted to /workspace; --network=host gives
# codex access to the proxy on 127.0.0.1:8022 and the open internet
# (apt/pip/curl/etc. usable from inside the task). Filesystem outside
# /workspace is the container's ephemeral overlay — codex cannot see the
# host's main repo, other attempts' workspaces, or any docs/reports.
UID_GID="$(id -u):$(id -g)"
# `--rm` already auto-removes the container on exit; the task runner's
# TimeoutExpired handler now reaps any codex-runner:v1 container left
# behind by a SIGKILL'd `docker run` (only one runs at a time because
# the pipeline is serial). No --name needed.
CODEX_TEMPLATE='docker run --rm --network=host -u '"$UID_GID"' -v {workspace}:/workspace:rw -e OPENAI_API_KEY=EMPTY -e OPENAI_BASE_URL={endpoint} -e HOME=/tmp -w /workspace codex-runner:v1 codex exec --json --skip-git-repo-check --dangerously-bypass-approvals-and-sandbox -C /workspace -c '"'"'model_provider="local-proxy"'"'"' -c '"'"'model_providers.local-proxy={{name="local-proxy",base_url="{endpoint}",env_key="OPENAI_API_KEY",wire_api="responses",stream_idle_timeout_ms=600000}}'"'"' -c '"'"'model_reasoning_effort="high"'"'"' -c '"'"'model_supports_reasoning_summaries=true'"'"' -c '"'"'model_reasoning_summary="auto"'"'"' --model {model} "Read the task prompt at /workspace/AGENTS.md and complete it in this workspace."'

# Ensure runtime flags = all on (T2/T3/T4 enabled).
docker exec lumo-vllm-track-b-suffix bash -lc 'printf "%s" "{\"T2\": false, \"T3\": false, \"T4\": false}" > /tmp/lumo_track_b_runtime_flags.json && cat /tmp/lumo_track_b_runtime_flags.json'
echo

export OPENAI_API_KEY="EMPTY"

exec .venv/bin/python scripts/run_track_b_e2e_round.py \
  --round 0 \
  --runtime-config-hash "$RUNTIME_CONFIG_HASH" \
  --codex-command-template "$CODEX_TEMPLATE" \
  --warmup-policy round_start \
  --warmup-system-prompt-json "$WARMUP_SP_JSON" \
  --reset-prefix-cache-url http://127.0.0.1:9950/reset_prefix_cache \
  --zero-token-retries 3 \
  --clock-skew-ms-p99 8 \
  --trace-emitter-correctness-verified-at "$NOW_ISO" \
  --protocol-hash-match \
  --repeat 4 \
  --timeout-s 1800 \
  --out-root "output/track_b_e2e_v4a_v2" \
  --endpoint "$ENDPOINT" \
  --vllm-request-metrics-jsonl /tmp/track_b_e2e_proxy_capture/request_metrics.jsonl \
  --hypothesis "v4a_v2 D-point re-baseline (T1+T2+T3+T4 on) under proxy stack §13-§17 (LUMO_PROXY_NONSTREAM_BYPASS=1, LUMO_PROXY_AUTO_CONTINUE=1, retries=5) on 11 active tasks." \
  --config-delta-vs-prior-round "v4a_v2 baseline vs degenerate v4a baseline: proxy stack §13-§17 fixes activated; corpus shrunk 13→11 (excluded plugin-scaffold-alignment + skill-router-contract-upgrade for missing AGENTS.md + seeded drift); --timeout-s 1800 (was 900)" \
  --auto-research-agent-recommendation "Compare wallclock and task_score against degenerate v4a baseline. Expect per-task wallclock to grow from ~19 s to several minutes (26.5 tool calls × tool-exec + ~3K decode tokens at ~30 tps + retries)." \
  --next-round-proposal "After v4a_v2 D-point lands, run a/b/c ablation against same 11 corpus." \
  --codex-smoke-timeout-s 600 \
  --defer-preflight-checks vllm_request_metrics_join_available codex_trace_out_supported dcgm_profile_fields_available codex_command_smoke
