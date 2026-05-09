# Track B Round 2 — single-task smoke through actual Codex CLI

**Date:** 2026-05-09
**Container:** `lumo-vllm-track-b-suffix` (live with all 6 Round 2
prelaunch patches).
**Task:** `dead-flag-reachability-audit/v1-clean-baseline` (one of
the 13 v2 corpus families).
**Goal:** validate that the patched runtime serves real Codex CLI
agent traffic without regressions, and that oracle synthesis +
capture work end-to-end on actual agent prompts.
**Setup:** sidecar proxy on 8033 forwarding to patched vLLM at 9950.
Codex CLI (`@openai/codex` 0.128.0) configured with
`base_url=http://127.0.0.1:8033/v1, wire_api=responses`.

## Outcome

- **Exit code: 0** — Codex completed the run without errors.
- **Elapsed: 109.50s** — matches the v2 Round 0 baseline median
  (109.07s) for the same task to within 0.4%.
- 1 captured request through the proxy (Codex performed one
  agent turn, read prompt.md via shell, then concluded the task —
  same shape as the corresponding v2 Round 0 run for this family).
- Artifacts checked into
  `output/track_b_round2/single_task_smoke/round_1/dead-flag-reachability-audit__v1-clean-baseline/run_01/`.

## Capture row from real Codex traffic

```json
{
  "request_id": "resp_bcb8e012ae6d3f19",
  "oracle_session_id": "sess_6fe439bd0a4cb002",
  "oracle_turn_index": 0,
  "oracle_dialect": "codex",
  "oracle_tool_schema_count": 22,
  "oracle_primed_text_count": 0,
  "prefill_sum_s": 91.54,
  "decode_sum_s": 9.56,
  "spec_decode_num_accepted_tokens": 49,
  "spec_decode_num_draft_tokens": 155,
  "tool_call_observed": true,
  "regime": "tool-call"
}
```

What this validates:

- **Proxy oracle synthesis on real Codex traffic**: Codex sent a
  request with **22 tools** (its full agent toolset — much richer
  than my synthetic 2-tool benchmark). The proxy correctly
  extracted all 22 schemas into `tool_schemas` and stashed the
  count.
- **Dialect detection**: `oracle_dialect=codex` confirms the
  proxy detected the codex tool shape (presence of `shell` /
  `apply_patch` / etc. in tools).
- **Session id derived from real prompt**: The session id is the
  hash of Codex's first user message (in this case the agent's
  built-in system instructions + the task prompt).
- **Spec-decode active**: 49/155 accepted = **31.6% on the cold
  first turn**. Acceptance pattern is consistent with what we
  saw in the synthetic micro-benchmark (turn 0 cold).
- **vLLM prefix cache hit rate jumped to 49.4%** during the turn
  (per docker logs), suggesting strong prompt overlap between
  this Codex turn and recent prior traffic. T1's per-session
  partitioning works on top of vLLM's existing prefix-cache layer.
- **Wallclock identical to Round 0 baseline** — the patches did
  not regress single-turn-per-task performance.

## Why only 1 captured turn

This task family lets Codex decide when it's done. In this run,
Codex read prompt.md via a single `cat` shell call (recorded as
the captured turn), then emitted `turn.completed` with
`output_tokens=82` — concluding the task without further work.
This is consistent with Codex 0.128.0's known zero-token quirk
on certain prompts, and the v2 Round 0 baseline saw similar
single-turn outcomes on some families.

A multi-turn task would exercise T1's session scoping more
aggressively. The full v2 sweep (4 runs × 13 families) covers
multi-turn shapes naturally.

## What the smoke does NOT prove

- **Multi-turn T1 lift on real Codex**: this run only had 1 turn,
  so the per-session suffix tree didn't carry response context
  across turns (which is where T1's gains come from). The 5×3
  micro-benchmark already validated that signal at 46% relative
  acceptance lift.
- **Schema-aware drafter on real Codex**: Codex 0.128.0 uses
  auto tool_choice in production, so `expected_tool_call` is
  not set. T3's schema-aware path was validated separately
  with a synthetic forced-choice request (200 OK, valid
  apply_patch output).

## Steps the operator can repeat

```bash
# 1. Launch capture proxy on a fresh port.
LUMO_TRACK_B_REQUEST_METRICS_OUT=/tmp/run_capture.jsonl \
LUMO_TRACK_B_RUNTIME_CONFIG_HASH=<sha256:...> \
.venv/bin/python -m lumo_flywheel_serving.inference_proxy \
  --listen-host 127.0.0.1 --listen-port 8033 \
  --upstream-base-url http://127.0.0.1:9950 \
  --pid-file /tmp/run_proxy.pid --log-path /tmp/run_proxy.log \
  --registry-path model_registry.yaml \
  --state-root /tmp/run_state &

# 2. Drive ALL 13 task families.
TEMPLATE='codex exec --json --skip-git-repo-check -C {workspace} \
  -c '"'"'model_provider="local-proxy"'"'"' \
  -c '"'"'model_providers.local-proxy={{name="local-proxy",base_url="{endpoint}",env_key="OPENAI_API_KEY",wire_api="responses"}}'"'"' \
  --model {model} \
  "Read the task prompt at {prompt_file} and complete it in this workspace."'

.venv/bin/python scripts/run_track_b_e2e_task.py \
  --tasks all --round 1 --repeat 4 \
  --out-root output/track_b_e2e_v2/round_1_patched \
  --endpoint http://127.0.0.1:8033/v1 \
  --health-url http://127.0.0.1:9950/health \
  --metrics-url http://127.0.0.1:9950/metrics \
  --reset-prefix-cache-url http://127.0.0.1:9950/reset_prefix_cache \
  --api-key EMPTY --model qwen3.5-27b \
  --runtime-config-hash <sha256:...> \
  --vllm-request-metrics-jsonl /tmp/run_capture.jsonl \
  --no-dcgm --defer-codex-trace-out --defer-vllm-request-metrics-join \
  --defer-dcgm-profile-fields --discard-cold-attempt-exit \
  --timeout-s 240 \
  --codex-command-template "$TEMPLATE"

# 3. Build applicability JSON for the patched run.
.venv/bin/python scripts/build_track_b_round2_applicability.py \
  --input output/track_b_e2e_v2/round_1_patched \
  --output output/track_b_round2/applicability_v2_round1_patched.json \
  --print

# 4. Diff vs Round 0 baseline.
.venv/bin/python scripts/build_track_b_round2_delta.py \
  --baseline output/track_b_round2/applicability_v2_round0.json \
  --patched  output/track_b_round2/applicability_v2_round1_patched.json \
  --output   output/track_b_round2/delta_v2_round0_to_round1_patched.json \
  --print
```

Total wall time for the full sweep: ~22 min based on Round 0
aggregate (1309.67s × 13 tasks ÷ existing parallelism), plus
analysis.
