---
name: auto-research-round-manager
description: Manage the Sprint 0 auto-research round for qwen3.5-27b on the proposal-ranking-manager-judgment V1 workload. Use when Codex should run the first L1-only auto-research loop, load the produced tuned bundle, bring the real vLLM/Qwen serving path to ready state, and verify the round live with the proposal-ranking-manager-judgment family using gpt-5.4 high subagents.
---

# Auto Research Round Manager

Use this skill for the first serving auto-research round defined by `docs/HLD-Serving-Backend-AutoResearch-v0_1.md`:

- Sprint 0 only
- `L1` action space only
- model `qwen3.5-27b`
- family `proposal-ranking-manager-judgment`
- live gate on `v1-clean-baseline`

The round is complete only when all of these succeed:

1. `auto-research run` emits a fresh tuned bundle.
2. `load-tuned-config` or `resume` activates that bundle.
3. Real serving reaches `/health` on the chosen port.
4. Live family verification through the real `/v1/responses` path finishes with `pass: true`.

## Hard Rules

- Use `gpt-5.4` with `high` reasoning for every spawned subagent.
- Keep one active research worker at a time.
- Do not use cheaper models.
- Do not create branches or worktrees.
- Prefer live verification over adding extra tests.
- If the live gate fails, inspect logs first; do not guess.

## Round Inputs

- HLD: `docs/HLD-Serving-Backend-AutoResearch-v0_1.md`
- Workload: `benchmark_blueprints/families/proposal-ranking-manager-judgment/serving_workload.yaml`
- Live runner: `scripts/run_live_proposal_family.py`
- Serving code:
  - `src/lumo_flywheel_serving/auto_research.py`
  - `src/lumo_flywheel_serving/model_server.py`
  - `src/lumo_flywheel_serving/cli.py`

## Default Paths

Use these unless the user overrides them:

```bash
ROOT=/home/mark/shared/lumoFlyWheel/output/sprint0_live
REPO=/home/mark/shared/lumoFlyWheel
BUNDLE_DIR="$ROOT/tuned_configs"
STATE_DIR="$ROOT/state"
LOGS_DIR="$ROOT/logs"
TRITON_DIR="$ROOT/triton"
BASE_URL=http://127.0.0.1:8101/v1
```

## Manager Workflow

### 1. Run the round

```bash
.venv/bin/lumoserve \
  --registry model_registry.yaml \
  --state-root "$STATE_DIR" \
  --tuned-config-root "$BUNDLE_DIR" \
  auto-research run qwen3.5-27b \
  --family-id proposal-ranking-manager-judgment \
  --workload-file benchmark_blueprints/families/proposal-ranking-manager-judgment/serving_workload.yaml
```

Record:

- produced bundle path
- auto-research `run_log.json`
- `search_trace.json`
- `measurement_trace.json`

### 2. Load the bundle

```bash
.venv/bin/lumoserve \
  --registry model_registry.yaml \
  --state-root "$STATE_DIR" \
  load-tuned-config "$BUNDLE"
```

### 3. Bring up real serving

```bash
LUMO_HOST_MEMORY_RECOVERY=0 \
LUMO_SKIP_VRAM_WAIT=1 \
LUMO_MIN_GPU_MEMORY_UTILIZATION=0.05 \
VLLM_API_KEY=EMPTY \
.venv/bin/lumoserve \
  --registry model_registry.yaml \
  --container-name lumo-vllm-sprint0 \
  --port 8100 \
  --proxy-port 8101 \
  --logs-root "$LOGS_DIR" \
  --triton-cache-root "$TRITON_DIR" \
  --state-root "$STATE_DIR" \
  serve qwen3.5-27b
```

Then verify:

```bash
curl -sf -H 'Authorization: Bearer EMPTY' http://127.0.0.1:8100/health
```

If health fails:

- inspect `$LOGS_DIR/vllm_qwen3.5-27b.log`
- inspect runtime state in `$STATE_DIR/serving_runtime_state.json`
- fix the repo code or run path
- rerun the round from step 1 unless the existing bundle is still valid

### 4. Run the live family gate

```bash
VLLM_API_KEY=EMPTY \
.venv/bin/python scripts/run_live_proposal_family.py \
  --base-url "$BASE_URL" \
  --model qwen3.5-27b \
  --json
```

The pass artifact is the latest `output/live_proposal_family/.../result.json`.

Required success shape:

- `codex_result.returncode == 0`
- `pass == true`
- `score >= 80`

## Research Worker Pattern

If you need a dedicated research subagent:

- Spawn exactly one worker at a time.
- Use `gpt-5.4` and `high`.
- Give it only the current blocker or loop step.
- Have it report:
  - commands run
  - logs inspected
  - files changed
  - artifact paths
  - one terminal status: `ROUND_PASSED` or `BLOCKED`

Suggested worker task shapes:

- diagnose live serving startup failure
- patch infeasible tuning/runtime contract
- rerun the live proposal family verifier after serve is healthy

## Round Completion Report

Report back with:

- bundle path
- auto-research run log path
- serving log path
- live result path
- live verify-result path
- whether the round passed
- exact blocker if it did not
