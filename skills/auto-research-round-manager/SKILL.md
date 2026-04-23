---
name: auto-research-round-manager
description: Run the Phase B auto-research round as a Python outer loop over `lumoserve auto-research ...` commands. Use only after the Phase A substrate is present and the preconditions pass.
---

# Auto Research Round Manager

This skill is the Phase B outer loop described in `docs/HLD-Serving-Backend-AutoResearch-v0_1-SubSpec-AutoResearchAgent.md`.

Scope:
- Sprint 0 only
- `L1` action space only
- model `qwen3.5-27b`
- family `proposal-ranking-manager-judgment`
- one active `codex exec` at a time

## Preconditions

Refuse to start unless all of these pass:
- `python -c "from lumo_flywheel_serving.measurement_harness import RealMeasurementHarness"` succeeds.
- `codex --version` succeeds.
- `git status --short` is empty.
- `lumoserve auto-research --help` lists `bootstrap-round`, `measure`, `commit-candidate`, `rescreen`, `validate-holdout`, `finalize-round`, `status`, and `run`.
- For each production subcommand, `lumoserve auto-research <name> --help-only` exits 0 and prints `{"subcommand":"<name>","status":"registered"}`.
- The workload yaml exists and its `seed_trace_ref` points at an existing jsonl file.
- `LUMO_AUTO_RESEARCH_ALLOW_NON_AGENT` is unset.

Block on failure. Do not bootstrap a round when any precondition is red.

## Round Bootstrap

Run:

```bash
.venv/bin/lumoserve \
  --registry model_registry.yaml \
  auto-research bootstrap-round \
  --model-id qwen3.5-27b \
  --family-id proposal-ranking-manager-judgment \
  --sprint sprint-0 \
  --workload-file benchmark_blueprints/families/proposal-ranking-manager-judgment/serving_workload.yaml \
  --round-root output/auto_research
```

Read the returned `round_id` and `round_dir`. From this point on:
- Python owns lifecycle, retries, stop criteria, rescreen, holdout, and finalize.
- Each `codex exec` owns exactly one iteration.

## Python Loop

1. Measure and commit the two pre-written baselines:
   - `candidates/baseline_a/candidate.yaml`
   - `candidates/baseline_b/candidate.yaml`
2. Compute `noise_floor = 2 * abs(baseline_a.objective_value - baseline_b.objective_value)` and persist it into `round_spec.yaml`.
3. For each main-loop iteration:
   - materialize `iteration_brief.md` placeholders in memory
   - spawn one `codex exec` with `--json`, `--cd <round_dir>`, and `--output-last-message`
   - capture stdout to `candidates/<NNN>/agent_session.jsonl`
   - if the iteration writes `BLOCKED.md`, stop with `ROUND_BLOCKED`
   - otherwise read the new `results.tsv` row and update stop criteria
4. After the main loop exits:
   - `lumoserve auto-research rescreen --round-id <id> --top-k 3 --profile full`
   - choose the winner by `objective_mean`
   - `lumoserve auto-research validate-holdout --round-id <id> --candidate-uuid <winner_uuid>`
5. Finalize exactly once:
   - `lumoserve auto-research finalize-round --round-id <id>`

## Watchdog

After every iteration:
- reject writes outside the round directory and bundle output path
- reject commits that lack `Signed-off-by: lumoserve-auto-research-cli <auto-research@lumo-flywheel>`
- reject commits that lack `Candidate-UUID: <uuid>` matching the staged row

## Forbidden

- Do not use subagents.
- Do not hold a long-running codex transcript across iterations.
- Do not let a codex iteration call `finalize-round`.
- Do not rerun blocked rounds automatically.

## Report

Return:
- `round_id`
- `round_branch`
- outcome
- `bundle_path` or `null`
- stopping reason
- blocker text if any
- counts for iterations, feasible rows, rescreened rows
