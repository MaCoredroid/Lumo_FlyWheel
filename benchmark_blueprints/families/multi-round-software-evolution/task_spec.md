# `multi-round-software-evolution` Task Spec

**Track:** 10 — Strategic Management & Long-Horizon Evolution
**Family id:** `multi-round-software-evolution`
**Spec version:** CNB-55 v1.0
**Variants:** 5 (v1 through v5)

## Task Prompt (canonical)

You are acting as the engineering manager for a product area that has already gone through several rounds of software evolution. The workspace contains:

- a frozen focus map of plausible next-round investments,
- prior round notes,
- current objective metrics,
- repo-state notes about partial fixes and blocked follow-on work,
- and, in later variants, release and incident context.

Produce a **round plan** using the family's structured-output CLI. The agent writes a small JSON file and runs `./bin/cnb55-evolution submit brief_input.json`; the CLI validates the input and then writes the canonical `brief/round_plan.json` (read by the grader) and `brief/round_plan.md` (human rendering).

The round plan must:

1. Pick exactly one next-round focus (`selected_focus.focus_id`).
2. Explain why that focus should win this round, grounded in the workspace.
3. Name at least one explicit `do_not_touch` boundary with evidence-backed reasoning.
4. Define a partial-progress metric with a concrete baseline, target, and guardrail.
5. Include an assumption ledger that separates observed facts from missing inputs.

The core managerial challenge is **not** "what looks most severe right now." The family is built around choosing the move that unlocks future progress while avoiding destabilizing cleanup, stale evidence, sunk-cost continuation, or incident-blind retries.

### Required structured-output CLI

Location: `./bin/cnb55-evolution` (shipped with every variant).

```bash
./bin/cnb55-evolution schema              # prints the JSON schema
./bin/cnb55-evolution validate FILE.json  # dry-run; exits non-zero on error
./bin/cnb55-evolution submit FILE.json    # validates and writes brief/round_plan.{json,md}
```

Schema version: `cnb55.evolution_plan.v1`.

Required top-level fields:

- `schema_version`
- `variant_id`
- `selected_focus.{focus_id,title,why_now,citations[]}`
- `do_not_touch[].{area,reason,citations[]}`
- `partial_progress_metric.{metric_id,baseline,target,guardrail,measurement_plan[]}`
- `assumption_ledger[].{topic,status,note}`

## Scenario Type

`strategic_management`

The agent is choosing what to do in the next round, not writing code. The signal comes from prioritization, deferral discipline, and whether the progress metric would let a strong team tell the difference between meaningful motion and churn.

## Required Surfaces

- `shell` for reading the bundle and running the CLI
- long-horizon reasoning over frozen evidence
- explicit boundary setting
- concrete partial-progress definition

No network, no MCP, no browser, no code edits inside the workspace bundle.

## Workspace Bundle (per variant)

Every variant ships the following under `workspace_bundle/<variant_id>/`:

```text
AGENTS.md
Dockerfile
.scenario_variant
bin/cnb55-evolution
focus_candidates/priority_map.md
round_history/round_log.md
objective_metrics/scoreboard.md
repo_state/open_threads.md
release_context/release_gate.md      # V4+
incident_context/rollback.md         # V5
artifacts/README.md
tests/test_evolution_plan.py
brief/
```

The `brief/` directory starts empty. The agent may write `brief_input.json` at workspace root and then use the CLI to populate `brief/round_plan.json` and `brief/round_plan.md`.

## Difficulty Ladder

### v1 — clean baseline

The visible severe regression is already contained; the real blocker is a quieter invariant failure that keeps follow-on work from sticking. The right answer is to prioritize `schema-snapshot-alignment`, not the flashier contained cleanup.

- **Primary stress:** choosing the true blocker, not the loudest issue.
- **Expected probe score:** 24–30.

### v2 — noisy distractor

V1 plus stale pager evidence that makes the contained issue look urgent again. The correct plan still prioritizes `schema-snapshot-alignment`, while explicitly not reopening `contained-pager-cleanup`.

- **Primary stress:** stale-evidence resistance.
- **Expected probe score:** 18–24.

### v3 — dirty state

V2 plus an abandoned partial patch for `cold-start-cache-tuning`. The tempting move is "finish what already exists"; the correct move is to defer that partial and still fix the prerequisite invariant.

- **Primary stress:** sunk-cost resistance.
- **Expected probe score:** 14–20.

### v4 — multi-corpus objective

V3 plus release context showing the upcoming launch is blocked more by `streaming-watermark-alignment` than by the base invariant work. The agent must re-weight the objective instead of repeating the old answer.

- **Primary stress:** objective drift under a new corpus.
- **Expected probe score:** 12–18.

### v5 — recovery in thread

V4 plus incident context showing the previously accepted watermark plan was rolled back because the schema snapshot invariant was still unstable underneath it. The next round must step back to the prerequisite `schema-snapshot-alignment`, while explicitly not retrying the rolled-back watermark work yet.

- **Primary stress:** recovery quality and incident-aware replanning.
- **Expected probe score:** 8–14.

### Ladder monotonicity target

Expected per-variant means on GPT-5.4/high after hardening: V1≈28, V2≈22, V3≈18, V4≈14, V5≈10. Family mean ≈18.

## Expected Deliverables

- `brief/round_plan.json` with the canonical structured output.
- `brief/round_plan.md` rendered by the CLI.

No workspace file outside `brief/` may be modified, except `brief_input.json` at workspace root. Ephemeral validation caches created by local test runs (`.pytest_cache/`, `__pycache__/`) are tolerated and ignored by the scorer.

## Grader Contract

### Phase 2 — visible checks

```bash
pytest -q tests/test_evolution_plan.py
```

Visible checks confirm:

- `brief/round_plan.json` exists and parses.
- `schema_version == "cnb55.evolution_plan.v1"`.
- `variant_id` matches `.scenario_variant`.
- `selected_focus` is populated.
- `do_not_touch` is non-empty and structured.
- `partial_progress_metric` has baseline, target, guardrail, and measurement plan.
- `assumption_ledger` includes at least one `status: "missing"` row.

### Hidden checks

The deterministic scorer verifies:

- next-round focus matches the gold focus for the variant,
- the boundary list includes the variant's required do-not-touch areas,
- cited files exist,
- the partial-progress metric is concrete rather than vague,
- stale/contained issues are not re-opened for the wrong reason,
- abandoned partial work is not treated as free progress,
- incident-blind retries are capped.

## Partial-Credit Ceilings

- `no_round_plan_file` ≤ 0
- `malformed_round_plan` ≤ 10
- `weak_focus_grounding` ≤ 20
- `vague_partial_progress_metric` ≤ 25
- `incident_blind_retry` ≤ 30
- `anchored_on_contained_regression` ≤ 35
- `sunk_cost_finish` ≤ 35
- `boundary_missing` ≤ 35
- `objective_drift` ≤ 40

## Quality Gate

- Oracle score ≥ 90
- Empty brief score = 0
- Shortcut score ≤ 35
- Deterministic scorer under `CNB55_SEED=42`
- Family-local verification matrices for V1 and one stress variant
- Probe run against the authored family using `codex exec`

## Saturation and Renewal Plan

If mean `P_benchmark` rises above 80 for two consecutive probe rounds, the family enters `saturation_renewal_due`.

Renewal queue:

1. Add a mid-run objective change where release context arrives after the initial read.
2. Add a contradictory release-vs-incident corpus that requires the plan to flag the conflict explicitly.
3. Retire the original V1 baseline if it becomes pure floor signal.
