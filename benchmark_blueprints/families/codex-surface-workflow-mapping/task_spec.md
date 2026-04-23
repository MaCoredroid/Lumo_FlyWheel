# `codex-surface-workflow-mapping` Task Spec

**Track:** 02 — Codebase Understanding
**Family id:** `codex-surface-workflow-mapping`
**Spec version:** CNB-55 v1.0
**Variants:** 5 (v1 through v5)

## Task Prompt (canonical)

You are dropped into a small repo that already runs a script-driven triage workflow. The benchmark is not asking you to invent a workflow. It is asking you to map the live repo workflow into Codex-native artifacts without anchoring on stale drafts, deprecated helpers, or outdated schedule notes.

Produce the family's four required artifacts through the structured-output CLI:

- `artifacts/SKILL.md`
- `artifacts/codex_triage.toml`
- `artifacts/automation_proposal.md`
- `artifacts/mapping_note.md`

The canonical submission path is:

```
./bin/cnb55-workflow-map schema
./bin/cnb55-workflow-map validate workflow_input.json
./bin/cnb55-workflow-map submit workflow_input.json
```

The agent writes `workflow_input.json` at workspace root. The CLI validates it, writes `artifacts/workflow_map.json` as the canonical scored payload, and renders the four human-facing artifacts listed above.

### Required structured-output schema

- `schema_version`: `cnb55.workflow_mapping.v1`
- `variant_id`: exact contents of `.scenario_variant`
- `skill.entrypoint_command_literal`
- `toml.entrypoint_command_literal`
- `automation.kind`
- `automation.schedule_literal`
- `automation.command_literal`
- `automation.task_prompt`
- `mapping_note.decisions[]`
- `rejected_candidates[]`

Every cited `source_paths[]` entry must be a real file inside the provided workspace bundle. Every `command_literal` must appear verbatim in at least one cited source file. Schedule literals are validated the same way. Out-of-bundle evidence is invalid.

## Scenario Type

`codebase_understanding` — the agent must read the repo, resolve which workflow path is live, distinguish stale vs. current surfaces, and express that mapping consistently across multiple deliverables.

## Required Surfaces

- Shell for repo inspection and optional test execution.
- File reads across `scripts/`, `docs/`, `ops/automation/`, `fixtures/`, and any variant-specific `release_context/` or `incident_context/`.
- Structured-output CLI usage via `./bin/cnb55-workflow-map`.
- Codex artifact authoring via the CLI-rendered outputs under `artifacts/`.

No network, no browser, no sibling-family evidence, no benchmark-authoring note scavenging.

## Workspace Bundle (per variant)

Every variant ships:

```
.scenario_variant
AGENTS.md
Dockerfile
Makefile
bin/cnb55-workflow-map
scripts/
docs/
ops/automation/
fixtures/
skills/legacy/
tests/test_workflow_map.py
artifacts/README.md
```

Variant-specific files add noise or state pressure:

- V2 adds a migration memo and a more tempting deprecated helper path.
- V3 adds an abandoned Codex patch draft under `drafts/`.
- V4 adds `release_context/` proving the workflow scope is blocker-first, not generic queue review.
- V5 adds `incident_context/` showing a weekend helper-based automation was rolled back.

## Difficulty Ladder

### v1 — clean-baseline

One live path (`make codex-daily-triage`), one stale skill draft, one conflicting legacy schedule note.

### v2 — noisy-distractor

Adds a migration memo and deprecated helper that look current on superficial grep.

### v3 — dirty-state

Adds an abandoned Codex patch that points at the wrong helper. The agent must not treat partial work as proof of the live path.

### v4 — multi-corpus-objective

Adds `release_context/` showing the workflow has shifted to blocker-first triage for the on-call rotation. A generic “daily queue sweep” mapping is now wrong.

### v5 — recovery-in-thread

Adds `incident_context/` showing the weekend helper-based automation caused noise and was rolled back. Re-using that path is a judgment failure, not an acceptable alternate interpretation.

## Expected Deliverables

- `artifacts/workflow_map.json` — canonical grader input written by the CLI.
- `artifacts/SKILL.md` — repo-local skill pointing at the live workflow.
- `artifacts/codex_triage.toml` — Codex config artifact consistent with the same entrypoint and scope.
- `artifacts/automation_proposal.md` — schedule and task semantics separated cleanly.
- `artifacts/mapping_note.md` — artifact-by-artifact explanation citing exact source paths and exact command literals.

## Grader Contract

### Visible checks

```
pytest -q tests/test_workflow_map.py
```

Visible checks are capped at 30 points. They verify the canonical JSON exists, parses, matches `cnb55.workflow_mapping.v1`, writes the four deliverables, keeps schedule semantics separate from the task section, and includes one rejected stale candidate.

### Hidden checks

Hidden scoring verifies:

- the live entrypoint is `make codex-daily-triage`, not `python3 scripts/triage_legacy.py --window today --emit-md reports/daily_triage.md`
- skill, TOML, and automation all point at the same live entrypoint
- the mapping is grounded in the shipped repo files
- schedule evidence resolves to `0 9 * * 1-5` and is kept out of `automation.task_prompt`
- stale and dirty-state candidates are explicitly rejected
- V4+ reads `release_context/` and preserves blocker-first scope
- V5 reads `incident_context/` and does not re-introduce the rolled-back helper path

## Partial-Credit Ceilings

- `no_submission` ≤ 0
- `malformed_submission` ≤ 10
- `inconsistent_entrypoints` ≤ 0
- `stale_entrypoint` ≤ 20
- `wrong_schedule` ≤ 25
- `schedule_in_prompt` ≤ 30
- `ungrounded_mapping` ≤ 25
- `generic_mapping` ≤ 25
- `dirty_state_reuse` ≤ 35
- `objective_drift` ≤ 30
- `incident_blind_reuse` ≤ 25

## Saturation and Renewal Plan

Trigger when mean `P_benchmark > 80` for two consecutive probe rounds.

Renewal queue:

1. Mid-run staffing/cadence change injected after the first inspection turn.
2. Contradictory repo-vs-incident automation evidence that must be flagged explicitly.
3. Retire V1 once the floor check saturates and promote a harder V2-derived baseline.
