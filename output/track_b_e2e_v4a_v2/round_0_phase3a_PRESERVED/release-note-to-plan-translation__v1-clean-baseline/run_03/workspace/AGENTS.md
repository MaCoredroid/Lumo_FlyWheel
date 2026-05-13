# Agent Instructions — `release-note-to-plan-translation`

## Task

Translate frozen release notes plus repo-state evidence into an ordered
implementation plan. The plan must identify the smallest meaningful first
milestone, the prerequisite dependency chain, and the main user-visible risk
if the work is sequenced incorrectly.

## Inputs

- `release_notes/` — the release-note bullets that must be translated into a
  concrete execution order. Read dates and status notes carefully; some notes
  are stale or superseded.
- `repo_inventory/` — current repo-state evidence: dependency notes, test
  status, partial patches, and rollout constraints. This is the source of
  hidden coupling and user-visible risk.
- `release_context/` — optional. If present, read it before deciding; the
  operating objective has shifted since the original notes were written.
- `incident_context/` — optional. If present, read it before deciding; a prior
  plan order was rolled back and should change what you pick first.
- `.scenario_variant` — the variant id string. Use it in your brief.
- `tests/test_plan_brief.py` — visible checks. Do not modify it.

## Output — use the CLI

Use `./bin/cnb55-brief` to validate and submit a structured JSON brief.
Do not hand-write `brief/manager_brief.md`.

1. Write `brief_input.json` at the workspace root.
2. Inspect the schema if needed:
   `./bin/cnb55-brief schema`
3. Validate:
   `./bin/cnb55-brief validate brief_input.json`
4. Submit:
   `./bin/cnb55-brief submit brief_input.json`

The canonical schema version is `cnb55.release_plan_brief.v1`.

Your brief must contain:
- `first_milestone_id`
- `ordered_steps` with contiguous ranks
- `dependency_notes` tying prerequisite work to downstream steps
- `primary_risk` naming the user-visible failure mode if the order is wrong
- `assumption_ledger` with at least one honest `missing` row

## What strong planning looks like

- The first milestone is bounded: it should reduce ambiguity or de-risk the
  path, not attempt the full launch.
- Dependencies are explicit. If dashboard or rollout work depends on schema,
  fixtures, or guardrails, that relation should appear both in ordering and in
  `dependency_notes`.
- Risk is user-visible. Tie it to the evidence in `repo_inventory/`, not to an
  abstract engineering concern.
- If a note is stale, a draft is abandoned, the objective changed, or an
  incident invalidated a prior order, say so in the plan rather than anchoring
  on the old path.

## Rules

- Do not modify `release_notes/`, `repo_inventory/`, `release_context/`,
  `incident_context/`, `tests/`, `AGENTS.md`, `Dockerfile`, `.scenario_variant`,
  or `bin/`.
- Do not write files outside `brief/`. The only extra root file allowed is
  `brief_input.json`, which the CLI reads.
- Do not fetch network resources.
- Do not add `sitecustomize.py`, `usercustomize.py`, or `pytest.py`.

Any of those trips the integrity detector and fails the run.
