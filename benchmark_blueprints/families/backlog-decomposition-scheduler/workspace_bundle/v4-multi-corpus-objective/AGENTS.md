# Agent Instructions — `backlog-decomposition-scheduler`

## Task

Build a dependency-aware execution schedule for this frozen backlog. The right
answer respects prerequisites, keeps the migration specialist out of conflicting
parallel work, and keeps the risky cutover after its observability and dry-run
gates.

## Inputs

- `backlog/` — one markdown file per backlog item.
- `repo_evidence/` — current objective, dependency notes, staffing, and risk context.
- `release_context/` — optional. If present, read it before locking the schedule.
- `incident_context/` — optional. If present, read it before locking the schedule.
- `.scenario_variant` — the variant id.
- `tests/test_schedule_brief.py` — visible checks. Do not modify it.

## Output — use the CLI

Use `./bin/cnb55-schedule`. Do not hand-write `brief/schedule_brief.json`.

1. Write `brief_input.json` at the workspace root.
2. Validate:

   ```bash
   ./bin/cnb55-schedule validate brief_input.json
   ```

3. Submit:

   ```bash
   ./bin/cnb55-schedule submit brief_input.json
   ```

The JSON must include:

- `schema_version = "cnb55.schedule_brief.v1"`
- `variant_id`
- `objective_focus`
- `schedule[]` with `item_id`, `slot`, `lane`, `summary`, `citations[]`, `constraint_tags[]`
- `scarce_role_plan`
- `risk_gate`
- `assumption_ledger[]`

## What strong work looks like

- Dependencies land before blocked work.
- Items needing the migration SRE do not share a slot.
- The risky rollout lands after the dry-run and observability work.
- If release or incident context changes the objective, the schedule reflects the current objective rather than stale planning.
- Any inferred staffing, timing, or rollout detail is marked in the assumption ledger.

## Rules

- Do not modify `backlog/`, `repo_evidence/`, `release_context/`, `incident_context/`, `tests/`, `AGENTS.md`, `Dockerfile`, `.scenario_variant`, or `bin/`.
- Do not write outside `brief/` other than the required root `brief_input.json`.
- Do not fetch network resources.
- Do not add test shims.
