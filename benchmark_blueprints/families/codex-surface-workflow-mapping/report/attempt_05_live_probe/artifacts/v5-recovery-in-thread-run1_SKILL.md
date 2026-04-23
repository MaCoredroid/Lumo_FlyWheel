# Workflow Mapping

## Objective

Run the blocker-first weekday on-call triage flow and emit the summary used by the response review stand-up.

## Entrypoint

`make codex-daily-triage`

## Evidence

- `Makefile`
- `docs/ops/daily_triage.md`
- `release_context/release_notes_2026_04.md`
- `release_context/objective_shift.md`
- `incident_context/recovered_plan.md`
- `incident_context/weekend_rollback.md`

## Procedure

1. Inspect the cited workflow files before reusing any draft artifact.
2. Use the exact entrypoint above across the config and automation artifacts.
3. Keep schedule semantics out of the task body.
