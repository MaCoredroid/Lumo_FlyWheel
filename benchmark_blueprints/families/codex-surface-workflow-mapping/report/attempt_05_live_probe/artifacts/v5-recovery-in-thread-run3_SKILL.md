# Workflow Mapping

## Objective

Run the blocker-first weekday daily triage for the on-call rotation via make codex-daily-triage so the summary feeds the response review stand-up instead of a broad queue sweep.

## Entrypoint

`make codex-daily-triage`

## Evidence

- `docs/ops/daily_triage.md`
- `Makefile`
- `scripts/triage.py`
- `release_context/release_notes_2026_04.md`
- `release_context/objective_shift.md`
- `incident_context/recovered_plan.md`
- `incident_context/weekend_rollback.md`

## Procedure

1. Inspect the cited workflow files before reusing any draft artifact.
2. Use the exact entrypoint above across the config and automation artifacts.
3. Keep schedule semantics out of the task body.
