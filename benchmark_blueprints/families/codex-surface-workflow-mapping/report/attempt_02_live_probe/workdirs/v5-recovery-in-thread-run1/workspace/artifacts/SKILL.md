# Workflow Mapping

## Objective

Run the weekday blocker-first triage workflow for the on-call rotation and emit the summary used by the response review stand-up without expanding back into a general or weekend queue sweep.

## Entrypoint

`make codex-daily-triage`

## Evidence

- `docs/ops/daily_triage.md`
- `incident_context/recovered_plan.md`
- `release_context/objective_shift.md`
- `release_context/release_notes_2026_04.md`
- `scripts/triage.py`

## Procedure

1. Inspect the cited workflow files before reusing any draft artifact.
2. Use the exact entrypoint above across the config and automation artifacts.
3. Keep schedule semantics out of the task body.
