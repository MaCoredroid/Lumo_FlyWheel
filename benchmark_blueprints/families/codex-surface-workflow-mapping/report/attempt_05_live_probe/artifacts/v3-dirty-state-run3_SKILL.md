# Workflow Mapping

## Objective

Run the live weekday Codex daily triage workflow for the dirty-state variant by invoking the Make entrypoint, summarizing active-window blocker incidents for the on-call rotation, and keeping rollback-only helpers out of the operator path.

## Entrypoint

`make codex-daily-triage`

## Evidence

- `docs/ops/daily_triage.md`
- `Makefile`
- `.scenario_variant`
- `fixtures/open_incidents.json`
- `docs/ops/migration_notes.md`

## Procedure

1. Inspect the cited workflow files before reusing any draft artifact.
2. Use the exact entrypoint above across the config and automation artifacts.
3. Keep schedule semantics out of the task body.
