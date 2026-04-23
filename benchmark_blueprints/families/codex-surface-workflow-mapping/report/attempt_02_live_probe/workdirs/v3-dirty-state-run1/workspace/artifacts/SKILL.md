# Workflow Mapping

## Objective

Run the live daily triage flow for the active incident window, summarize current blocker incidents for the on-call rotation, and keep dirty-state blocker handling tied to the live queue rather than rollback-only helpers or seven-day legacy sweeps.

## Entrypoint

`make codex-daily-triage`

## Evidence

- `docs/ops/daily_triage.md`
- `Makefile`
- `docs/ops/migration_notes.md`
- `fixtures/open_incidents.json`
- `.scenario_variant`

## Procedure

1. Inspect the cited workflow files before reusing any draft artifact.
2. Use the exact entrypoint above across the config and automation artifacts.
3. Keep schedule semantics out of the task body.
