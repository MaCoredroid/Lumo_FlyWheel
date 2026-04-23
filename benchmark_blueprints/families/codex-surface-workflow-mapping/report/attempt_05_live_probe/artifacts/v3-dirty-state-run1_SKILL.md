# Workflow Mapping

## Objective

Triage the open incident queue for the on-call rotation and summarize the live blockers without falling back to rollback-only legacy helpers.

## Entrypoint

`make codex-daily-triage`

## Evidence

- `docs/ops/daily_triage.md`
- `Makefile`
- `docs/ops/migration_notes.md`

## Procedure

1. Inspect the cited workflow files before reusing any draft artifact.
2. Use the exact entrypoint above across the config and automation artifacts.
3. Keep schedule semantics out of the task body.
