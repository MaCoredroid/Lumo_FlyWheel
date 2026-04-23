# Workflow Mapping

## Objective

Triage the open incident queue for the on-call rotation and summarize the live blockers. In this dirty-state variant, keep abandoned Codex drafts and rollback-only helpers as review evidence rather than as runnable sources of truth.

## Entrypoint

`make codex-daily-triage`

## Evidence

- `docs/ops/daily_triage.md`
- `Makefile`
- `docs/ops/migration_notes.md`
- `.scenario_variant`

## Procedure

1. Inspect the cited workflow files before reusing any draft artifact.
2. Use the exact entrypoint above across the config and automation artifacts.
3. Keep schedule semantics out of the task body.
