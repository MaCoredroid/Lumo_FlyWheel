# Workflow Mapping

## Objective

Triage the open incident queue for the on-call rotation and summarize the live blockers. In the v3-dirty-state bundle, keep the Codex mapping pinned to the live Make target and treat rollback-only legacy helpers and abandoned drafts as comparison evidence, not as the runnable workflow.

## Entrypoint

`make codex-daily-triage`

## Evidence

- `Makefile`
- `docs/ops/daily_triage.md`
- `docs/ops/migration_notes.md`
- `.scenario_variant`

## Procedure

1. Inspect the cited workflow files before reusing any draft artifact.
2. Use the exact entrypoint above across the config and automation artifacts.
3. Keep schedule semantics out of the task body.
