# Workflow Mapping

## Objective

Triage the open incident queue for the on-call rotation and summarize the live blockers. In this dirty-state variant, inspect and reject abandoned migration drafts and rollback-only helpers before reusing any Codex artifact.

## Entrypoint

`make codex-daily-triage`

## Evidence

- `docs/ops/daily_triage.md`
- `docs/ops/migration_notes.md`
- `drafts/codex_skill_patch.md`
- `drafts/codex_triage.toml`

## Procedure

1. Inspect the cited workflow files before reusing any draft artifact.
2. Use the exact entrypoint above across the config and automation artifacts.
3. Keep schedule semantics out of the task body.
