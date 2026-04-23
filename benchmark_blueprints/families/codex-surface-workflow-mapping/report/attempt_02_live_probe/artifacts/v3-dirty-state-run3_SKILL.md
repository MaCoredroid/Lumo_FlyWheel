# Workflow Mapping

## Objective

Run the live Codex-facing daily triage workflow for the on-call rotation through `make codex-daily-triage`, summarizing active blockers from the open incident queue. In the `v3-dirty-state` bundle, preserve the live Make target as canonical even though rollback-only legacy helpers and abandoned Codex drafts remain present for migration diffing.

## Entrypoint

`make codex-daily-triage`

## Evidence

- `Makefile`
- `docs/ops/daily_triage.md`
- `docs/ops/migration_notes.md`

## Procedure

1. Inspect the cited workflow files before reusing any draft artifact.
2. Use the exact entrypoint above across the config and automation artifacts.
3. Keep schedule semantics out of the task body.
