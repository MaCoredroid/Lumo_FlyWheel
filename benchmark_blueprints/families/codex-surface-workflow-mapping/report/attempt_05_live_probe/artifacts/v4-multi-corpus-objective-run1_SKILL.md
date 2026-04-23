# Workflow Mapping

## Objective

Run the live Codex triage flow against blocker incidents for the on-call rotation, preserve owner clarity, and emit a response-review-stand-up summary before any general queue cleanup.

## Entrypoint

`make codex-daily-triage`

## Evidence

- `docs/ops/daily_triage.md`
- `Makefile`
- `release_context/objective_shift.md`
- `release_context/release_notes_2026_04.md`
- `docs/ops/migration_notes.md`

## Procedure

1. Inspect the cited workflow files before reusing any draft artifact.
2. Use the exact entrypoint above across the config and automation artifacts.
3. Keep schedule semantics out of the task body.
