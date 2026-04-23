# Workflow Mapping

## Objective

Run the blocker-first daily triage workflow for the on-call rotation and produce the markdown summary that feeds the response review stand-up. Treat the old broad queue sweep and seven-day weekend sweep as deprecated context, not the live task.

## Entrypoint

`make codex-daily-triage`

## Evidence

- `docs/ops/daily_triage.md`
- `Makefile`
- `scripts/triage.py`
- `release_context/release_notes_2026_04.md`
- `release_context/objective_shift.md`

## Procedure

1. Inspect the cited workflow files before reusing any draft artifact.
2. Use the exact entrypoint above across the config and automation artifacts.
3. Keep schedule semantics out of the task body.
