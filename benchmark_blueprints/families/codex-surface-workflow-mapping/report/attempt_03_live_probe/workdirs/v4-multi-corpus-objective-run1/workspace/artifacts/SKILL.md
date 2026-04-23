# Workflow Mapping

## Objective

Triage blocker incidents for the on-call rotation, keep owner clarity explicit, and produce the daily summary that feeds the response review stand-up before any general backlog cleanup or broad queue sweep.

## Entrypoint

`make codex-daily-triage`

## Evidence

- `Makefile`
- `docs/ops/daily_triage.md`
- `release_context/objective_shift.md`
- `release_context/release_notes_2026_04.md`

## Procedure

1. Inspect the cited workflow files before reusing any draft artifact.
2. Use the exact entrypoint above across the config and automation artifacts.
3. Keep schedule semantics out of the task body.
