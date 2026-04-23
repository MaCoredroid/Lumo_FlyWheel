# Workflow Mapping

## Objective

Run the blocker-first daily triage for the on-call rotation, keep ownership clarity explicit, and produce summary output that can feed the response review stand-up.

## Entrypoint

`make codex-daily-triage`

## Evidence

- `docs/ops/daily_triage.md`
- `Makefile`
- `release_context/objective_shift.md`
- `release_context/release_notes_2026_04.md`
- `scripts/triage.py`

## Procedure

1. Inspect the cited workflow files before reusing any draft artifact.
2. Use the exact entrypoint above across the config and automation artifacts.
3. Keep schedule semantics out of the task body.
