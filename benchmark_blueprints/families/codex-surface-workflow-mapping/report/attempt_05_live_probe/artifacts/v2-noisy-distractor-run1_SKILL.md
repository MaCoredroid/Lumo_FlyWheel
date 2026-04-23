# Workflow Mapping

## Objective

Run the live daily triage workflow for the on-call rotation: use the active incident window, summarize blocker incidents, and keep non-blocking routing drift as noise instead of remapping the task.

## Entrypoint

`make codex-daily-triage`

## Evidence

- `docs/ops/daily_triage.md`
- `Makefile`
- `scripts/triage.py`
- `fixtures/open_incidents.json`
- `.scenario_variant`

## Procedure

1. Inspect the cited workflow files before reusing any draft artifact.
2. Use the exact entrypoint above across the config and automation artifacts.
3. Keep schedule semantics out of the task body.
