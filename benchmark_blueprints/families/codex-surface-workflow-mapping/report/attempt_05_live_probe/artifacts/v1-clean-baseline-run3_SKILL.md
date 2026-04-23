# Workflow Mapping

## Objective

Run the live daily triage workflow for the on-call rotation, summarize blocker incidents from the active queue, and keep the clean-baseline mapping anchored only to the current repo workflow sources.

## Entrypoint

`make codex-daily-triage`

## Evidence

- `docs/ops/daily_triage.md`
- `Makefile`
- `scripts/triage.py`
- `fixtures/open_incidents.json`

## Procedure

1. Inspect the cited workflow files before reusing any draft artifact.
2. Use the exact entrypoint above across the config and automation artifacts.
3. Keep schedule semantics out of the task body.
