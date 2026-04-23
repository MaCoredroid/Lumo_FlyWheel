# Workflow Mapping

## Objective

Run the live daily triage workflow against the active incident window, summarize current blockers for the on-call rotation, and treat fixtures/open_incidents.json as the local dry-run evidence surface for this clean baseline variant.

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
