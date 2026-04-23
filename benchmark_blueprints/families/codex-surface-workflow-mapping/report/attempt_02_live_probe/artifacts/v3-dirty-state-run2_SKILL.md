# Workflow Mapping

## Objective

Run the live daily triage workflow for the on-call rotation, summarize the active blocker queue, and use fixtures/open_incidents.json as the local dry-run evidence surface. For the v3-dirty-state variant, keep the mapping centered on active blockers and the rollback-only legacy helper out of the Codex entrypoint.

## Entrypoint

`make codex-daily-triage`

## Evidence

- `Makefile`
- `docs/ops/daily_triage.md`
- `scripts/triage.py`
- `fixtures/open_incidents.json`

## Procedure

1. Inspect the cited workflow files before reusing any draft artifact.
2. Use the exact entrypoint above across the config and automation artifacts.
3. Keep schedule semantics out of the task body.
