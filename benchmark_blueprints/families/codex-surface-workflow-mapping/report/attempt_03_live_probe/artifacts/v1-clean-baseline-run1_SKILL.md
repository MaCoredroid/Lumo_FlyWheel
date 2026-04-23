# Workflow Mapping

## Objective

Triage the open incident queue for the on-call rotation and summarize the live blockers. Use fixtures/open_incidents.json as the sample evidence surface for local dry runs.

## Entrypoint

`make codex-daily-triage`

## Evidence

- `docs/ops/daily_triage.md`
- `Makefile`
- `fixtures/open_incidents.json`

## Procedure

1. Inspect the cited workflow files before reusing any draft artifact.
2. Use the exact entrypoint above across the config and automation artifacts.
3. Keep schedule semantics out of the task body.
