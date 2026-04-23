# Workflow Mapping

## Objective

Triage the open incident queue for the on-call rotation and summarize the live blockers. In this v3-dirty-state bundle, prefer the live Make target and current docs over rollback-only helpers or abandoned migration drafts.

## Entrypoint

`make codex-daily-triage`

## Evidence

- `docs/ops/daily_triage.md`
- `Makefile`
- `scripts/triage.py`
- `.scenario_variant`

## Procedure

1. Inspect the cited workflow files before reusing any draft artifact.
2. Use the exact entrypoint above across the config and automation artifacts.
3. Keep schedule semantics out of the task body.
