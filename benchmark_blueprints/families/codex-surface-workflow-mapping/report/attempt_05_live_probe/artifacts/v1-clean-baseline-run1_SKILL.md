# Workflow Mapping

## Objective

Map the live Codex daily triage workflow for the open incident queue. The skill should direct Codex to inspect the active workflow evidence, run the repo-supported Make entrypoint, and avoid the legacy helper kept only for comparison.

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
