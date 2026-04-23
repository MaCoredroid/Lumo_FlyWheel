# Workflow Mapping

## Objective

Run the weekday blocker-first daily triage for the on-call rotation before any general backlog cleanup, producing summary output for the response review stand-up.

## Entrypoint

`make codex-daily-triage`

## Evidence

- `Makefile`
- `docs/ops/daily_triage.md`
- `release_context/objective_shift.md`
- `incident_context/recovered_plan.md`

## Procedure

1. Inspect the cited workflow files before reusing any draft artifact.
2. Use the exact entrypoint above across the config and automation artifacts.
3. Keep schedule semantics out of the task body.
