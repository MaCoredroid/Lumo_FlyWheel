# Workflow Mapping

## Objective

Run the live blocker-first daily triage flow for the on-call rotation, keep ownership clarity visible in the output, and produce a summary that can feed the response review stand-up without expanding back into broad queue sweeps.

## Entrypoint

`make codex-daily-triage`

## Evidence

- `docs/ops/daily_triage.md`
- `release_context/objective_shift.md`
- `release_context/release_notes_2026_04.md`
- `incident_context/recovered_plan.md`
- `Makefile`

## Procedure

1. Inspect the cited workflow files before reusing any draft artifact.
2. Use the exact entrypoint above across the config and automation artifacts.
3. Keep schedule semantics out of the task body.
