# Workflow Mapping

## Objective

Run the recovered weekday blocker-first triage flow for the on-call rotation, preserving ownership clarity and a stand-up-ready summary instead of widening back into the older broad queue sweep.

## Entrypoint

`make codex-daily-triage`

## Evidence

- `docs/ops/daily_triage.md`
- `release_context/objective_shift.md`
- `release_context/release_notes_2026_04.md`
- `incident_context/recovered_plan.md`

## Procedure

1. Inspect the cited workflow files before reusing any draft artifact.
2. Use the exact entrypoint above across the config and automation artifacts.
3. Keep schedule semantics out of the task body.
