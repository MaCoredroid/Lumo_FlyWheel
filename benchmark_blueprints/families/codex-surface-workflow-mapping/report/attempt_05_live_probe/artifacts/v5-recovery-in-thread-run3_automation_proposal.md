# Automation Proposal

## Task

Run make codex-daily-triage to triage blocker incidents for the on-call rotation before any backlog cleanup and emit the summary for the response review stand-up. Use fixtures/open_incidents.json for local dry runs, and keep cadence configuration out of this task body.

## Schedule

- kind: `cron`
- schedule_literal: `0 9 * * 1-5`
- timezone: `America/Los_Angeles`

## Command

`make codex-daily-triage`

## Evidence

- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `scripts/triage.py`
- `fixtures/open_incidents.json`
- `release_context/release_notes_2026_04.md`
- `release_context/objective_shift.md`
- `incident_context/recovered_plan.md`
