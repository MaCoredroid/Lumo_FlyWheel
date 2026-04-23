# Automation Proposal

## Task

Run the blocker-first triage flow for the on-call rotation and produce the summary used in the response review stand-up by invoking `make codex-daily-triage`. Use `fixtures/open_incidents.json` as the local dry-run evidence surface, and keep this task body separate from the weekday schedule.

## Schedule

- kind: `cron`
- schedule_literal: `0 9 * * 1-5`
- timezone: `America/Los_Angeles`

## Command

`make codex-daily-triage`

## Evidence

- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `incident_context/recovered_plan.md`
- `release_context/objective_shift.md`
- `fixtures/open_incidents.json`
