# Automation Proposal

## Task

Run `make codex-daily-triage` on the weekday cadence, triage blocker incidents for on-call ownership clarity, and produce the summary artifact for the response review stand-up. Do not turn this task into a weekend helper run or a broad backlog sweep.

## Schedule

- kind: `cron`
- schedule_literal: `0 9 * * 1-5`
- timezone: `America/Los_Angeles`

## Command

`make codex-daily-triage`

## Evidence

- `docs/ops/daily_triage.md`
- `incident_context/recovered_plan.md`
- `ops/automation/triage.cron`
