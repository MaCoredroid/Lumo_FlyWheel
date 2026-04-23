# Automation Proposal

## Task

Run `make codex-daily-triage` to triage blocker incidents for the on-call rotation first and emit the summary for the response review stand-up. Keep the task focused on the blocker-first workflow and do not fold the weekday schedule into the task body or revive the rollback-only legacy helper.

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
- `incident_context/weekend_rollback.md`
