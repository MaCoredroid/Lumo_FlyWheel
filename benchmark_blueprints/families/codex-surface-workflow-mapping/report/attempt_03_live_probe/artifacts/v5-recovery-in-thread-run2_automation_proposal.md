# Automation Proposal

## Task

Run the blocker-first on-call triage workflow via `make codex-daily-triage`, review blocker incidents before any general queue cleanup, and emit the summary used for the response review stand-up.

## Schedule

- kind: `cron`
- schedule_literal: `0 9 * * 1-5`
- timezone: `America/Los_Angeles`

## Command

`make codex-daily-triage`

## Evidence

- `Makefile`
- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `incident_context/recovered_plan.md`
