# Automation Proposal

## Task

Run the live daily triage workflow for the on-call rotation, summarize the current blocker incidents, and keep the weekday schedule separate from the task instructions.

## Schedule

- kind: `cron`
- schedule_literal: `0 9 * * 1-5`
- timezone: `America/Los_Angeles`

## Command

`make codex-daily-triage`

## Evidence

- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `fixtures/open_incidents.json`
- `Makefile`
