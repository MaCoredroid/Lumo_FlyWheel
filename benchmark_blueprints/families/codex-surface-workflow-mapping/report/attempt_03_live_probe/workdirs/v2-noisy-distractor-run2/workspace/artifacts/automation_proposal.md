# Automation Proposal

## Task

Run the live daily triage workflow for the on-call rotation, summarize the open blockers, and keep the task body focused on incident triage instead of schedule policy.

## Schedule

- kind: `cron`
- schedule_literal: `0 9 * * 1-5`
- timezone: `America/Los_Angeles`

## Command

`make codex-daily-triage`

## Evidence

- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `Makefile`
