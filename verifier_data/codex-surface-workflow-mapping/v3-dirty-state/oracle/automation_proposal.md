# Automation Proposal

## Task

Triage the open incident queue for the on-call rotation and summarize the live blockers.

## Schedule

- kind: `heartbeat`
- schedule_literal: `0 9 * * 1-5`
- timezone: `America/Los_Angeles`

## Command

`make codex-daily-triage`

## Evidence

- `ops/automation/triage.cron`
- `docs/ops/daily_triage.md`
