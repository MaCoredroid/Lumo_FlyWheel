# Automation Proposal

## Task

Triage the open incident queue for the on-call rotation and summarize the live blockers. Use fixtures/open_incidents.json as the sample evidence surface for local dry runs.

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
