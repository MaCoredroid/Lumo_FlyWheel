# Automation Proposal

## Task

Run `make codex-daily-triage` to triage the open incident queue for the on-call rotation and summarize the live blockers. Keep the task body focused on the workflow itself; the weekday 09:00 cadence belongs in scheduler configuration, not in the task instructions. Use `fixtures/open_incidents.json` as the sample evidence surface for local dry runs.

## Schedule

- kind: `cron`
- schedule_literal: `0 9 * * 1-5`
- timezone: `America/Los_Angeles`

## Command

`make codex-daily-triage`

## Evidence

- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
