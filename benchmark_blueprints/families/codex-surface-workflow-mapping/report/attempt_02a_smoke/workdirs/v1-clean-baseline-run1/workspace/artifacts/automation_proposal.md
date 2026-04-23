# Automation Proposal

## Task

Run `make codex-daily-triage` to triage the open incident queue for the on-call rotation and summarize the live blockers. Keep task semantics separate from the weekday 09:00 schedule and use `fixtures/open_incidents.json` only as the local dry-run evidence surface.

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
- `fixtures/open_incidents.json`
