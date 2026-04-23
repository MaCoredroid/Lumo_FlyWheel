# Automation Proposal

## Task

Run `make codex-daily-triage` to triage blocker incidents for the on-call rotation, preserve clear ownership in the daily summary, and feed the response review stand-up before any general backlog cleanup. Keep cadence and cron details outside the task body.

## Schedule

- kind: `cron`
- schedule_literal: `0 9 * * 1-5`
- timezone: `America/Los_Angeles`

## Command

`make codex-daily-triage`

## Evidence

- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `release_context/objective_shift.md`
- `release_context/release_notes_2026_04.md`
