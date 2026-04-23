# Automation Proposal

## Task

Run the live daily triage workflow for the active incident window, summarize the current blocker incidents for the on-call rotation, and use the repo fixture only as a local dry-run evidence surface when validating output phrasing.

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
