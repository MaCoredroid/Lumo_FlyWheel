# Automation Proposal

## Task

Run the live Codex daily triage workflow for the on-call rotation. Keep the task on the active incident window and blocker-only summary, use fixtures/open_incidents.json only as the local dry-run evidence surface, and keep schedule semantics out of the task body.

## Schedule

- kind: `cron`
- schedule_literal: `0 9 * * 1-5`
- timezone: `America/Los_Angeles`

## Command

`make codex-daily-triage`

## Evidence

- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `scripts/triage.py`
- `fixtures/open_incidents.json`
