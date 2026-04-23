# Automation Proposal

## Task

Run the live daily triage workflow for the on-call rotation, summarize blocker incidents from the open incident queue, and keep the task body separate from the cron schedule. Preserve the weekday-only cadence because the local incident evidence includes weekend automation page noise, and use fixtures/open_incidents.json as the dry-run evidence surface.

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
- `.scenario_variant`
