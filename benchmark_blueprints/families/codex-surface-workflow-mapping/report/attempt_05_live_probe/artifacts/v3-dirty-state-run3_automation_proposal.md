# Automation Proposal

## Task

Run the live Codex daily triage workflow through the repo Make entrypoint, not the rollback-only helper. Summarize the active-window blocker incidents for the on-call rotation in the dirty-state variant, and keep schedule configuration out of this task body.

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
- `.scenario_variant`
- `fixtures/open_incidents.json`
