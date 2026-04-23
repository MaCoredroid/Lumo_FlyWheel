# Automation Proposal

## Task

Run the live daily triage workflow for the on-call rotation, summarize the current blocker incidents, and ignore rollback-only legacy helpers or abandoned migration drafts when producing Codex-facing output.

## Schedule

- kind: `cron`
- schedule_literal: `0 9 * * 1-5`
- timezone: `America/Los_Angeles`

## Command

`make codex-daily-triage`

## Evidence

- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `docs/ops/migration_notes.md`
- `fixtures/open_incidents.json`
