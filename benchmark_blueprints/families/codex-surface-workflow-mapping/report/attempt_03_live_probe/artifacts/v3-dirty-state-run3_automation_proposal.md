# Automation Proposal

## Task

Run the live daily triage workflow for the on-call rotation, summarize current blocker incidents, and preserve the task definition independently from the weekday schedule literal. Use the current Make target rather than the rollback-only helper or abandoned Codex drafts.

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
- `docs/ops/migration_notes.md`
