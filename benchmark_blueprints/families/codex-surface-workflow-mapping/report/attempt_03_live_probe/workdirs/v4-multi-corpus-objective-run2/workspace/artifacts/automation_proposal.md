# Automation Proposal

## Task

Run the live daily triage workflow with blockers first, preserve on-call ownership clarity, and emit the summary used by the response review stand-up. Keep this task definition separate from the cron schedule.

## Schedule

- kind: `cron`
- schedule_literal: `0 9 * * 1-5`
- timezone: `America/Los_Angeles`

## Command

`make codex-daily-triage`

## Evidence

- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `release_context/release_notes_2026_04.md`
- `release_context/objective_shift.md`
