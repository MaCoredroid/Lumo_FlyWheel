# Automation Proposal

## Task

Run the live daily triage workflow through `make codex-daily-triage`. Preserve the live task semantics: prioritize blocker incidents for the on-call rotation before any general backlog cleanup, and produce summary output that can feed the response review stand-up. Do not restate or embed the cron cadence inside the task body.

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
