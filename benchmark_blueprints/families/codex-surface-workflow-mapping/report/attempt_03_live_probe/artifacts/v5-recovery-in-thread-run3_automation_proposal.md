# Automation Proposal

## Task

Run the live blocker-first triage workflow via `make codex-daily-triage`. Prioritize blocker incidents for the on-call rotation before any broader cleanup, preserve ownership clarity in the emitted summary, and keep this task definition independent from the scheduler configuration.

## Schedule

- kind: `cron`
- schedule_literal: `0 9 * * 1-5`
- timezone: `America/Los_Angeles`

## Command

`make codex-daily-triage`

## Evidence

- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `incident_context/recovered_plan.md`
- `release_context/objective_shift.md`
- `release_context/release_notes_2026_04.md`
