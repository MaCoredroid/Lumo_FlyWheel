# Automation Proposal

## Task

Run the blocker-first daily triage for the active window, keep on-call ownership explicit, and produce the markdown summary for the response review stand-up. Keep the task semantics separate from the cron cadence.

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
- `release_context/objective_shift.md`
- `release_context/release_notes_2026_04.md`
- `scripts/triage.py`
