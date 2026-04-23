# Automation Proposal

## Task

Run the live blocker-first triage workflow for the on-call rotation and refresh the markdown summary used by the response review stand-up. Keep task semantics about triage ownership and blocker handling only; keep cadence in the scheduler config.

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
- `release_context/release_notes_2026_04.md`
- `release_context/objective_shift.md`
