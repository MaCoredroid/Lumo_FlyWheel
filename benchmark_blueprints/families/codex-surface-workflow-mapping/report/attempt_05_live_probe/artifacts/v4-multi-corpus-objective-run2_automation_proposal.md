# Automation Proposal

## Task

Run the live daily triage workflow through the Make target, prioritize blocker incidents for the on-call rotation before any general queue cleanup, and keep the task body focused on blocker-first triage while leaving cadence to the separate schedule field.

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
- `Makefile`
