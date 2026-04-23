# Automation Proposal

## Task

Run the live blockers-first daily triage workflow for the on-call rotation. Keep the task semantics focused on triaging blocker incidents and producing the stand-up summary; treat the weekday 09:00 cadence as scheduler configuration, not part of the task body. Use the live Make entrypoint and not the rollback-only legacy helper.

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
