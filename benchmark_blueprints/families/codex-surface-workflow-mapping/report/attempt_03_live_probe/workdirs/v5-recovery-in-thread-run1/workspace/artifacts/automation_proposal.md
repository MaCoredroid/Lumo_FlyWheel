# Automation Proposal

## Task

Run `make codex-daily-triage` to produce the blocker-first daily triage summary for the on-call rotation and the response review stand-up. Keep the task scoped to the live Make entrypoint, and do not substitute the rollback-only legacy helper or reintroduce a broad queue sweep.

## Schedule

- kind: `cron`
- schedule_literal: `0 9 * * 1-5`
- timezone: `America/Los_Angeles`

## Command

`make codex-daily-triage`

## Evidence

- `docs/ops/daily_triage.md`
- `incident_context/recovered_plan.md`
- `ops/automation/triage.cron`
- `release_context/objective_shift.md`
- `release_context/release_notes_2026_04.md`
- `incident_context/weekend_rollback.md`
