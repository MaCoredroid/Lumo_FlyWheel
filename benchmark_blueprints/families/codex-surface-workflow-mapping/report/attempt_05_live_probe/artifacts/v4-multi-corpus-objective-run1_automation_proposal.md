# Automation Proposal

## Task

Run the blockers-first Codex daily triage workflow for the on-call rotation, use the repo evidence surface for local dry runs, and produce a stand-up-ready summary without broad queue-sweep behavior.

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
