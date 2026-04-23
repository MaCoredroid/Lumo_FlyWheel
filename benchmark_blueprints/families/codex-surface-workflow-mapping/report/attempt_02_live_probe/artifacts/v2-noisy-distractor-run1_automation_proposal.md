# Automation Proposal

## Task

Run `make codex-daily-triage` to triage the open incident queue for the on-call rotation and summarize the live blockers. Use `fixtures/open_incidents.json` as the sample evidence surface for local dry runs. Keep task semantics separate from schedule semantics when drafting the Codex automation prompt. In this noisy-distractor variant, ignore rollback-only helpers and pre-current schedule notes and keep the mapping anchored on the live Make target.

## Schedule

- kind: `cron`
- schedule_literal: `0 9 * * 1-5`
- timezone: `America/Los_Angeles`

## Command

`make codex-daily-triage`

## Evidence

- `Makefile`
- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `docs/ops/migration_notes.md`
- `docs/ops/legacy_schedule.md`
