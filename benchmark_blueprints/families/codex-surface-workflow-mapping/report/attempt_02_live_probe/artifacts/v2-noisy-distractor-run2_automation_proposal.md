# Automation Proposal

## Task

Run the live daily triage workflow via `make codex-daily-triage` to triage the open incident queue for the on-call rotation and summarize the live blockers. Do not fold the weekday cadence into the task body, and ignore rollback-only legacy helpers when mapping this noisy-distractor variant.

## Schedule

- kind: `cron`
- schedule_literal: `0 9 * * 1-5`
- timezone: `America/Los_Angeles`

## Command

`make codex-daily-triage`

## Evidence

- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `docs/ops/migration_notes.md`
- `.scenario_variant`
