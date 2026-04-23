# Automation Proposal

## Task

Run the live weekday daily triage for the on-call rotation by using the current workflow entrypoint and summarizing active blockers from the open incident queue. In this dirty-state migration variant, keep rollback-only legacy helpers out of the task mapping even if they remain in the workspace for diffing.

## Schedule

- kind: `cron`
- schedule_literal: `0 9 * * 1-5`
- timezone: `America/Los_Angeles`

## Command

`python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## Evidence

- `ops/automation/triage.cron`
- `docs/ops/daily_triage.md`
- `Makefile`
