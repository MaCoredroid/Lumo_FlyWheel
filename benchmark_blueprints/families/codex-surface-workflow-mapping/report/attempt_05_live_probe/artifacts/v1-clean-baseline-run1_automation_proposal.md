# Automation Proposal

## Task

Run the active Codex daily triage workflow for the open incident queue, summarize blocker incidents from the live evidence surface, and emit the daily triage report. Keep this task body independent from cadence; cadence is owned by the schedule field.

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
- `scripts/triage.py`
- `fixtures/open_incidents.json`
