# Automation Proposal

## Task

Run the live daily triage workflow via `make codex-daily-triage`, which resolves to `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`, to triage the open incident queue for the on-call rotation and summarize the live blockers. Keep the task body limited to the workflow itself; carry the weekday 09:00 cadence separately as schedule metadata.

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
