# Automation Proposal

## Task

Invoke the live daily triage workflow through make codex-daily-triage, which resolves to python3 scripts/triage.py --window active --emit-md reports/daily_triage.md, so the on-call rotation gets a blocker-focused summary from the active incident window. Keep this task description separate from the weekday 09:00 cadence.

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
- `scripts/triage.py`
