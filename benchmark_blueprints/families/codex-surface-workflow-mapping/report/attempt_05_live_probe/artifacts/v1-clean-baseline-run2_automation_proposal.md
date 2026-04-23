# Automation Proposal

## Task

Run `make codex-daily-triage` for the live daily triage workflow, which shells out to `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md` to triage the open incident queue for the on-call rotation and summarize the live blockers. Keep this task body separate from the weekday schedule literal.

## Schedule

- kind: `cron`
- schedule_literal: `0 9 * * 1-5`
- timezone: `America/Los_Angeles`

## Command

`make codex-daily-triage`

## Evidence

- `AGENTS.md`
- `Makefile`
- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `scripts/triage.py`
