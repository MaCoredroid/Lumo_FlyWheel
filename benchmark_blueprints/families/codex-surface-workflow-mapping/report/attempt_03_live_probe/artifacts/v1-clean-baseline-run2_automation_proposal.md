# Automation Proposal

## Task

Run the live Codex-facing daily triage workflow for the on-call rotation, use the repo's active triage path, and summarize the live blockers from the open incident queue. Keep schedule semantics separate from the task body.

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
