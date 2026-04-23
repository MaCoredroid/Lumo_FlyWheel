# Automation Proposal

## Task

Run `make codex-daily-triage` to triage the open incident queue for the on-call rotation and summarize the live blockers. Use `fixtures/open_incidents.json` as the local dry-run evidence surface. Keep schedule semantics out of this task body so cadence stays in automation metadata.

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
- `fixtures/open_incidents.json`
