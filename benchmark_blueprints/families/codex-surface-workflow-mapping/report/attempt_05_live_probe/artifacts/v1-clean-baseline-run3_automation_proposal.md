# Automation Proposal

## Task

Run `make codex-daily-triage` to triage the open incident queue for the on-call rotation, summarize the live blockers, and use `fixtures/open_incidents.json` as the local dry-run evidence surface. Keep task semantics separate from schedule semantics.

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
