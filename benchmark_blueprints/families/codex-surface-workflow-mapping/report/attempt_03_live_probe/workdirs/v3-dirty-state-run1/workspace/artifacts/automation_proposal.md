# Automation Proposal

## Task

Run the live daily triage workflow for the on-call rotation, summarize the live blocker incidents, and use `fixtures/open_incidents.json` as the local dry-run evidence surface. Keep the task body focused on triage semantics; keep cadence in the schedule configuration instead of the prompt.

## Schedule

- kind: `cron`
- schedule_literal: `0 9 * * 1-5`
- timezone: `America/Los_Angeles`

## Command

`make codex-daily-triage`

## Evidence

- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `fixtures/open_incidents.json`
- `.scenario_variant`
