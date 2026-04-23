# Automation Proposal

## Task

Run the live daily triage workflow via `make codex-daily-triage`, then summarize the blocker-only incident queue for the on-call rotation using the same task semantics documented in `docs/ops/daily_triage.md`. Keep the cadence in config only; do not restate the cron schedule inside the task body.

## Schedule

- kind: `cron`
- schedule_literal: `0 9 * * 1-5`
- timezone: `America/Los_Angeles`

## Command

`python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## Evidence

- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `scripts/triage.py`
- `fixtures/open_incidents.json`
