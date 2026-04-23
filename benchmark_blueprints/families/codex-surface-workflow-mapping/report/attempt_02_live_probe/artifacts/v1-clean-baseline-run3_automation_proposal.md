# Automation Proposal

## Task

Run the live daily triage workflow by invoking `make codex-daily-triage`, which shells out to `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`, then summarize the live blockers for the on-call rotation. Keep this task body focused on the triage work itself and treat the schedule as separate automation metadata. For local dry runs, use `fixtures/open_incidents.json` as the sample evidence surface.

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
- `Makefile`
