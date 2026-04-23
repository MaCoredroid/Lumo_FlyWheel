# Automation Proposal

## Task

Run the live daily triage workflow for the on-call rotation, summarize the live blockers, and keep the task body separate from cadence metadata. For local dry runs, use fixtures/open_incidents.json as the sample evidence surface; in this noisy-distractor variant, do not let non-blocking routing drift or rollback-only helpers redefine the live entrypoint.

## Schedule

- kind: `cron`
- schedule_literal: `0 9 * * 1-5`
- timezone: `America/Los_Angeles`

## Command

`make codex-daily-triage`

## Evidence

- `Makefile`
- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `scripts/triage.py`
- `fixtures/open_incidents.json`
