# Automation Proposal

## Task

Run the live daily triage workflow by invoking `make codex-daily-triage`. Prioritize blocker incidents for the on-call rotation before any general backlog cleanup, keep on-call ownership explicit, and produce a summary that can feed the response review stand-up. For local dry runs, use `fixtures/open_incidents.json` as the evidence surface.

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
- `release_context/objective_shift.md`
- `release_context/release_notes_2026_04.md`
