# Automation Proposal

## Task

Run the live daily triage workflow by invoking `make codex-daily-triage` so the Make target drives `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`. Keep task semantics separate from schedule semantics, and in this dirty-state bundle do not swap in rollback-only or draft legacy helpers.

## Schedule

- kind: `cron`
- schedule_literal: `0 9 * * 1-5`
- timezone: `America/Los_Angeles`

## Command

`make codex-daily-triage`

## Evidence

- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `docs/ops/migration_notes.md`
- `.scenario_variant`
