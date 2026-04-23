# Mapping Note

## skill

The skill artifact should expose the live Codex-facing entrypoint from docs/ops/daily_triage.md and keep the scope on blocker triage for the on-call rotation instead of carrying forward the legacy draft wording.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`

Command literals:
- `make codex-daily-triage`

## toml

The TOML should mirror the same live entrypoint and read set so the config stays aligned with the active Make target and the pinned weekday automation source.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`
- `ops/automation/triage.cron`

Command literals:
- `make codex-daily-triage`

## automation

The automation artifact should keep task semantics on the live Codex entrypoint while taking schedule semantics from the pinned weekday cadence rather than embedding the Python helper or stale seven-day sweep directly into the task body.

Source paths:
- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`

Command literals:
- `make codex-daily-triage`
- `0 9 * * 1-5`

## mapping_note

The mapping note should document the live command chain from the Codex entrypoint to the active Python helper and record that the v1-clean-baseline variant keeps the baseline live workflow instead of switching to rollback-only legacy helpers.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`
- `ops/automation/triage.cron`
- `.scenario_variant`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## Rejected Candidates

- `docs/ops/legacy_schedule.md`: Rejected because this note says it pre-dates the current automation and explicitly says not to use it as the source of truth when mapping the live workflow. It also pins the stale daily cron literal `0 9 * * *` instead of the live weekday cadence.
- `skills/legacy/daily_triage/SKILL.md`: Rejected because the file labels itself as a legacy draft that predates the current repo migration, while docs/ops/daily_triage.md names `make codex-daily-triage` as the live Codex-facing entrypoint.
- `Makefile#daily-triage-legacy`: Rejected because scripts/triage_legacy.py marks this invocation as deprecated and preserved for rollback diffing only, while the live Make target is `codex-daily-triage` and points at the active `scripts/triage.py` flow.
