# Mapping Note

## skill

The skill should expose the live Codex-facing entrypoint as `make codex-daily-triage`, because the workflow doc names that target as live and the Makefile resolves it to the active `scripts/triage.py` implementation rather than the rollback helper.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`
- `docs/ops/migration_notes.md`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## toml

The TOML should preserve the same live entrypoint and point readers at the Make target, live workflow doc, cron file, and migration note so operators do not anchor on the deprecated helper during the noisy-distractor variant.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`
- `ops/automation/triage.cron`
- `docs/ops/migration_notes.md`

Command literals:
- `make codex-daily-triage`
- `0 9 * * 1-5`

## automation

The automation artifact should keep task semantics separate from schedule semantics: the task stays the live blocker triage prompt, while the weekday cadence comes from `ops/automation/triage.cron`. The command stays on the same live Make entrypoint instead of embedding the deprecated helper.

Source paths:
- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `Makefile`

Command literals:
- `make codex-daily-triage`
- `0 9 * * 1-5`

## mapping_note

Because `.scenario_variant` is `v2-noisy-distractor`, the mapping note needs to document why the legacy schedule and legacy skill draft are rejected. The distractors still exist for rollback diffing, but the migration note says they never became the live entrypoint.

Source paths:
- `.scenario_variant`
- `docs/ops/legacy_schedule.md`
- `skills/legacy/daily_triage/SKILL.md`
- `docs/ops/migration_notes.md`

Command literals:
- `python3 scripts/triage_legacy.py --window today --emit-md reports/daily_triage.md`
- `0 9 * * *`

## Rejected Candidates

- `skills/legacy/daily_triage/SKILL.md`: This is a legacy draft that still points at the deprecated helper. The migration note says the helper was retained for rollback diffing only and did not become the live entrypoint.
- `docs/ops/legacy_schedule.md`: This note explicitly predates the current automation and documents the legacy seven-day sweep. The live cadence is the weekday cron literal in `ops/automation/triage.cron`.
