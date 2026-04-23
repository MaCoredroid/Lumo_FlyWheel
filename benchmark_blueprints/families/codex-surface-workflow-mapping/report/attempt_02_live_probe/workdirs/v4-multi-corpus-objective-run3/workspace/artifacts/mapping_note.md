# Mapping Note

## skill

Map the skill to `make codex-daily-triage` because the daily triage ops doc names it as the live Codex-facing entrypoint, the Make target resolves to the current `scripts/triage.py` invocation, and the active script filters blocker incidents rather than doing a broad queue sweep.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`
- `scripts/triage.py`
- `release_context/release_notes_2026_04.md`
- `release_context/objective_shift.md`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## toml

Keep the TOML aligned to the same live entrypoint and narrow workflow scope. The migration note says the legacy helper remained rollback-only during migration, so the config should read current workflow files first and not route through deprecated helpers.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`
- `docs/ops/migration_notes.md`
- `release_context/release_notes_2026_04.md`
- `release_context/objective_shift.md`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## automation

Use the live Codex entrypoint as the automation command, but keep the weekday schedule separate as scheduler metadata. The cron file provides the cadence literal while the release context and live workflow doc explain that the job now serves blocker-first on-call triage and the response review stand-up.

Source paths:
- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `Makefile`
- `release_context/release_notes_2026_04.md`
- `release_context/objective_shift.md`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## mapping_note

Call out stale candidates explicitly. The legacy schedule note, legacy skill, draft patch, and deprecated helper all point at `python3 scripts/triage_legacy.py --window today --emit-md reports/daily_triage.md`, but those files say the helper is rollback-only and not the live workflow, which conflicts with the current blocker-first objective and weekday cadence.

Source paths:
- `docs/ops/daily_triage.md`
- `docs/ops/migration_notes.md`
- `docs/ops/legacy_schedule.md`
- `skills/legacy/daily_triage/SKILL.md`
- `drafts/codex_skill_patch.md`
- `scripts/triage_legacy.py`
- `release_context/release_notes_2026_04.md`
- `release_context/objective_shift.md`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage_legacy.py --window today --emit-md reports/daily_triage.md`

## Rejected Candidates

- `docs/ops/legacy_schedule.md`: Rejected because this note explicitly pre-dates the current automation, keeps the deprecated helper, and preserves a seven-day sweep that conflicts with the current blocker-first weekday workflow.
- `skills/legacy/daily_triage/SKILL.md`: Rejected because the file labels itself as legacy and still teaches the rollback-only helper instead of the live Codex-facing Make entrypoint.
- `drafts/codex_skill_patch.md`: Rejected because the draft says it was an abandoned patch around the deprecated helper and was never shipped, so it is evidence of drift rather than proof of the live workflow.
