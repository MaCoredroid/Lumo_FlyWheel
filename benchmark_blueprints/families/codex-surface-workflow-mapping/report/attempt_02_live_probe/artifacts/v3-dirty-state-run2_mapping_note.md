# Mapping Note

## skill

Point the skill at `make codex-daily-triage` because the live Codex-facing entrypoint is named that way in the ops doc, and the Makefile resolves it to the active-window triage script that reads the local incident fixture and emits the markdown report.

Source paths:
- `Makefile`
- `docs/ops/daily_triage.md`
- `scripts/triage.py`
- `fixtures/open_incidents.json`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## toml

Keep the TOML entrypoint on the same Make target and require the migration notes so the config does not drift back to the rollback-only helper during the dirty-state variant.

Source paths:
- `Makefile`
- `docs/ops/daily_triage.md`
- `docs/ops/migration_notes.md`

Command literals:
- `make codex-daily-triage`

## automation

Use the live weekday cron cadence from the automation file, but keep the task body aligned to the Codex-facing Make target so task semantics remain separate from schedule semantics.

Source paths:
- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `Makefile`

Command literals:
- `make codex-daily-triage`
- `0 9 * * 1-5`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## mapping_note

Record the live mapping and the stale rejections together: the ops docs state that the Make target stayed live during migration, while the legacy note, draft, and helper are preserved only for rollback diffing or historical comparison.

Source paths:
- `docs/ops/daily_triage.md`
- `docs/ops/migration_notes.md`
- `docs/ops/legacy_schedule.md`
- `drafts/codex_skill_patch.md`
- `skills/legacy/daily_triage/SKILL.md`
- `scripts/triage_legacy.py`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage_legacy.py --window today --emit-md reports/daily_triage.md`

## Rejected Candidates

- `docs/ops/legacy_schedule.md`: Rejected because this note says it pre-dates the current automation and explicitly says not to use it as the source of truth for the live workflow.
- `skills/legacy/daily_triage/SKILL.md`: Rejected because the file identifies itself as a legacy draft that predates the current repo migration and keeps the deprecated helper only for comparison.
- `drafts/codex_skill_patch.md`: Rejected because the draft says it was an abandoned half-finished patch that should not be treated as proof of the live workflow.
- `scripts/triage_legacy.py`: Rejected because the script comments and output both mark it as deprecated and kept only for rollback diffs, not as the live Codex entrypoint.
