# Mapping Note

## skill

The skill should expose the Codex-facing entrypoint instead of the raw Python helper because the live workflow document names `make codex-daily-triage` as the entrypoint and the Make target defines the same workflow scope.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`

Command literals:
- `make codex-daily-triage`

## toml

The TOML should pin the same live entrypoint and keep the workflow scope aligned with the blocker-focused daily triage procedure documented in the repo, while treating the cited ops docs as must-read context.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`
- `fixtures/open_incidents.json`

Command literals:
- `make codex-daily-triage`

## automation

The automation mapping should keep task semantics separate from schedule semantics: the task remains daily blocker triage, while the weekday 09:00 cadence is sourced from the cron file and mirrored in the live workflow doc.

Source paths:
- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `Makefile`

Command literals:
- `make codex-daily-triage`
- `0 9 * * 1-5`

## mapping_note

The note should explain why the mapping favors the live Make target and rejects the stale helper surfaces. This variant includes noisy distractors, so the note needs to record that legacy commands remain in-tree only for rollback diffing and old-vs-new wording comparison.

Source paths:
- `docs/ops/daily_triage.md`
- `docs/ops/legacy_schedule.md`
- `docs/ops/migration_notes.md`
- `skills/legacy/daily_triage/SKILL.md`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage_legacy.py --window today --emit-md reports/daily_triage.md`

## Rejected Candidates

- `docs/ops/legacy_schedule.md`: Rejected because this note explicitly says it pre-dates the current automation, cites the deprecated helper, and says not to use it as the source of truth for the live mapping.
- `skills/legacy/daily_triage/SKILL.md`: Rejected because the legacy skill is a preserved draft from before the migration and exists only for wording comparison, not as the current Codex-native workflow surface.
- `scripts/triage_legacy.py`: Rejected because the script is marked deprecated and rollback-only, so mapping it as the live entrypoint would anchor on a retained helper instead of the live Make target.
