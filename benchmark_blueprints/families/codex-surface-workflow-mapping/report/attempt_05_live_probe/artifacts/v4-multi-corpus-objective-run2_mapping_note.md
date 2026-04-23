# Mapping Note

## skill

Point the skill at the live Codex-facing Make target rather than the underlying helper so the rendered skill matches the documented entrypoint and the blocker-first release objective.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`
- `release_context/objective_shift.md`
- `release_context/release_notes_2026_04.md`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## toml

Use the same Make entrypoint in TOML, and require the current daily-triage and release-context docs so consumers see the narrowed blocker-first scope instead of the older broad sweep narrative.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`
- `release_context/objective_shift.md`
- `release_context/release_notes_2026_04.md`
- `docs/ops/migration_notes.md`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## automation

Keep automation schedule semantics in the cron literal while task semantics stay on the live Make entrypoint and the blocker-first release scope; this avoids reintroducing the deprecated daily seven-day sweep framing.

Source paths:
- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `Makefile`
- `release_context/objective_shift.md`
- `release_context/release_notes_2026_04.md`

Command literals:
- `make codex-daily-triage`
- `0 9 * * 1-5`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## mapping_note

Record the live Make target and explicitly reject abandoned or rollback-only legacy artifacts so the note preserves why the deprecated helper and old every-day cron are not the Codex-native workflow.

Source paths:
- `docs/ops/daily_triage.md`
- `docs/ops/legacy_schedule.md`
- `docs/ops/migration_notes.md`
- `drafts/codex_triage.toml`
- `drafts/codex_skill_patch.md`
- `skills/legacy/daily_triage/SKILL.md`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage_legacy.py --window today --emit-md reports/daily_triage.md`
- `0 9 * * *`

## Rejected Candidates

- `docs/ops/legacy_schedule.md`: Legacy note explicitly says it pre-dates the current automation and must not be used as the source of truth; it also carries the obsolete every-day cron `0 9 * * *`.
- `skills/legacy/daily_triage/SKILL.md`: The legacy skill draft says it predates the current repo migration and is kept only for comparing old vs. new wording, so it is evidence of deprecated language rather than the live entrypoint.
- `drafts/codex_triage.toml`: The draft TOML is marked abandoned and points at the deprecated helper plus the stale daily cron, while the migration notes say to resume from the live Make target instead.
- `drafts/codex_skill_patch.md`: The abandoned patch states it tried to wrap the deprecated helper and was never shipped, so it cannot anchor the mapped Codex workflow.
