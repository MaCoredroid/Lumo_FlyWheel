# Mapping Note

## skill

The skill anchors on `make codex-daily-triage` because the ops doc names it as the live Codex-facing entrypoint, while the release context narrows the workflow to blocker-first on-call triage that feeds the stand-up rather than a broad queue sweep.

Source paths:
- `Makefile`
- `docs/ops/daily_triage.md`
- `release_context/objective_shift.md`
- `release_context/release_notes_2026_04.md`

Command literals:
- `make codex-daily-triage`

## toml

The TOML keeps the same live make entrypoint and must-read set so the rendered config stays aligned with the active workflow docs and the April objective shift, instead of reviving the deprecated helper or stale daily sweep language.

Source paths:
- `Makefile`
- `docs/ops/daily_triage.md`
- `release_context/objective_shift.md`
- `release_context/release_notes_2026_04.md`
- `docs/ops/migration_notes.md`

Command literals:
- `make codex-daily-triage`

## automation

The automation mapping keeps task semantics separate from schedule semantics: the task calls the same live make target, while the weekday 09:00 cadence stays sourced from the cron file and ops doc. The release notes also rule out weekend sweep behavior.

Source paths:
- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `release_context/release_notes_2026_04.md`

Command literals:
- `make codex-daily-triage`

## mapping_note

The note explicitly records why stale candidates were rejected: legacy docs, legacy skill text, and abandoned drafts all point at the rollback-only helper, while migration notes state that the live workflow remained on the Make target throughout the codex-native migration.

Source paths:
- `docs/ops/daily_triage.md`
- `docs/ops/legacy_schedule.md`
- `docs/ops/migration_notes.md`
- `drafts/codex_skill_patch.md`
- `drafts/codex_triage.toml`
- `skills/legacy/daily_triage/SKILL.md`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage_legacy.py --window today --emit-md reports/daily_triage.md`

## Rejected Candidates

- `drafts/codex_triage.toml`: The file marks itself as an abandoned draft, points at the deprecated helper, and carries the stale every-day cron instead of the live weekday schedule. Migration notes say the live workflow stayed on the Make target.
- `skills/legacy/daily_triage/SKILL.md`: This skill is labeled as a legacy draft that predates the current migration and exists only for comparison, so it is evidence of the old wording rather than the live Codex workflow.
- `drafts/codex_skill_patch.md`: The patch was never shipped and explicitly says it should not be treated as proof of the live workflow, so it cannot override the current make-based entrypoint and blocker-first release context.
