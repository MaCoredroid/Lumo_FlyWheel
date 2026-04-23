# Mapping Note

## skill

The skill uses the documented live Codex-facing entrypoint instead of wrapping the rollback-only helper. Its scope follows the release-context objective shift toward blocker-first triage, explicit on-call ownership, and a summary for the response review stand-up.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`
- `release_context/objective_shift.md`
- `release_context/release_notes_2026_04.md`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## toml

The TOML pins the same make-based entrypoint and points readers at the live workflow files plus release context before they reuse any migration-era artifact. That keeps the config aligned with the current blocker-first weekday workflow instead of a stale broad queue sweep.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`
- `ops/automation/triage.cron`
- `release_context/objective_shift.md`
- `release_context/release_notes_2026_04.md`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## automation

The automation proposal keeps task semantics separate from schedule semantics: the task stays focused on blocker-first on-call triage and stand-up-ready output, while the weekday cadence stays in the cron fields. That matches the live weekday schedule and the release note that weekend queue sweeps were deprioritized after March.

Source paths:
- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `release_context/release_notes_2026_04.md`
- `Makefile`

Command literals:
- `make codex-daily-triage`
- `0 9 * * 1-5`

## mapping_note

The mapping note records why the live workflow stays on the make target and why legacy or abandoned candidates are rejected. Migration notes, legacy docs, and draft files all show the deprecated helper was kept only for rollback diffing and never became the live entrypoint.

Source paths:
- `docs/ops/legacy_schedule.md`
- `docs/ops/migration_notes.md`
- `drafts/codex_triage.toml`
- `drafts/codex_skill_patch.md`
- `skills/legacy/daily_triage/SKILL.md`

Command literals:
- `python3 scripts/triage_legacy.py --window today --emit-md reports/daily_triage.md`
- `0 9 * * *`

## Rejected Candidates

- `scripts/triage_legacy.py`: The script marks its own invocation as deprecated and rollback-only, so it is evidence of a fallback path rather than the live Codex-facing workflow.
- `docs/ops/legacy_schedule.md`: This note explicitly says it pre-dates the current automation and still carries the seven-day legacy schedule, which conflicts with the current blocker-first weekday workflow and the release note that weekend sweeps were deprioritized.
- `skills/legacy/daily_triage/SKILL.md`: The file is labeled as a legacy draft that predates the current repo migration and still points at the deprecated helper instead of the live make target.
- `drafts/codex_triage.toml`: This abandoned draft says do not ship, points at the legacy helper, and keeps the obsolete daily cron instead of resuming from the live make target and weekday cadence.
- `drafts/codex_skill_patch.md`: The abandoned patch wraps the deprecated helper and was never shipped, so it is evidence of a paused migration attempt rather than the current workflow.
