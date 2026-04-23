# Mapping Note

## skill

The skill should anchor on `make codex-daily-triage` because the live Codex-facing entrypoint is named explicitly in the workflow doc, and the release context narrows the scope to blocker-first, on-call-oriented triage rather than a broad queue sweep.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`
- `release_context/objective_shift.md`
- `release_context/release_notes_2026_04.md`

Command literals:
- `make codex-daily-triage`

## toml

The TOML should keep the same live entrypoint while marking the current workflow scope as blocker incidents first, on-call ownership clarity, and stand-up-ready summary output. The Make target remains the live wrapper even though the underlying Python command still exists.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`
- `release_context/objective_shift.md`
- `release_context/release_notes_2026_04.md`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## automation

The automation artifact should keep task semantics and schedule semantics separate: the task invokes the live Make entrypoint, while the weekday-only cadence stays in the cron configuration. The weekday narrowing also matches the release note that weekend queue sweeps were deprioritized after noisy pages in March.

Source paths:
- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `release_context/release_notes_2026_04.md`

Command literals:
- `make codex-daily-triage`
- `0 9 * * 1-5`

## mapping_note

The mapping note should explicitly record why the live workflow wins over stale candidates: the legacy helper is retained only for rollback diffing, the legacy schedule note is not the source of truth, and the abandoned Codex patch never shipped. That evidence belongs in the note rather than being silently ignored.

Source paths:
- `docs/ops/daily_triage.md`
- `docs/ops/legacy_schedule.md`
- `docs/ops/migration_notes.md`
- `drafts/codex_skill_patch.md`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage_legacy.py --window today --emit-md reports/daily_triage.md`

## Rejected Candidates

- `scripts/triage_legacy.py`: Deprecated rollback-only helper; migration notes say it was kept for diffing outputs and never replaced the live Make entrypoint.
- `skills/legacy/daily_triage/SKILL.md`: Legacy skill draft still points at the deprecated helper and predates the current repo migration, so it is evidence of stale wording rather than the live workflow.
- `drafts/codex_skill_patch.md`: Abandoned Codex patch that wrapped the deprecated helper and was never shipped, so it cannot be reused as the canonical Codex-native mapping.
