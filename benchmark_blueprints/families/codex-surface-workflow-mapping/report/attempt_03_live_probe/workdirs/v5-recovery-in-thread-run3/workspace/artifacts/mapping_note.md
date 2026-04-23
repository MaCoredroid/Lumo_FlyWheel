# Mapping Note

## skill

The skill maps to the live Codex-facing entrypoint `make codex-daily-triage`, while the cited release context narrows the task to blocker-first triage, on-call ownership clarity, and a stand-up-ready summary instead of the older broad queue sweep narrative.

Source paths:
- `docs/ops/daily_triage.md`
- `release_context/objective_shift.md`
- `release_context/release_notes_2026_04.md`
- `incident_context/recovered_plan.md`
- `Makefile`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## toml

The TOML should keep the same live Make entrypoint while still citing the underlying implementation and the rollback recovery files, so the config remains anchored on the approved weekday blocker-first path rather than any legacy helper invocation.

Source paths:
- `docs/ops/daily_triage.md`
- `incident_context/recovered_plan.md`
- `ops/automation/triage.cron`
- `Makefile`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## automation

The automation artifact keeps task semantics separate from schedule semantics: the task remains the live Make entrypoint and blocker-first on-call workflow, while the weekday cadence comes from the recovered plan and current cron literal after the weekend rollback.

Source paths:
- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `incident_context/recovered_plan.md`
- `incident_context/weekend_rollback.md`
- `release_context/release_notes_2026_04.md`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## mapping_note

The mapping note needs to explain why legacy and abandoned candidates stay rejected in this recovery variant: migration notes keep the helper as rollback-only, the incident rollback documents noisy weekend regressions, and the stale drafts still wrap the deprecated helper instead of the restored Make-based workflow.

Source paths:
- `docs/ops/daily_triage.md`
- `docs/ops/migration_notes.md`
- `incident_context/recovered_plan.md`
- `incident_context/weekend_rollback.md`
- `drafts/codex_skill_patch.md`
- `drafts/codex_triage.toml`
- `skills/legacy/daily_triage/SKILL.md`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage_legacy.py --window today --emit-md reports/daily_triage.md`

## Rejected Candidates

- `docs/ops/legacy_schedule.md`: This note explicitly says it pre-dates the current automation, still carries the deprecated seven-day helper path, and says not to use it as the source of truth for the live workflow.
- `skills/legacy/daily_triage/SKILL.md`: The legacy skill states that it predates the current migration and exists only for wording comparison, so it cannot define the current Codex-native mapping.
- `drafts/codex_triage.toml`: The draft is marked abandoned, points at the legacy helper and seven-day schedule, and its own note says to resume from the live Make target instead.
- `drafts/codex_skill_patch.md`: This abandoned patch wrapped the deprecated helper and was never shipped, so in the recovery variant it is evidence of a rejected migration attempt rather than a valid workflow source.
