# Mapping Note

## skill

The skill maps to the live Codex-facing entrypoint rather than the deprecated helper because both `Makefile` and `docs/ops/daily_triage.md` name `make codex-daily-triage` as the current workflow, and `docs/ops/migration_notes.md` says the legacy helper stayed rollback-only. In the noisy-distractor variant, the right response is to ignore stale drafts and anchor the skill on the live Make target.

Source paths:
- `.scenario_variant`
- `Makefile`
- `docs/ops/daily_triage.md`
- `docs/ops/migration_notes.md`

Command literals:
- `make codex-daily-triage`

## toml

The TOML keeps the same live entrypoint and preserves the weekday automation cadence as separate config so the rendered automation can reuse one authoritative workflow. This variant exposes distractor files but no separate release-context or rollback bundle to override the live Make target, so the must-read set stays focused on the current workflow sources.

Source paths:
- `.scenario_variant`
- `Makefile`
- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`

Command literals:
- `make codex-daily-triage`
- `0 9 * * 1-5`

## automation

The automation uses the exact script command and weekday cron literal quoted from current repo files, while the task prompt remains about triaging blockers for the on-call queue instead of embedding schedule semantics. The fixture file is evidence for local dry runs, not a replacement entrypoint.

Source paths:
- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `scripts/triage.py`
- `fixtures/open_incidents.json`

Command literals:
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`
- `0 9 * * 1-5`

## mapping_note

The mapping note explicitly rejects stale legacy sources because this variant is noisy through deprecated helper references, not through an alternate live workflow. The note therefore documents both the live Make entrypoint and the deprecated legacy command so reviewers can see why the latter was excluded.

Source paths:
- `.scenario_variant`
- `Makefile`
- `docs/ops/daily_triage.md`
- `docs/ops/migration_notes.md`
- `docs/ops/legacy_schedule.md`
- `skills/legacy/daily_triage/SKILL.md`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage_legacy.py --window today --emit-md reports/daily_triage.md`

## Rejected Candidates

- `docs/ops/legacy_schedule.md`: Rejected because the note says it pre-dates the current automation, still describes the deprecated helper and seven-day sweep, and explicitly says not to use it as the source of truth for the live workflow mapping.
- `skills/legacy/daily_triage/SKILL.md`: Rejected because it is a legacy draft kept only so reviewers can compare old versus new workflow wording, which makes it rollback evidence rather than the live Codex-native entrypoint.
