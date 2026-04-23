# Mapping Note

## skill

The skill uses the live Codex-facing entrypoint from the current ops doc and keeps its scope aligned with the blocker-focused triage implemented by the active Make target and triage script.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`
- `scripts/triage.py`
- `fixtures/open_incidents.json`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## toml

The TOML keeps the same live entrypoint and must-read set so downstream Codex runs inspect the active workflow files first; in the clean-baseline variant there is no extra release, rollback, or dirty-state evidence to include.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`
- `ops/automation/triage.cron`
- `fixtures/open_incidents.json`

Command literals:
- `make codex-daily-triage`
- `0 9 * * 1-5`

## automation

The automation proposal preserves the live weekday cron cadence from the automation file while keeping the task body tied to the Codex-facing Make entrypoint instead of inlining schedule semantics into the task prompt.

Source paths:
- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `Makefile`

Command literals:
- `make codex-daily-triage`
- `0 9 * * 1-5`

## mapping_note

The note documents why the mapping follows the current daily triage sources and rejects the deprecated legacy helper, legacy draft skill, and legacy daily sweep schedule that the workspace marks as non-authoritative.

Source paths:
- `docs/ops/daily_triage.md`
- `docs/ops/legacy_schedule.md`
- `scripts/triage_legacy.py`
- `skills/legacy/daily_triage/SKILL.md`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage_legacy.py --window today --emit-md reports/daily_triage.md`
- `0 9 * * *`

## Rejected Candidates

- `docs/ops/legacy_schedule.md`: Rejected because the note explicitly says it pre-dates the current automation and should not be used as the source of truth for the live workflow mapping.
- `skills/legacy/daily_triage/SKILL.md`: Rejected because the legacy skill draft says it predates the current repo migration and exists only for old-vs-new wording comparison, not for the live Codex artifact mapping.
- `docs/ops/legacy_schedule.md`: Rejected because the live cadence is the weekday-only cron in `ops/automation/triage.cron`, while this legacy note still carries the stale seven-day sweep schedule.
