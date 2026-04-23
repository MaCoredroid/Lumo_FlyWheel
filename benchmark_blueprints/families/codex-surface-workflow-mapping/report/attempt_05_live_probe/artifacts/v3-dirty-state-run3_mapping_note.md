# Mapping Note

## skill

The skill maps to the documented live Codex-facing entrypoint make codex-daily-triage and keeps the underlying active-window triage command visible as implementation evidence. In the dirty-state variant, the fixture contains blocker incidents and weekend automation noise, so the scope stays on live blockers instead of broad legacy sweep language.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`
- `.scenario_variant`
- `fixtures/open_incidents.json`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## toml

The TOML keeps the same live Make entrypoint and workflow scope so the config does not drift away from the repo-owned command. The weekday cadence belongs in automation metadata, but the referenced files still show the active-window Python implementation and the pinned weekday schedule for operator context.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`
- `ops/automation/triage.cron`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## automation

The automation proposal keeps task semantics separate from schedule semantics by using the live Make entrypoint as the task command and the weekday cron literal as scheduling metadata. That matches the current repo automation instead of the stale daily seven-day sweep language.

Source paths:
- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `Makefile`
- `docs/ops/legacy_schedule.md`

Command literals:
- `make codex-daily-triage`
- `0 9 * * 1-5`

## mapping_note

The mapping note records why the legacy helper and abandoned drafts were rejected: migration notes say the helper stayed rollback-only, the legacy note says it is not the source of truth, and the draft artifacts explicitly describe themselves as abandoned or comparison-only. That rejection evidence belongs in the note rather than being reused as live workflow input.

Source paths:
- `docs/ops/migration_notes.md`
- `docs/ops/legacy_schedule.md`
- `drafts/codex_triage.toml`
- `drafts/codex_skill_patch.md`
- `skills/legacy/daily_triage/SKILL.md`

Command literals:
- `python3 scripts/triage_legacy.py --window today --emit-md reports/daily_triage.md`

## Rejected Candidates

- `drafts/codex_triage.toml`: The file is marked abandoned and do not ship, still points at the deprecated helper, and carries the stale daily schedule instead of the live weekday Make entrypoint.
- `drafts/codex_skill_patch.md`: This draft says it was a half-finished abandoned patch that wrapped the deprecated helper and was never shipped.
- `skills/legacy/daily_triage/SKILL.md`: The legacy skill explicitly says it predates the migration and exists only for old versus new workflow comparison, so it is evidence to reject rather than a live Codex artifact source.
- `docs/ops/legacy_schedule.md`: The note pre-dates the current automation, still documents the deprecated helper and old daily cadence, and explicitly says not to use it as the source of truth.
