# Mapping Note

## skill

The skill should anchor on the live Codex-facing entrypoint rather than the underlying Python helper because the repo documentation names the Make target as the canonical interface and the migration notes say the legacy helper stayed rollback-only. For the dirty-state variant, that means the skill must steer readers away from stale diffing surfaces that still exist in-tree.

Source paths:
- `Makefile`
- `docs/ops/daily_triage.md`
- `docs/ops/migration_notes.md`

Command literals:
- `make codex-daily-triage`

## toml

The TOML should expose the same live Make entrypoint as the skill and carry the weekday schedule literal separately so task semantics and cadence do not collapse into one field. The live schedule is the weekday cron, not the deprecated seven-day sweep from the legacy note.

Source paths:
- `Makefile`
- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`

Command literals:
- `make codex-daily-triage`
- `0 9 * * 1-5`

## automation

The automation artifact should keep the concrete task command and the weekday cron literal exactly as the repo pins them today. In the dirty-state variant, the prompt should mention live blockers and on-call triage while avoiding rollback-only helpers that are still present only for migration diffing.

Source paths:
- `ops/automation/triage.cron`
- `docs/ops/daily_triage.md`
- `scripts/triage.py`

Command literals:
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`
- `0 9 * * 1-5`

## mapping_note

The mapping note should document both the accepted live entrypoint and the stale rollback-only alternative so reviewers can see why the live mapping stayed on the Make target. That distinction matters more in this dirty-state bundle because both commands are still visible in the repo, but only one is described as live.

Source paths:
- `Makefile`
- `docs/ops/daily_triage.md`
- `docs/ops/migration_notes.md`
- `scripts/triage_legacy.py`
- `docs/ops/legacy_schedule.md`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage_legacy.py --window today --emit-md reports/daily_triage.md`

## Rejected Candidates

- `skills/legacy/daily_triage/SKILL.md`: This legacy skill draft still points at the deprecated helper, while the repo says the live Codex-facing entrypoint is `make codex-daily-triage` and the helper remained rollback-only during migration.
- `drafts/codex_skill_patch.md`: The draft is explicitly abandoned and wraps the deprecated helper rather than the live Make target, so it is evidence of stale Codex work, not a reusable mapping source.
- `docs/ops/legacy_schedule.md`: This note predates the current automation, carries the wrong daily schedule, and explicitly says not to use it as the source of truth for the live workflow mapping.
