# Mapping Note

## skill

Use the Codex-facing Make entrypoint instead of the deprecated helper because the live workflow doc names `make codex-daily-triage` as the entrypoint while the Python command behind it stays an implementation detail. In the dirty-state variant, keep the scope centered on active-window blocker triage so rollback-only artifacts do not redefine the task.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`
- `docs/ops/migration_notes.md`
- `fixtures/open_incidents.json`
- `.scenario_variant`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## toml

Keep the TOML entrypoint aligned with the same live Make target and point must-read context at the workflow doc, Makefile, migration note, weekday cron, and incident fixture so operators can see both the public entrypoint and the underlying live implementation.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`
- `docs/ops/migration_notes.md`
- `ops/automation/triage.cron`
- `fixtures/open_incidents.json`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`
- `0 9 * * 1-5`

## automation

Model automation as a weekday cron because the repo pins cadence in `ops/automation/triage.cron`, but keep the task body focused on triage semantics instead of embedding schedule text. The command remains the same live Make entrypoint so the automation surface stays consistent with the skill and TOML artifacts.

Source paths:
- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `Makefile`
- `fixtures/open_incidents.json`

Command literals:
- `make codex-daily-triage`
- `0 9 * * 1-5`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## mapping_note

Use the note artifact to document why the live mapping excludes rollback-only and abandoned Codex surfaces: the legacy note warns it is not the source of truth, the migration note says the helper remained rollback-only, the legacy skill is a comparison draft, and the abandoned patch was never shipped. That rejection matters more in the dirty-state variant because the active blocker queue must stay anchored to the live workflow rather than stale rollback references.

Source paths:
- `docs/ops/daily_triage.md`
- `docs/ops/legacy_schedule.md`
- `docs/ops/migration_notes.md`
- `skills/legacy/daily_triage/SKILL.md`
- `drafts/codex_skill_patch.md`
- `.scenario_variant`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage_legacy.py --window today --emit-md reports/daily_triage.md`

## Rejected Candidates

- `docs/ops/legacy_schedule.md`: This note explicitly says it pre-dates the current automation, still carries the deprecated helper and legacy `0 9 * * *` cadence, and should not be used as the source of truth for the live mapping.
- `skills/legacy/daily_triage/SKILL.md`: The legacy skill is kept only for wording comparison, still points at the deprecated helper, and the migration note says that helper remained rollback-only rather than becoming the live entrypoint.
- `drafts/codex_skill_patch.md`: The draft is marked as an abandoned patch, says it was never shipped, and therefore cannot be treated as evidence of the current Codex workflow.
