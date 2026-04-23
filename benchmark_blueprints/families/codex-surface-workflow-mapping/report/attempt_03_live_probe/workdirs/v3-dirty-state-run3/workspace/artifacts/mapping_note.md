# Mapping Note

## skill

The skill should start from `make codex-daily-triage` because the live Codex-facing entrypoint is named explicitly in the daily triage doc, while the dirty-state variant still contains rollback-only and abandoned files that need to be treated as evidence instead of reusable implementation.

Source paths:
- `docs/ops/daily_triage.md`
- `docs/ops/migration_notes.md`
- `.scenario_variant`

Command literals:
- `make codex-daily-triage`

## toml

The TOML should keep the same Make entrypoint and point readers at the live workflow files plus the cron evidence, because the repo migration note says the helper remained rollback-only while the Make target stayed live.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`
- `docs/ops/migration_notes.md`
- `ops/automation/triage.cron`

Command literals:
- `make codex-daily-triage`

## automation

The automation artifact should preserve schedule semantics separately from task semantics: use the weekday cron literal from `ops/automation/triage.cron`, but run the same Codex-facing Make entrypoint that the live workflow doc names. The dirty-state variant increases the need to avoid inheriting the stale daily-all-week schedule from legacy notes.

Source paths:
- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `docs/ops/legacy_schedule.md`

Command literals:
- `make codex-daily-triage`
- `0 9 * * 1-5`

## mapping_note

The mapping note must explicitly reject stale candidates because this dirty-state workspace keeps both legacy and abandoned Codex artifacts nearby. The note should show that `python3 scripts/triage_legacy.py --window today --emit-md reports/daily_triage.md` is preserved only for rollback diffs and that the mapping stays anchored on the live Make target instead.

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

- `docs/ops/legacy_schedule.md`: Rejected because the file labels itself as a legacy note, carries the deprecated helper and a seven-day schedule, and says not to use it as the source of truth for the live workflow.
- `skills/legacy/daily_triage/SKILL.md`: Rejected because the skill is explicitly marked as a legacy draft kept only for comparing old vs. new workflow wording during migration.
- `drafts/codex_skill_patch.md`: Rejected because it is an abandoned Codex patch that was never shipped and explicitly says it should not be treated as proof of the live workflow.
- `drafts/codex_triage.toml`: Rejected because the abandoned TOML draft hard-codes the deprecated helper and the stale `0 9 * * *` schedule instead of the live Make target and weekday cron.
