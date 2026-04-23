# Mapping Note

## skill

Use the Codex-facing Make target as the skill entrypoint because the live workflow doc names `make codex-daily-triage` directly, while the Makefile shows the underlying script invocation and `scripts/triage.py` confirms the blocker-focused behavior over `fixtures/open_incidents.json`.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`
- `scripts/triage.py`
- `fixtures/open_incidents.json`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## toml

Keep the TOML aligned to the same Make entrypoint so config and skill do not diverge. Include the current weekday automation evidence in the must-read set because the live schedule is pinned separately from the task definition.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`
- `ops/automation/triage.cron`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## automation

Map automation as a cron schedule with the weekday literal from `ops/automation/triage.cron`, but keep the Codex task body pointed at `make codex-daily-triage` because the docs explicitly say to separate task semantics from schedule semantics.

Source paths:
- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `Makefile`

Command literals:
- `make codex-daily-triage`
- `0 9 * * 1-5`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## mapping_note

The v3 dirty-state variant does not introduce a new entrypoint; it raises the bar for rejecting rollback-only and abandoned migration material. The note should therefore preserve the live Make target while documenting why legacy helper references remain evidence only.

Source paths:
- `.scenario_variant`
- `docs/ops/daily_triage.md`
- `docs/ops/legacy_schedule.md`
- `docs/ops/migration_notes.md`
- `drafts/codex_skill_patch.md`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage_legacy.py --window today --emit-md reports/daily_triage.md`

## Rejected Candidates

- `scripts/triage_legacy.py`: Rejected because the script labels the invocation deprecated and rollback-only, and the migration note says the helper was kept only for diffing outputs rather than as the live entrypoint.
- `docs/ops/legacy_schedule.md`: Rejected because the note explicitly says it pre-dates the current automation and must not be used as the source of truth for mapping the live workflow.
- `skills/legacy/daily_triage/SKILL.md`: Rejected because the legacy skill is a draft that predates the repo migration and exists only for old-vs-new wording comparison.
- `drafts/codex_skill_patch.md`: Rejected because the draft is marked abandoned and says the legacy helper patch was never shipped.
- `drafts/codex_triage.toml`: Rejected because the TOML is explicitly labeled abandoned, points at the deprecated helper, and says the migration should resume from the live Make target instead.
