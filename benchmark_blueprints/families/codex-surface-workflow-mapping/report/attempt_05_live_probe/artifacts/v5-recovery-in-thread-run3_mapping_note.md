# Mapping Note

## skill

The skill keeps the operator-facing entrypoint on make codex-daily-triage because the live workflow doc names that target directly, the Makefile resolves it to the active triage script, and the release plus objective context narrow the task to blocker-first on-call triage for the stand-up.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`
- `scripts/triage.py`
- `release_context/release_notes_2026_04.md`
- `release_context/objective_shift.md`
- `incident_context/recovered_plan.md`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## toml

The TOML preserves the Make target as the canonical entrypoint and records the rollback-aware must-read set so future Codex runs load the release narrowing, migration warning, and recovery plan before touching automation wording.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`
- `docs/ops/migration_notes.md`
- `incident_context/recovered_plan.md`
- `incident_context/weekend_rollback.md`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## automation

The automation artifact splits the weekday schedule from the task body: the live cron stays 0 9 * * 1-5, while the task itself stays focused on blocker-first triage through the recovered Make route rather than embedding timing or broad queue-sweep semantics.

Source paths:
- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `release_context/objective_shift.md`
- `incident_context/recovered_plan.md`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## mapping_note

The mapping note calls out the live recovery path and the stale helper path together so reviewers can see why rollback-only and abandoned migration artifacts were excluded from the shipped Codex mapping.

Source paths:
- `docs/ops/migration_notes.md`
- `incident_context/weekend_rollback.md`
- `incident_context/recovered_plan.md`
- `drafts/codex_skill_patch.md`
- `docs/ops/legacy_schedule.md`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage_legacy.py --window today --emit-md reports/daily_triage.md`

## Rejected Candidates

- `drafts/codex_triage.toml`: This abandoned draft hard-codes the rollback-only helper and the seven-day schedule, while the current recovery plan restores weekday-only automation through make codex-daily-triage.
- `drafts/codex_skill_patch.md`: The patch was never shipped and only wrapped the deprecated helper; migration notes say the helper stayed around for diffing and rollback, not as the live entrypoint.
- `skills/legacy/daily_triage/SKILL.md`: The legacy skill is retained as historical wording, but the repo docs and rollback context put the current workflow back on the Make target and weekday schedule.
- `docs/ops/legacy_schedule.md`: The note explicitly says it pre-dates the current automation and must not be used as the source of truth because it preserves the deprecated helper and seven-day sweep.
