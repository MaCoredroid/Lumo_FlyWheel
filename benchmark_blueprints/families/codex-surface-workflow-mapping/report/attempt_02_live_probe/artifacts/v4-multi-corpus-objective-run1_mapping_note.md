# Mapping Note

## skill

The skill anchors on `make codex-daily-triage` because the repo calls it the live Codex-facing entrypoint, while the release context narrows the workflow to blocker incidents, on-call ownership clarity, and a stand-up summary rather than a broad queue sweep.

Source paths:
- `docs/ops/daily_triage.md`
- `release_context/objective_shift.md`
- `release_context/release_notes_2026_04.md`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## toml

The TOML keeps the same live entrypoint and records the release-context files as must-read inputs so Codex reads the narrowed blockers-first scope before running. The direct Python command is cited as the Make target expansion, not as a replacement entrypoint.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`
- `release_context/objective_shift.md`
- `release_context/release_notes_2026_04.md`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## automation

The automation mapping keeps task and schedule separate: the scheduler owns the weekday `0 9 * * 1-5` cadence from the cron file, while the automation task body points to the same live Make entrypoint and blockers-first objective from the docs and release notes.

Source paths:
- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `release_context/release_notes_2026_04.md`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## mapping_note

The note explicitly rejects stale artifacts because this variant includes release-context narrowing but no `incident_context/` rollback evidence. That means the rollback-only helper and abandoned Codex draft remain evidence to reject, not live workflow anchors.

Source paths:
- `docs/ops/migration_notes.md`
- `docs/ops/legacy_schedule.md`
- `drafts/codex_skill_patch.md`
- `skills/legacy/daily_triage/SKILL.md`
- `docs/ops/daily_triage.md`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage_legacy.py --window today --emit-md reports/daily_triage.md`

## Rejected Candidates

- `docs/ops/legacy_schedule.md`: Rejected because the note says it pre-dates the current automation, still describes the deprecated helper and old daily cron, and should not be used as the source of truth.
- `scripts/triage_legacy.py`: Rejected because the script itself says the invocation is deprecated and preserved for rollback diffing only, while migration notes say the live workflow stayed on the Make target.
- `skills/legacy/daily_triage/SKILL.md`: Rejected because the skill is labeled a legacy draft that predates the current repo migration and exists only for comparing old and new wording.
- `drafts/codex_skill_patch.md`: Rejected because the draft says it was an abandoned half-finished patch around the deprecated helper and was never shipped.
