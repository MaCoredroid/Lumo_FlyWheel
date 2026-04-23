# Mapping Note

## skill

The skill stays anchored on `make codex-daily-triage` because the daily triage doc names it as the live Codex-facing entrypoint, and the release context narrows the scope to blocker-first on-call triage instead of a broad queue sweep.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`
- `release_context/objective_shift.md`
- `release_context/release_notes_2026_04.md`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## toml

The TOML mirrors the same live Make entrypoint and records the live workflow scope, while must-read files point reviewers to the migration note and release-context objective shift instead of any legacy helper draft.

Source paths:
- `docs/ops/daily_triage.md`
- `docs/ops/migration_notes.md`
- `Makefile`
- `release_context/objective_shift.md`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## automation

Automation uses the same live Make entrypoint but keeps scheduling separate: the weekday cron literal comes from the pinned automation files, while the task body stays focused on blockers-first triage and stand-up output.

Source paths:
- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `release_context/objective_shift.md`
- `release_context/release_notes_2026_04.md`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## mapping_note

The mapping note documents why this v4 release-context variant changes workflow scope around objective drift but does not switch entrypoints: migration notes keep the legacy helper rollback-only while the live Make target remains canonical.

Source paths:
- `docs/ops/daily_triage.md`
- `docs/ops/migration_notes.md`
- `release_context/objective_shift.md`
- `Makefile`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## Rejected Candidates

- `drafts/codex_triage.toml`: This abandoned draft points at the deprecated helper and legacy daily cadence, while the draft itself says to resume from the live Make target instead.
- `docs/ops/legacy_schedule.md`: The legacy schedule note explicitly says not to use it as the source of truth and describes the older queue-sweep behavior that the release context deprioritized.
- `skills/legacy/daily_triage/SKILL.md`: The legacy skill is retained only for old-vs-new wording comparison and still cites the rollback-only helper instead of the live Codex entrypoint.
- `drafts/codex_skill_patch.md`: This patch was never shipped and says it wraps the deprecated helper, so it is evidence of an abandoned migration attempt, not the live workflow.
