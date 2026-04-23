# Mapping Note

## skill

The skill should name the live Codex-facing entrypoint as `make codex-daily-triage`, because the ops doc marks that command as the current entrypoint and the Make target resolves to the active `scripts/triage.py` invocation. The release context narrows the scope to blocker incidents first for on-call ownership clarity rather than a broad queue sweep.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`
- `release_context/release_notes_2026_04.md`
- `release_context/objective_shift.md`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## toml

The TOML should carry the same live entrypoint and workflow scope as the skill, while the must-read list points operators at the live workflow doc, Make target, automation cadence, and migration note that explains why the legacy helper stayed rollback-only.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`
- `ops/automation/triage.cron`
- `docs/ops/migration_notes.md`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## automation

The automation artifact should keep the task body focused on blocker-first on-call triage and stand-up output, while the weekday cron literal stays in the schedule section. The cited cron and workflow docs show the weekday cadence and the live underlying command without changing the Codex-facing entrypoint.

Source paths:
- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `release_context/release_notes_2026_04.md`
- `release_context/objective_shift.md`

Command literals:
- `make codex-daily-triage`
- `0 9 * * 1-5`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## mapping_note

The mapping note needs to document why the live mapping stays on the Make target and rejects legacy rollback material. The migration notes and legacy references all show that the old helper was retained only for rollback diffing and comparison, not as proof of the current workflow.

Source paths:
- `docs/ops/daily_triage.md`
- `docs/ops/migration_notes.md`
- `docs/ops/legacy_schedule.md`
- `drafts/codex_skill_patch.md`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage_legacy.py --window today --emit-md reports/daily_triage.md`

## Rejected Candidates

- `docs/ops/legacy_schedule.md`: Rejected because this note predates the current automation, preserves the old seven-day sweep framing, and explicitly says not to use it as the source of truth for the live workflow.
- `skills/legacy/daily_triage/SKILL.md`: Rejected because the legacy skill draft predates the repo migration and is kept only for wording comparison, not for the current Codex-native entrypoint.
- `drafts/codex_triage.toml`: Rejected because the draft is labeled abandoned, points at the deprecated helper and old daily cron, and tells the reader to resume from the live Make target instead.
- `drafts/codex_skill_patch.md`: Rejected because the patch was never shipped and explicitly says the deprecated helper is not proof of the live workflow.
