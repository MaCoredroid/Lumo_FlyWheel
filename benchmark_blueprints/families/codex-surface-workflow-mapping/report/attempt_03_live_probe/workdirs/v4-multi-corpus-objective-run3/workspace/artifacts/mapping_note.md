# Mapping Note

## skill

The skill should expose the repo's Codex-facing entrypoint as `make codex-daily-triage`, then describe the narrowed blocker-first, on-call-owned, stand-up-ready workflow scope from the current release context rather than the older broad queue sweep.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`
- `release_context/objective_shift.md`
- `release_context/release_notes_2026_04.md`
- `scripts/triage.py`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## toml

The TOML should keep the same live Make entrypoint and workflow scope, while the must-read list points operators at the live workflow note, current release context, and migration note so they do not swap in the rollback-only helper during codex-native runs.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`
- `release_context/objective_shift.md`
- `release_context/release_notes_2026_04.md`
- `docs/ops/migration_notes.md`

Command literals:
- `make codex-daily-triage`

## automation

The automation artifact should keep schedule semantics in the cron metadata and task semantics in the prompt. The live cadence is the weekday 09:00 cron, but the executable command still maps to the same Codex entrypoint rather than the deprecated direct legacy helper.

Source paths:
- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `Makefile`
- `release_context/objective_shift.md`
- `release_context/release_notes_2026_04.md`
- `scripts/triage.py`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## mapping_note

The mapping note should explain why the legacy helper and abandoned draft artifacts are rejected: they were retained only for rollback diffing and still encode the pre-shift daily sweep instead of the current blocker-first workflow driven by the live Make target.

Source paths:
- `docs/ops/daily_triage.md`
- `docs/ops/legacy_schedule.md`
- `docs/ops/migration_notes.md`
- `drafts/codex_skill_patch.md`
- `drafts/codex_triage.toml`
- `skills/legacy/daily_triage/SKILL.md`
- `scripts/triage_legacy.py`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage_legacy.py --window today --emit-md reports/daily_triage.md`

## Rejected Candidates

- `drafts/codex_triage.toml`: This file is explicitly an abandoned draft, points at the deprecated legacy helper, and carries the stale daily cron instead of the live Make-based workflow.
- `skills/legacy/daily_triage/SKILL.md`: The legacy skill predates the current repo migration and is preserved for wording comparison only, so it should not be promoted over the live Codex-facing Make target.
- `docs/ops/legacy_schedule.md`: This note is marked as pre-current automation, still describes the deprecated seven-day sweep, and conflicts with the blocker-first weekday cadence captured in the current release context and cron file.
