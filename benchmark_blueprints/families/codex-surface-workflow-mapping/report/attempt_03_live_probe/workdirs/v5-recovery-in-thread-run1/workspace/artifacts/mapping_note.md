# Mapping Note

## skill

The skill points at `make codex-daily-triage` because the live Codex-facing entrypoint is the recovered Make target, while release context narrows the workflow to blocker incidents, on-call ownership clarity, and stand-up-ready output.

Source paths:
- `docs/ops/daily_triage.md`
- `release_context/objective_shift.md`
- `release_context/release_notes_2026_04.md`
- `incident_context/recovered_plan.md`

Command literals:
- `make codex-daily-triage`

## toml

The TOML keeps the same live entrypoint so the generated config does not drift back to the paused migration draft or the rollback-only helper path.

Source paths:
- `docs/ops/daily_triage.md`
- `docs/ops/migration_notes.md`
- `incident_context/recovered_plan.md`

Command literals:
- `make codex-daily-triage`

## automation

The automation artifact separates the weekday cron cadence from the task body: the schedule remains `0 9 * * 1-5`, while the task itself stays anchored on the live Make target that recovery explicitly approved.

Source paths:
- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `incident_context/recovered_plan.md`
- `incident_context/weekend_rollback.md`

Command literals:
- `make codex-daily-triage`
- `0 9 * * 1-5`

## mapping_note

The note records why stale candidates are rejected in this recovery variant: migration notes keep the legacy helper only for diffing, and the incident rollback documents why the old weekend helper path cannot be reused as the canonical Codex workflow.

Source paths:
- `docs/ops/daily_triage.md`
- `docs/ops/migration_notes.md`
- `incident_context/weekend_rollback.md`
- `docs/ops/legacy_schedule.md`
- `drafts/codex_triage.toml`
- `drafts/codex_skill_patch.md`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage_legacy.py --window today --emit-md reports/daily_triage.md`

## Rejected Candidates

- `drafts/codex_triage.toml`: This abandoned draft hard-codes the deprecated helper and the seven-day cron shape, which conflicts with the recovered weekday-only Make-target workflow.
- `skills/legacy/daily_triage/SKILL.md`: The legacy skill is explicitly marked as predating the migration and still points at the rollback-diff helper instead of the live Codex-facing Make entrypoint.
- `docs/ops/legacy_schedule.md`: This note is not source of truth for the recovery variant because it preserves the pre-migration seven-day sweep and deprecated helper that were deprioritized after noisy pages.
