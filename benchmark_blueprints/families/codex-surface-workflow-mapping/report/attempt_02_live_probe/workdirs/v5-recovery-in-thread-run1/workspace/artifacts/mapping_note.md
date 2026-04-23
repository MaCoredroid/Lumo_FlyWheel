# Mapping Note

## skill

The skill should start from `make codex-daily-triage` because the live workflow doc and the recovered post-rollback plan both name that Make target as the approved Codex-facing route, while the release context narrows the job to blocker-first on-call triage for the stand-up summary.

Source paths:
- `docs/ops/daily_triage.md`
- `incident_context/recovered_plan.md`
- `release_context/objective_shift.md`
- `release_context/release_notes_2026_04.md`
- `scripts/triage.py`

Command literals:
- `make codex-daily-triage`

## toml

The TOML stays on the same recovered Make entrypoint and carries the rollback context in its must-read set so the configuration preserves the weekday blocker-first recovery scope instead of drifting back to the deprecated helper shape.

Source paths:
- `docs/ops/daily_triage.md`
- `incident_context/recovered_plan.md`
- `incident_context/weekend_rollback.md`
- `release_context/objective_shift.md`

Command literals:
- `make codex-daily-triage`

## automation

The automation artifact should keep task semantics separate from schedule semantics by using the same Make entrypoint for execution while grounding the weekday cadence independently in the cron evidence; that preserves the recovery away from weekend helper-based pages.

Source paths:
- `docs/ops/daily_triage.md`
- `incident_context/recovered_plan.md`
- `ops/automation/triage.cron`

Command literals:
- `make codex-daily-triage`
- `0 9 * * 1-5`

## mapping_note

The mapping note needs to show both the recovered live route and the stale helper route so reviewers can see why the weekday Make target wins and why legacy notes, legacy skills, and abandoned drafts remain evidence to reject rather than reuse.

Source paths:
- `incident_context/recovered_plan.md`
- `incident_context/weekend_rollback.md`
- `docs/ops/legacy_schedule.md`
- `drafts/codex_skill_patch.md`
- `skills/legacy/daily_triage/SKILL.md`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage_legacy.py --window today --emit-md reports/daily_triage.md`

## Rejected Candidates

- `docs/ops/legacy_schedule.md`: Rejected because this note explicitly says it pre-dates the current automation, still carries the seven-day sweep shape, and says not to use it as the source of truth when mapping the live workflow.
- `skills/legacy/daily_triage/SKILL.md`: Rejected because it is a legacy draft kept for old-versus-new wording comparison and still points at the deprecated helper instead of the recovered Make target.
- `drafts/codex_skill_patch.md`: Rejected because it is an abandoned Codex patch around the deprecated helper and explicitly says it was never shipped, so it cannot define the live mapping.
