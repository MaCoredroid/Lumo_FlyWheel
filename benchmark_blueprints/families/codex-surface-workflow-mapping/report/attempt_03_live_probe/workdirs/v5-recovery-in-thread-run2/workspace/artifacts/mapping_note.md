# Mapping Note

## skill

The skill stays anchored on the recovered live entrypoint rather than the rollback helper because the workflow scope narrowed to blocker-first on-call triage and the approved route is the Make target.

Source paths:
- `Makefile`
- `docs/ops/daily_triage.md`
- `release_context/objective_shift.md`
- `incident_context/recovered_plan.md`

Command literals:
- `make codex-daily-triage`

## toml

The TOML mirrors the same Make entrypoint and points operators at the current migration and rollback notes so they do not resume from the deprecated helper or stale queue-sweep narrative.

Source paths:
- `Makefile`
- `docs/ops/migration_notes.md`
- `incident_context/recovered_plan.md`

Command literals:
- `make codex-daily-triage`

## automation

Automation remains a weekday cron that invokes the recovered Make target while keeping schedule semantics separate from task semantics; the underlying implementation still shells out to the active Python triage command.

Source paths:
- `Makefile`
- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `incident_context/recovered_plan.md`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## mapping_note

The mapping note records why the live route stays on the recovered Make target and why the rollback-only helper and abandoned Codex draft are explicitly rejected in this variant.

Source paths:
- `docs/ops/migration_notes.md`
- `docs/ops/legacy_schedule.md`
- `incident_context/weekend_rollback.md`
- `incident_context/recovered_plan.md`
- `drafts/codex_triage.toml`
- `drafts/codex_skill_patch.md`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage_legacy.py --window today --emit-md reports/daily_triage.md`

## Rejected Candidates

- `docs/ops/legacy_schedule.md`: This note explicitly pre-dates the current automation, documents the deprecated helper, and carries the superseded seven-day sweep instead of the recovered weekday blocker-first route.
- `drafts/codex_triage.toml`: This abandoned draft wraps the deprecated helper and even notes that the live migration should resume from the Make target instead of shipping the legacy command.
