# Mapping Note

## skill

The skill maps to the Codex-facing entrypoint `make codex-daily-triage` because the live workflow doc names that command as the entrypoint, the Makefile defines the active implementation behind it, and the release plus incident recovery notes narrow the scope to blocker-first on-call triage instead of a broad queue sweep.

Source paths:
- `Makefile`
- `docs/ops/daily_triage.md`
- `release_context/release_notes_2026_04.md`
- `release_context/objective_shift.md`
- `incident_context/recovered_plan.md`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## toml

The TOML keeps the same live entrypoint as the skill and elevates the release and rollback-recovery documents into must-read evidence so the config preserves the blocker-first weekday workflow instead of drifting back to the older seven-day helper narrative.

Source paths:
- `Makefile`
- `docs/ops/daily_triage.md`
- `release_context/objective_shift.md`
- `incident_context/recovered_plan.md`
- `incident_context/weekend_rollback.md`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## automation

The automation proposal uses a cron schedule because the live cadence is explicitly pinned to weekday mornings, while the task body stays separate and continues to invoke the same Make-based entrypoint instead of embedding schedule semantics or the deprecated helper directly.

Source paths:
- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `incident_context/recovered_plan.md`
- `Makefile`

Command literals:
- `make codex-daily-triage`
- `0 9 * * 1-5`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## mapping_note

The mapping note explicitly records why the legacy helper and abandoned Codex drafts were rejected: every stale source still points at `python3 scripts/triage_legacy.py --window today --emit-md reports/daily_triage.md`, while the rollback context says that helper is rollback-only and the recovered plan reinstates `make codex-daily-triage` as the only approved automation route.

Source paths:
- `docs/ops/legacy_schedule.md`
- `drafts/codex_triage.toml`
- `drafts/codex_skill_patch.md`
- `incident_context/weekend_rollback.md`
- `incident_context/recovered_plan.md`

Command literals:
- `python3 scripts/triage_legacy.py --window today --emit-md reports/daily_triage.md`
- `make codex-daily-triage`

## Rejected Candidates

- `drafts/codex_triage.toml`: The draft is explicitly marked abandoned and still couples the deprecated helper with the old seven-day automation shape instead of the recovered blocker-first weekday flow.
- `drafts/codex_skill_patch.md`: This patch never shipped and only wraps the deprecated helper, which the rollback notes preserve for diffing and rollback evidence rather than live automation reuse.
- `skills/legacy/daily_triage/SKILL.md`: The legacy skill is retained for comparison only and points at the deprecated helper, so it cannot define the current Codex-native mapping for the recovery variant.
