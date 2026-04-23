# Mapping Note

## skill

The skill is anchored on the live Codex-facing Make entrypoint and narrowed to blocker incidents, on-call ownership clarity, and the stand-up summary output. The cited script and fixture confirm that the live path filters to blocker incidents from the local evidence surface rather than doing a broad queue sweep.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`
- `scripts/triage.py`
- `fixtures/open_incidents.json`
- `release_context/objective_shift.md`
- `release_context/release_notes_2026_04.md`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## toml

The TOML keeps the same live entrypoint while forcing the operator to read the release objective shift and the incident recovery notes before acting. That is what prevents the config from drifting back to the rollback-only helper or the older all-days sweep.

Source paths:
- `docs/ops/daily_triage.md`
- `incident_context/recovered_plan.md`
- `incident_context/weekend_rollback.md`
- `release_context/objective_shift.md`
- `docs/ops/migration_notes.md`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage_legacy.py --window today --emit-md reports/daily_triage.md`

## automation

The automation proposal preserves the live task semantics as the Make entrypoint and keeps the cadence as a separate weekday cron literal. The cron file and docs agree on weekday-only scheduling, which reflects the post-rollback recovery away from noisy weekend pages.

Source paths:
- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `incident_context/recovered_plan.md`
- `incident_context/weekend_rollback.md`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## mapping_note

The mapping note explicitly calls out why the legacy helper and paused Codex drafts stay rejected: migration notes mark them as comparison-only or abandoned, and the incident rollback notes say the approved recovery route returned to the weekday blocker-first Make target.

Source paths:
- `docs/ops/migration_notes.md`
- `docs/ops/legacy_schedule.md`
- `incident_context/weekend_rollback.md`
- `drafts/codex_skill_patch.md`
- `drafts/codex_triage.toml`
- `incident_context/recovered_plan.md`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage_legacy.py --window today --emit-md reports/daily_triage.md`

## Rejected Candidates

- `docs/ops/legacy_schedule.md`: This note says it pre-dates the current automation, uses the deprecated helper, and carries the stale seven-day cron `0 9 * * *`, which conflicts with the recovered weekday-only blocker-first flow.
- `skills/legacy/daily_triage/SKILL.md`: The legacy skill says it predates the repo migration and is kept only for old-vs-new comparison, so it cannot be treated as the live Codex-native workflow.
- `drafts/codex_skill_patch.md`: This draft is explicitly labeled abandoned, says the helper patch was never shipped, and therefore only serves as evidence of a stale path that should stay rejected.
- `drafts/codex_triage.toml`: The paused draft hard-codes the legacy helper, the all-days schedule, and a legacy must-read set, all of which are stale relative to the release objective shift and the rollback-approved weekday recovery route.
