# Mapping Note

## skill

The skill should point at `make codex-daily-triage` because the repo names that as the live Codex-facing entrypoint, while migration notes say the legacy helper stayed rollback-only during the codex-native transition.

Source paths:
- `docs/ops/daily_triage.md`
- `docs/ops/migration_notes.md`
- `Makefile`

Command literals:
- `make codex-daily-triage`

## toml

The TOML should preserve the same Make entrypoint so Codex reads the workflow through the live wrapper instead of binding directly to an abandoned script-level draft.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`

Command literals:
- `make codex-daily-triage`

## automation

The automation mapping should keep the weekday cron literal `0 9 * * 1-5` separate from the task itself. That weekday scope matches the live cron file and avoids the legacy seven-day sweep, which is especially important in this dirty-state bundle because `fixtures/open_incidents.json` includes blocker `INC-412` for weekend automation page noise.

Source paths:
- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `fixtures/open_incidents.json`

Command literals:
- `make codex-daily-triage`
- `0 9 * * 1-5`

## mapping_note

The mapping note should explicitly document that dirty-state evidence does not change the live entrypoint away from the Make target; it only strengthens the rejection of rollback-only and abandoned legacy paths.

Source paths:
- `docs/ops/daily_triage.md`
- `docs/ops/migration_notes.md`
- `docs/ops/legacy_schedule.md`
- `drafts/codex_triage.toml`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage_legacy.py --window today --emit-md reports/daily_triage.md`

## Rejected Candidates

- `docs/ops/legacy_schedule.md`: Rejected because this note explicitly pre-dates the current automation, points to the deprecated helper, and carries the stale seven-day sweep rather than the live weekday cadence.
- `skills/legacy/daily_triage/SKILL.md`: Rejected because the file labels itself as a legacy draft kept only for comparing old versus new wording, so it is evidence of a stale mapping candidate rather than the live Codex workflow.
- `drafts/codex_skill_patch.md`: Rejected because the draft says it was a half-finished abandoned patch that never shipped and wrapped the deprecated helper instead of the live Make entrypoint.
- `drafts/codex_triage.toml`: Rejected because the draft TOML is marked do-not-ship, uses the legacy helper and stale daily cron, and instructs the reader to resume from the live Make target instead.
