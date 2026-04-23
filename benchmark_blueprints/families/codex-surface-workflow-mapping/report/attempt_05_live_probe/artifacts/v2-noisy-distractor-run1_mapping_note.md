# Mapping Note

## skill

The skill should expose the live Codex-facing entrypoint exactly as the repo documents it and keep the workflow focused on active blocker triage for the on-call rotation. The noisy-distractor variant does not change the entrypoint; it only reinforces that warning-only routing drift stays out of the mapped task.

Source paths:
- `docs/ops/daily_triage.md`
- `scripts/triage.py`
- `fixtures/open_incidents.json`
- `.scenario_variant`

Command literals:
- `make codex-daily-triage`

## toml

The TOML should keep the same live entrypoint as the skill and make the operator read the Make target, the weekday cron source, and the blocker-filtering script before reusing any draft wording. That preserves one workflow definition across config and execution artifacts.

Source paths:
- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `scripts/triage.py`
- `Makefile`

Command literals:
- `make codex-daily-triage`

## automation

The automation artifact should preserve the live task separately from the live weekday schedule. The command stays on the Codex-facing Make target, while the cron literal comes from the current automation file instead of the deprecated all-days schedule.

Source paths:
- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`

Command literals:
- `make codex-daily-triage`
- `0 9 * * 1-5`

## mapping_note

The mapping note needs to explain why the migration kept stale helpers in the repo without promoting them to live entrypoints. It should also record that the noisy-distractor variant still maps to the active blocker workflow, because the live script filters blocker incidents and the legacy helper remains rollback-only.

Source paths:
- `docs/ops/daily_triage.md`
- `docs/ops/migration_notes.md`
- `docs/ops/legacy_schedule.md`
- `scripts/triage.py`
- `scripts/triage_legacy.py`
- `skills/legacy/daily_triage/SKILL.md`
- `fixtures/open_incidents.json`
- `.scenario_variant`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage_legacy.py --window today --emit-md reports/daily_triage.md`

## Rejected Candidates

- `docs/ops/legacy_schedule.md`: This note says it pre-dates the current automation, keeps the deprecated helper and old all-days cron view, and explicitly says not to use it as the source of truth for the live workflow mapping.
- `skills/legacy/daily_triage/SKILL.md`: The legacy skill is labeled as a draft that predates the current migration, while the migration notes say the helper remained rollback-only and the live workflow stayed on the Make target.
- `scripts/triage_legacy.py`: The file itself marks the invocation as deprecated and preserved for rollback diffing only, so it cannot anchor the Codex-native mapping even though it is still present in the workspace.
