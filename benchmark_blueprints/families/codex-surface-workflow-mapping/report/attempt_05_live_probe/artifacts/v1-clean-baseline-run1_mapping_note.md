# Mapping Note

## skill

The skill artifact should use the live Codex-facing entrypoint because docs/ops/daily_triage.md states that the live entrypoint is make codex-daily-triage. In the v1-clean-baseline variant there is no variant-specific dirty-state, release, or rollback evidence that overrides the baseline active-incident workflow.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`
- `scripts/triage.py`
- `.scenario_variant`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## toml

The TOML configuration should point at the same live Make entrypoint as the skill while preserving the active script command as the underlying executable evidence. This keeps Codex configuration aligned with the repo-supported target rather than bypassing it with a stale helper.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## automation

The automation artifact should separate task semantics from schedule semantics: the task is active incident triage, while the schedule literal is the weekday 09:00 cadence from the cron evidence. The cron command uses the active triage script and active window.

Source paths:
- `ops/automation/triage.cron`
- `docs/ops/daily_triage.md`

Command literals:
- `0 9 * * 1-5`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## mapping_note

The mapping note should document why the live path was selected and why stale candidates were rejected. The active command is named in the daily triage docs and Makefile, while the legacy command and seven-day cadence are explicitly marked as pre-current or rollback-only evidence.

Source paths:
- `docs/ops/daily_triage.md`
- `docs/ops/legacy_schedule.md`
- `skills/legacy/daily_triage/SKILL.md`
- `scripts/triage_legacy.py`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage_legacy.py --window today --emit-md reports/daily_triage.md`

## Rejected Candidates

- `docs/ops/legacy_schedule.md`: This note says it pre-dates the current automation, uses a seven-day cadence, and should not be used as the source of truth for the live workflow.
- `skills/legacy/daily_triage/SKILL.md`: The legacy skill draft predates the current repo migration and is kept only so reviewers can compare old versus new workflow wording.
- `scripts/triage_legacy.py`: The script itself marks the deprecated invocation as preserved for rollback diffing only, not as the live Codex triage workflow.
