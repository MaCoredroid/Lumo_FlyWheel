# Mapping Note

## skill

The skill should anchor on the live Codex-facing entrypoint `make codex-daily-triage` because `docs/ops/daily_triage.md` names it as the current workflow and the Make target resolves to the active triage helper rather than the deprecated rollback-only helper.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`
- `scripts/triage.py`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## toml

The TOML should preserve the same live entrypoint and read-set so Codex is directed to the authoritative workflow note, the Make target, the pinned weekday cron, and the local incident fixture used for dry runs. In this clean-baseline variant there is no dirty-state, release, or rollback corpus that would justify changing the mapping away from the live baseline.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`
- `ops/automation/triage.cron`
- `fixtures/open_incidents.json`

Command literals:
- `make codex-daily-triage`
- `0 9 * * 1-5`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## automation

The automation mapping should keep schedule semantics separate from task semantics: the cron cadence comes from `ops/automation/triage.cron`, while the task prompt should describe running the live triage workflow and summarizing blockers. The command literal itself is the active `scripts/triage.py` invocation quoted in both the docs and the cron file.

Source paths:
- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `scripts/triage.py`

Command literals:
- `0 9 * * 1-5`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## mapping_note

The mapping note should explicitly document why the live workflow wins and why stale sources are rejected: the legacy note says not to use it as source of truth, the legacy helper is preserved only for rollback diffing, and the legacy skill is a draft that predates the current repo migration.

Source paths:
- `docs/ops/legacy_schedule.md`
- `scripts/triage_legacy.py`
- `skills/legacy/daily_triage/SKILL.md`
- `docs/ops/daily_triage.md`

Command literals:
- `python3 scripts/triage_legacy.py --window today --emit-md reports/daily_triage.md`
- `make codex-daily-triage`

## Rejected Candidates

- `docs/ops/legacy_schedule.md`: Rejected because the note explicitly says it pre-dates the current automation and must not be used as the source of truth; it also carries the stale daily cron literal `0 9 * * *` instead of the pinned weekday schedule.
- `scripts/triage_legacy.py`: Rejected because the file labels this invocation as deprecated and preserved for rollback diffing only, and the implementation only emits a rollback stub rather than live blocker triage.
- `skills/legacy/daily_triage/SKILL.md`: Rejected because the skill is explicitly a legacy draft that predates the repo migration and still points at the deprecated helper instead of the live Make entrypoint.
