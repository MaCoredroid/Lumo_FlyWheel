# Mapping Note

## skill

The skill should expose the live Codex-facing entrypoint instead of the underlying helper directly, because `docs/ops/daily_triage.md` names `make codex-daily-triage` as the live entrypoint and the Make target resolves to the current `scripts/triage.py` invocation.

Source paths:
- `Makefile`
- `docs/ops/daily_triage.md`
- `scripts/triage.py`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## toml

The TOML config should keep the same live entrypoint as the skill and list the current workflow files to read before execution so the config stays anchored on the live Make target and weekday automation evidence.

Source paths:
- `Makefile`
- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## automation

The automation artifact should preserve the live task entrypoint while sourcing the cadence separately from the cron file, because the docs explicitly say to keep task semantics separate from schedule semantics and the weekday cron is pinned as `0 9 * * 1-5`.

Source paths:
- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## mapping_note

This noisy-distractor variant includes rollback-only legacy materials, so the mapping note must explicitly reject them with evidence: the old helper and seven-day schedule are documented as pre-current, deprecated, and not the source of truth for the live mapping.

Source paths:
- `docs/ops/legacy_schedule.md`
- `docs/ops/migration_notes.md`
- `skills/legacy/daily_triage/SKILL.md`
- `scripts/triage_legacy.py`

Command literals:
- `python3 scripts/triage_legacy.py --window today --emit-md reports/daily_triage.md`

## Rejected Candidates

- `docs/ops/legacy_schedule.md`: The note says it pre-dates the current automation, still mentions a seven-day sweep, and says not to use it as the source of truth when mapping the live workflow.
- `skills/legacy/daily_triage/SKILL.md`: The legacy skill is explicitly labeled a draft that predates the current repo migration and is kept only for comparing old versus new wording.
- `Makefile`: The `daily-triage-legacy` target exists, but `docs/ops/migration_notes.md` says the helper remained rollback-only while the live workflow stayed on the Make target for `codex-daily-triage`.
