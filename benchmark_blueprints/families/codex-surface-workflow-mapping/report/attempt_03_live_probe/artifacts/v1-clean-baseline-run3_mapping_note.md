# Mapping Note

## skill

The skill anchors on the live Codex-facing entrypoint because `docs/ops/daily_triage.md` names `make codex-daily-triage` as the workflow entrypoint, and `Makefile` resolves that target to the active `scripts/triage.py` invocation that triages blocker incidents.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`
- `scripts/triage.py`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## toml

The TOML keeps the same live entrypoint as the skill and records the clean-baseline variant so the generated config reads the current workflow files first instead of inheriting rollback-only helpers or stale draft wording.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`
- `.scenario_variant`

Command literals:
- `make codex-daily-triage`

## automation

The automation maps the weekday scheduler literal from `ops/automation/triage.cron` onto the same live entrypoint while leaving the task prompt focused on triage behavior rather than embedding schedule semantics in the task body.

Source paths:
- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`

Command literals:
- `make codex-daily-triage`
- `0 9 * * 1-5`

## mapping_note

The note explicitly rejects stale candidates because the clean-baseline variant should not pivot to rollback-only files: the legacy skill says it predates the current migration, the legacy schedule says it is not the source of truth, and `scripts/triage_legacy.py` is preserved for rollback diffs only.

Source paths:
- `skills/legacy/daily_triage/SKILL.md`
- `docs/ops/legacy_schedule.md`
- `scripts/triage_legacy.py`
- `.scenario_variant`

Command literals:
- `python3 scripts/triage_legacy.py --window today --emit-md reports/daily_triage.md`
- `0 9 * * *`

## Rejected Candidates

- `skills/legacy/daily_triage/SKILL.md`: Rejected because this draft says it predates the current repo migration and keeps the legacy helper only so reviewers can compare old vs. new workflow wording.
- `docs/ops/legacy_schedule.md`: Rejected because the note says it pre-dates the current automation and must not be used as the source of truth when mapping the live workflow.
- `Makefile:daily-triage-legacy`: Rejected because the legacy target points at the deprecated helper, while the live Codex-facing workflow is documented as `make codex-daily-triage`.
