# Mapping Note

## skill

The skill should point at the documented live Codex-facing entrypoint rather than any rollback helper. The daily triage doc names `make codex-daily-triage` as the live entrypoint, and the Make target resolves to the active triage script with the active window.

Source paths:
- `Makefile`
- `docs/ops/daily_triage.md`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## toml

The TOML should preserve the same live entrypoint and keep the operator focused on the current workflow scope. The same live Make entrypoint is the right configuration anchor because the workflow doc presents it as the Codex-facing command while the Makefile shows the exact script behind it.

Source paths:
- `Makefile`
- `docs/ops/daily_triage.md`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## automation

The automation proposal should separate task semantics from schedule semantics. The workflow doc says to keep them separate, the cron file pins the weekday cadence, and the active script plus fixture show that this variant still maps to the blocker-focused live workflow even when warning-level noise is present.

Source paths:
- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `scripts/triage.py`
- `fixtures/open_incidents.json`

Command literals:
- `make codex-daily-triage`
- `0 9 * * 1-5`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## mapping_note

The mapping note should explain why legacy candidates remain rejected. Migration notes call the legacy helper rollback-only, the legacy schedule note says it pre-dates the current automation and must not be used as source of truth, and the legacy skill is preserved only for wording comparison. In the noisy-distractor variant, that matters because warning-level or deprecated signals must not override the live blocker-oriented mapping.

Source paths:
- `docs/ops/migration_notes.md`
- `docs/ops/legacy_schedule.md`
- `docs/ops/daily_triage.md`
- `skills/legacy/daily_triage/SKILL.md`
- `scripts/triage.py`
- `fixtures/open_incidents.json`

Command literals:
- `python3 scripts/triage_legacy.py --window today --emit-md reports/daily_triage.md`
- `0 9 * * *`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## Rejected Candidates

- `docs/ops/legacy_schedule.md`: Rejected because this note explicitly pre-dates the current automation, carries the deprecated helper, and says not to use it as the source of truth when mapping the live workflow.
- `skills/legacy/daily_triage/SKILL.md`: Rejected because the legacy skill is kept only for comparison wording, while migration notes say the helper remained rollback-only and did not become the live entrypoint.
- `Makefile#daily-triage-legacy`: Rejected because the Makefile still exposes the legacy target for rollback diffing, but the live workflow docs and the current cron pin the active mapping to the codex-daily-triage path instead.
