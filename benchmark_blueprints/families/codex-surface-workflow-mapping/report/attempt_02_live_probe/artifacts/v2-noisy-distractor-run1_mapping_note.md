# Mapping Note

## skill

The skill points to `make codex-daily-triage` because the Makefile defines that target and the live operations doc names it as the Codex-facing entrypoint. The workflow scope is copied from the current workflow note instead of the legacy draft.

Source paths:
- `Makefile`
- `docs/ops/daily_triage.md`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## toml

The TOML keeps the same live entrypoint as the skill and lists the current workflow files as must-read inputs so the generated config stays anchored on the live path rather than the deprecated helper. The migration note is included in must-read because this variant carries stale distractors that need explicit rejection.

Source paths:
- `Makefile`
- `docs/ops/daily_triage.md`
- `docs/ops/migration_notes.md`

Command literals:
- `make codex-daily-triage`

## automation

The automation artifact keeps task semantics on the live Make target while sourcing cadence separately from the cron file. That matches the current workflow doc, which explicitly tells us to keep schedule semantics out of the task body. For this noisy-distractor variant, the automation text also says to ignore rollback-only helpers and stale schedule notes.

Source paths:
- `Makefile`
- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `docs/ops/legacy_schedule.md`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## mapping_note

The mapping note calls out the noisy-distractor condition explicitly: this bundle contains a rollback-only helper, a legacy schedule note, and a legacy draft skill. Those are documented as stale in the workspace itself, so the note records the rejection and keeps every rendered artifact aligned to the live Make target.

Source paths:
- `docs/ops/migration_notes.md`
- `docs/ops/legacy_schedule.md`
- `skills/legacy/daily_triage/SKILL.md`
- `scripts/triage_legacy.py`

Command literals:
- `python3 scripts/triage_legacy.py --window today --emit-md reports/daily_triage.md`

## Rejected Candidates

- `Makefile#daily-triage-legacy`: This target is not the live workflow. The migration note says the legacy helper was kept only for diffing outputs and remained rollback-only while the live workflow stayed on the Make target.
- `docs/ops/legacy_schedule.md`: This note explicitly says it pre-dates the current automation and must not be used as the source of truth. It also carries the stale seven-day cadence instead of the current weekday cron.
- `skills/legacy/daily_triage/SKILL.md`: The file is labeled a legacy draft and says it is kept only for comparing old versus new workflow wording, so it should not drive the Codex-native mapping.
