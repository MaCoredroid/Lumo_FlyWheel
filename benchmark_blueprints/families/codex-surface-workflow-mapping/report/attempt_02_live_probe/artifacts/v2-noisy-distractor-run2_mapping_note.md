# Mapping Note

## skill

The skill should expose the Codex-facing live entrypoint, not the rollback helper. In this noisy-distractor variant, the workflow map stays on `make codex-daily-triage`, and the supporting evidence shows that the underlying script reads `fixtures/open_incidents.json` and summarizes blocker incidents for on-call triage.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`
- `scripts/triage.py`
- `fixtures/open_incidents.json`
- `.scenario_variant`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## toml

The TOML config needs the same live entrypoint as the skill so the generated config does not drift onto the deprecated helper. The must-read list keeps the live Make target, live ops note, cadence file, migration warning, and scenario variant together.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`
- `ops/automation/triage.cron`
- `docs/ops/migration_notes.md`
- `.scenario_variant`

Command literals:
- `make codex-daily-triage`
- `0 9 * * 1-5`

## automation

The automation proposal keeps task semantics separate from schedule semantics by pairing the live entrypoint with the weekday cron literal instead of embedding the schedule into the task text. That matters here because the bundle contains stale seven-day references that should be treated as distractors.

Source paths:
- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `docs/ops/legacy_schedule.md`
- `.scenario_variant`

Command literals:
- `make codex-daily-triage`
- `0 9 * * 1-5`
- `0 9 * * *`

## mapping_note

The mapping note should explicitly document why the deprecated surfaces were rejected: the migration notes call the helper rollback-only, the legacy skill is labeled as a legacy draft, and the legacy schedule note explicitly says not to use it as the source of truth. That rejection is part of the correct mapping for this noisy-distractor variant.

Source paths:
- `docs/ops/migration_notes.md`
- `docs/ops/legacy_schedule.md`
- `skills/legacy/daily_triage/SKILL.md`
- `scripts/triage_legacy.py`
- `.scenario_variant`

Command literals:
- `python3 scripts/triage_legacy.py --window today --emit-md reports/daily_triage.md`
- `0 9 * * *`

## Rejected Candidates

- `skills/legacy/daily_triage/SKILL.md`: This file is explicitly a legacy draft that predates the current repo migration, so it should not override the live Codex-facing entrypoint.
- `docs/ops/legacy_schedule.md`: This note says it pre-dates the current automation and explicitly warns not to use it as the source of truth when mapping the live workflow.
- `scripts/triage_legacy.py`: The script comments mark the invocation as deprecated and preserved for rollback diffing only, which makes it a distractor rather than the live entrypoint.
