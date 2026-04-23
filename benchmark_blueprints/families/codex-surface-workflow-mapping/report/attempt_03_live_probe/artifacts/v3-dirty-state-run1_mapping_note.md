# Mapping Note

## skill

The skill should expose the live Codex-facing entrypoint as `make codex-daily-triage`, because the workflow doc names that exact command and the migration note says the helper stayed rollback-only. For the dirty-state variant, stale migration leftovers are evidence to reject, not alternative entrypoints to promote.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`
- `docs/ops/migration_notes.md`
- `.scenario_variant`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## toml

The TOML should stay on the same live Make entrypoint and cite the weekday cron separately, so the config matches the repo's current automation instead of the abandoned legacy-draft path. The must-read list stays anchored on live workflow files plus the sample incident surface.

Source paths:
- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `docs/ops/migration_notes.md`
- `fixtures/open_incidents.json`
- `.scenario_variant`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## automation

The automation proposal should separate the work itself from when it runs: the task is incident triage and blocker summarization, while the schedule remains the weekday cron literal. The command stays aligned with the same live Make entrypoint used by the skill and TOML.

Source paths:
- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `fixtures/open_incidents.json`
- `.scenario_variant`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## mapping_note

The mapping note should document both the live mapping and the stale rollback-only alternatives so reviewers can see why the dirty-state variant still resolves to the Make target. The abandoned draft and legacy helper remain explicitly rejected because the migration note keeps them in comparison-only status.

Source paths:
- `docs/ops/daily_triage.md`
- `docs/ops/migration_notes.md`
- `docs/ops/legacy_schedule.md`
- `drafts/codex_skill_patch.md`
- `.scenario_variant`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage_legacy.py --window today --emit-md reports/daily_triage.md`

## Rejected Candidates

- `docs/ops/legacy_schedule.md`: This note explicitly says it pre-dates the current automation, still carries the deprecated helper and old daily cron, and should not be used as the live source of truth.
- `skills/legacy/daily_triage/SKILL.md`: The file is labeled a legacy draft and says it is kept only to compare old versus new wording, so it is not the runnable Codex skill for the current workflow.
- `drafts/codex_triage.toml`: The draft is marked abandoned and do not ship, and it hard-codes the deprecated helper instead of the live Make-target workflow.
- `drafts/codex_skill_patch.md`: This half-finished patch never shipped and only wraps the deprecated helper, so it is evidence of abandoned migration work rather than the current workflow.
