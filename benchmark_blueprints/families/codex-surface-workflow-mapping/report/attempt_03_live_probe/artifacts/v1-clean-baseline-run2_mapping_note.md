# Mapping Note

## skill

The skill should expose the repo's Codex-facing entrypoint as `make codex-daily-triage`, because the live operations doc names that command directly and the Make target resolves to the active `scripts/triage.py` invocation over the blocker-focused incident set.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`
- `scripts/triage.py`
- `fixtures/open_incidents.json`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## toml

The TOML should preserve the same live entrypoint and workflow scope as the skill so the config and generated artifacts stay aligned on one command instead of drifting to a helper or raw schedule file. The clean-baseline variant does not introduce a release or rollback override, so the baseline entrypoint remains authoritative.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`
- `.scenario_variant`

Command literals:
- `make codex-daily-triage`

## automation

The automation should still point at the same live entrypoint, while the weekday cadence stays in dedicated schedule fields. The repo's live doc explicitly says to keep task semantics separate from schedule semantics, and the cron file pins the current weekday run window.

Source paths:
- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`

Command literals:
- `make codex-daily-triage`
- `0 9 * * 1-5`

## mapping_note

The mapping note should explain why the legacy helper and legacy wording are rejected: the legacy schedule note says it pre-dates the current automation and should not be the source of truth, the legacy skill is a draft kept for comparison, and the legacy script is preserved only for rollback diffing. In this clean-baseline bundle, that evidence means the note should document rejection rather than switching the workflow.

Source paths:
- `docs/ops/legacy_schedule.md`
- `skills/legacy/daily_triage/SKILL.md`
- `scripts/triage_legacy.py`
- `.scenario_variant`

Command literals:
- `python3 scripts/triage_legacy.py --window today --emit-md reports/daily_triage.md`

## Rejected Candidates

- `docs/ops/legacy_schedule.md`: Rejected because the note says it pre-dates the current automation and explicitly says not to use it as the source of truth when mapping the live workflow.
- `skills/legacy/daily_triage/SKILL.md`: Rejected because it is labeled a legacy draft that predates the current repo migration and is kept only for comparing old versus new wording.
- `scripts/triage_legacy.py`: Rejected because the file marks the invocation as deprecated and says the helper is kept for rollback diffs only, so it is not the live Codex entrypoint.
