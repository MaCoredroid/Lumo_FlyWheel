# Mapping Note

## skill

The skill uses make codex-daily-triage because docs/ops/daily_triage.md names it as the live Codex-facing entrypoint, Makefile defines the target, and scripts/triage.py shows the active workflow inspects fixtures/open_incidents.json for blocker triage in this clean baseline variant.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`
- `scripts/triage.py`
- `fixtures/open_incidents.json`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## toml

The TOML keeps the same make codex-daily-triage entrypoint so the config surface matches the live workflow, and the v1-clean-baseline variant does not introduce dirty-state, release-context, or rollback overrides that would justify switching away from the active path.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`
- `.scenario_variant`

Command literals:
- `make codex-daily-triage`

## automation

Automation keeps task semantics separate from schedule semantics by using the weekday cron literal from ops/automation/triage.cron while pointing the task itself at make codex-daily-triage, which daily_triage.md identifies as the live entrypoint and Makefile resolves to the active triage.py command.

Source paths:
- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `Makefile`
- `scripts/triage.py`

Command literals:
- `0 9 * * 1-5`
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## mapping_note

The mapping note records the live command chain and the evidence-backed stale rejections so reviewers can see why the clean baseline variant stays on the active workflow instead of the rollback-only legacy helper or its outdated seven-day schedule note.

Source paths:
- `docs/ops/daily_triage.md`
- `docs/ops/legacy_schedule.md`
- `skills/legacy/daily_triage/SKILL.md`
- `scripts/triage_legacy.py`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage_legacy.py --window today --emit-md reports/daily_triage.md`
- `0 9 * * *`

## Rejected Candidates

- `skills/legacy/daily_triage/SKILL.md`: Rejected because the file labels itself as a legacy draft that predates the current repo migration and keeps the triage_legacy.py helper only for comparing old versus new wording.
- `docs/ops/legacy_schedule.md`: Rejected because the note explicitly says it pre-dates the current automation, pairs the old seven-day cadence with the deprecated triage_legacy.py helper, and says not to use it as the source of truth for the live workflow map.
- `Makefile::daily-triage-legacy`: Rejected because Makefile retains the daily-triage-legacy target only alongside the live codex-daily-triage target, while docs/ops/daily_triage.md names the codex target as the live Codex-facing entrypoint.
