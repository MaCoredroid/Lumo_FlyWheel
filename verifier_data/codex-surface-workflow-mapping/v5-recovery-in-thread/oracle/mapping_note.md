# Mapping Note

## skill

The skill uses the live Make target and cites the direct script only as backing evidence.

Source paths:
- `Makefile`
- `docs/ops/daily_triage.md`
- `scripts/triage.py`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## toml

The TOML config keeps the same live entrypoint and points the agent at the repo surfaces it must read.

Source paths:
- `Makefile`
- `docs/ops/daily_triage.md`
- `fixtures/open_incidents.json`

Command literals:
- `make codex-daily-triage`

## automation

The automation proposal keeps task text separate from the weekday schedule literal grounded in the cron file.

Source paths:
- `ops/automation/triage.cron`
- `docs/ops/daily_triage.md`

Command literals:
- `make codex-daily-triage`
- `0 9 * * 1-5`

## mapping_note

The mapping note ties each artifact back to the exact repo files and exact literals that justify the decision.

Source paths:
- `Makefile`
- `docs/ops/daily_triage.md`
- `fixtures/open_incidents.json`
- `incident_context/recovered_plan.md`
- `incident_context/weekend_rollback.md`
- `ops/automation/triage.cron`
- `release_context/objective_shift.md`
- `release_context/release_notes_2026_04.md`
- `scripts/triage.py`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`
- `0 9 * * 1-5`

## Rejected Candidates

- `skills/legacy/daily_triage/SKILL.md`: Rejected because this path is stale relative to the live workflow mapping.
- `drafts/codex_skill_patch.md`: Rejected because this path is abandoned relative to the live workflow mapping.
- `incident_context/weekend_rollback.md`: Rejected because this path is rollback relative to the live workflow mapping.
