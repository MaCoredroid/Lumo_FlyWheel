# Mapping Note

## skill

The skill stays on the live Codex-facing entrypoint because both the workspace instructions and the ops note identify `make codex-daily-triage` as the current workflow, and the underlying script scopes the run to active blockers from `fixtures/open_incidents.json` rather than the deprecated legacy sweep.

Source paths:
- `AGENTS.md`
- `Makefile`
- `docs/ops/daily_triage.md`
- `scripts/triage.py`
- `fixtures/open_incidents.json`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## toml

The TOML uses the same live `make codex-daily-triage` entrypoint so the config matches the documented Codex-facing workflow, while the must-read paths keep the operator anchored on the live Make target, the current ops note, and the weekday automation source instead of any stale draft.

Source paths:
- `AGENTS.md`
- `Makefile`
- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`

Command literals:
- `make codex-daily-triage`
- `0 9 * * 1-5`

## automation

The automation proposal keeps task semantics and schedule semantics separate by making the task call the live `make codex-daily-triage` entrypoint while citing the weekday cron literal independently. In this `v1-clean-baseline` bundle there is no release-context or rollback override file to change that mapping, so the baseline path stays on the current weekday triage flow.

Source paths:
- `AGENTS.md`
- `Makefile`
- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `scripts/triage.py`

Command literals:
- `make codex-daily-triage`
- `0 9 * * 1-5`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## mapping_note

The mapping note records the live entrypoint and the explicit stale rejections together, because the workspace docs say not to use the legacy schedule note as the source of truth and the legacy helper is preserved only for rollback diffing. That makes the clean-baseline mapping evidence-backed without inventing a replacement workflow.

Source paths:
- `AGENTS.md`
- `docs/ops/daily_triage.md`
- `docs/ops/legacy_schedule.md`
- `scripts/triage_legacy.py`
- `skills/legacy/daily_triage/SKILL.md`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage_legacy.py --window today --emit-md reports/daily_triage.md`

## Rejected Candidates

- `docs/ops/legacy_schedule.md`: Rejected because this note explicitly says it pre-dates the current automation and must not be used as the source of truth for the live workflow mapping.
- `skills/legacy/daily_triage/SKILL.md`: Rejected because the file labels itself as a legacy draft, and the repo keeps that helper only for comparing old versus new wording rather than for the live Codex-facing entrypoint.
- `Makefile#daily-triage-legacy`: Rejected because the Makefile exposes a separate `daily-triage-legacy` target while the live entrypoint is the distinct `codex-daily-triage` target called out by the current ops documentation.
