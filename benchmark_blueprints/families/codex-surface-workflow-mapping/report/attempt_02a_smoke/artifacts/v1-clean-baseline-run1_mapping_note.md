# Mapping Note

## skill

Use the Codex-facing entrypoint rather than the deprecated helper. The live workflow doc names `make codex-daily-triage` as the entrypoint, while the Makefile grounds the underlying command. Because `.scenario_variant` is `v1-clean-baseline`, keep the baseline workflow without release or rollback add-ons.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`
- `.scenario_variant`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## toml

Mirror the same live entrypoint in config and require the current workflow files before reuse. The baseline variant does not introduce dirty-state or rollback-specific overrides, so the TOML stays aligned to the normal daily triage path.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`
- `.scenario_variant`
- `ops/automation/triage.cron`

Command literals:
- `make codex-daily-triage`
- `0 9 * * 1-5`

## automation

Keep task semantics separate from schedule semantics. Use the live Codex entrypoint as the task command and preserve the weekday cron literal from the active automation source, not the legacy seven-day sweep.

Source paths:
- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `Makefile`

Command literals:
- `make codex-daily-triage`
- `0 9 * * 1-5`

## mapping_note

Document both the live entrypoint and the concrete shell-out so stale drafts can be rejected with direct evidence. The baseline variant keeps the mapping focused on the active incident queue flow backed by the current fixture surface.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`
- `fixtures/open_incidents.json`
- `.scenario_variant`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## Rejected Candidates

- `docs/ops/legacy_schedule.md`: Legacy note explicitly says it pre-dates the current automation, uses the deprecated helper, and shows the obsolete daily cron instead of the active weekday schedule.
- `skills/legacy/daily_triage/SKILL.md`: Legacy skill draft says it predates the current repo migration and is kept only for old-vs-new wording comparison, so it is not the live Codex entrypoint.
- `Makefile`: The `daily-triage-legacy` target is present for backward compatibility, but the active Codex-facing target is `codex-daily-triage` and the benchmark should not anchor on the legacy target.
- `scripts/triage_legacy.py`: The script header marks this invocation as deprecated and kept for rollback diffing only; it writes a legacy stub instead of performing live blocker triage.
