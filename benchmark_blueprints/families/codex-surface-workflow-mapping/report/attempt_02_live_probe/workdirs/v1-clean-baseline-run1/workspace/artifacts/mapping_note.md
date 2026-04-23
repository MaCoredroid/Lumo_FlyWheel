# Mapping Note

## skill

The skill should point at the repo's live Codex-facing entrypoint because `docs/ops/daily_triage.md` says the live entrypoint is `make codex-daily-triage`, and the Makefile binds that target to the current `scripts/triage.py` command. The variant is `v1-clean-baseline`, so there is no variant-specific release or rollback override to add on top of the base blocker-triage workflow.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`
- `.scenario_variant`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## toml

The TOML should reuse the same live `make codex-daily-triage` entrypoint instead of inlining a different workflow, while listing the current docs, Makefile, cron file, fixtures, and variant marker as must-read context. That keeps the config aligned with the repo's current mapping and preserves the clean-baseline scope for this submission.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`
- `ops/automation/triage.cron`
- `.scenario_variant`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## automation

The automation artifact should keep task semantics separate from schedule semantics: the task stays on the live `make codex-daily-triage` workflow named in the docs, while the weekday cadence comes from `ops/automation/triage.cron` as `0 9 * * 1-5`. That avoids reviving the stale seven-day schedule and keeps the command surface consistent across artifacts.

Source paths:
- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `Makefile`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## mapping_note

The mapping note should explicitly document why the current workflow wins and why the preserved legacy helper and legacy schedule are rejected. For this `v1-clean-baseline` variant, the note should describe the base live mapping only and record the stale candidates as evidence-backed rejections rather than blending them into the active workflow.

Source paths:
- `docs/ops/daily_triage.md`
- `skills/legacy/daily_triage/SKILL.md`
- `docs/ops/legacy_schedule.md`
- `.scenario_variant`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage_legacy.py --window today --emit-md reports/daily_triage.md`

## Rejected Candidates

- `skills/legacy/daily_triage/SKILL.md`: This draft explicitly says it predates the current repo migration and still references the deprecated helper, so it is evidence of the old wording rather than the live Codex-native workflow.
- `docs/ops/legacy_schedule.md`: The note says it pre-dates the current automation and should not be used as the source of truth, and its daily cron literal conflicts with the live weekday cadence in `ops/automation/triage.cron`.
- `Makefile`: The `daily-triage-legacy` target is preserved for comparison, but the strong mapping target is the separate `codex-daily-triage` entrypoint wired to `scripts/triage.py`.
