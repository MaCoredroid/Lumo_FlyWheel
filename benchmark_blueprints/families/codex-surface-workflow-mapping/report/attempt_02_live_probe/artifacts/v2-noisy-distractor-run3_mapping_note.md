# Mapping Note

## skill

The skill maps to the live Codex-facing entrypoint named in docs and backed by the Make target, while keeping the workflow scope anchored on blocker-focused on-call triage from the local incident fixture instead of on stale helper text.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`
- `fixtures/open_incidents.json`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## toml

The TOML keeps the same live entrypoint as the skill and forces readers through the migration note so rollback-only helpers are not mistaken for the active workflow during reuse.

Source paths:
- `docs/ops/daily_triage.md`
- `Makefile`
- `docs/ops/migration_notes.md`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## automation

The automation proposal keeps task semantics separate from schedule semantics: the task points at the live Make entrypoint, while the schedule stays the weekday cron literal already pinned in automation. That weekday boundary matters here because the fixture includes weekend automation page noise and the noisy-distractor variant should not regress to a seven-day sweep.

Source paths:
- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `fixtures/open_incidents.json`
- `.scenario_variant`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## mapping_note

This variant includes noisy legacy distractors, so the note records the evidence chain that keeps the rendered artifacts on the live Make target and rejects rollback-only or pre-migration drafts as non-authoritative mapping inputs.

Source paths:
- `docs/ops/daily_triage.md`
- `docs/ops/legacy_schedule.md`
- `docs/ops/migration_notes.md`
- `scripts/triage_legacy.py`
- `skills/legacy/daily_triage/SKILL.md`
- `.scenario_variant`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage_legacy.py --window today --emit-md reports/daily_triage.md`

## Rejected Candidates

- `Makefile`: The file also defines `daily-triage-legacy`, but the live target in the same file is `codex-daily-triage`; the legacy target exists only as a fallback path and is not the Codex-facing entrypoint.
- `docs/ops/legacy_schedule.md`: This note explicitly says it pre-dates the current automation, carries the stale seven-day sweep, and says not to use it as the source of truth for the live workflow mapping.
- `scripts/triage_legacy.py`: The script labels its own invocation as deprecated and rollback-diff only, and the migration note confirms the helper remained rollback-only rather than becoming the live entrypoint.
- `skills/legacy/daily_triage/SKILL.md`: This is a legacy draft retained for wording comparison. In the noisy-distractor variant it should be treated as non-authoritative legacy context, not as the skill source for the live mapping.
