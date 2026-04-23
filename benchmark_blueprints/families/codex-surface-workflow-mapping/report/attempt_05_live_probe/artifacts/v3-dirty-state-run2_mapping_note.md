# Mapping Note

## skill

The skill points at the live Codex-facing Make target because the current ops doc names `make codex-daily-triage` as the entrypoint. For the dirty-state variant, the skill also tells the operator to inspect abandoned draft artifacts before reusing any prior Codex wording.

Source paths:
- `docs/ops/daily_triage.md`
- `docs/ops/migration_notes.md`
- `drafts/codex_skill_patch.md`
- `drafts/codex_triage.toml`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage_legacy.py --window today --emit-md reports/daily_triage.md`

## toml

The TOML keeps the same live entrypoint while preserving the current workflow scope from the repo evidence. Its must-read list prioritizes the live Make target and migration notes over stale legacy schedule material.

Source paths:
- `Makefile`
- `docs/ops/daily_triage.md`
- `docs/ops/migration_notes.md`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## automation

The automation keeps task semantics separate from schedule semantics: the task body stays on live blocker triage, while the weekday cadence comes from the cron file. The mapped command still points at the same live Make entrypoint instead of the rollback-only helper.

Source paths:
- `docs/ops/daily_triage.md`
- `ops/automation/triage.cron`
- `docs/ops/legacy_schedule.md`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`

## mapping_note

The mapping note explicitly records why dirty in-tree state should not be mistaken for the live workflow: legacy notes, legacy skill text, and abandoned migration drafts remain as comparison evidence only. That variant-specific warning is part of the mapping because stale artifacts are present alongside the live path.

Source paths:
- `docs/ops/daily_triage.md`
- `docs/ops/legacy_schedule.md`
- `docs/ops/migration_notes.md`
- `drafts/codex_triage.toml`
- `drafts/codex_skill_patch.md`
- `skills/legacy/daily_triage/SKILL.md`

Command literals:
- `make codex-daily-triage`
- `python3 scripts/triage_legacy.py --window today --emit-md reports/daily_triage.md`

## Rejected Candidates

- `drafts/codex_triage.toml`: This abandoned draft hard-codes the deprecated helper and obsolete daily schedule, and its own comment says to resume from the live Make target instead.
- `drafts/codex_skill_patch.md`: This half-finished patch wraps the deprecated helper and is explicitly kept as abandoned evidence, which is unsafe to reuse in the dirty-state variant.
- `skills/legacy/daily_triage/SKILL.md`: The legacy skill predates the current migration and is kept only for wording comparison, not as the live Codex workflow.
- `docs/ops/legacy_schedule.md`: The legacy schedule note is explicitly not the source of truth: it uses the deprecated helper and a seven-day cadence instead of the live weekday automation.
