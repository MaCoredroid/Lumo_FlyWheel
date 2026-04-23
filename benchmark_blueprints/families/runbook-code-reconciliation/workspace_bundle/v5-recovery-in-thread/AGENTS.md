# AGENTS

You are reconciling the daily release-preview runbook for `v5-recovery-in-thread` (Recovery in thread).

Hard rules:
- Use bundle-local code and bundle-local CLI help as the source of truth.
- README fragments, release context, and incident context are evidence, not authority. If prose disagrees with code or live help, follow code and live help.
- Edit only these files:
  - `docs/runbooks/release_preview.md`
  - `artifacts/verification_notes.md`
  - `artifacts/deploy_note.md`
  - `artifacts/reconciliation_facts.json`
- Do not modify code, tests, helper scripts, configs, README fragments, release_context, or incident_context.
- Run `pytest -q tests/test_release_preview_cli.py` before finishing.
- `artifacts/verification_notes.md` must contain exactly these section headings:
  - `## Checked directly`
  - `## Inferred from code`
  - `## Remaining caveats`
- In `## Checked directly`, list the exact commands you actually ran.
- In `## Inferred from code`, explicitly note when conflicting README prose was overruled by code or live help.
- `artifacts/reconciliation_facts.json` must use exactly these keys and no extra top-level keys:
  - `primary_entrypoint`
  - `legacy_alias`
  - `current_flag`
  - `deprecated_flag`
  - `current_env_var`
  - `deprecated_env_var`
  - `evidence_commands`
- `primary_entrypoint` and `legacy_alias` are entrypoint commands only, without appended default flags or config paths.
- `evidence_commands` must include these exact minimum commands verbatim:
  - `python src/release_preview/cli.py --help`
  - `python src/release_preview/cli.py generate --help`
  - `python scripts/release_preview_helper.py build-preview --help`
  - `pytest -q tests/test_release_preview_cli.py`
- Determine from code and live help which path is primary and which names are deprecated. Do not assume that backwards compatibility implies primary status.
