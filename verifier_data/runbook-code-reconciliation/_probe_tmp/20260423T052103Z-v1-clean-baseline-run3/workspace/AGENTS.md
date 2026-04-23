# AGENTS

You are reconciling the daily release-preview runbook for `v1-clean-baseline` (Clean baseline).

Hard rules:
- Use bundle-local code and bundle-local CLI help as the source of truth.
- Edit only these files:
  - `docs/runbooks/release_preview.md`
  - `artifacts/verification_notes.md`
  - `artifacts/deploy_note.md`
  - `artifacts/reconciliation_facts.json`
- Do not modify code, tests, helper scripts, configs, README fragments, release_context, or incident_context.
- Run `pytest -q tests/test_release_preview_cli.py` before finishing.
- Record exact checked commands in `artifacts/verification_notes.md` and `artifacts/reconciliation_facts.json`.
- The helper alias still exists for compatibility, but it must not be the primary runbook path.
