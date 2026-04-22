# AGENTS

You are repairing the nightly regression digest for `v5-recovery-in-thread`.

Hard rules:
- Edit only these files:
  - `ops_digest/src/schema.py`
  - `ops_digest/src/digest_builder.py`
  - `ops_digest/automation/nightly_regression_watch.toml`
  - `ops_digest/docs/escalation_runbook.md`
  - `ops_digest/fixtures/inbox/generated_digest.md`
- Do not modify tests, fixtures under `ops_digest/fixtures/runs/`, local operator notes, release_context, or incident_context.
- Preserve the existing automation identity instead of creating a sibling file.
- Preserve unrelated local note edits in `ops_digest/fixtures/inbox/local_operator_notes.md`.
- Run `pytest -q ops_digest/tests` before finishing.

Focus:
- Parse `final_verdict.pass` and `milestones.results`.
- Treat missing required milestones as blocking even when the final verdict passes.
- Keep advisory warnings non-blocking.
- Pick the latest completed run for each `report_date`.
- Regenerate `ops_digest/fixtures/inbox/generated_digest.md`.

Variant note:
- incident_context/ documents a prior bad fix that added a sibling automation and changed delivery wording. Repair only the existing watch.
