# Oncall Runbook

Canonical structured handoff path:

`python scripts/run_handoff.py --input fixtures/handoff_input.json --output out.md`

Use `skills/oncall_handoff/SKILL.md` for operator guidance and `automations/handoff-primary.toml` for automation wiring.

If `release_context` or `incident_context` files are present in the workspace, read them as input context and leave them unchanged.
