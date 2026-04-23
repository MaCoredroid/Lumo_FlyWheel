# Oncall Runbook

Use `skills/oncall_handoff/SKILL.md`.

Canonical command:

`python scripts/run_handoff.py --input fixtures/handoff_input.json --output out.md`

Canonical automation:

`automations/handoff-primary.toml`

If release or incident context is present in the bundle, follow it during handoff preparation without editing the context files themselves.
