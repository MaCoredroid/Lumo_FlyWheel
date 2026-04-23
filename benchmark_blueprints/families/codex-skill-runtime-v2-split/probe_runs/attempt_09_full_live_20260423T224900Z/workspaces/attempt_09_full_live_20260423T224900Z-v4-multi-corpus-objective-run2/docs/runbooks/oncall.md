# Oncall Runbook

Use the canonical structured handoff path:
- `skills/oncall_handoff/SKILL.md`
- `automations/handoff-primary.toml`
- `config/runtime.toml`
- `python scripts/run_handoff.py --input fixtures/handoff_input.json --output out.md`

Follow-up escalation handoffs should reuse the same canonical contract rather
than cloning a second skill path.
