# Oncall Handoff

Use the structured oncall handoff runtime for the primary escalation path.

Canonical runtime:
- Skill: `skills/oncall_handoff/SKILL.md`
- Automation: `automations/handoff-primary.toml`
- Mutable runtime config: `config/runtime.toml`
- Command: `python scripts/run_handoff.py --input fixtures/handoff_input.json --output out.md`

Operational notes:
- Reuse the canonical oncall handoff contract for follow-up escalation handoffs.
- Do not route the live bundle through the legacy monolith artifact.
- Do not clone the legacy prompt into a second skill.
