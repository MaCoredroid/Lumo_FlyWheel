# Oncall Runbook

Use the canonical structured handoff path:

- Skill bundle: `skills/oncall_handoff/SKILL.md`
- Automation: `automations/handoff-primary.toml`
- Direct command: `python scripts/run_handoff.py --input fixtures/handoff_input.json --output out.md`

If release or incident context files are present in the bundle, follow them as
read-only context and do not edit them during the handoff run.
