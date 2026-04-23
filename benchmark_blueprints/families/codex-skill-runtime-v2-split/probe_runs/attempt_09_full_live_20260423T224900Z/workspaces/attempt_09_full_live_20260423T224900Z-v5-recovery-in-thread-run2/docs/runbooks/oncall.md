# Oncall Runbook

Use `skills/oncall_handoff/SKILL.md` with `automations/handoff-primary.toml`.

Canonical handoff command:
`python scripts/run_handoff.py --input fixtures/handoff_input.json --output out.md`

Follow-up escalation handoffs reuse the same canonical contract and command path.
Do not restore `automations/handoff-copy.toml` as a live alias; it remains retired
after the duplicate-page rollback incident.
