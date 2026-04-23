# Oncall Handoff

Use the canonical structured handoff runtime only.

- Live automation: `automations/handoff-primary.toml`
- Live command: `python scripts/run_handoff.py --input fixtures/handoff_input.json --output out.md`
- Live runbook: `docs/runbooks/oncall.md`

Follow-up escalation handoffs reuse the same command path and contract.
Do not route live handoffs through the legacy monolith, the copied prompt, or the
retired duplicate automation alias.
