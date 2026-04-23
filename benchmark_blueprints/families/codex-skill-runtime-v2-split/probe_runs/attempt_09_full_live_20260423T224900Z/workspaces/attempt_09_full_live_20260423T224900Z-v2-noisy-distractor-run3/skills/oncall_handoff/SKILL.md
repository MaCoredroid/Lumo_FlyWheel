# Oncall Handoff

Use the structured handoff flow for all live runs.

Canonical command:

`python scripts/run_handoff.py --input fixtures/handoff_input.json --output out.md`

Runtime alignment:

- Live config: `config/runtime.toml`
- Canonical automation: `automations/handoff-primary.toml`
- Live runbook: `docs/runbooks/oncall.md`

Execution rules:

- Produce the handoff through `scripts/run_handoff.py`; do not fall back to the legacy monolith note or the copied legacy prompt.
- If `release_context` or `incident_context` files are present, incorporate them as read-only context.
- Treat `automations/handoff-copy.toml` as a retired alias; use the primary automation instead.
