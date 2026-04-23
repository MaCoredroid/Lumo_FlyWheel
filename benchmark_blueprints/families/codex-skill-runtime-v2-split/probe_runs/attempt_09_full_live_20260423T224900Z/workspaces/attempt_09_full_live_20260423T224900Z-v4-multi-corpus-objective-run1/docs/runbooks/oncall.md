# Oncall Runbook

Use `config/runtime.toml` as the mutable source of truth.
Run the live handoff through `automations/handoff-primary.toml`.
Canonical command: `python scripts/run_handoff.py --input fixtures/handoff_input.json --output out.md`
