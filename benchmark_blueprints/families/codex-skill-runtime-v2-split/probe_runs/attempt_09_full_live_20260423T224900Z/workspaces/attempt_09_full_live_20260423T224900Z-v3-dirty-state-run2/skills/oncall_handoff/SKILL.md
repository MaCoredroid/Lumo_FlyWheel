# Oncall Handoff

Use the structured runtime path only.

- Mutable runtime config: `config/runtime.toml`
- Canonical automation: `automations/handoff-primary.toml`
- Runnable handoff command: `python scripts/run_handoff.py --input fixtures/handoff_input.json --output out.md`

Do not route live runs through retired aliases.
