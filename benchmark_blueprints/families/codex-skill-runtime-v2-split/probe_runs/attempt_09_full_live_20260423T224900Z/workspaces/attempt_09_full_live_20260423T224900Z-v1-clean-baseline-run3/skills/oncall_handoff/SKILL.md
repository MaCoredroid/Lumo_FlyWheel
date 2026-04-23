# Oncall Handoff

Use the structured oncall handoff runtime.

Canonical automation: `automations/handoff-primary.toml`
Mutable runtime config: `config/runtime.toml`
Runnable handoff command: `python scripts/run_handoff.py --input fixtures/handoff_input.json --output out.md`

Inputs:
- Read incident and release context when present.
- Use `fixtures/handoff_input.json` as the default local handoff payload.

Outputs:
- Write the rendered handoff markdown to the requested output path.

Notes:
- `automations/handoff-copy.toml` is retired and kept only as a disabled compatibility stub.
- Do not use the legacy monolith or copied prompt as the active runtime path.
