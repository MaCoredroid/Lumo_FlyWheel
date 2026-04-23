# Oncall Handoff

Use this skill as the canonical structured handoff path for the runtime bundle.

## Runtime entrypoints

- Mutable runtime config: `config/runtime.toml`
- Canonical automation: `automations/handoff-primary.toml`
- Runnable command: `python scripts/run_handoff.py --input fixtures/handoff_input.json --output out.md`

## Handoff flow

1. Read the handoff payload from the configured JSON input.
2. Render the structured markdown handoff with `scripts/run_handoff.py`.
3. Write the output markdown to the configured output path.
4. If a release or incident context file is present in the bundle, follow that context without editing the context file itself.
