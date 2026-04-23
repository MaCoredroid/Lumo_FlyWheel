# Oncall Handoff

Use this structured handoff path for the live runtime bundle.

## Canonical command

`python scripts/run_handoff.py --input fixtures/handoff_input.json --output out.md`

## Runtime contract

- Use `automations/handoff-primary.toml` as the only canonical automation.
- Keep `config/runtime.toml` aligned to this skill and command path.
- Follow `release_context/q3_bundle_reuse.md` when preparing escalation follow-ups.
- If incident context is present in the bundle, apply it during handoff generation without editing the context file.

## Output expectation

Generate a concise handoff document from the provided input payload and write it
to the requested output path.
