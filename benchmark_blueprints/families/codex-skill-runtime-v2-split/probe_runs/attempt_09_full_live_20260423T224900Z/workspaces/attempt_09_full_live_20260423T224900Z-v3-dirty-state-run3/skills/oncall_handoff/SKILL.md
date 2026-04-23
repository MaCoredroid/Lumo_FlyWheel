# Oncall Handoff

Use the canonical structured handoff flow for this bundle.

## Runtime Path

- Mutable runtime config: `config/runtime.toml`
- Canonical automation: `automations/handoff-primary.toml`
- Runnable command: `python scripts/run_handoff.py --input fixtures/handoff_input.json --output out.md`

## Procedure

1. Read any release or incident context files present in the bundle as
   operational input only.
2. Run the canonical handoff command exactly as defined above or via
   `automations/handoff-primary.toml`.
3. Write the generated markdown handoff output to the requested output path.

## Notes

- `automations/handoff-copy.toml` is a retired duplicate alias and must not be
  treated as live.
- Do not route execution through legacy notes or copied prompts.
