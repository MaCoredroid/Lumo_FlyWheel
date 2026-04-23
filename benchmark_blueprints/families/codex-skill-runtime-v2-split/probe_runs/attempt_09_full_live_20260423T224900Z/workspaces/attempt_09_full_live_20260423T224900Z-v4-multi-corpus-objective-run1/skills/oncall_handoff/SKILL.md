# Oncall Handoff

## Purpose

Produce the live oncall handoff from the structured runtime bundle.

## Canonical Runtime

- Mutable runtime config: `config/runtime.toml`
- Canonical automation: `automations/handoff-primary.toml`
- Runnable handoff command: `python scripts/run_handoff.py --input fixtures/handoff_input.json --output out.md`

## Operator Notes

1. Start from `config/runtime.toml` when checking the active runtime wiring.
2. Run the command from the canonical automation path.
3. If release or incident context files are present in the bundle, follow their guidance without editing those context files.
4. Treat any legacy alias automation as retired and non-live.
