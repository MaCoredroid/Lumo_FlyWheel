# Oncall Handoff

Canonical structured runtime path for the live oncall handoff bundle.

Use this sequence:
- Load the mutable runtime pointers from `config/runtime.toml`.
- Run `python scripts/run_handoff.py --input fixtures/handoff_input.json --output out.md`.
- Route live automation through `automations/handoff-primary.toml`.
- Reuse this same contract for follow-up escalations; do not restore the
  retired duplicate alias in `automations/handoff-copy.toml`.
