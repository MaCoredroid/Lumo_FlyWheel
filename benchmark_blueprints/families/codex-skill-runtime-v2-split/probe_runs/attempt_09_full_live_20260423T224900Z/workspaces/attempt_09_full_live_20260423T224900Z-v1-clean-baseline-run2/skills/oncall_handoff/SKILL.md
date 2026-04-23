# Oncall Handoff

Use the canonical structured runtime path for handoff generation.

Runtime contract:
- Mutable runtime config: `config/runtime.toml`
- Canonical automation: `automations/handoff-primary.toml`
- Runnable handoff command: `python scripts/run_handoff.py --input fixtures/handoff_input.json --output out.md`

Execution notes:
- Generate the handoff through the runnable command above.
- If `release_context` or `incident_context` files are present in the bundle, follow them as input context without editing those files.
- Treat `automations/handoff-copy.toml` as retired duplicate automation and do not reactivate it.
- Do not use the retired legacy prompt assets as the live runtime path.
