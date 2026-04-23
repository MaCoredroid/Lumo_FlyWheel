# Oncall Handoff

Use the structured runtime bundle for the live handoff path.

Primary handoff command:
`python scripts/run_handoff.py --input fixtures/handoff_input.json --output out.md`

Rules:
- Treat `config/runtime.toml` as the mutable runtime entrypoint.
- Use `automations/handoff-primary.toml` as the only canonical automation.
- Keep follow-up escalation handoffs on the same command contract instead of restoring a duplicate automation.
- Record rollback context in `docs/runbooks/oncall.md` and leave incident/release context files unchanged.
