# Oncall Handoff

Use the structured runtime entrypoint for this bundle.

Canonical runtime files:
- `config/runtime.toml`
- `automations/handoff-primary.toml`
- `docs/runbooks/oncall.md`

Canonical handoff command:

```bash
python scripts/run_handoff.py --input fixtures/handoff_input.json --output out.md
```

If `release_context` or `incident_context` files are present in the workspace, follow them during handoff preparation without editing those context files.
