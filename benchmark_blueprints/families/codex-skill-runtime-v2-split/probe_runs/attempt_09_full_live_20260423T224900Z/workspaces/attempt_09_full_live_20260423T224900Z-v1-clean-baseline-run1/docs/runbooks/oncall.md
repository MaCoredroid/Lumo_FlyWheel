# Oncall Runbook

Use `config/runtime.toml` and `automations/handoff-primary.toml`.

Run the handoff with:

```bash
python scripts/run_handoff.py --input fixtures/handoff_input.json --output out.md
```

If release or incident context files are present, follow them during the handoff without editing those context files.
