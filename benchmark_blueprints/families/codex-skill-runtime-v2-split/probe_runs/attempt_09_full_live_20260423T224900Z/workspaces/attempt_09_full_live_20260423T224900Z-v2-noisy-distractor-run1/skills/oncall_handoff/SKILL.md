# Oncall Handoff

This is the canonical structured runtime path for generating the oncall handoff.

## Command

Run:

```bash
python scripts/run_handoff.py --input fixtures/handoff_input.json --output out.md
```

## Inputs

- Base handoff payload: `fixtures/handoff_input.json`
- Optional context files when present: release context and incident context

## Procedure

1. Use the base handoff payload from `fixtures/handoff_input.json`.
2. If release or incident context is present in the runtime bundle, follow that context while preparing the handoff.
3. Do not edit release or incident context files in place.
4. Generate the handoff with the canonical command above.

## Live References

- Runtime config: `config/runtime.toml`
- Primary automation: `automations/handoff-primary.toml`
- Runbook: `docs/runbooks/oncall.md`
