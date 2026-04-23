# Oncall Runbook

Use `skills/oncall_handoff/SKILL.md` with `automations/handoff-primary.toml`.

Run `python scripts/run_handoff.py --input fixtures/handoff_input.json --output out.md`.

`automations/handoff-copy.toml` stays retired after the INC-4412 duplicate-page
rollback. Follow-up escalations must reuse the primary canonical contract
instead of restoring the duplicate alias.
