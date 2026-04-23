# Oncall Runbook

Use `config/runtime.toml` as the mutable runtime entrypoint.

Canonical handoff command:
`python scripts/run_handoff.py --input fixtures/handoff_input.json --output out.md`

Canonical automation:
`automations/handoff-primary.toml`

Rollback note:
INC-4412 rolled back the duplicate heartbeat after `automations/handoff-copy.toml`
was re-enabled and created duplicate pages. Keep `automations/handoff-copy.toml`
retired. Any follow-up escalation handoff must reuse the same canonical command
contract as the primary path instead of restoring a second live automation.
