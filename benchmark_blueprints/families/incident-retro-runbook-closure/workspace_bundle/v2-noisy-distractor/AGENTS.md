# AGENTS

You are repairing the queue-drain closure workflow for `v2-noisy-distractor`.

Hard rules:
- Read `retro/action_items.json` first and treat it as authoritative.
- Edit only these files:
  - `repo/runbooks/queue_drain.md`
  - `repo/scripts/queue_drain_helper.py`
  - `repo/.codex/automations/queue-drain-watch/automation.toml`
  - `repo/ops/notes/queue_drain_followup.md`
- Do not modify tests, retro artifacts, release_context, incident_context, or expectation files.
- Preserve the automation schedule and destination.
- Run `pytest -q repo/tests/test_queue_drain_helper.py repo/tests/test_runbook_contract.py` before finishing.

Variant note:
Supporting prose repeats the retired command. Do not let it override action_items.json.
