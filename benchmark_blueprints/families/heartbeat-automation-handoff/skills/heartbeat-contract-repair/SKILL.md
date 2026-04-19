---
name: heartbeat-contract-repair
description: Use when attempting the heartbeat-automation-handoff benchmark family. Read the family task spec and evaluator, then reconcile the visible serializer, prompt template, and example artifacts while preserving heartbeat semantics.
---

# Heartbeat Contract Repair

1. Read `../../task_spec.md` and `../../evaluator_contract.md`.
2. Work only inside `../../workspace_bundle/`.
3. Reconcile:
   - `automation/serializer.py`
   - `templates/review_digest_prompt.md.j2`
   - `fixtures/heartbeat/review_digest_expected.toml`
   - `docs/automations/review_digest.md`
4. Do not convert the flow into cron semantics.
