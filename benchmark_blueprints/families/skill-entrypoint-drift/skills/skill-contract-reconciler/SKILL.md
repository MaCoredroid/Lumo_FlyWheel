---
name: skill-contract-reconciler
description: Use when attempting the skill-entrypoint-drift benchmark family. Read the family task spec and evaluator, then make the visible skill, example, and helper contract agree without weakening the smoke harness.
---

# Skill Contract Reconciler

1. Read `../../task_spec.md` and `../../evaluator_contract.md`.
2. Work only inside `../../workspace_bundle/`.
3. Reconcile:
   - `skills/ops-digest/SKILL.md`
   - `skills/ops-digest/examples/weekly_digest.md`
   - `tools/digest_runner.py`
4. Do not weaken `scripts/smoke_skill.py`.
5. Prefer one clearly runnable command path and one clearly documented output shape.
