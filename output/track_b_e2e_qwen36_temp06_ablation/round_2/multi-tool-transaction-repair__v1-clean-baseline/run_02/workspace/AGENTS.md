# Agent Instructions - multi-tool-transaction-repair

Repair the local transaction workflow while preserving atomicity and avoiding duplicate notification side effects.

Expected work:
- Patch service code under `services/`.
- Update `docs/transaction_ops.md`.
- Run `pytest -q tests/test_transaction_flow.py tests/test_atomicity.py`.

Do not edit `fixtures/`, `tests/`, `.scenario_variant`, or `AGENTS.md`.
