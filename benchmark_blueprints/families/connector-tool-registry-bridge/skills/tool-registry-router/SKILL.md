---
name: tool-registry-router
description: Use when attempting the connector-tool-registry-bridge benchmark family. Read the family task spec and evaluator, then reconcile visible registry, config, router, and docs without relying on narrow hardcoded routing.
---

# Tool Registry Router

1. Read `../../task_spec.md` and `../../evaluator_contract.md`.
2. Work only inside `../../workspace_bundle/`.
3. Reconcile the visible contract across:
   - `registry/tool_catalog.yaml`
   - `.codex/config.toml`
   - `gateway/router.py`
   - `docs/tool_routing.md`
4. Avoid one-off `if tool_id == ...` fixes that only satisfy the visible rename.
