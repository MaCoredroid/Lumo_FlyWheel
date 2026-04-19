---
name: marketplace-bundle-alignment
description: Use when attempting the plugin-marketplace-validation benchmark family. Read the family task spec and evaluator first, then reconcile manifest, install config, and release docs without editing the trusted validator or build harness.
---

# Marketplace Bundle Alignment

1. Read `../../task_spec.md` and `../../evaluator_contract.md` before changing anything.
2. Treat `../../workspace_bundle/` as the only visible benchmark workspace.
3. Align these visible surfaces first:
   - `.codex-plugin/plugin.json`
   - `fixtures/marketplace/sample_install.toml`
   - `docs/plugin_release.md`
4. Do not edit `scripts/validate_bundle.py` or `scripts/build_bundle.py`.
5. Prefer one canonical slug and one canonical startup command across all visible files.
6. Leave a short note about what still appears risky under hidden marketplace validation.
