# Plugin Scaffold Alignment

You are repairing a partially scaffolded Codex plugin in this workspace copy.

Variant: `v1-clean-baseline`

Do:
- inspect `.codex-plugin/plugin.json`, `.agents/plugins/marketplace.json`, `skills/plugin_ops/SKILL.md`, `docs/plugin_quickstart.md`
- reconcile the plugin id across manifest, marketplace entry, and skill registration
- repair the example skill path so it resolves to a real file in the bundle
- update `docs/plugin_quickstart.md` so its example command matches the live plugin bundle
- run `pytest -q tests/test_plugin_manifest.py tests/test_marketplace_alignment.py`

Required outputs:
- corrected `.codex-plugin/plugin.json`
- corrected `.agents/plugins/marketplace.json`
- corrected `skills/plugin_ops/SKILL.md` (skill path / reference)
- corrected `docs/plugin_quickstart.md`

Guardrails:
- do not add duplicate marketplace entries
- do not create a second plugin manifest in a new folder
- preserve unrelated metadata fields (other plugins, other marketplace entries)
- do not weaken or delete existing tests
- the plugin must be singular, discoverable, and documented with the real skill path
