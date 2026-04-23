
# AGENTS

Repair the plugin loader as a real ESM migration, not a docs-only cleanup.

Constraints:
- Keep mixed-module helpers working.
- Regenerate CLI evidence from the provided capture script.
- Do not edit tests, plugins, repo evidence, release context, or incident context.
- The built `dist/` output must be runnable on its own.
