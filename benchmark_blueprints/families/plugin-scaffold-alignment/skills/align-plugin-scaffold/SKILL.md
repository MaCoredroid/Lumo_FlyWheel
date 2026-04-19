# Align Plugin Scaffold

Use this skill when a partially scaffolded Codex plugin has matching-looking files but still cannot be discovered or has stale path wiring.

## Do

- Trace plugin id, display metadata, marketplace wiring, and skill path together
- Check discoverability rather than only string equality
- Preserve unrelated metadata fields and comments
- Keep docs tied to one real example path

## Do Not

- Do not duplicate marketplace entries or manifests
- Do not stop at mirroring ids across files
- Do not rewrite docs into vague placeholders

## Working Pattern

1. Confirm the single intended plugin id.
2. Confirm the marketplace entry resolves to that plugin.
3. Confirm the documented skill path exists and matches the repaired bundle.
4. Preserve the rest of the metadata untouched.

## Success Signal

The plugin can be discovered through marketplace metadata, the skill path resolves, and only one repaired bundle exists.
