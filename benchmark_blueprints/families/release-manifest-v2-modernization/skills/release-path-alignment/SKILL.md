# Release Path Alignment

Use this skill when a task modernizes release workflows, manifests, and deploy smoke behavior together.

## Workflow
1. Verify the live dry-run path before editing docs.
2. Align workflow contract, manifest fields, and environment names.
3. Treat staging smoke as a semantic gate, not a cosmetic step.
4. Preserve unrelated release-note and environment edits.

## Anti-Patterns
- Syntax-only workflow migration.
- Bypassing smoke verification.
- Updating docs/config while workflow and manifest remain inconsistent.

## Done Signal
- Visible checks pass and the release path is semantically aligned.
