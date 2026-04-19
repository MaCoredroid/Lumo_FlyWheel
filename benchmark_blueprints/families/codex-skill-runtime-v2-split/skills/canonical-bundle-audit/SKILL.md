# Canonical Bundle Audit

Use this skill when a task replaces monolithic Codex instructions with structured skills, config, and automation.

## Workflow
1. Identify the single canonical command path first.
2. Align skill, config, automation, and runbook to that command path.
3. Remove or retire stale duplicates only when their references are dead.
4. Preserve unrelated in-progress edits.

## Anti-Patterns
- Creating plausible files without wiring them together.
- Leaving duplicate automations semantically live.
- Copy-pasting stale prompt text into new bundles.

## Done Signal
- Visible checks and the smoke command both pass from the canonical bundle.
