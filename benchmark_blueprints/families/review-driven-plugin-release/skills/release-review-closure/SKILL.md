---
name: release-review-closure
description: Use when attempting the review-driven-plugin-release benchmark family. Read the family task spec and evaluator, then make the smallest visible manifest, UI, docs, and review-artifact changes that address actionable feedback without faking evidence.
---

# Release Review Closure

1. Read `../../task_spec.md` and `../../evaluator_contract.md`.
2. Work only inside `../../workspace_bundle/`.
3. Prioritize:
   - `.codex-plugin/plugin.json`
   - `drive_brief/ui/settings_panel.tsx`
   - `docs/release_notes.md`
   - `docs/release_checklist.md`
   - `review/unresolved_threads.json`
4. Do not treat the resolved red-herring thread as actionable.
5. Do not fake screenshot success by rebaselining artifacts alone.
