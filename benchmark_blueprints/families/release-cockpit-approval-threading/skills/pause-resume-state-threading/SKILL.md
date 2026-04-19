# Pause Resume State Threading

Use this skill when solving `release-cockpit-approval-threading`.

## Objective
Implement `paused_for_approval` as a real cross-surface state, not a badge alias.

## Required Approach
1. Trace pause and resume semantics through backend transitions, UI rendering, and automation behavior.
2. Preserve resumable provenance when a release is paused.
3. Resume to the exact prior active substate and work item or thread.
4. Validate both the browser-facing release state and the automation behavior.
5. Update runbook and automation artifacts to match the true semantics.

## Do Not
- Add only a UI label or badge mapping.
- Suppress the automation globally.
- Resume to a generic `active` state.
- Rewrite fixtures or screenshots to fake the state.

## Completion Standard
The task is solved only if pause preserves resumable provenance, resume restores the exact prior target, and the live cockpit plus automation both reflect that behavior.
