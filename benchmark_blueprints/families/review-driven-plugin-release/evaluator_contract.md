# Evaluator Contract

## Scope
- Family: `review-driven-plugin-release`
- Task: `t9_review_driven_plugin_release_drive_brief`
- Visible working area: `workspace_bundle/`
- Scoring target for a naive `gpt-5.4` / `high` solver: `15-25/100`

## Scoring Breakdown
- `5` points: restores the obviously missing compatibility field in the visible manifest.
- `5` points: attempts a visible UI fix for the fallback toggle instead of docs-only edits.
- `5` points: updates visible release notes or checklist so they are less contradictory.
- `5` points: provides a bounded review reply that distinguishes actionable from non-actionable visible feedback.
- `20` points visible cap: no visible-only submission may exceed `20/100`.
- `20` points hidden: compatibility field is restored with exact name, type, and release-window behavior.
- `20` points hidden: UI fix survives two-state screenshot verification, including the optional-field-unset state.
- `20` points hidden: release evidence output agrees with docs and screenshot artifacts without rebaselining.
- `20` points hidden: only the two actionable unresolved thread ids are addressed; the resolved red herring is not treated as work.

## Integrity Rules
- Zero credit for screenshot rebaselining without a corresponding UI code fix.
- No credit above the visible cap unless manifest, UI, and evidence all agree under hidden checks.

## Hidden Checks
- Render the settings panel in two states.
- Reject broad manifest rewrites.
- Reject review replies that resolve the wrong thread ids.

## Score Interpretation
- `0-10`: manifest-only, docs-only, or fake-evidence attempt.
- `15-25`: meaningful visible repair but still misses hidden compatibility, screenshot, or thread-id checks.
- `30+`: too easy; harden the task and rerun.
