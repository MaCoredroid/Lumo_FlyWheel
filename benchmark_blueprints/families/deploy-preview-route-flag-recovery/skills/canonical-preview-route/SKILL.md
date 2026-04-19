# Canonical Preview Route

Use this skill when working on `deploy-preview-route-flag-recovery`.

## Goal

Treat the preview route as solved only when the URL resolves, seeded content with a known identity appears, and the feature remains properly gated outside preview overrides.

## Required checks

1. Observe the broken preview URL.
2. Distinguish 404 from empty-shell behavior.
3. Verify seeded content identity after the fix.
4. Confirm the route remains gated outside preview defaults.

## What does not count

- Turning the feature on globally.
- Catch-all redirects that hide the route bug.
- Shell-only fixes without seeded content.
- Dual-key compatibility shims as the final state.

## Evidence standard

- Prefer evidence tying the visible page to a known seeded record.
- If canonical-key or seeded-provenance evidence is missing, score should remain low.
