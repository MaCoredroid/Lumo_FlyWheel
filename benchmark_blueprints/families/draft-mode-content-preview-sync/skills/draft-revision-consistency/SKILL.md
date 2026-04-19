# Draft Revision Consistency

Use this skill when working on `draft-mode-content-preview-sync`.

## Goal

Treat draft preview as correct only when the body content and visible status both reflect the same draft revision. Banner text alone is not sufficient.

## Required checks

1. Inspect the known draft article in preview mode.
2. Verify draft body freshness.
3. Verify the status chip.
4. Compare against at least one published-only article requested in preview mode.

## What does not count

- Global cache disablement.
- Hardcoded draft content.
- Always forcing draft state under `preview=1`.
- Banner-only fixes.

## Evidence standard

- Prefer evidence that ties body and status to a revision identity or timestamp.
- If cross-article or same-revision evidence is missing, score should remain low.
