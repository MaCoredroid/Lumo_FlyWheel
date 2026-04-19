# Dashboard Brief Capture

Use this skill when the task is to repair or validate an automated dashboard capture flow that feeds a human-readable ops brief.

## Inputs
- Automation config or selector map
- Local dashboard UI with seeded data
- Screenshot manifest or capture directory
- Brief template and operator note

## Workflow
1. Verify which panels are required before editing capture selectors.
2. Reproduce the capture path in the browser, including narrower viewport behavior.
3. Generate fresh screenshot artifacts and record a manifest token proving freshness.
4. Update the brief and operator note only after capture evidence is correct.

## Avoid
- Reusing stale screenshots.
- Capturing the first alert-like panel instead of the required panel.
- Ignoring below-the-fold behavior on narrow layouts.

## Expected output
- Repaired capture mapping or automation
- Fresh screenshot evidence
- Brief and operator note aligned to the live UI
