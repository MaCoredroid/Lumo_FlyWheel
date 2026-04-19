# Network Origin Discipline

Use this skill when working on `preview-form-webhook-cors`.

## Goal

Treat the preview browser network path as authoritative. A passing solution must repair preview submission, preserve validation behavior, and keep cross-origin policy narrow and intentional.

## Required checks

1. Observe the real preview request target from the browser.
2. Check preflight behavior.
3. Check a valid submission.
4. Check an invalid submission and its browser-visible validation response.

## What does not count

- Pointing preview to production.
- Wildcard or reflected-origin CORS.
- Client-only fake success.
- Validation removal.

## Evidence standard

- Prefer browser-network evidence over config text alone.
- If runtime/network evidence is missing, score should be capped low.
