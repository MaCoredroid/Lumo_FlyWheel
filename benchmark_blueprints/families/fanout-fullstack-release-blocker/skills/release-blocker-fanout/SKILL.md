# Release Blocker Fanout

Use this skill when a release-blocking bug spans backend payloads, frontend submission flow, and operator documentation.

## Required workflow

1. Split ownership by backend, frontend, and docs.
2. Verify the rename on the live request path, not only in static code.
3. Treat compatibility shims as suspicious unless tests require them.
4. Capture proof that the submitted payload and persisted or echoed record agree.

## Deliverable checklist

- backend parser and emitted schema aligned
- frontend form and request client aligned
- docs mention retired token, new token, and rollout order
- proof artifact tied to the seeded flow

