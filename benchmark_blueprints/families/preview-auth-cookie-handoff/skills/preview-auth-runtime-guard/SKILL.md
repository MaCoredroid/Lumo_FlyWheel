# Preview Auth Runtime Guard

Use this skill when working on `preview-auth-cookie-handoff`.

## Goal

Treat hosted preview behavior as the source of truth. The task is not complete when the login form merely submits; it is complete only when a fresh browser profile reaches `/cases`, reload persists the session, and cookie clearing restores the protected redirect.

## Required checks

1. Observe unauthenticated `/cases` redirect behavior.
2. Log in through the preview flow.
3. Confirm authenticated `/cases` loads.
4. Reload and confirm the session persists.
5. Clear cookies and confirm protection returns.

## What does not count

- Preview-only auth bypasses.
- Middleware disablement.
- Arbitrary-origin callback acceptance.
- Wildcard or parent-domain cookie broadening.
- Static screenshots without runtime provenance.

## Evidence standard

- Prefer fresh browser/network evidence.
- Use file-level patch targets only when they are tied back to observed preview behavior.
- If runtime evidence is missing, the attempt should score low under the family evaluator.
