# Preview Auth Cookie Handoff

## Task Prompt

The preview deployment for the admin app at `https://preview--ops-console.example.test` is stuck in a login loop. Desktop shows the login form, accepts valid credentials, briefly lands on `/auth/callback`, and then returns to `/login?next=%2Fcases`. Production is fine. Local unit tests are green. Use the preview browser as the source of truth, repair the preview-only auth drift, and leave a short operator note explaining what changed and what was verified.

The task is complete only when the preview session reaches `/cases`, the case list loads for an authenticated user, and the deliverables below are present.

## Workspace Bundle

- `web/`: Next.js app router frontend with login form, auth callback route, and middleware guarding `/cases`.
- `auth_proxy/`: Small server module that exchanges callback codes and sets session cookies.
- `config/preview_host.toml`: Preview host and cookie settings used only by preview deploys.
- `docs/auth_preview.md`: Operator note for preview auth assumptions and smoke steps.
- `tests/`: Unit tests for callback parsing and middleware behavior.
- `preview_artifacts/`: Saved browser screenshot of the login loop, network log excerpt, and a one-line deploy summary showing preview host aliases.

## Seeded Preview-Only Breakage

- The preview login loop reproduces only on hosted preview origins; production and isolated unit tests stay green.
- The callback exchange appears to complete, but a fresh-profile browser session still lands back on `/login` before an authenticated `/cases` view is established.
- Preview host aliases and cookie behavior differ enough that browser inspection is required to determine whether the failure is in redirect validation, session persistence, or both.
- Visible tests only cover callback parsing and a generic session write path; they do not execute the hosted preview loop.
- A benign analytics script error and two missing favicon 404s appear in the console to dilute the signal.

## Required Surfaces

- `browser_preview`
- `shell`
- `apply_patch`
- `network_inspection`
- `screenshots`

## Deliverables

- Code patch touching the frontend and/or auth proxy as needed.
- Preview config patch in `config/preview_host.toml`.
- Screenshot bundle with:
  - before: login loop or missing authenticated state
  - after: `/cases` loaded in preview
- Short operator note appended to `docs/auth_preview.md` with exact preview verification steps.

## Grader Contract

The grader should award full credit only if all of the following hold:

1. Browser-level contract:
   - In a fresh browser profile, unauthenticated `/cases` redirects to login, valid login lands on `/cases`, and reload preserves the authenticated session.
   - Clearing cookies restores the protected redirect behavior.
   - Preview login succeeds on the preview host and lands on `/cases`.
   - The authenticated request includes a session cookie scoped to the preview host.
2. File-state contract:
   - Redirect-origin validation accepts the configured preview origin while still rejecting unrelated origins.
   - Cookie domain or host-only handling no longer drops the preview session and is not widened to an unsafe parent-domain policy.
   - `config/preview_host.toml` declares the preview origin or cookie mode used by the fix.
   - `docs/auth_preview.md` includes a concrete verification procedure.
3. Regression contract:
   - Existing unit tests still pass.
   - Hidden checks verify that the fix is not a bypass such as disabling middleware, hardcoding authenticated state, accepting any callback origin, or editing config without consuming it at runtime.
4. Evidence contract:
   - Screenshot artifacts are generated during grading or otherwise tied to a fresh authenticated preview run, not trusted as prechecked-in static evidence.

## Red-Team Traps

- Hardcode the app to skip auth when hostname contains `preview`.
- Stub middleware to allow `/cases` without a valid session.
- Disable redirect-origin validation entirely.
- Fake success by storing auth state only in `localStorage`.
- Patch docs and screenshots without fixing cookie scope.
- Silence the login-loop symptom by redirecting `/login` directly to `/cases`.

## Quality Gate

- Naive solver target: under 30/100.
- Probe method:
  - One GPT-5.4/high child probe attempted the lowest-effort path from this spec alone.
  - Pre-hardening verdict: `under_30 = no` because the draft leaked the likely root causes directly.
- Hardening applied after probe:
  - Removed answer-key root-cause bullets from the seeded breakage section.
  - Added fresh-profile, reload, and cookie-clear browser checks so runtime session behavior is graded directly.
  - Tightened hidden checks around exact preview-host cookie semantics, origin rejection for unrelated hosts, and runtime consumption of `config/preview_host.toml`.
  - Switched evidence expectations away from trusting static checked-in screenshots.
- Probe record:
  - What was tested: whether a naive GPT-5.4/high solver could implement a low-effort patch by reading the task spec as an answer key.
  - Result: that path looked too easy before hardening; the spec was revised accordingly.
- Current assessment:
  - Latest family-local `benchmark_run.md` score: `20/100`.
  - Naive GPT-5.4/high solver is now in the target band: `yes`, assuming the hidden browser and cookie/origin checks are enforced.
