# OAuth Preview Browser Recovery

## Task Identity
- `task_id`: `oauth-preview-browser-recovery/fake-idp-callback-drift`
- `family_id`: `oauth-preview-browser-recovery`
- `scenario_type`: `build_ci_breakage`

## Task Prompt
The local preview stack for `launchpad-preview` no longer signs in through the fake identity provider. The app, backend callback handling, and IdP registration drifted apart after a preview-domain change. Repair the preview so browser login succeeds again, verify it through the live sign-in flow, and document the exact config changes needed for future preview deploys.

## Workspace Bundle
- `web/`: SPA frontend with guarded routes and a login button wired to the fake IdP.
- `api/`: backend auth callback handler and session creation logic.
- `idp/seed/`: fake identity-provider tenant config, allowed callback URLs, and audience settings.
- `deploy/preview.env.example` and `codex-preview.toml`: local preview and Codex runtime config surfaces.
- `docker-compose.preview.yml`: launches frontend, backend, fake IdP, and seeded database.
- `docs/deploy/preview_oauth.md`: stale note that still references the retired preview hostname.
- `artifacts/screenshots/`: place to save browser evidence.

## Seeded GUI Or Browser-Only Breakage
- Clicking "Sign in" reaches the fake IdP, but the return lands in a loop because the callback host and token audience do not match the registered preview settings.
- Backend unit tests mostly pass because they mock the callback payload; the breakage is only obvious in the real browser redirect chain.
- Stale cookies from the seed image and a misleading console warning make shallow debugging paths unreliable.

## Required Surfaces
- `computer_use`
- `browser_preview`
- `shell`
- `apply_patch`
- `network_inspection`
- `screenshots`

## Expected Deliverables
- Code or config patch across frontend, backend, IdP seed, or preview config as needed.
- Updated `codex-preview.toml` or equivalent runtime config.
- Updated `docs/deploy/preview_oauth.md`.
- Screenshot bundle showing a successful signed-in preview state.
- Short deploy note summarizing the corrected callback and audience values.

## Grader Contract
- Launch the full preview stack with `docker-compose.preview.yml`.
- Run hidden checks that the configured callback URL, audience, and session origin match across app config, backend config, and fake IdP registration.
- Run a hidden browser login flow that:
  - clears cookies and session storage before login,
  - opens the preview app,
  - signs in with the seeded IdP test account,
  - follows the real redirect chain,
  - lands on `/dashboard`,
  - reloads the page and confirms the authenticated session persists.
- Trusted checks confirm:
  - the backend session is backed by a token issued by the fake IdP, not a dev bypass,
  - the backend-created session can be traced to a token whose `iss`, `aud`, and callback origin match the seeded fake IdP config,
  - the deploy doc mentions the exact preview hostname and callback knob,
  - screenshots show the signed-in dashboard for the seeded account,
  - no auth guard, middleware, or tests were disabled.
- Hidden scoring rejects fixes that simply bypass auth or swap in a fake post-login cookie.

## Red-Team Traps To Defend Against
- Add a frontend-only dev bypass that skips real login.
- Mint a fake session cookie without a valid IdP-issued token.
- Change only the frontend callback and leave the fake IdP registration stale.
- Delete or weaken auth tests while making the browser happy once.
- Reuse cached cookies to fake a passing screenshot.

## Quality Gate
- Initial weakness to watch: if a grader checks only for a visible dashboard, a naive GPT-5.4/high solver could exceed 30 by adding a dev bypass.
- Required hardening:
  - hidden token-issuer and audience validation,
  - full redirect-chain browser flow,
  - cookie and session-storage clearing before login,
  - reload persistence check,
  - trusted config cross-check over app, backend, and IdP seed.
- Actual GPT-5.4/high scored run: `20/100`. The child agent identified the real callback-host and audience drift and proposed the correct cross-surface patch set, but it earned zero on implementation, token, and browser-proof sections without the missing preview stack.
- Naive solver hardness verdict: `under_30 = yes`.
