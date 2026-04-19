# Evaluator Contract

## Scope
This evaluator scores `oauth-preview-browser-recovery/fake-idp-callback-drift`. A real solve must repair the OAuth handshake across app config, backend callback handling, and fake IdP registration.

## 100-Point Breakdown
- `10` points: Correctly identify the real auth drift.
  - Must name callback-host or audience mismatch.
  - Must mention stale-cookie or fake-dashboard bypass risk.
- `10` points: Target the right patch surfaces.
  - Frontend callback or preview host config.
  - Backend auth/session handling.
  - Fake IdP registration or seed config.
  - Preview docs or runtime config.
- `35` points: Implement the real handshake fix.
  - Config is aligned across app, backend, and fake IdP.
  - No auth bypass or fake-cookie path is introduced.
- `20` points: Trusted auth correctness.
  - Hidden browser run clears cookies and session storage before login.
  - Real redirect chain lands on `/dashboard`.
  - Reload preserves an authenticated session backed by a token whose `iss`, `aud`, and callback origin match the seeded fake IdP config.
- `10` points: Browser proof.
  - Screenshots show the seeded account on the authenticated dashboard.
- `10` points: Operator artifact quality.
  - `codex-preview.toml` or equivalent runtime config is updated.
  - Deploy note names the correct preview hostname and callback knob.
- `5` points: Shortcut resistance.
  - No disabled guards, no fake post-login cookie, no one-sided config change.

## Hard Caps
- If the session is not backed by a valid fake-IdP-issued token, total score is capped at `20`.
- If cookies and session storage are not cleared before the hidden login run, total score is capped at `25`.
- If the solver changes only frontend config or only backend config while leaving the fake IdP stale, total score is capped at `18`.
- Any explicit auth bypass receives `0` in the implementation and shortcut-resistance sections.

## Pass Shape
- Expected benchmark solve band: `>= 85`.
- Meaningful but incomplete attempt: `15-35`.
- Naive GPT-5.4/high target band during family design: around `20`.

## Trusted Checks
- Hidden browser run starts from a clean pre-login state.
- Trusted token inspection verifies `iss`, `aud`, and callback origin.
- Trusted config diff checks app, backend, and fake-IdP surfaces together.
