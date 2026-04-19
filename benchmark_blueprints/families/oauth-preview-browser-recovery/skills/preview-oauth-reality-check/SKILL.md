# Preview OAuth Reality Check

Use this skill when solving `oauth-preview-browser-recovery`.

## Objective
Repair the real preview OAuth handshake, not just the appearance of a successful login.

## Required Approach
1. Clear browser cookies and session storage before validation.
2. Follow the real redirect chain through the fake identity provider.
3. Align frontend config, backend callback handling, and fake IdP registration.
4. Validate that the resulting session comes from a real token with matching `iss`, `aud`, and callback origin.
5. Update runtime config and deploy notes with the exact corrected values.

## Do Not
- Add a dev-only post-login bypass.
- Mint or inject a fake session cookie.
- Fix only one side of the handshake.
- Treat a single dashboard render as sufficient evidence.

## Completion Standard
The task is solved only if a clean browser session can log in through the fake IdP, reload on `/dashboard`, and remain authenticated with trusted token provenance.
