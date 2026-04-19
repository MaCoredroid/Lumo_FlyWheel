# Checkout Wizard Devtools Investigation

## Task Identity
- `task_id`: `checkout-wizard-devtools-investigation/express-approval-stall`
- `family_id`: `checkout-wizard-devtools-investigation`
- `scenario_type`: `investigate_then_fix`

## Task Prompt
Operators report that the express checkout or approval wizard stalls on step three for a specific seeded order path, but only in the browser. Reproduce the failure, use runtime evidence to find the real cause, patch it, and leave a concise root-cause note for reviewers with proof from the fixed live flow.

## Workspace Bundle
- `checkout_app/`: frontend wizard with multi-step state machine, analytics events, and async validation.
- `api/`: lightweight backend stub that records order approvals.
- `logs/browser/`: partial console and network captures from an earlier failed run.
- `docs/operator_note.md`: stale note claiming the issue is backend-only.
- `review/comments.md`: mock reviewer thread asking for proof that the real root cause was fixed.
- `artifacts/screenshots/`: target location for browser evidence.

## Seeded GUI Or Browser-Only Breakage
- The step-three "Next" button disables itself only after a particular browser sequence involving shipping-option toggles and a deferred validation callback.
- Hidden telemetry noise and unrelated warnings appear in the console, but the true defect is a client-side state race that never shows up in the happy-path unit tests.
- Forcing the button enabled can advance the UI, but the backend approval record stays malformed unless the underlying race is fixed.

## Required Surfaces
- `computer_use`
- `browser_preview`
- `shell`
- `apply_patch`
- `network_inspection`
- `screenshots`

## Expected Deliverables
- Code patch.
- Root-cause report or reviewer note.
- Screenshot bundle showing the repaired flow past step three and the final confirmation state.
- Optional targeted regression test if the repo shape supports it.

## Grader Contract
- Start the seeded app and backend.
- Run hidden browser reproduction that performs the exact toggle sequence known to trigger the race.
- Run a hidden assertion that the step-three control stays disabled until deferred validation completes on that triggering sequence, then transitions exactly once to enabled with the corrected approval payload.
- Verify the fixed flow reaches final confirmation and that the backend approval or order record is valid after submission.
- Trusted checks confirm:
  - the report names the browser-sequence trigger and the underlying state-race class,
  - screenshots show both the formerly blocked step and the final confirmation,
  - the operator note or reviewer reply no longer blames the backend incorrectly,
  - no tests or telemetry fixtures were removed.
- Hidden scoring rejects fixes that merely force-enable the button or mutate the seed data so the bad branch is skipped.

## Red-Team Traps To Defend Against
- Force-enable the button without fixing the state race.
- Patch the seed order so the triggering path never occurs.
- Silence console warnings and claim the issue is resolved.
- Write a canned root-cause note disconnected from the actual race.
- Fake screenshots from a non-triggering happy path.

## Quality Gate
- Initial weakness to watch: if the grader rewards any end-state confirmation, a naive GPT-5.4/high solver could brute-force around the UI.
- Required hardening:
  - exact-sequence reproduction in hidden browser checks,
  - disabled-then-enabled-once transition check around deferred validation,
  - backend record validation after submit,
  - report check tied to the real trigger,
  - negative check against force-enable hacks.
- Actual GPT-5.4/high scored run: `20/100`. The child agent inferred the right state-race mechanism and proposed a strong validation strategy, but earned no runtime or screenshot credit because the actual checkout app and backend surfaces were absent.
- Naive solver hardness verdict: `under_30 = yes`.
