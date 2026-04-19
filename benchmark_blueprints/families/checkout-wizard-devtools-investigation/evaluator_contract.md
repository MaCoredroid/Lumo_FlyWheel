# Evaluator Contract

## Scope
This evaluator scores `checkout-wizard-devtools-investigation/express-approval-stall`. A real solve must fix the browser-sequence-triggered state race rather than route around the disabled control.

## 100-Point Breakdown
- `10` points: Correctly identify the trigger and defect class.
  - Must name the step-three toggle sequence.
  - Must identify a client-side state race or deferred-validation ordering issue.
- `10` points: Target the right patch surfaces.
  - Frontend wizard state machine or validation logic.
  - Reviewer or operator-facing root-cause note.
  - Optional targeted regression test.
- `35` points: Implement the real fix.
  - The exact triggering browser sequence no longer stalls.
  - No brute-force force-enable hack is used.
- `20` points: Trusted runtime correctness.
  - Hidden assertion verifies the step-three control stays disabled until deferred validation completes, then transitions exactly once to enabled.
  - Backend approval or order record is valid after submission.
- `10` points: Browser proof.
  - Screenshots show the formerly blocked step and the final confirmation.
- `10` points: Root-cause artifact quality.
  - Report names the triggering sequence and defect class.
  - Reviewer or operator note no longer misattributes the issue to the backend.
- `5` points: Shortcut resistance.
  - No seed mutation, no warning suppression theater, no fake screenshots.

## Hard Caps
- If the solver force-enables the button or bypasses the sequence guard, total score is capped at `15`.
- If the hidden transition-shape assertion fails, total score is capped at `20`.
- If the backend record remains malformed after submit, total score is capped at `20`.
- If the report does not mention the browser trigger and state-race class, artifact-quality points are `0`.

## Pass Shape
- Expected benchmark solve band: `>= 85`.
- Meaningful but incomplete attempt: `15-35`.
- Naive GPT-5.4/high target band during family design: around `20`.

## Trusted Checks
- Hidden browser run reproduces the exact shipping-toggle sequence.
- Trusted assertion verifies disabled-then-enabled-once behavior.
- Trusted backend check validates the approval or order record after submit.
