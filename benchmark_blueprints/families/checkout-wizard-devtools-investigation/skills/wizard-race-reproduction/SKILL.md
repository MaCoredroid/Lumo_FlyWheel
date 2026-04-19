# Wizard Race Reproduction

Use this skill when solving `checkout-wizard-devtools-investigation`.

## Objective
Reproduce the exact browser-only stall, identify the real state-race mechanism, and fix it without brute-forcing the control state.

## Required Approach
1. Reproduce the exact shipping-toggle or step-three trigger sequence.
2. Observe the deferred-validation behavior and distinguish signal from telemetry noise.
3. Fix the ordering or state-race issue in the wizard logic.
4. Verify the control remains disabled until validation completes, then enables exactly once.
5. Confirm the backend approval or order record is valid after the final submission.

## Do Not
- Force-enable the button.
- Patch the seed path to avoid the trigger.
- Silence warnings and claim the issue is fixed.
- Produce screenshots from a non-triggering happy path.

## Completion Standard
The task is solved only if the real triggering sequence works, the transition shape is correct, and the backend record is valid.
